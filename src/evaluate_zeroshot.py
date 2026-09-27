import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm

from normalizer import clean_text
from metrics import evaluate_benchmark, FullBenchmarkReport

# Гарантия корректного вывода UTF-8 (азербайджанские символы: ə, ş, ç, ğ, ı, ö, ü) в Windows PowerShell
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def load_audio_16k(path: str) -> np.ndarray:
    """
    Надежная загрузка аудио с дискретизацией 16 кГц в память (float32 numpy array).
    Не требует внешнего системного ffmpeg в PATH на Windows.
    """
    try:
        if path.lower().endswith(".wav"):
            data, sr = sf.read(path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)  # Стерео -> моно
            if sr != 16000:
                data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            return data
        else:
            data, _ = librosa.load(path, sr=16000, mono=True)
            return data
    except Exception as e:
        # Универсальный fallback
        data, _ = librosa.load(path, sr=16000, mono=True)
        return data


def get_audio_duration(path: str) -> float:
    """Быстрое чтение длительности аудио в секундах из метаданных файла."""
    try:
        info = sf.info(path)
        return float(info.duration)
    except Exception:
        try:
            return float(librosa.get_duration(path=path))
        except Exception:
            return 0.0


def resolve_audio_path(path: str) -> str:
    """
    Автоматическое самовосстановление пути, если репозиторий перенесли в другую папку.
    """
    if os.path.exists(path):
        return path
    normalized = str(path).replace("\\", "/")
    if "/data/" in normalized:
        rel = normalized.split("/data/", 1)[1]
        candidate = os.path.join("data", rel.replace("/", os.sep))
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return path


def get_clean_names(model_id: str, manifest_path: str):
    """
    Возвращает короткие чистые имена для папок:
    openai/whisper-tiny -> whisper-tiny
    data/manifests/test_common_voice.csv -> common_voice
    """
    model_name = model_id.split("/")[-1].lower()
    stem = Path(manifest_path).stem.lower()
    dataset_name = stem.replace("test_", "")
    return model_name, dataset_name


def sanitize_name(name: str) -> str:
    """Преобразование model_id в безопасное имя файла (openai/whisper-tiny -> whisper-tiny)."""
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")


def load_asr_pipeline(model_id: str, device: str = "cuda", batch_size: int = 16):
    """
    Загрузка Hugging Face ASR pipeline для Whisper или Meta MMS с оптимизацией FP16 на GPU.
    """
    torch_device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    torch_dtype = torch.float16 if torch_device.startswith("cuda") else torch.float32

    print(f"\n🚀 Загрузка модели: {model_id} на {torch_device.upper()} (dtype={torch_dtype})...")

    is_mms = "mms" in model_id.lower()

    if is_mms:
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id).to(torch_device)
        # Установка целевого азербайджанского языка для MMS
        processor.tokenizer.set_target_lang("aze-script_latin")
        model.load_adapter("aze-script_latin")

        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=torch_device,
            batch_size=batch_size,
        )
        generate_kwargs = {}
    else:
        # Семейство Whisper
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(torch_device)

        processor = AutoProcessor.from_pretrained(model_id)

        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=torch_device,
            chunk_length_s=30,
            batch_size=batch_size,
        )
        # Whisper: принудительно задаем азербайджанский язык и задачу транскрипции
        generate_kwargs = {
            "language": "azerbaijani",
            "task": "transcribe"
        }

    return asr_pipe, generate_kwargs, torch_device


def run_evaluation(
    model_id: str,
    manifest_path: str,
    asr_pipe,
    generate_kwargs: dict,
    results_dir: str = "results",
    batch_size: int = 16,
) -> FullBenchmarkReport:
    """
    Запуск инференса на одном манифесте, сбор предикшенов и расчет всех метрик.
    """
    model_folder, dataset_folder = get_clean_names(model_id, manifest_path)
    
    # Иерархическая папка: results/zeroshot/<model>/<dataset>/
    dataset_out_dir = os.path.join(results_dir, "zeroshot", model_folder, dataset_folder)
    model_out_dir = os.path.join(results_dir, "zeroshot", model_folder)
    os.makedirs(dataset_out_dir, exist_ok=True)

    df = pd.read_csv(manifest_path)
    required_cols = {"audio_path", "sentence"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Манифест {manifest_path} должен содержать колонки: {required_cols}")

    print(f"\n📋 Оценка бенчмарка: [{dataset_folder}] для модели [{model_folder}] ({len(df)} аудио)")

    audio_paths = [resolve_audio_path(p) for p in df["audio_path"].tolist()]
    references = df["sentence"].astype(str).tolist()

    # Быстрый параллельный замер длительностей аудио
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        durations = list(ex.map(get_audio_duration, audio_paths))

    # Многопоточный генератор для параллельной загрузки аудио с диска (CPU не ждет GPU)
    def audio_dict_stream(paths: List[str]) -> Generator[Dict[str, Any], None, None]:
        with ThreadPoolExecutor(max_workers=8) as executor:
            for audio_array in executor.map(load_audio_16k, paths):
                yield {"raw": audio_array, "sampling_rate": 16000}

    # Замер времени чистого инференса модели
    print(f"⏳ Запуск инференса модели (batch_size={batch_size})...")
    start_time = time.perf_counter()

    hypotheses = []
    stream = audio_dict_stream(audio_paths)
    for out in tqdm(asr_pipe(stream, generate_kwargs=generate_kwargs, batch_size=batch_size), total=len(audio_paths), desc=f"Eval {dataset_folder}"):
        hypotheses.append(out.get("text", ""))

    processing_time_sec = time.perf_counter() - start_time

    # Расчет всех метрик через metrics.py
    report = evaluate_benchmark(
        references=references,
        hypotheses=hypotheses,
        durations=durations,
        processing_time_sec=processing_time_sec,
        total_audio_sec=sum(durations),
        model_name=model_id,
        dataset_name=dataset_folder,
        apply_normalization=True
    )

    # 1. Сохранение детальных предсказаний построчно: results/zeroshot/<model>/<dataset>/predictions.csv
    pred_records = []
    for idx in range(len(audio_paths)):
        ref_clean = clean_text(references[idx])
        hyp_clean = clean_text(hypotheses[idx])
        exact = (ref_clean == hyp_clean and len(ref_clean) > 0)
        pred_records.append({
            "index": idx,
            "audio_path": audio_paths[idx],
            "ground_truth": references[idx],
            "normalized_reference": ref_clean,
            "hypothesis": hypotheses[idx],
            "normalized_hypothesis": hyp_clean,
            "duration_sec": round(durations[idx], 2),
            "is_exact_match": exact
        })
    pred_df = pd.DataFrame(pred_records)
    preds_file = os.path.join(dataset_out_dir, "predictions.csv")
    pred_df.to_csv(preds_file, index=False, encoding="utf-8")

    # 2. Сохранение полного JSON отчета: results/zeroshot/<model>/<dataset>/metrics.json
    json_file = os.path.join(dataset_out_dir, "metrics.json")
    report.to_json(json_file)

    # 3. Добавление / Обновление в локальную сводку модели: results/zeroshot/<model>/summary.csv
    model_summary_file = os.path.join(model_out_dir, "summary.csv")
    summary_row = {
        "model": model_folder,
        "dataset": dataset_folder,
        "samples": report.samples_count,
        "audio_hours": round(report.total_audio_sec / 3600.0, 2),
        "wer": round(report.wer, 2),
        "wer_ci_95": f"[{report.sentence_stats.wer_ci_95_lower:.2f} - {report.sentence_stats.wer_ci_95_upper:.2f}]",
        "cer": round(report.cer, 2),
        "sub_rate": round(report.wer_details.sub_rate, 2),
        "del_rate": round(report.wer_details.del_rate, 2),
        "ins_rate": round(report.wer_details.ins_rate, 2),
        "wil": round(report.wil, 2),
        "exact_match_pct": round(report.exact_match_rate, 2),
        "rtf": round(report.rtf, 4),
        "speedup": round(report.speedup, 1)
    }

    if os.path.exists(model_summary_file):
        m_df = pd.read_csv(model_summary_file)
        mask = (m_df["dataset"] == dataset_folder)
        m_df = m_df[~mask]
        m_df = pd.concat([m_df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        m_df = pd.DataFrame([summary_row])
    m_df.to_csv(model_summary_file, index=False, encoding="utf-8")

    # 4. Добавление / Обновление в глобальную сводную таблицу: results/benchmark_summary.csv
    global_summary_file = os.path.join(results_dir, "benchmark_summary.csv")
    if os.path.exists(global_summary_file):
        summary_df = pd.read_csv(global_summary_file)
        mask = (summary_df["model"] == model_folder) & (summary_df["dataset"] == dataset_folder)
        summary_df = summary_df[~mask]
        summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([summary_row])
    summary_df.to_csv(global_summary_file, index=False, encoding="utf-8")

    # 5. Добавление строки в LaTeX-файл (results/latex_tables.txt)
    latex_file = os.path.join(results_dir, "latex_tables.txt")
    with open(latex_file, "a", encoding="utf-8") as f:
        f.write(f"% {model_folder} on {dataset_folder}\n")
        f.write(report.to_latex_row() + "\n")

    # Печать красивой сводки
    print(f"\n{'='*70}")
    print(f"📊 РЕЗУЛЬТАТЫ: {model_folder} | {dataset_folder}")
    print(f"{'='*70}")
    print(f" • WER:             {report.wer:.2f}% (95% CI: [{report.sentence_stats.wer_ci_95_lower:.2f} - {report.sentence_stats.wer_ci_95_upper:.2f}])")
    print(f" • CER:             {report.cer:.2f}%")
    print(f" • Exact Match:     {report.exact_match_rate:.2f}% ({report.sentence_stats.exact_matches}/{report.samples_count})")
    print(f" • Errors (S/D/I):  Sub={report.wer_details.sub_rate:.1f}% | Del={report.wer_details.del_rate:.1f}% | Ins={report.wer_details.ins_rate:.1f}%")
    print(f" • RTF / Speed:     RTF={report.rtf:.4f} ({report.speedup:.1f}x real-time)")
    print(f"💾 Файлы сохранены:")
    print(f"   ├─ {preds_file}")
    print(f"   ├─ {json_file}")
    print(f"   └─ {model_summary_file}")
    print(f"{'='*70}\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot ASR Benchmark Evaluation for Azerbaijani")
    parser.add_argument("--model_id", type=str, default="openai/whisper-tiny", help="HuggingFace model ID")
    parser.add_argument("--manifest", type=str, default=None, help="Path to single test CSV manifest")
    parser.add_argument("--all_manifests", action="store_true", help="Run benchmark across all 3 test domains")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device (cuda or cpu)")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for output results")

    args = parser.parse_args()

    # Определение списка манифестов
    if args.all_manifests or args.manifest is None:
        manifest_list = [
            "data/manifests/test_common_voice.csv",
            "data/manifests/test_fleurs.csv",
            "data/manifests/test_localdoc.csv"
        ]
    else:
        manifest_list = [args.manifest]

    # Проверка наличия манифестов
    for m in manifest_list:
        if not os.path.exists(m):
            raise FileNotFoundError(f"Файл манифеста не найден: {m}. Сначала запустите create_manifests.py!")

    # Инициализация пайплайна модели
    asr_pipe, generate_kwargs, device = load_asr_pipeline(
        model_id=args.model_id,
        device=args.device,
        batch_size=args.batch_size
    )

    # Прогон по каждому датасету
    for manifest_path in manifest_list:
        run_evaluation(
            model_id=args.model_id,
            manifest_path=manifest_path,
            asr_pipe=asr_pipe,
            generate_kwargs=generate_kwargs,
            results_dir=args.results_dir,
            batch_size=args.batch_size
        )

    print("\n🎉 Все запланированные бенчмарки успешно завершены!")


if __name__ == "__main__":
    main()
