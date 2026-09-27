import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import jiwer
from normalizer import clean_text

# Специфические фонемы / символы азербайджанского алфавита для отдельного аудита
AZ_SPECIAL_LETTERS = ["ə", "ı", "ğ", "ş", "ç", "ö", "ü"]

@dataclass
class WERDecomposition:
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    total_words: int
    sub_rate: float
    del_rate: float
    ins_rate: float
    wil: float  # Word Information Lost
    wip: float  # Word Information Preserved

@dataclass
class CERDecomposition:
    cer: float
    char_substitutions: int
    char_deletions: int
    char_insertions: int
    char_hits: int
    total_chars: int

@dataclass
class SentenceLevelStats:
    total_sentences: int
    exact_matches: int
    exact_match_rate: float
    ser: float  # Sentence Error Rate
    wer_mean: float
    wer_std: float
    wer_median: float
    wer_ci_95_lower: float  # 95% Bootstrap Confidence Interval Lower
    wer_ci_95_upper: float  # 95% Bootstrap Confidence Interval Upper

@dataclass
class FullBenchmarkReport:
    model_name: str
    dataset_name: str
    samples_count: int
    total_audio_sec: float
    processing_time_sec: float
    rtf: float
    speedup: float
    # Основные результаты
    wer: float
    cer: float
    wil: float
    ser: float
    exact_match_rate: float
    # Подробности
    wer_details: WERDecomposition
    cer_details: CERDecomposition
    sentence_stats: SentenceLevelStats
    duration_stratification: Dict[str, Any]
    az_letter_accuracies: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        s = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(s)
        return s

    def to_latex_row(self) -> str:
        """
        Готовая строка для LaTeX-таблицы в статью:
        Model & WER (95% CI) & CER & Exact Match & WIL & RTF
        """
        ci_str = f"{self.wer:.1f} \\footnotesize{{[{self.sentence_stats.wer_ci_95_lower:.1f}-{self.sentence_stats.wer_ci_95_upper:.1f}]}}"
        return f"{self.model_name} & {ci_str} & {self.cer:.2f}\\% & {self.exact_match_rate:.1f}\\% & {self.wil:.2f}\\% & {self.rtf:.4f} \\\\"


def compute_bootstrap_ci(
    sentence_wers: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Вычисление 95% доверительного интервала (Bootstrap Confidence Interval)
    для WER (стандарт топовых публикаций ACL / Interspeech).
    """
    if len(sentence_wers) == 0:
        return (0.0, 0.0)

    n = len(sentence_wers)
    # Генерируем бутстрэп-выборки
    rng = np.random.default_rng(seed=42)
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = np.mean(sentence_wers[boot_indices], axis=1)

    lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    return (round(lower, 2), round(upper, 2))


def compute_detailed_wer(references: List[str], hypotheses: List[str]) -> WERDecomposition:
    out = jiwer.process_words(references, hypotheses)
    total_words = int(out.hits + out.substitutions + out.deletions)

    sub_rate = round((out.substitutions / total_words) * 100.0, 2) if total_words > 0 else 0.0
    del_rate = round((out.deletions / total_words) * 100.0, 2) if total_words > 0 else 0.0
    ins_rate = round((out.insertions / total_words) * 100.0, 2) if total_words > 0 else 0.0

    return WERDecomposition(
        wer=round(out.wer * 100.0, 2),
        substitutions=int(out.substitutions),
        deletions=int(out.deletions),
        insertions=int(out.insertions),
        hits=int(out.hits),
        total_words=total_words,
        sub_rate=sub_rate,
        del_rate=del_rate,
        ins_rate=ins_rate,
        wil=round(getattr(out, "wil", 0.0) * 100.0, 2),
        wip=round(getattr(out, "wip", 0.0) * 100.0, 2),
    )


def compute_detailed_cer(references: List[str], hypotheses: List[str]) -> CERDecomposition:
    out = jiwer.process_characters(references, hypotheses)
    total_chars = int(out.hits + out.substitutions + out.deletions)

    return CERDecomposition(
        cer=round(out.cer * 100.0, 2),
        char_substitutions=int(out.substitutions),
        char_deletions=int(out.deletions),
        char_insertions=int(out.insertions),
        char_hits=int(out.hits),
        total_chars=total_chars,
    )


def compute_sentence_stats(references: List[str], hypotheses: List[str]) -> SentenceLevelStats:
    total = len(references)
    sentence_wers = []
    exact_matches = 0

    for r, h in zip(references, hypotheses):
        if r == h:
            exact_matches += 1
            sentence_wers.append(0.0)
        else:
            w = jiwer.wer(r, h)
            sentence_wers.append(min(w * 100.0, 100.0))

    arr = np.array(sentence_wers)
    ser = round(((total - exact_matches) / total) * 100.0, 2) if total > 0 else 0.0
    exact_rate = round((exact_matches / total) * 100.0, 2) if total > 0 else 0.0

    ci_low, ci_high = compute_bootstrap_ci(arr, n_bootstrap=1000)

    return SentenceLevelStats(
        total_sentences=total,
        exact_matches=exact_matches,
        exact_match_rate=exact_rate,
        ser=ser,
        wer_mean=round(float(np.mean(arr)), 2),
        wer_std=round(float(np.std(arr)), 2),
        wer_median=round(float(np.median(arr)), 2),
        wer_ci_95_lower=ci_low,
        wer_ci_95_upper=ci_high
    )


def analyze_az_phonemes(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    accuracies = {}
    for letter in AZ_SPECIAL_LETTERS:
        target_count = 0
        diff_count = 0
        for r, h in zip(references, hypotheses):
            c_ref = r.count(letter)
            c_hyp = h.count(letter)
            target_count += c_ref
            diff_count += abs(c_ref - c_hyp)

        if target_count > 0:
            acc = max(0.0, round((1.0 - (diff_count / target_count)) * 100.0, 2))
        else:
            acc = 100.0
        accuracies[letter] = acc
    return accuracies


def stratify_by_duration(
    references: List[str],
    hypotheses: List[str],
    durations: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Стратификация ошибок по длительности аудио:
    - short: < 3 сек
    - medium: 3 - 8 сек
    - long: > 8 сек (где у Whisper часто возникают галлюцинации)
    """
    if not durations or len(durations) != len(references):
        return {}

    buckets = {"short (<3s)": [], "medium (3-8s)": [], "long (>8s)": []}

    for r, h, d in zip(references, hypotheses, durations):
        if d < 3.0:
            buckets["short (<3s)"].append((r, h))
        elif d <= 8.0:
            buckets["medium (3-8s)"].append((r, h))
        else:
            buckets["long (>8s)"].append((r, h))

    summary = {}
    for b_name, pairs in buckets.items():
        if pairs:
            b_refs = [p[0] for p in pairs]
            b_hyps = [p[1] for p in pairs]
            b_wer = round(jiwer.wer(b_refs, b_hyps) * 100.0, 2)
            summary[b_name] = {"count": len(pairs), "wer": b_wer}
        else:
            summary[b_name] = {"count": 0, "wer": None}

    return summary


def evaluate_benchmark(
    references: List[str],
    hypotheses: List[str],
    durations: Optional[List[float]] = None,
    processing_time_sec: float = 0.0,
    total_audio_sec: float = 0.0,
    model_name: str = "Unknown Model",
    dataset_name: str = "Unknown Dataset",
    apply_normalization: bool = True
) -> FullBenchmarkReport:
    """
    Флагманская точка входа для оценки любого эксперимента.
    """
    if apply_normalization:
        clean_refs = [clean_text(r) for r in references]
        clean_hyps = [clean_text(h) for h in hypotheses]
    else:
        clean_refs = references
        clean_hyps = hypotheses

    # Фильтрация пустых
    valid_data = [(r, h, d) for r, h, d in zip(clean_refs, clean_hyps, durations or [0.0] * len(clean_refs)) if len(r.strip()) > 0]
    final_refs = [x[0] for x in valid_data]
    final_hyps = [x[1] for x in valid_data]
    final_durs = [x[2] for x in valid_data]

    # Расчет всех блоков метрик
    wer_res = compute_detailed_wer(final_refs, final_hyps)
    cer_res = compute_detailed_cer(final_refs, final_hyps)
    sent_res = compute_sentence_stats(final_refs, final_hyps)
    phoneme_accs = analyze_az_phonemes(final_refs, final_hyps)
    strat_res = stratify_by_duration(final_refs, final_hyps, final_durs)

    rtf = round(processing_time_sec / total_audio_sec, 4) if total_audio_sec > 0 else 0.0
    speedup = round(1.0 / rtf, 1) if rtf > 0 else 0.0

    return FullBenchmarkReport(
        model_name=model_name,
        dataset_name=dataset_name,
        samples_count=len(final_refs),
        total_audio_sec=round(total_audio_sec, 2),
        processing_time_sec=round(processing_time_sec, 2),
        rtf=rtf,
        speedup=speedup,
        wer=wer_res.wer,
        cer=cer_res.cer,
        wil=wer_res.wil,
        ser=sent_res.ser,
        exact_match_rate=sent_res.exact_match_rate,
        wer_details=wer_res,
        cer_details=cer_res,
        sentence_stats=sent_res,
        duration_stratification=strat_res,
        az_letter_accuracies=phoneme_accs
    )


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 70)
    print("🔬 ТЕСТИРОВАНИЕ АКАДЕМИЧЕСКОГО МОДУЛЯ ОЦЕНКИ ASR (PhD / Interspeech Level)")
    print("=" * 70)

    demo_refs = [
        "bizim evimizdə qonaqlar var və çay içirlər",
        "işıq və istilik mənbəyidir",
        "python proqramlaşdırma dili çox məşhurdur",
        "qədim alban abidələri qorunur"
    ]
    demo_hyps = [
        "bizim evimizdən qonaqlar var və çay içirlər",
        "işıq və istilik mənbəyidir",
        "python proqram dili çox məşhurdur",
        "qədim alban abidələri qorunur"
    ]
    demo_durs = [4.2, 2.1, 3.8, 9.1]

    rep = evaluate_benchmark(
        references=demo_refs,
        hypotheses=demo_hyps,
        durations=demo_durs,
        processing_time_sec=0.45,
        total_audio_sec=19.2,
        model_name="Whisper-Small (LoRA)",
        dataset_name="FLEURS Test",
        apply_normalization=True
    )

    print(f"📌 Модель:                {rep.model_name}")
    print(f"📊 WER:                   {rep.wer}% [95% CI: {rep.sentence_stats.wer_ci_95_lower}% - {rep.sentence_stats.wer_ci_95_upper}%]")
    print(f"📊 CER:                   {rep.cer}%")
    print(f"🎯 Exact Match Rate:      {rep.exact_match_rate}%")
    print(f"⚡ RTF / Ускорение:        {rep.rtf} ({rep.speedup}x реального времени)")
    print(f"🔍 Замены/Выпадения/Вставки: S={rep.wer_details.substitutions}, D={rep.wer_details.deletions}, I={rep.wer_details.insertions}")
    print(f"📜 LaTeX Row:             {rep.to_latex_row()}")
    print("\n🔤 Точность азербайджанских букв:")
    for l, acc in rep.az_letter_accuracies.items():
        print(f"   '{l}': {acc}%")

    print("\n⏱ Стратификация по длительности:")
    for b_name, b_info in rep.duration_stratification.items():
        print(f"   {b_name}: {b_info['count']} сэмплов, WER={b_info['wer']}%")

    print("\n✅ Фреймворк метрик готов к публикации на конференциях!")
