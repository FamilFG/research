import os
import sys
import pandas as pd
import soundfile as sf

# Обеспечиваем корректный вывод UTF-8 в Windows-консоли
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def analyze_dataset(name: str, base_dir: str, audio_subdir: str, splits: list):
    print(f"\n" + "=" * 60)
    print(f"📊 АНАЛИЗ ДАТАСЕТА: {name}")
    print("=" * 60)

    audio_dir = os.path.join(base_dir, audio_subdir)
    results = []

    for s in splits:
        tsv_path = os.path.join(base_dir, s)
        if not os.path.exists(tsv_path):
            continue

        df = pd.read_csv(tsv_path, sep="\t")
        total_samples = len(df)
        existing_samples = 0
        missing_samples = 0
        total_duration_sec = 0.0
        all_words = set()

        for _, row in df.iterrows():
            clip_name = str(row["path"])
            # Убираем поддиректорию, если она есть в пути
            clip_basename = os.path.basename(clip_name)
            full_clip_path = os.path.join(audio_dir, clip_basename)

            if os.path.exists(full_clip_path):
                existing_samples += 1
                try:
                    info = sf.info(full_clip_path)
                    total_duration_sec += info.duration
                except Exception:
                    pass
            else:
                missing_samples += 1

            sentence = str(row.get("sentence", ""))
            for w in sentence.split():
                all_words.add(w.lower())

        mins = total_duration_sec / 60.0
        hours = total_duration_sec / 3600.0

        results.append({
            "dataset": name,
            "split": s,
            "samples": existing_samples,
            "missing": missing_samples,
            "duration_min": round(mins, 2),
            "duration_hours": round(hours, 2),
            "vocab_size": len(all_words)
        })

    return results

def main():
    cv_results = analyze_dataset(
        name="Mozilla Common Voice",
        base_dir="data/common_voice",
        audio_subdir="clips",
        splits=["test.tsv", "dev.tsv", "train.tsv"]
    )

    fleurs_results = analyze_dataset(
        name="Google FLEURS",
        base_dir="data/fleurs",
        audio_subdir="audio",
        splits=["test.tsv", "dev.tsv", "train.tsv"]
    )

    all_results = cv_results + fleurs_results

    print("\n" + "=" * 80)
    print("📋 СВОДНАЯ ТАБЛИЦА ДАТАСЕТОВ ДЛЯ СТАТЬИ (DATASET BENCHMARK SUMMARY):")
    print("=" * 80)
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))
    
    # Сохраняем сводную таблицу в results
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv("results/datasets_summary.csv", index=False)
    print("\n✅ Сводка сохранена в results/datasets_summary.csv")

if __name__ == "__main__":
    main()
