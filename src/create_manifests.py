import os
import sys
import pandas as pd
import pyarrow.parquet as pq

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

def create_unified_manifests(data_dir: str = "data", output_dir: str = "data/manifests"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("🏗 СОЗДАНИЕ ЕДИНЫХ МАНИФЕСТОВ ДЛЯ 3 ДОМЕНОВ (BENCHMARK SPLITS)")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Common Voice (Домашняя краудсорсинговая речь)
    # -------------------------------------------------------------
    cv_dir = os.path.join(data_dir, "common_voice")
    cv_clips_dir = os.path.join(cv_dir, "clips")

    # CV Test
    cv_test_df = pd.read_csv(os.path.join(cv_dir, "test.tsv"), sep="\t")
    cv_test_records = []
    for _, row in cv_test_df.iterrows():
        clip = os.path.basename(str(row["path"]))
        full_path = os.path.abspath(os.path.join(cv_clips_dir, clip))
        if os.path.exists(full_path):
            cv_test_records.append({"audio_path": full_path, "sentence": str(row["sentence"]), "domain": "crowdsourced_home"})
    df_cv_test = pd.DataFrame(cv_test_records)
    df_cv_test.to_csv(os.path.join(output_dir, "test_common_voice.csv"), index=False, encoding="utf-8")
    print(f"✅ Common Voice Test (Домашняя речь): {len(df_cv_test)} записей")

    # CV Train & Dev
    cv_train_df = pd.read_csv(os.path.join(cv_dir, "train.tsv"), sep="\t")
    cv_train_records = []
    for _, row in cv_train_df.iterrows():
        clip = os.path.basename(str(row["path"]))
        full_path = os.path.abspath(os.path.join(cv_clips_dir, clip))
        if os.path.exists(full_path):
            cv_train_records.append({"audio_path": full_path, "sentence": str(row["sentence"]), "domain": "crowdsourced_home"})
    df_cv_train = pd.DataFrame(cv_train_records)

    cv_dev_df = pd.read_csv(os.path.join(cv_dir, "dev.tsv"), sep="\t")
    cv_dev_records = []
    for _, row in cv_dev_df.iterrows():
        clip = os.path.basename(str(row["path"]))
        full_path = os.path.abspath(os.path.join(cv_clips_dir, clip))
        if os.path.exists(full_path):
            cv_dev_records.append({"audio_path": full_path, "sentence": str(row["sentence"]), "domain": "crowdsourced_home"})
    df_cv_dev = pd.DataFrame(cv_dev_records)

    # -------------------------------------------------------------
    # 2. Google FLEURS (Студийная профессиональная речь)
    # -------------------------------------------------------------
    fleurs_dir = os.path.join(data_dir, "fleurs")
    fleurs_audio_dir = os.path.join(fleurs_dir, "audio")

    # FLEURS Test
    fleurs_test_df = pd.read_csv(os.path.join(fleurs_dir, "test.tsv"), sep="\t")
    fleurs_test_records = []
    for _, row in fleurs_test_df.iterrows():
        clip = os.path.basename(str(row["path"]))
        full_path = os.path.abspath(os.path.join(fleurs_audio_dir, clip))
        if os.path.exists(full_path):
            fleurs_test_records.append({"audio_path": full_path, "sentence": str(row["sentence"]), "domain": "studio_read"})
    df_fleurs_test = pd.DataFrame(fleurs_test_records)
    df_fleurs_test.to_csv(os.path.join(output_dir, "test_fleurs.csv"), index=False, encoding="utf-8")
    print(f"✅ Google FLEURS Test (Студийная речь): {len(df_fleurs_test)} записей")

    # FLEURS Train & Dev
    fleurs_train_df = pd.read_csv(os.path.join(fleurs_dir, "train.tsv"), sep="\t")
    fleurs_train_records = []
    for _, row in fleurs_train_df.iterrows():
        clip = os.path.basename(str(row["path"]))
        full_path = os.path.abspath(os.path.join(fleurs_audio_dir, clip))
        if os.path.exists(full_path):
            fleurs_train_records.append({"audio_path": full_path, "sentence": str(row["sentence"]), "domain": "studio_read"})
    df_fleurs_train = pd.DataFrame(fleurs_train_records)

    fleurs_dev_df = pd.read_csv(os.path.join(fleurs_dir, "dev.tsv"), sep="\t")
    fleurs_dev_records = []
    for _, row in fleurs_dev_df.iterrows():
        clip = os.path.basename(str(row["path"]))
        full_path = os.path.abspath(os.path.join(fleurs_audio_dir, clip))
        if os.path.exists(full_path):
            fleurs_dev_records.append({"audio_path": full_path, "sentence": str(row["sentence"]), "domain": "studio_read"})
    df_fleurs_dev = pd.DataFrame(fleurs_dev_records)

    # -------------------------------------------------------------
    # 3. LocalDoc (Лекционная и техническая речь - Screencasts)
    # Из 4548 записей берем 500 в Test (Benchmark 3), а 4048 - в Train
    # -------------------------------------------------------------
    localdoc_dir = os.path.join(data_dir, "localdoc")
    parquet_path = os.path.join(localdoc_dir, "train_subset.parquet")

    # Для LocalDoc аудио лежит внутри Parquet (bytes). Извлечем аудио для test сплита:
    localdoc_audio_dir = os.path.join(localdoc_dir, "audio")
    os.makedirs(localdoc_audio_dir, exist_ok=True)

    print("\n📦 Извлечение аудио LocalDoc для манифестов...")
    table = pq.read_table(parquet_path)
    localdoc_df = table.to_pandas()

    # Детерминированный сплит LocalDoc: 
    # ~11% Test (500), ~10% Dev (450), ~79% Train (3598) - стандарт ML (80/10/10)
    localdoc_test_slice = localdoc_df.iloc[:500]
    localdoc_dev_slice = localdoc_df.iloc[500:950]
    localdoc_train_slice = localdoc_df.iloc[950:]

    localdoc_test_records = []
    for idx, row in localdoc_test_slice.iterrows():
        audio_name = f"localdoc_test_{idx:05d}.wav"
        file_path = os.path.abspath(os.path.join(localdoc_audio_dir, audio_name))
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(row["audio"]["bytes"])
        localdoc_test_records.append({"audio_path": file_path, "sentence": str(row["text"]), "domain": "lecture_screencast"})
    
    df_localdoc_test = pd.DataFrame(localdoc_test_records)
    df_localdoc_test.to_csv(os.path.join(output_dir, "test_localdoc.csv"), index=False, encoding="utf-8")
    print(f"✅ LocalDoc Test (Технические видеолекции): {len(df_localdoc_test)} записей")

    # LocalDoc Dev
    localdoc_dev_records = []
    for idx, row in localdoc_dev_slice.iterrows():
        audio_name = f"localdoc_dev_{idx:05d}.wav"
        file_path = os.path.abspath(os.path.join(localdoc_audio_dir, audio_name))
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(row["audio"]["bytes"])
        localdoc_dev_records.append({"audio_path": file_path, "sentence": str(row["text"]), "domain": "lecture_screencast"})
    df_localdoc_dev = pd.DataFrame(localdoc_dev_records)
    print(f"✅ LocalDoc Dev (Валидация лекционной речи): {len(df_localdoc_dev)} записей")

    # LocalDoc Train
    localdoc_train_records = []
    for idx, row in localdoc_train_slice.iterrows():
        audio_name = f"localdoc_train_{idx:05d}.wav"
        file_path = os.path.abspath(os.path.join(localdoc_audio_dir, audio_name))
        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(row["audio"]["bytes"])
        localdoc_train_records.append({"audio_path": file_path, "sentence": str(row["text"]), "domain": "lecture_screencast"})
    df_localdoc_train = pd.DataFrame(localdoc_train_records)
    print(f"✅ LocalDoc Train (Обучение лекционной речи): {len(df_localdoc_train)} записей")

    # -------------------------------------------------------------
    # 4. Создание единых TRAIN и DEV выборок для Fine-Tuning
    # -------------------------------------------------------------
    df_train_combined = pd.concat([df_fleurs_train, df_cv_train, df_localdoc_train], ignore_index=True)
    df_train_combined.to_csv(os.path.join(output_dir, "train_combined.csv"), index=False, encoding="utf-8")

    df_dev_combined = pd.concat([df_fleurs_dev, df_cv_dev, df_localdoc_dev], ignore_index=True)
    df_dev_combined.to_csv(os.path.join(output_dir, "dev_combined.csv"), index=False, encoding="utf-8")

    print("\n" + "=" * 70)
    print("📊 ИТОГОВАЯ КАРТИНА МАНИФЕСТОВ ДЛЯ СТАТЬИ:")
    print("=" * 70)
    summary = [
        {"Манифест": "test_common_voice.csv", "Назначение": "Экзамен 1: Домашняя речь", "Фраз": len(df_cv_test)},
        {"Манифест": "test_fleurs.csv", "Назначение": "Экзамен 2: Студийные дикторы", "Фраз": len(df_fleurs_test)},
        {"Манифест": "test_localdoc.csv", "Назначение": "Экзамен 3: Видеолекции (IT-сленг)", "Фраз": len(df_localdoc_test)},
        {"Манифест": "dev_combined.csv", "Назначение": "Валидация при обучении (Dev)", "Фраз": len(df_dev_combined)},
        {"Манифест": "train_combined.csv", "Назначение": "Обучение (Train Combined)", "Фраз": len(df_train_combined)},
    ]
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    create_unified_manifests()
