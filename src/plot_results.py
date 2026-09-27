import os
import sys
import json
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Научный стиль графиков (Interspeech / ACL standard)
plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})


def plot_all_figures(results_dir: str = "results", out_dir: str = "results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    summary_file = os.path.join(results_dir, "benchmark_summary.csv")

    if not os.path.exists(summary_file):
        print(f"⚠️ Файл {summary_file} не найден. Сначала запустите evaluate_zeroshot.py!")
        return

    df = pd.read_csv(summary_file)
    if len(df) == 0:
        print("⚠️ Таблица пуста.")
        return

    print(f"\n📊 Генерация научных графиков из {summary_file}...")

    # Цветовая палитра для доменов
    domain_palette = {
        "common_voice": "#E74C3C",  # Красный: Домашний шум
        "fleurs": "#2ECC71",        # Зеленый: Студия
        "localdoc": "#3498DB",      # Синий: Видеолекции
        "custom_whatsapp": "#9B59B6" # Фиолетовый: Мобильные
    }
    domain_labels = {
        "common_voice": "Common Voice (Domestic)",
        "fleurs": "FLEURS (Studio Read)",
        "localdoc": "LocalDoc (Lectures/IT)",
        "custom_whatsapp": "In-the-Wild (Mobile)"
    }

    # -------------------------------------------------------------
    # График 1: WER по моделям и доменам
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=df,
        x="model",
        y="wer",
        hue="dataset",
        palette=domain_palette,
        edgecolor="black",
        linewidth=0.8
    )
    plt.title("Cross-Domain Word Error Rate (WER %)", weight="bold", pad=12)
    plt.xlabel("ASR Model", weight="bold")
    plt.ylabel("WER (%) [Lower is better]", weight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Замена меток легенды на красивые
    handles, labels = ax.get_legend_handles_labels()
    pretty_labels = [domain_labels.get(l, l) for l in labels]
    plt.legend(handles, pretty_labels, title="Speech Domain", frameon=True)

    # Добавление значений над столбцами
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.1f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom",
                        fontsize=8, rotation=0, xytext=(0, 2),
                        textcoords="offset points")

    plt.tight_layout()
    fig1_png = os.path.join(out_dir, "fig1_wer_cross_domain.png")
    fig1_pdf = os.path.join(out_dir, "fig1_wer_cross_domain.pdf")
    plt.savefig(fig1_png, dpi=300)
    plt.savefig(fig1_pdf)
    plt.close()
    print(f"✅ График 1 сохранен: {fig1_png}")

    # -------------------------------------------------------------
    # График 2: Декомпозиция ошибок (Substitutions, Deletions, Insertions)
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    bar_width = 0.25
    x = np.arange(len(df))

    p1 = plt.bar(x, df["sub_rate"], width=bar_width, label="Substitutions (S)", color="#E67E22", edgecolor="black")
    p2 = plt.bar(x, df["del_rate"], width=bar_width, bottom=df["sub_rate"], label="Deletions (D)", color="#34495E", edgecolor="black")
    p3 = plt.bar(x, df["ins_rate"], width=bar_width, bottom=df["sub_rate"] + df["del_rate"], label="Insertions / Hallucinations (I)", color="#E74C3C", edgecolor="black")

    labels_x = [f"{m}\n({d})" for m, d in zip(df["model"], df["dataset"])]
    plt.xticks(x, labels_x, rotation=15, ha="right", fontsize=9)
    plt.title("Error Decomposition: Explaining Zero-Shot Hallucinations", weight="bold", pad=12)
    plt.ylabel("Relative Error Rate (%)", weight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()

    fig2_png = os.path.join(out_dir, "fig2_error_decomposition.png")
    fig2_pdf = os.path.join(out_dir, "fig2_error_decomposition.pdf")
    plt.savefig(fig2_png, dpi=300)
    plt.savefig(fig2_pdf)
    plt.close()
    print(f"✅ График 2 сохранен: {fig2_png}")

    # -------------------------------------------------------------
    # График 3: Trade-off: Скорость (RTF) vs Качество (WER)
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    scatter = sns.scatterplot(
        data=df,
        x="rtf",
        y="wer",
        hue="dataset",
        style="model",
        s=140,
        palette=domain_palette,
        edgecolor="black"
    )
    plt.title("Inference Efficiency Trade-Off: RTF vs. Accuracy", weight="bold", pad=12)
    plt.xlabel("Real-Time Factor (RTF) [Lower is faster]", weight="bold")
    plt.ylabel("WER (%) [Lower is better]", weight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    for _, row in df.iterrows():
        plt.annotate(f"{row['model']}", (row['rtf'], row['wer']), textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.tight_layout()
    fig3_png = os.path.join(out_dir, "fig3_efficiency_tradeoff.png")
    fig3_pdf = os.path.join(out_dir, "fig3_efficiency_tradeoff.pdf")
    plt.savefig(fig3_png, dpi=300)
    plt.savefig(fig3_pdf)
    plt.close()
    print(f"✅ График 3 сохранен: {fig3_png}")

    # -------------------------------------------------------------
    # График 4: Радарный график аудита азербайджанских букв (ə, ı, ğ, ş, ç, ö, ü)
    # -------------------------------------------------------------
    metrics_files = glob.glob(os.path.join(results_dir, "zeroshot", "*", "*", "metrics.json"))
    if metrics_files:
        letters = ["ə", "ı", "ğ", "ş", "ç", "ö", "ü"]
        angles = np.linspace(0, 2 * np.pi, len(letters), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        for mf in metrics_files:
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                letter_accs = data.get("az_letter_accuracies", {})
                values = [letter_accs.get(l, 0.0) for l in letters]
                values += values[:1]

                label = f"{data.get('model_name', 'M').split('/')[-1]} ({data.get('dataset_name', 'D')})"
                ax.plot(angles, values, linewidth=1.5, label=label)
                ax.fill(angles, values, alpha=0.1)
            except Exception:
                pass

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(letters, fontsize=12, weight="bold")
        ax.set_ylim(0, 100)
        plt.title("Azerbaijani Phoneme / Letter Accuracy Audit (%)", weight="bold", pad=20)
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        plt.tight_layout()

        fig4_png = os.path.join(out_dir, "fig4_phoneme_radar.png")
        fig4_pdf = os.path.join(out_dir, "fig4_phoneme_radar.pdf")
        plt.savefig(fig4_png, dpi=300)
        plt.savefig(fig4_pdf)
        plt.close()
        print(f"✅ График 4 сохранен: {fig4_png}")

    print(f"\n🎉 Все графики успешно сохранены в папку: {out_dir}")


if __name__ == "__main__":
    plot_all_figures()
