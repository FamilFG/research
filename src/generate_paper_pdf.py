import os
import sys
import subprocess
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def generate_academic_pdf(
    html_output_path: str = "paper.html",
    pdf_output_path: str = "paper.pdf"
):
    print("=" * 70)
    print("📑 ГЕНЕРАЦИЯ АКАДЕМИЧЕСКОГО PDF СТАТЬИ (INTERSPEECH / ACL FORMAT)")
    print("=" * 70)

    # Пути к изображениям графиков
    fig1_path = Path("results/plots/fig1_wer_cross_domain.png").resolve().as_uri()
    fig2_path = Path("results/plots/fig2_error_decomposition.png").resolve().as_uri()

    template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Can Efficient Models Outperform Giants? A Multi-Domain ASR Benchmark and Linguistic Analysis for Azerbaijani</title>
<style>
    @page {
        size: A4;
        margin: 16mm 14mm 16mm 14mm;
        @bottom-right {
            content: counter(page);
            font-size: 8pt;
            font-family: "Times New Roman", Times, serif;
        }
    }

    body {
        font-family: "Times New Roman", Times, Georgia, serif;
        font-size: 9.2pt;
        line-height: 1.22;
        color: #111;
        background: #fff;
        margin: 0;
        padding: 0;
    }

    /* Заголовочный блок (на всю ширину двух колонок) */
    .header-box {
        text-align: center;
        margin-bottom: 10pt;
    }

    h1.paper-title {
        font-size: 15.5pt;
        font-weight: bold;
        margin: 0 0 5pt 0;
        line-height: 1.18;
    }

    .author-name {
        font-size: 11pt;
        font-weight: bold;
        margin-bottom: 2pt;
    }

    .author-affiliation {
        font-size: 9pt;
        font-style: italic;
        color: #333;
        margin-bottom: 2pt;
    }

    .author-email {
        font-family: "Courier New", Courier, monospace;
        font-size: 8.5pt;
        color: #003366;
        margin-bottom: 10pt;
    }

    /* Аннотация (Abstract) */
    .abstract-container {
        width: 92%;
        margin: 0 auto 12pt auto;
        font-size: 8.8pt;
        text-align: justify;
        line-height: 1.2;
        padding: 0 8pt;
    }

    .abstract-title {
        font-weight: bold;
        font-style: italic;
        display: inline;
    }

    .keywords {
        margin-top: 4pt;
        font-size: 8.2pt;
    }

    .keywords-title {
        font-weight: bold;
        font-style: italic;
        display: inline;
    }

    /* Двухколоночный макет для основного текста */
    .two-column-body {
        column-count: 2;
        column-gap: 6mm;
        text-align: justify;
        text-justify: inter-word;
    }

    h2.sec-heading {
        font-size: 10pt;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin: 8pt 0 3pt 0;
        border-bottom: 0.5pt solid #333;
        padding-bottom: 1pt;
        break-after: avoid;
    }

    h3.subsec-heading {
        font-size: 9.2pt;
        font-weight: bold;
        font-style: italic;
        margin: 6pt 0 2pt 0;
        break-after: avoid;
    }

    p {
        margin: 0 0 4pt 0;
        text-indent: 9pt;
    }

    p.no-indent {
        text-indent: 0;
    }

    /* Таблицы в стиле LaTeX Booktabs */
    .table-container {
        margin: 6pt 0 8pt 0;
        width: 100%;
        break-inside: avoid;
    }

    .full-width {
        column-span: all;
    }

    table.booktabs {
        width: 100%;
        border-collapse: collapse;
        font-size: 7.6pt;
        line-height: 1.12;
        margin: 3pt 0;
        border-top: 1.2pt solid #000;
        border-bottom: 1.2pt solid #000;
    }

    table.booktabs th {
        border-bottom: 0.7pt solid #000;
        padding: 2.5pt 3pt;
        font-weight: bold;
        text-align: left;
    }

    table.booktabs td {
        padding: 2pt 3pt;
        border-bottom: 0.3pt solid #eee;
    }

    table.booktabs tr:last-child td {
        border-bottom: none;
    }

    .caption {
        font-size: 7.8pt;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2.5pt;
    }

    /* Блоки с примерами кода / транскрипций */
    .quote-box {
        background: #f8f9fa;
        border-left: 2pt solid #2c3e50;
        padding: 2.5pt 5pt;
        margin: 3.5pt 0;
        font-size: 7.5pt;
        font-family: "Courier New", Courier, monospace;
        line-height: 1.18;
        break-inside: avoid;
    }

    /* Иллюстрации */
    .figure-box {
        text-align: center;
        margin: 6pt 0;
        break-inside: avoid;
    }

    .figure-box img {
        max-width: 100%;
        height: auto;
        border: 0.3pt solid #ccc;
    }

    .fig-caption {
        font-size: 7.5pt;
        margin-top: 2.5pt;
        text-align: justify;
    }

    /* Ссылки */
    .references {
        font-size: 7.6pt;
        line-height: 1.15;
    }

    .references ol {
        padding-left: 11pt;
        margin: 0;
    }

    .references li {
        margin-bottom: 2pt;
    }
</style>
</head>
<body>

<div class="header-box">
    <h1 class="paper-title">Can Efficient Models Outperform Giants?<br>A Multi-Domain ASR Benchmark and Linguistic Analysis for Azerbaijani</h1>
    <div class="author-name">Famil Gulmammadov</div>
    <div class="author-affiliation">Department of Computer Science, French-Azerbaijani University (UFAZ) &middot; Baku, Azerbaijan</div>
    <div class="author-email">famil.gulmammadov@ufaz.az</div>
</div>

<div class="abstract-container">
    <div class="abstract-title">Abstract—</div>
    Recent breakthroughs in massively multilingual speech foundation models, notably OpenAI Whisper and Meta MMS, have demonstrated impressive zero-shot performance across major world languages. However, agglutinative and low-resource languages such as Azerbaijani continue to suffer from acute acoustic brittleness, uncurated benchmarks, and pathological decoding failures. In this paper, we introduce the first standardized, multi-domain speech recognition benchmark for Azerbaijani across four distinct acoustic and lexical environments: <strong>Studio Speech (Google FLEURS)</strong>, <strong>Crowdsourced Domestic Speech (Mozilla Common Voice)</strong>, <strong>Educational Screencasts (LocalDoc)</strong>, and <strong>In-the-Wild Mobile Messages (Custom WhatsApp)</strong>, encompassing <strong>8,970 verified recordings (~18.5 hours)</strong>. 
    Our empirical zero-shot evaluation uncovers critical linguistic vulnerabilities in sub-100M parameter models: on <em>Whisper-Tiny (39M)</em>, domestic WER degrades to <strong>201.24%</strong> driven by catastrophic autoregressive repetition loops (Insertion Rate = 118.08%) and complete blindness (0.0% accuracy) to five vital Azerbaijani graphemes (<code>ə</code>, <code>ğ</code>, <code>ş</code>, <code>ç</code>, <code>ü</code>). Scaling to <em>Whisper-Base (74M)</em> induces a sharp linguistic phase transition, dropping insertion rates by 4.3&times; (to 27.17% on Common Voice and 15.33% on FLEURS) and recovering graphemic representations up to 78.6% for <code>ö</code> and 75.0% for <code>ş</code>. We present statistical validation with 95% Bootstrap Confidence Intervals, character-level metrics (CER), phonetic audits, and real-time efficiency metrics on a local 16GB GPU (achieving up to 48.8&times; real-time throughput). Finally, we outline our Parameter-Efficient Fine-Tuning (LoRA) framework designed to enable lightweight architectures to outperform billion-parameter models.
    <div class="keywords">
        <div class="keywords-title">Keywords—</div>
        Automatic Speech Recognition, Low-Resource NLP, Azerbaijani Language, Whisper, Multi-Domain Benchmark, Agglutinative Morphology, Error Decomposition, Parameter-Efficient Adaptation.
    </div>
</div>

<div class="two-column-body">

<h2 class="sec-heading">1. Introduction</h2>
<p class="no-indent">
Speech foundation models trained on hundreds of thousands of weakly-supervised audio hours have fundamentally transformed speech recognition [1, 2]. However, the assumption that foundation models generalize universally to low-resource languages masks severe structural vulnerabilities.
</p>
<p>
Azerbaijani, an Oghuz Turkic language spoken by over 30 million people, represents an archetype of agglutinative morphology combined with low digital resource availability. Words are synthesized through long cascades of grammatical suffixes attached to invariant lexical stems (e.g., <em>şəhər-lər-imiz-dəki-lər-dən-dir</em>). A single transcription error at an inflectional boundary alters the identity of an entire token. Consequently, traditional Word Error Rate (WER) severely penalizes morphological near-misses, mandating Character Error Rate (CER) as an indispensable co-primary metric.
</p>
<p>
Furthermore, Azerbaijani Latin orthography includes seven language-specific non-ASCII characters (<code>ə</code>, <code>ı</code>, <code>ğ</code>, <code>ş</code>, <code>ç</code>, <code>ö</code>, <code>ü</code>). Notably, the near-open front unrounded vowel <code>ə</code> accounts for over 10% of all graphemes in natural Azerbaijani text. Multilingual subword tokenizers trained overwhelmingly on Western Indo-European corpora exhibit poor token boundaries and insufficient acoustic representations for these phonemes.
</p>

<h3 class="subsec-heading">1.1 Research Questions</h3>
<p class="no-indent">This study investigates four concrete research questions:</p>
<p><strong>RQ1 (Cross-Domain Gap):</strong> How severely does acoustic mismatch between clean studio dictation, reverberant home environments, and spontaneous technical screencasts degrade zero-shot recognition?</p>
<p><strong>RQ2 (Pathological Hallucination Dynamics):</strong> What causes small zero-shot models to yield WER exceeding 100%, and how does error decomposition (S + D + I) diagnose decoding failure?</p>
<p><strong>RQ3 (Graphemic Blindness):</strong> At what parameter scale does a foundation model acquire representation for language-specific phonemes?</p>
<p><strong>RQ4 (Engineering &amp; Compute Economics):</strong> Can Parameter-Efficient Fine-Tuning (LoRA) on a consumer 16GB GPU enable a lightweight 244M model to match or exceed billion-parameter giants, reducing enterprise serving costs?</p>

<h2 class="sec-heading">2. Multi-Domain Corpus Curation</h2>
<p class="no-indent">
Previous speech benchmarks have evaluated Azerbaijani almost exclusively on clean, studio-recorded sentences (e.g., Google FLEURS), providing an overly optimistic appraisal of real-world viability. To address this, we curated four distinct domains into a deterministic corpus of <strong>8,970 audio recordings</strong>.
</p>

<div class="table-container full-width">
    <div class="caption">Table 1: Multi-Domain Azerbaijani Speech Benchmark Corpus</div>
    <table class="booktabs">
        <thead>
            <tr>
                <th>Domain ID</th>
                <th>Source Dataset</th>
                <th>Partition</th>
                <th>Clips</th>
                <th>Duration</th>
                <th>Acoustic Condition</th>
                <th>Linguistic &amp; Lexical Characteristics</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Domain 1</strong></td>
                <td><strong>Google FLEURS</strong></td>
                <td>Test (Closed)</td>
                <td>923</td>
                <td>3.24 h</td>
                <td>Studio mics, high SNR (&gt;25 dB), zero noise</td>
                <td>Literary reading (Wikipedia, news), complex syntax (up to 52 words)</td>
            </tr>
            <tr>
                <td><strong>Domain 2</strong></td>
                <td><strong>Mozilla Common Voice</strong></td>
                <td>Test (Closed)</td>
                <td>126</td>
                <td>0.20 h</td>
                <td>Laptops, webcams, room reverberation</td>
                <td>Crowdsourced colloquial phrases, speaker diversity (5–14 words)</td>
            </tr>
            <tr>
                <td><strong>Domain 3</strong></td>
                <td><strong>LocalDoc Screencasts</strong></td>
                <td>Test (Closed)</td>
                <td>500</td>
                <td>0.80 h</td>
                <td>Lavalier/USB mics, keystrokes, breathing</td>
                <td>Educational IT lectures, technical English loanwords (<em>fayl, kod, brauzer</em>)</td>
            </tr>
            <tr>
                <td><strong>Domain 4</strong></td>
                <td><strong>In-the-Wild Mobile (Ours)</strong></td>
                <td>Test (Closed)</td>
                <td>100</td>
                <td>~0.15 h</td>
                <td>Smartphones, Opus/AAC, domestic/street noise</td>
                <td>Spontaneous voice messages, colloquial phonetics</td>
            </tr>
            <tr>
                <td>—</td>
                <td>Combined FLEURS + CV + LocalDoc</td>
                <td><strong>Train</strong></td>
                <td><strong>6,478</strong></td>
                <td><strong>~15.4 h</strong></td>
                <td>Multi-acoustic mixture</td>
                <td>Standard training split for Parameter-Efficient LoRA adaptation</td>
            </tr>
            <tr>
                <td>—</td>
                <td>Combined FLEURS + CV + LocalDoc</td>
                <td><strong>Dev</strong></td>
                <td><strong>943</strong></td>
                <td><strong>~1.9 h</strong></td>
                <td>Multi-acoustic mixture</td>
                <td>Cross-domain validation set for checkpoint selection and early stopping</td>
            </tr>
        </tbody>
    </table>
</div>

<h3 class="subsec-heading">2.1 Quality Control &amp; Contamination Prevention</h3>
<p class="no-indent">
Our curation enforces three strict data-hygiene policies:
</p>
<p>
<strong>1. Elimination of Voice-Over Dubbing:</strong> Third-party corpora (e.g., VOA / WorldSpeech) were evaluated and rejected due to English voice-over dubbing in the background under low SNR (~5.0 dB), which triggered severe language hallucinations in Whisper.
</p>
<p>
<strong>2. Deduplication of Common Voice Releases:</strong> Mozilla Common Voice releases are cumulative (v18 subsumes v1–v17). Merging releases blindly introduces 85–90% data duplication and causes severe train-to-test contamination. We enforced strict disjoint speaker and sentence hashes.
</p>
<p>
<strong>3. Deterministic LocalDoc Partitioning:</strong> The 4,548 screencast clips extracted from raw parquet were partitioned 80/10/10: 500 clips to closed test, 450 to dev, and 3,598 to train.
</p>

<h2 class="sec-heading">3. Evaluation Framework &amp; Metrics</h2>
<p class="no-indent">
<strong>Normalization Protocol:</strong> We implement the official OpenAI Whisper <code>BasicTextNormalizer(remove_diacritics=False)</code> paired with strict Azerbaijani casing rules. Standard Unicode lowercasing erroneously maps <code>I</code> to <code>i</code> (destroying the distinction with dotless <code>ı</code>), and <code>İ</code> to <code>i</code>. Our pre-processing preserves linguistic identity prior to evaluation.
</p>
<p>
<strong>Error Decomposition:</strong> Standard WER reports only aggregate error (S + D + I) / N. We explicitly decompose errors into Substitution (S), Deletion (D), and Insertion (I) rates, which proved vital for isolating autoregressive hallucination loops.
</p>
<p>
<strong>Statistical Significance:</strong> To guard against random sample fluctuations, all WER reports are coupled with 95% non-parametric Bootstrap Confidence Intervals (B = 1000 iterations).
</p>
<p>
<strong>Inference Efficiency:</strong> Real-Time Factor is measured as RTF = T_processing / T_audio, calculated on an NVIDIA RTX 5060 Ti GPU with TensorFloat-32 Tensor Cores.
</p>

<div class="table-container full-width">
    <div class="caption">Table 2: Empirical Zero-Shot Benchmark Results Across Models and Acoustic Domains</div>
    <table class="booktabs">
        <thead>
            <tr>
                <th>Model Architecture</th>
                <th>Params</th>
                <th>Domain / Test Set</th>
                <th>Hours</th>
                <th>WER (%)</th>
                <th>95% Bootstrap CI</th>
                <th>CER (%)</th>
                <th>Sub (%)</th>
                <th>Del (%)</th>
                <th>Ins (%)</th>
                <th>WIL (%)</th>
                <th>Exact (%)</th>
                <th>RTF</th>
                <th>Speedup</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Whisper-Tiny</strong></td>
                <td>39M</td>
                <td>Common Voice (Home)</td>
                <td>0.20</td>
                <td><strong>201.24</strong></td>
                <td>[85.35 &ndash; 90.12]</td>
                <td>157.00</td>
                <td>72.62</td>
                <td>10.54</td>
                <td><strong>118.08</strong></td>
                <td>98.63</td>
                <td>0.0</td>
                <td>0.1713</td>
                <td>5.8&times;</td>
            </tr>
            <tr>
                <td><strong>Whisper-Tiny</strong></td>
                <td>39M</td>
                <td>Google FLEURS (Studio)</td>
                <td>3.24</td>
                <td><strong>145.63</strong></td>
                <td>[87.39 &ndash; 88.80]</td>
                <td>95.47</td>
                <td>72.06</td>
                <td>10.40</td>
                <td><strong>63.17</strong></td>
                <td>97.99</td>
                <td>0.0</td>
                <td><strong>0.0205</strong></td>
                <td><strong>48.8&times;</strong></td>
            </tr>
            <tr>
                <td><strong>Whisper-Tiny</strong></td>
                <td>39M</td>
                <td>LocalDoc (Lectures)</td>
                <td>0.80</td>
                <td><strong>120.91</strong></td>
                <td>[74.85 &ndash; 77.75]</td>
                <td>78.26</td>
                <td>62.31</td>
                <td>9.24</td>
                <td><strong>49.36</strong></td>
                <td>94.23</td>
                <td>0.0</td>
                <td><strong>0.0457</strong></td>
                <td><strong>21.9&times;</strong></td>
            </tr>
            <tr style="background: #f2f4f8;">
                <td><strong>Whisper-Base</strong></td>
                <td>74M</td>
                <td>Common Voice (Home)</td>
                <td>0.20</td>
                <td><strong>102.17</strong></td>
                <td>[76.56 &ndash; 83.27]</td>
                <td><strong>47.14</strong></td>
                <td>68.08</td>
                <td>6.92</td>
                <td><strong>27.17</strong></td>
                <td>94.80</td>
                <td>0.0</td>
                <td>0.0850</td>
                <td>11.8&times;</td>
            </tr>
            <tr style="background: #f2f4f8;">
                <td><strong>Whisper-Base</strong></td>
                <td>74M</td>
                <td>Google FLEURS (Studio)</td>
                <td>3.24</td>
                <td><strong>87.58</strong></td>
                <td>[75.82 &ndash; 77.56]</td>
                <td><strong>34.27</strong></td>
                <td>65.18</td>
                <td>7.07</td>
                <td><strong>15.33</strong></td>
                <td>92.89</td>
                <td>0.0</td>
                <td><strong>0.0374</strong></td>
                <td><strong>26.7&times;</strong></td>
            </tr>
            <tr style="background: #f2f4f8;">
                <td><strong>Whisper-Base</strong></td>
                <td>74M</td>
                <td>LocalDoc (Lectures)</td>
                <td>0.80</td>
                <td><strong>81.52</strong></td>
                <td>[65.93 &ndash; 69.29]</td>
                <td><strong>38.08</strong></td>
                <td>54.72</td>
                <td>7.92</td>
                <td><strong>18.88</strong></td>
                <td>87.42</td>
                <td>0.0</td>
                <td><strong>0.0664</strong></td>
                <td><strong>15.1&times;</strong></td>
            </tr>
            <tr style="color: #666; font-style: italic;">
                <td>Whisper-Small</td>
                <td>244M</td>
                <td>All 3 Domains</td>
                <td>—</td>
                <td><em>[Phase 3 Next]</em></td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
            </tr>
            <tr style="color: #666; font-style: italic;">
                <td>Whisper-Medium</td>
                <td>769M</td>
                <td>All 3 Domains</td>
                <td>—</td>
                <td><em>[Queued]</em></td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
            </tr>
            <tr style="color: #666; font-style: italic;">
                <td>Whisper-Large-v3</td>
                <td>1550M</td>
                <td>All 3 Domains</td>
                <td>—</td>
                <td><em>[Queued]</em></td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
            </tr>
            <tr style="font-weight: bold; background: #e8f8f5;">
                <td>LoRA-Small (Ours)</td>
                <td>244M</td>
                <td>All Domains Target</td>
                <td>—</td>
                <td><em>[Phase 4 Goal &lt;18%]</em></td>
                <td>—</td>
                <td><em>[&lt;8%]</em></td>
                <td>—</td>
                <td>—</td>
                <td><em>[&lt;3%]</em></td>
                <td>—</td>
                <td><em>[&gt;35%]</em></td>
                <td><strong>~0.03</strong></td>
                <td><strong>~33&times;</strong></td>
            </tr>
        </tbody>
    </table>
</div>

<h2 class="sec-heading">4. Empirical Findings &amp; Discussion</h2>

<h3 class="subsec-heading">4.1 Autoregressive Hallucination Pathology</h3>
<p class="no-indent">
The benchmark results in Table 2 resolve a long-standing puzzle: why zero-shot models on low-resource languages yield WER values exceeding 100%. Our error decomposition proves this is driven by <strong>repetition loops</strong> rather than phonetic misclassifications.
</p>
<p>
On domestic speech, <em>Whisper-Tiny</em> exhibits an Insertion Rate of <strong>118.08%</strong>. When the acoustic encoder produces low-confidence feature maps, the autoregressive decoder fails to emit the <code>&lt;|endoftranscript|&gt;</code> token. The generation collapses into self-reinforcing n-gram repetition until reaching the maximum sequence buffer (448 tokens):
</p>

<div class="quote-box">
<strong>Clip CV-34229675:</strong><br>
<strong>REF:</strong> Orta məktəbi bitirdikdən sonra Ukraynanın Lvov şəhərində hərbi xidmətdə olmuşdur.<br>
<strong>HYP:</strong> Orta məkterdi bir dirdikten sonra ukranalını bu şərində <em>şərində şərində şərində şərində şədində şədində... [438 insertion tokens]</em>
</div>

<div class="quote-box">
<strong>Clip LocalDoc-Test-00005:</strong><br>
<strong>REF:</strong> Bitirilmiş kurs barəsində rəyi burada yazmaq mümkündür.<br>
<strong>HYP:</strong> İtirilmiş kurs payrəsindərəyi burada yazmak <em>mümkür mümkür mümkür... mümkün mümkün [100 repetitions]</em>
</div>

<h3 class="subsec-heading">4.2 The 74M Parameter Linguistic Transition</h3>
<p class="no-indent">
Doubling parameter scale from 39M to 74M (Base) induces a qualitative phase transition. Insertion rates fall by over 4-fold (from 118% to 27% on Common Voice, and 63% to 15% on FLEURS). Base possesses sufficient language modeling capacity to terminate generation, slashing CER by 3.3&times; (157.0% &rarr; 47.1%).
</p>

<div class="table-container">
    <div class="caption">Table 3: Phoneme / Letter Accuracy Audit on FLEURS (%)</div>
    <table class="booktabs">
        <thead>
            <tr>
                <th>Grapheme</th>
                <th>Phonetic Class</th>
                <th>Rank</th>
                <th>Tiny (39M)</th>
                <th>Base (74M)</th>
                <th>Gain (&Delta;)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong><code>ə</code></strong></td>
                <td>Near-open front vowel</td>
                <td>#1</td>
                <td><strong>0.0%</strong></td>
                <td><strong>42.8%</strong></td>
                <td><strong>+42.8%</strong></td>
            </tr>
            <tr>
                <td><strong><code>ş</code></strong></td>
                <td>Voiceless postalveolar fricative</td>
                <td>#9</td>
                <td><strong>0.0%</strong></td>
                <td><strong>75.0%</strong></td>
                <td><strong>+75.0%</strong></td>
            </tr>
            <tr>
                <td><strong><code>ç</code></strong></td>
                <td>Voiceless postalveolar affricate</td>
                <td>#14</td>
                <td><strong>0.0%</strong></td>
                <td><strong>54.8%</strong></td>
                <td><strong>+54.8%</strong></td>
            </tr>
            <tr>
                <td><strong><code>ö</code></strong></td>
                <td>Close-mid front rounded vowel</td>
                <td>#21</td>
                <td>32.6%</td>
                <td><strong>78.6%</strong></td>
                <td><strong>+46.0%</strong></td>
            </tr>
            <tr>
                <td><strong><code>ü</code></strong></td>
                <td>Close front rounded vowel</td>
                <td>#17</td>
                <td><strong>0.0%</strong></td>
                <td><strong>59.5%</strong></td>
                <td><strong>+59.5%</strong></td>
            </tr>
            <tr>
                <td><strong><code>ı</code></strong></td>
                <td>Close back unrounded vowel</td>
                <td>#8</td>
                <td>24.8%</td>
                <td><strong>66.5%</strong></td>
                <td><strong>+41.7%</strong></td>
            </tr>
            <tr>
                <td><strong><code>ğ</code></strong></td>
                <td>Voiced velar fricative</td>
                <td>#28</td>
                <td><strong>0.0%</strong></td>
                <td><strong>4.6%</strong></td>
                <td><strong>+4.6%</strong></td>
            </tr>
        </tbody>
    </table>
</div>

<h3 class="subsec-heading">4.3 Graphemic Audit: The <code>ə</code> Phenomenon</h3>
<p class="no-indent">
Table 3 details the recognition accuracy of language-specific letters. While <em>Whisper-Tiny</em> is completely blind to 5 of the 7 characters (0% recall), <em>Whisper-Base</em> restores recognition across most vowels and consonants, reaching 78.6% for <code>ö</code> and 75.0% for <code>ş</code>. However, the voiced velar fricative <code>ğ</code> remains virtually unlearned (4.6%), proving that zero-shot scaling alone cannot fully resolve low-resource phonetic deficits.
</p>

<h3 class="subsec-heading">4.4 Screencast Lexical Anomalies</h3>
<p class="no-indent">
In the educational screencast domain (LocalDoc), inspection of predictions reveals two distinct error archetypes:
</p>
<p>
<strong>1. Phonetic Transliteration of English Loanwords:</strong> The model maps technical vocabulary to Turkish/Azerbaijani sound-alikes: <em>Python</em> &rarr; <code>paytın</code>, <em>Java</em> &rarr; <code>Cava</code>, <em>Fortran</em> &rarr; <code>Fartan</code>, and <em>proqramı</em> &rarr; <code>programı</code>.
</p>
<p>
<strong>2. Numerical Verbalization Cascades:</strong> Rather than recognizing digits, Whisper spells out calendar years phonetically: <code>1991-ci ilin</code> &rarr; <em>Mindokguz dosan bilinzinin</em>, multiplying token-level edit distance penalties.
</p>

<div class="figure-box">
    <img src="FIG1_PLACEHOLDER" alt="Cross-Domain Word Error Rate">
    <div class="fig-caption"><strong>Figure 1:</strong> Cross-domain Word Error Rate comparison across speech environments with 95% Bootstrap Confidence Interval error bars.</div>
</div>

<div class="figure-box">
    <img src="FIG2_PLACEHOLDER" alt="Error Decomposition">
    <div class="fig-caption"><strong>Figure 2:</strong> Error decomposition into Substitutions, Deletions, and Insertions, demonstrating the dramatic suppression of repetition loops from Tiny to Base.</div>
</div>

<h2 class="sec-heading">5. Proposed LoRA Adaptation Framework</h2>
<p class="no-indent">
Our empirical baseline confirms that zero-shot foundation models are insufficient for reliable deployment in Azerbaijani. In Phase 4, we introduce Low-Rank Adaptation (LoRA) [5] applied to the query, key, value, and output projection matrices of the self-attention and cross-attention sublayers (r = 16, &alpha; = 32).
</p>
<p>
By freezing the base encoder-decoder weights and tuning only ~1% of parameters on our curated <strong>6,478 training recordings</strong>, we hypothesize that <strong>LoRA-adapted Whisper-Small (244M)</strong> will surpass zero-shot <strong>Whisper-Large-v3 (1.55B)</strong> while maintaining a 33&times; real-time serving advantage on local consumer hardware.
</p>

<h2 class="sec-heading">6. Conclusion</h2>
<p class="no-indent">
This study provides the first comprehensive multi-domain speech recognition benchmark for the Azerbaijani language. Through rigorous curation, error decomposition, and phonemic auditing, we demonstrated the severe limitations of small zero-shot models and documented the linguistic phase transition occurring at 74M parameters. Our ongoing parameter-efficient adaptation will establish whether compact models can systematically outperform industrial giants for low-resource agglutinative speech.
</p>

<h2 class="sec-heading">References</h2>
<div class="references">
    <ol>
        <li>A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, &ldquo;Robust speech recognition via large-scale weak supervision,&rdquo; in <em>Proc. ICML</em>, 2023.</li>
        <li>V. Pratap, A. Tjandra, B. Shi, et al., &ldquo;Scaling speech technology to 1,000+ languages,&rdquo; <em>arXiv:2305.13516</em>, 2023.</li>
        <li>A. Conneau, M. Ma, S. Khanuja, et al., &ldquo;FLEURS: Few-shot speech evaluation in 102 languages,&rdquo; in <em>Proc. IEEE SLT</em>, 2022.</li>
        <li>R. Ardila, M. Branson, K. Davis, et al., &ldquo;Common Voice: A massively-multilingual speech corpus,&rdquo; in <em>Proc. LREC</em>, 2020.</li>
        <li>E. J. Hu, Y. Shen, P. Wallis, et al., &ldquo;LoRA: Low-rank adaptation of large language models,&rdquo; in <em>Proc. ICLR</em>, 2022.</li>
        <li>F. Gulmammadov, &ldquo;Cross-domain speech recognition and linguistic adaptation for Azerbaijani,&rdquo; <em>Tech. Monograph, French-Azerbaijani University (UFAZ)</em>, 2026.</li>
    </ol>
</div>

</div>

</body>
</html>
"""

    html_content = template.replace("FIG1_PLACEHOLDER", fig1_path).replace("FIG2_PLACEHOLDER", fig2_path)

    with open(html_output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"✅ HTML макет статьи сохранен: {html_output_path}")

    # Компиляция в PDF через headless Edge
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    abs_html = os.path.abspath(html_output_path)
    abs_pdf = os.path.abspath(pdf_output_path)

    cmd = [
        edge_path,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={abs_pdf}",
        "--no-pdf-header-footer",
        abs_html
    ]

    print("🚀 Компиляция PDF через Chromium rendering engine...")
    subprocess.run(cmd, check=True)

    if os.path.exists(abs_pdf):
        size_kb = os.path.getsize(abs_pdf) // 1024
        print(f"🎉 Научный PDF успешно сгенерирован: {abs_pdf} ({size_kb} KB)")
    else:
        print("❌ Ошибка при создании PDF")

if __name__ == "__main__":
    generate_academic_pdf()
