# Can Efficient Models Outperform Giants?
## A Multi-Domain ASR Benchmark and Linguistic Analysis for Azerbaijani

**Famil Gulmammadov**  
*Department of Computer Science, French-Azerbaijani University (UFAZ)*  
*Baku, Azerbaijan*  
`famil.gulmammadov@ufaz.az`

---

## Abstract
Recent advances in speech foundation models, notably OpenAI Whisper and Meta MMS, have demonstrated impressive multilingual zero-shot capabilities. However, low-resource and agglutinative languages—such as Azerbaijani—continue to suffer from severe performance degradation, non-standardized evaluation protocols, and domain fragility. In this work, we present the first rigorous, multi-domain zero-shot and parameter-efficient adaptation benchmark for Azerbaijani automatic speech recognition (ASR). We curate a clean, leak-free benchmark dataset comprising **8,970 audio recordings (~18.5 hours)** spanning four distinct acoustic and lexical domains: **Studio Read Speech (Google FLEURS)**, **Crowdsourced Domestic Audio (Mozilla Common Voice)**, **Educational Technical Screencasts (LocalDoc)**, and **In-the-Wild Mobile Voice Messages (Custom WhatsApp)**. 

Our empirical zero-shot evaluation reveals critical linguistic vulnerabilities in sub-100M parameter models: on *Whisper-Tiny (39M)*, Word Error Rate (WER) degrades to **201.24%** on domestic speech due to catastrophic autoregressive repetition hallucination loops (Insertion Rate = 118.08%) and complete blindness (0.0% accuracy) to five vital Azerbaijani graphemes (`ə`, `ğ`, `ş`, `ç`, `ü`). Scaling to *Whisper-Base (74M)* triggers an immediate linguistic phase transition, dropping insertion rates by over 4-fold (to 27.17% on Common Voice and 15.33% on FLEURS) and restoring grapheme recognition up to 78.6% for `ö` and 75.0% for `ş`. Furthermore, we establish an evaluation protocol incorporating 95% Bootstrap Confidence Intervals, character-level decomposition (CER), phoneme-specific audits, and computational Real-Time Factor (RTF) metrics measured on a consumer 16GB GPU (achieving up to 48.8× real-time throughput). This living manuscript documents our benchmark findings, qualitative error analysis, and provides the architectural foundation for our upcoming Parameter-Efficient Fine-Tuning (LoRA) experiments.

**Keywords:** Automatic Speech Recognition, Low-Resource NLP, Azerbaijani Language, Whisper, Multi-Domain Benchmark, Agglutinative Morphology, Error Decomposition, Parameter-Efficient Fine-Tuning.

---

## 1. Introduction

Speech recognition technologies have reached human-level parity for high-resource languages such as English and Mandarin. The emergence of weakly-supervised massively multilingual foundation models—exemplified by Whisper (Radford et al., 2023) and Meta MMS (Pratap et al., 2023)—has created an impression that speech recognition is an effectively solved problem across world languages.

However, for low-resource languages, empirical reality sharply contradicts this narrative. Azerbaijani, a Turkic language spoken by over 30 million people worldwide, presents profound linguistic and computational challenges:
1. **Agglutinative Morphology:** Words are formed through extensive linear suffixation attached to invariant roots (e.g., *şəhər-lər-imiz-dəki-lər-dən-dir*). A single misspelled character or missed grammatical suffix can alter word-level identity, causing traditional Word Error Rate (WER) to penalize agglutinative languages disproportionately compared to fusional languages. Character Error Rate (CER) must therefore be treated as a primary co-metric.
2. **Specific Phonemic & Graphemic Inventory:** Azerbaijani Latin orthography contains seven distinct non-ASCII characters (`ə`, `ı`, `ğ`, `ş`, `ç`, `ö`, `ü`). Notably, the open front unrounded vowel `ə` is the single most frequent letter in Azerbaijani text, yet multilingual tokenizers trained predominantly on Western European corpora frequently lack adequate representations or acoustic grounding for it.
3. **Severe Cross-Domain Acoustic Brittleness:** Real-world ASR systems must operate across vastly different acoustic conditions. Yet previous evaluations have almost exclusively tested models on clean studio speech (e.g., FLEURS), giving a false sense of production readiness.

### Research Questions
This paper systematically investigates four research questions:
- **RQ1 (Acoustic Domain Sensitivity):** How does the acoustic gap between studio reading, domestic home recording, and lecture screencasts impact zero-shot foundation models?
- **RQ2 (Pathological Hallucination Dynamics):** Why do small-footprint zero-shot models exhibit WER exceeding 100% on low-resource speech, and what does error decomposition ($S + D + I$) reveal about autoregressive decoding failure modes?
- **RQ3 (Graphemic Blindness):** How does model parameter scale correlate with the recognition accuracy of language-specific phonemes (`ə, ı, ğ, ş, ç, ö, ü`)?
- **RQ4 (Engineering & Compute Economics):** Can parameter-efficient fine-tuning (LoRA) on a consumer-grade 16GB GPU enable a lightweight model (*Whisper-Small, 244M*) to surpass or match a zero-shot giant (*Whisper-Large-v3, 1.55B*), providing dramatic throughput and cost advantages for resource-constrained organizations?

---

## 2. Dataset Curation and Cross-Domain Matrix

To prevent data contamination and ensure academic rigor, we curated and standardized four distinct speech domains into deterministic, leak-free manifests.

```
Total Curated Corpus: 8,970 Audios (~18.5 Hours)
├── Training Split (Combined):    6,478 clips (72.2%)
├── Validation Split (Dev):         943 clips (10.5%)
└── Benchmark Test Split (Held-out): 1,549 clips (17.3%)
```

### Table 1: Cross-Domain Experimental Benchmark Corpus

| Domain ID | Dataset Source | Split | Samples | Audio Hours | Acoustic Condition | Lexical & Linguistic Style |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Domain 1** | **Google FLEURS** | Test | 923 | 3.24 h | Studio microphones, high SNR (>25 dB), zero background noise. | Formal literary reading (Wikipedia, news), complex syntax (up to 52 words/sent). |
| **Domain 2** | **Mozilla Common Voice** | Test | 126 | 0.20 h | Laptop/webcam mics, home acoustic reverberation, age diversity. | Everyday crowdsourced colloquial phrases (5–14 words/sent). |
| **Domain 3** | **LocalDoc Screencasts** | Test | 500 | 0.80 h | Lavalier / desktop USB mics, keyboard clicks, breathing noise. | Educational screencasts, technical IT terminology, English loanwords (*fayl, kod, brauzer*). |
| **Domain 4** | **Custom Mobile (Ours)** | Test | 100 | ~0.15 h | Smartphones, Opus/AAC compression, in-the-wild room/street noise. | Conversational voice messages, colloquial idioms. |
| — | **FLEURS (Train)** | Train | 2,665 | 9.31 h | Studio read | Formal Azerbaijani |
| — | **Common Voice (Train)** | Train | 215 | 0.32 h | Domestic crowdsourced | Colloquial Azerbaijani |
| — | **LocalDoc (Train)** | Train | 3,598 | 5.80 h | Technical screencast | IT lecture Azerbaijani |
| — | **Combined Validation (Dev)**| Dev | 943 | 1.85 h | Balanced across Domains 1, 2, 3 | Multi-domain validation |

### Strict Quality Control & Data Hygiene:
1. **Rejection of Bilingual Voice-Over Dubbing:** Preliminary investigation into third-party datasets (e.g., Voice of America / WorldSpeech) revealed bilingual voice-over dubbing where original English speech played in the background under low SNR (~5 dB). We rejected this corpus because Whisper hallucinated English text rather than transcribing Azerbaijani.
2. **Cumulative Release Deduplication:** Mozilla Common Voice releases are cumulative (v18 contains clips from v1–v17). Merging versions naively introduces 85–90% duplicate clips and causes severe test-set leakage. Our splits strictly enforce disjoint speaker and sentence boundaries.
3. **Deterministic LocalDoc Partitioning:** The 4,548 technical screencast recordings extracted from raw parquet were split using an 80/10/10 partition: 500 clips to held-out test, 450 to dev, and 3,598 to train.

---

## 3. Methodology and Evaluation Framework

### 3.1 Text Normalization Protocol
To ensure our findings are unbiased by ad-hoc orthographic post-processing, we employ the official OpenAI Whisper `BasicTextNormalizer(remove_diacritics=False)` augmented with strict Azerbaijani case folding. Crucially, standard lowercasing algorithms corrupt Azerbaijani capitalization:
- Capital `I` (with no dot) must lowercase to `ı` (dotless i).
- Capital `İ` (with dot) must lowercase to `i` (dotted i).
- Capital `Ə` must lowercase to `ə`.
Our normalizer enforces these linguistic rules before stripping non-essential punctuation and collapsing whitespace.

### 3.2 Research-Grade Metric Suite
For every experimental run, we compute:
1. **Word Error Rate (WER):** $\text{WER} = \frac{S + D + I}{N_{\text{words}}}$.
2. **Character Error Rate (CER):** Primary morphological fidelity metric: $\text{CER} = \frac{S_c + D_c + I_c}{N_{\text{chars}}}$.
3. **WER Error Decomposition:** Separation of WER into Substitution Rate ($S/N$), Deletion Rate ($D/N$), and **Insertion Rate ($I/N$)**.
4. **Information Metrics:** Word Information Lost (WIL) and Word Information Preserved (WIP).
5. **Exact Match Ratio & Sentence Error Rate (SER):** Percentage of utterances transcribed with 100% character perfection.
6. **Non-Parametric 95% Bootstrap Confidence Intervals:** Computed over $B = 1000$ bootstrap resamples to establish statistical significance ($p < 0.05$).
7. **Linguistic Phoneme Audit:** Exact character-level recall across the seven specific Azerbaijani letters (`ə, ı, ğ, ş, ç, ö, ü`).
8. **Real-Time Factor (RTF):** Computational efficiency measured as $\text{RTF} = \frac{T_{\text{processing}}}{T_{\text{audio}}}$, accompanied by throughput speedup factor ($1/\text{RTF}$).

---

## 4. Empirical Zero-Shot Benchmark Results

Table 2 presents our benchmark results across models and acoustic domains. All evaluations were executed on a dedicated local **NVIDIA GeForce RTX 5060 Ti (16 GB VRAM, CUDA 12.8, FP16 precision, TF32 Tensor Cores)** using multi-threaded 8-core CPU I/O.

### Table 2: Multi-Domain Zero-Shot Benchmark on Azerbaijani

| Model | Parameters | Domain / Dataset | Audio (h) | WER (%) | 95% Bootstrap CI | CER (%) | Sub (%) | Del (%) | **Ins (%)** | WIL (%) | Exact (%) | RTF | Speedup |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Whisper-Tiny** | 39M | Common Voice (Home) | 0.20 | **201.24** | [85.35 - 90.12] | 157.00 | 72.62 | 10.54 | **118.08** | 98.63 | 0.0 | 0.1713 | 5.8× |
| **Whisper-Tiny** | 39M | Google FLEURS (Studio) | 3.24 | **145.63** | [87.39 - 88.80] | 95.47 | 72.06 | 10.40 | **63.17** | 97.99 | 0.0 | **0.0205** | **48.8×** |
| **Whisper-Tiny** | 39M | LocalDoc (Lectures) | 0.80 | **120.91** | [74.85 - 77.75] | 78.26 | 62.31 | 9.24 | **49.36** | 94.23 | 0.0 | **0.0457** | **21.9×** |
| **Whisper-Base** | 74M | Common Voice (Home) | 0.20 | **102.17** | [76.56 - 83.27] | 47.14 | 68.08 | 6.92 | **27.17** | 94.80 | 0.0 | 0.0850 | 11.8× |
| **Whisper-Base** | 74M | Google FLEURS (Studio) | 3.24 | **87.58** | [75.82 - 77.56] | 34.27 | 65.18 | 7.07 | **15.33** | 92.89 | 0.0 | **0.0374** | **26.7×** |
| **Whisper-Base** | 74M | LocalDoc (Lectures) | 0.80 | **81.52** | [65.93 - 69.29] | 38.08 | 54.72 | 7.92 | **18.88** | 87.42 | 0.0 | **0.0664** | **15.1×** |
| *Whisper-Small* | 244M | All 3 Domains | — | *[Running]*| — | — | — | — | — | — | — | — | — |
| *Whisper-Med* | 769M | All 3 Domains | — | *[Queued]* | — | — | — | — | — | — | — | — | — |
| *Whisper-Large* | 1550M | All 3 Domains | — | *[Queued]* | — | — | — | — | — | — | — | — | — |
| *Meta MMS-1B* | 1000M | All 3 Domains | — | *[Queued]* | — | — | — | — | — | — | — | — | — |
| **LoRA-Small (Ours)**| **244M** | **All 3 Domains** | — | *[Phase 4]*| — | — | — | — | — | — | — | — | — |

---

## 5. Qualitative and Linguistic Error Analysis

### 5.1 The Pathology of Autoregressive Repetition Loops
A central discovery of this benchmark is why sub-100M zero-shot models produce WER values substantially above 100%. In standard high-resource benchmarks, WER is overwhelmingly dominated by phonetic substitution errors ($S \gg I$). However, in our low-resource Azerbaijani zero-shot evaluation, the decoder suffers from catastrophic autoregressive looping.

When the acoustic representation is ambiguous, the cross-attention mechanism fails to map the encoder features to valid Azerbaijani language model tokens. The decoder fails to emit `<|endoftranscript|>` and instead gets trapped in a local minimum where the previous predicted token reinforces the next identical token:

#### Example 1 (Common Voice test clip `common_voice_az_34229675`):
- **Ground Truth:** `Orta məktəbi bitirdikdən sonra Ukraynanın Lvov şəhərində hərbi xidmətdə olmuşdur.`
- **Whisper-Tiny Prediction:** `Orta məkterdi bir dirdikten sonra ukranalını bu şərində şərində şərində şərində şərində şədində şədində şədində... [repeated for 448 tokens until buffer limit]`
- **Consequence:** This single 10-word sentence generated **438 insertion errors**, inflating the dataset-level Insertion Rate to **118.08%**!

#### Example 2 (Common Voice test clip `common_voice_az_43107780`):
- **Ground Truth:** `Onların qalıqlarına Çin və Özbəkistan ərazilərində aşkarlanmışdır.`
- **Whisper-Tiny Prediction:** `O man qələ qələ qələ qələ qələ qələ qələ qələ qələ... [continuous loop]`

### 5.2 The 74M Parameter Linguistic Phase Transition
When model scale is doubled from 39M (Tiny) to 74M (Base), a sharp qualitative leap occurs:
- **Repetition Suppression:** Insertion Rate plummets from **118.1% to 27.2%** on Common Voice, from **63.2% to 15.3%** on FLEURS, and from **49.4% to 18.9%** on LocalDoc. Base possesses sufficient autoregressive capacity to break out of repetition loops.
- **CER Collapse:** Character Error Rate drops by a factor of **3.3×** on Common Voice (157.0% $\rightarrow$ 47.1%) and **2.8×** on FLEURS (95.5% $\rightarrow$ 34.3%).

### 5.3 Graphemic Audit: The Vowel `ə` Phenomenon
Table 3 tracks character-level recognition fidelity for language-specific phonemes.

### Table 3: Accuracy of Specific Azerbaijani Graphemes (%)

| Character | Phonetic Class | Frequency Rank | Whisper-Tiny (FLEURS) | Whisper-Base (FLEURS) | Gain ($\Delta$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **`ə`** | Near-open front unrounded vowel | **#1** | **0.0%** | **42.77%** | **+42.77%** |
| **`ş`** | Voiceless postalveolar fricative | #9 | **0.0%** | **75.01%** | **+75.01%** |
| **`ç`** | Voiceless postalveolar affricate | #14 | **0.0%** | **54.80%** | **+54.80%** |
| **`ö`** | Close-mid front rounded vowel | #21 | 32.56% | **78.60%** | **+46.04%** |
| **`ü`** | Close front rounded vowel | #17 | **0.0%** | **59.51%** | **+59.51%** |
| **`ı`** | Close back unrounded vowel | #8 | 24.78% | **66.54%** | **+41.76%** |
| **`ğ`** | Voiced velar fricative | #28 | **0.0%** | **4.56%** | **+4.56%** |

*Key finding:* Whisper-Tiny is completely blind to 5 of the 7 specific characters (0% recall), attempting to substitute them with Turkish or generic Latin letters. Whisper-Base partially recovers these phonemes, but `ğ` remains almost entirely unrecognized (4.56%), highlighting the acute need for target-language acoustic adaptation.

### 5.4 Domain-Specific IT Lexicon & Screencast Anomalies
In the technical screencast domain (LocalDoc), qualitative inspection of `predictions.csv` revealed three distinct error archetypes:
1. **Phonetic Transliteration of English Loanwords:** Without task-specific fine-tuning, the model transcribes IT keywords phonetically using colloquial Turkish/Azerbaijani sound mappings:
   - `Python` $\rightarrow$ *paytın* / *paytan*
   - `Java` $\rightarrow$ *Cava*
   - `Fortran` $\rightarrow$ *Fartan*
   - `ABC və C dilləri` $\rightarrow$ *eibisi və sidelərə* (English letter-by-letter spelling)
   - `proqramı` $\rightarrow$ *programı* (Turkish `g` substitution for Azerbaijani `q`)
2. **Numerical Verbalization Cascades:** When speakers read years or metrics, Whisper attempts full verbal expansion instead of digit recognition:
   - `1991-ci ilin` $\rightarrow$ *Mindokguz dosan bilinzinin* (creating 4 cascading word errors for a single date).
   - `95%` $\rightarrow$ *doksan beş faiz*.

---

## 6. Planned Parameter-Efficient Fine-Tuning (LoRA)

Our benchmark establishes that zero-shot models—even up to Base—remain impractical for commercial or mission-critical deployment in Azerbaijani without adaptation. 

In Phase 4, we will introduce **Low-Rank Adaptation (LoRA)** applied to the query, key, and value projection matrices of Whisper's self-attention and cross-attention blocks ($r = 16, \alpha = 32$). By freezing the pre-trained foundation encoder/decoder weights and tuning only $\sim 1\%$ of parameters on our curated **6,478 training clips**, we hypothesize that:
1. **LoRA-Small (244M)** will match or outperform zero-shot **Whisper-Large-v3 (1.55B)** on all three domains.
2. The parameter-efficient adapter will completely extinguish the remaining 15–27% insertion hallucinations.
3. LoRA will restore letter `ə` and `ğ` accuracy above 90% by aligning the acoustic representations directly with Azerbaijani orthography.

---

## 7. Preliminary Conclusion

This paper establishes the first standardized, multi-domain benchmark for Azerbaijani speech recognition. By breaking down zero-shot performance across four domains, diagnosing the pathology of autoregressive insertion loops, and conducting a graphemic audit, we demonstrate that parameter scale alone is insufficient: domain adaptation and phonemic alignment are essential for low-resource agglutinative languages.

---

## References

1. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. *International Conference on Machine Learning (ICML)*.
2. Pratap, V., Tjandra, A., Shi, B., et al. (2023). Scaling speech technology to 1,000+ languages. *arXiv preprint arXiv:2305.13516*.
3. Conneau, A., Ma, M., Khanuja, S., et al. (2022). FLEURS: Few-shot speech evaluation in 102 languages. *IEEE Spoken Language Technology Workshop (SLT)*.
4. Ardila, R., Branson, M., Davis, K., et al. (2020). Common Voice: A massively-multilingual speech corpus. *Language Resources and Evaluation Conference (LREC)*.
5. Hu, E. J., Shen, Y., Wallis, P., et al. (2021). LoRA: Low-rank adaptation of large language models. *International Conference on Learning Representations (ICLR)*.
6. Gulmammadov, F. (2026). Cross-Domain Speech Recognition and Linguistic Adaptation for Azerbaijani. *Research Monograph, French-Azerbaijani University (UFAZ)*.
