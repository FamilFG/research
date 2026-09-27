# Research Abstract

**Title:**  
# Can Efficient Models Outperform Giants? A Multi-Domain ASR Benchmark and Linguistic Analysis for Azerbaijani

**Author:** Famil Gulmammadov  
**Target Venues:** ACL / EACL Workshop on Low-Resource NLP, Interspeech, or LREC-COLING  

---

## Abstract

Despite recent breakthroughs in multilingual Automatic Speech Recognition (ASR) driven by foundation models such as OpenAI Whisper and Meta MMS, low-resource and morphologically rich languages like Azerbaijani remain under-benchmarked. Existing efforts predominantly rely on small, isolated evaluation splits without systematic comparison across model scales, acoustic environments, or adaptation paradigms. 

In this work, we present the first comprehensive, multi-domain empirical benchmark of modern ASR architectures for the Azerbaijani language. We systematically evaluate eight state-of-the-art model variants—ranging from 39M to 1.5B parameters—across three distinct acoustic domains: crowdsourced everyday speech (Mozilla Common Voice), controlled studio recordings (Google FLEURS), and real-world educational lecture screencasts (LocalDoc), aggregating over 23 hours of speech. 

We investigate whether parameter-efficient fine-tuning (PEFT / LoRA) applied to compact architectures (e.g., Whisper-Small with 244M parameters) can match or surpass massive zero-shot models (Whisper-Large with 1.5B parameters) while maintaining low inference latency (Real-Time Factor). Furthermore, we conduct an in-depth linguistic error analysis tailored to the agglutinative nature of Azerbaijani, examining why Character Error Rate (CER) provides a more representative evaluation than standard Word Error Rate (WER) and highlighting acoustic confusion patterns across specific phonemes (`ə`, `ğ`, `x`, `q`). 

Our study delivers:
1. The first open, reproducible multi-domain benchmark for Azerbaijani speech recognition;
2. Practical architectural recommendations for production deployment under compute constraints;
3. Open-source fine-tuned model checkpoints and reproducible evaluation pipelines.

---

**Keywords:** Automatic Speech Recognition (ASR), Low-Resource Languages, Azerbaijani, Multilingual Models, Whisper, LoRA, Agglutinative Morphology, Benchmark.
