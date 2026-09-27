# Research Plan: Azerbaijani ASR Benchmark Paper

**Title:** *Can Efficient Models Outperform Giants? A Multi-Domain ASR Benchmark and Linguistic Analysis for Azerbaijani*

**Author:** Famil Gulmammadov  
**Target:** Conference (Interspeech / LREC-COLING / EACL workshop) or Journal (MDPI Applied Sciences)  
**Timeline:** ~3–4 months

---

## 📌 Цель

Создать первый систематический публичный benchmark ASR моделей для азербайджанского языка.
Сравнить zero-shot, prompted и fine-tuned подходы. Дать практическую рекомендацию: какую модель использовать и когда.

---

## 🗂 Фазы проекта

---

### Фаза 1 — Подготовка данных (Неделя 1–2)

#### 1.1 Основной датасет
- [ ] Mozilla Common Voice 17.0 (Azerbaijani) — уже есть в репо
- [ ] Разбить на train / validation / test (стратифицированно по полу, возрасту если есть)
- [ ] Зафиксировать test set — он не меняется до конца эксперимента

#### 1.2 Дополнительные источники (опционально, но сильно повышает ценность)
- [ ] FLEURS (Google) — есть азербайджанский сплит, ~1h аудио
- [ ] VoxPopuli AZ — проверить наличие
- [ ] Свои записи — 50–100 коротких фраз, записанных вручную (out-of-domain test)

#### 1.3 Предобработка
- [ ] Нормализация текста (уже есть в part_b — переиспользовать)
- [ ] Фильтрация слишком коротких/длинных аудио
- [ ] Проверка кодировки (UTF-8, азербайджанские символы ə, ı, ğ, ş, ç, ö, ü)
- [ ] Статистика датасета: длина аудио, словарный запас, покрытие фонем

---

### Фаза 2 — Модели (Неделя 2–4)

#### 2.1 Zero-shot модели (без обучения, просто запуск)

| Модель | Размер | Источник |
|---|---|---|
| Whisper tiny | 39M | OpenAI |
| Whisper base | 74M | OpenAI |
| Whisper small | 244M | OpenAI |
| Whisper medium | 769M | OpenAI |
| Whisper large-v2 | 1.5B | OpenAI (уже есть) |
| Whisper large-v3 | 1.5B | OpenAI |
| MMS-300M | 300M | Meta |
| MMS-1B | 1B | Meta |

> Цель: показать scaling curve — как WER меняется с размером модели

#### 2.2 Prompted inference (уже частично есть)
- [ ] initial_prompt на азербайджанском для всех Whisper моделей
- [ ] Сравнить effect of prompt по размеру модели

#### 2.3 Fine-tuned модели (основной вклад paper'а)

| Модель для fine-tune | Почему |
|---|---|
| Whisper small | Быстрый, реалистичный для продакшна |
| Whisper medium | Баланс качество/ресурсы |
| wav2vec2-base (multilingual) | Другая архитектура для сравнения |

- [ ] Fine-tune через HuggingFace Trainer + `transformers`
- [ ] 10–20 эпох, early stopping по WER на validation
- [ ] Сохранить checkpoint для воспроизводимости

---

### Фаза 3 — Эксперименты (Неделя 3–5)

#### 3.1 Основные метрики
- **WER** (Word Error Rate) — главная метрика
- **CER** (Character Error Rate) — важно для агглютинативного языка
- **RTF** (Real-Time Factor) — скорость инференса (практическая важность)

#### 3.2 Матрица экспериментов

```
Для каждой модели:
  → Zero-shot WER/CER на Common Voice test
  → Zero-shot WER/CER на FLEURS (если есть)
  → + Prompt → WER/CER
  → + Fine-tune → WER/CER
  → RTF (CPU и GPU)
```

#### 3.3 Анализ ошибок (это делает paper сильнее)
- [ ] Топ-50 слов с наибольшим числом ошибок
- [ ] Confusion matrix по символам (какие буквы путает)
- [ ] Ошибки на специфичных азербайджанских фонемах: ə, ğ, x, q
- [ ] OOV (Out-of-Vocabulary) анализ
- [ ] Примеры плохих предсказаний с разбором почему

---

### Фаза 4 — Написание paper'а (Неделя 5–8)

#### Структура статьи

```
1. Abstract (~150 слов)
2. Introduction
   - Зачем ASR для азербайджанского
   - Что уже есть (мало)
   - Наш вклад (bullet points)
3. Related Work
   - Whisper, MMS, wav2vec2
   - Low-resource ASR в целом
   - Azerbaijani NLP (что есть)
4. Data
   - Common Voice, FLEURS
   - Статистика, предобработка
5. Models & Methods
   - Описание каждой модели
   - Параметры fine-tuning
   - Метрики
6. Results
   - Таблица с WER/CER всех моделей
   - График scaling curve
   - Анализ ошибок
7. Discussion
   - Что работает лучше и почему
   - Практические рекомендации
8. Conclusion
References
```

#### Разделение работы (примерно)
- **Ты:** zero-shot эксперименты, fine-tuning, код, таблицы результатов
- **Партнёр:** анализ ошибок, написание Related Work, Introduction

---

### Фаза 5 — Публикация (Неделя 8–12)

#### Цели для подачи (от реалистичного к амбициозному)

| Venue | Дедлайн (ориентир) | Сложность |
|---|---|---|
| EACL / ACL Workshop (low-resource NLP) | Следи за 2027 | ⭐⭐ |
| Interspeech | Март каждого года | ⭐⭐⭐ |
| LREC-COLING | Раз в 2 года | ⭐⭐ |
| MDPI Applied Sciences | Rolling | ⭐ (проще всего) |
| arXiv | В любое время | Бесплатно, сразу |

> **Стратегия:** сначала выложить на arXiv, потом подать на workshop или MDPI

#### Что опубликовать вместе с paper'ом
- [ ] Код на GitHub (уже есть репо — расширить)
- [ ] Fine-tuned модели на HuggingFace Hub
- [ ] Test set (если лицензия Common Voice позволяет)

---

## 📊 Ожидаемые результаты

```
Гипотеза 1: Fine-tuned Whisper small > Zero-shot Whisper large-v2 по WER
Гипотеза 2: Prompt дает +1–3% WER улучшение для всех Whisper моделей  
Гипотеза 3: CER более информативен чем WER для азербайджанского (агглютинация)
```

---

## 🛠 Технический стек

```python
# Основное
transformers>=4.40
datasets
jiwer          # WER/CER (уже есть)
evaluate

# Fine-tuning
accelerate
torch>=2.0

# Анализ
pandas
matplotlib
seaborn
```

**Железо:** Google Colab Pro или Kaggle (бесплатный T4) для fine-tuning

---

## ⚠️ Риски и как их избежать

| Риск | Решение |
|---|---|
| Common Voice AZ слишком мал для fine-tune | Использовать data augmentation (speed perturbation, noise) |
| Fine-tuned модель хуже zero-shot | Это тоже результат — публикуется как negative finding |
| Нет доступа к GPU | Kaggle / Colab дают бесплатно T4 |
| Мало новизны | Акцент на анализ ошибок + практические рекомендации |

---

## ✅ Чеклист по неделям

| Неделя | Задача |
|---|---|
| 1 | Собрать все датасеты, зафиксировать test set |
| 2 | Запустить все zero-shot модели, собрать baseline таблицу |
| 3 | Fine-tune Whisper small + medium |
| 4 | Fine-tune wav2vec2, промеры RTF |
| 5 | Анализ ошибок |
| 6 | Черновик paper'а |
| 7 | Ревизия, таблицы, графики |
| 8 | Финальный paper + выложить на arXiv |
| 9–12 | Подача на conference/journal |
