# 📝 Научный дневник и аналитические инсайты для статьи
**Автор:** Famil Gulmammadov (French-Azerbaijani University - UFAZ)  
**Тема:** Cross-Domain Zero-Shot and Parameter-Efficient Adaptation of Speech Foundation Models for Azerbaijani ASR

---

## 📌 Зачем нужен этот документ:
Здесь в реальном времени аккумулируются все качественные наблюдения, лингвистический анализ ошибок, аномалии и готовые формулировки на английском/русском языках. Когда эксперименты закончатся, отсюда будут напрямую браться абзацы для секций **Methodology**, **Results**, **Error Analysis** и **Discussion**.

---

## 1. Доменная матрица экспериментов (Benchmark Setup)

| Домен в статье | Датасет | Размер (аудио / часы) | Акустический профиль | Лексический профиль |
| :--- | :--- | :--- | :--- | :--- |
| **Domain 1: Studio / Read Speech** | **Google FLEURS** | 923 аудио / 3.24 ч | Профессиональные студийные микрофоны, идеальное SNR (>25 dB), литературное произношение. | Формальный литературный азербайджанский, Wikipedia, новости, длинные сложноподчиненные предложения (до 52 слов). |
| **Domain 2: Crowdsourced Domestic** | **Mozilla Common Voice** | 126 аудио / 0.20 ч | Домашние микрофоны ноутбуков, комнатная реверберация, разный возраст дикторов. | Повседневные разговорные фразы (5–14 слов), разнообразные акценты. |
| **Domain 3: Educational Screencast** | **LocalDoc** | 500 аудио / 0.80 ч | Петличные/настольные USB микрофоны, звук кликов мыши, фоновое дыхание. | Полуспонтанная лекционная речь, технический сленг, англицизмы (*fayl, kod, funksiya, interfeys, klaster*). |
| **Domain 4: In-the-Wild Mobile** *(В процессе сбора)* | **Custom WhatsApp** | 100 аудио / ~0.15 ч | Смартфоны, компрессия аудиокодеков Opus/AAC, уличный шум, эхо кухни/комнаты. | Разговорные интонации, диалектные словечки, сокращения. |

---

## 2. Эмпирические инсайты Zero-Shot моделей

### Инсайт 2.1: Феномен «Катастрофического авторегрессионного зацикливания» (Repetition Hallucination)
- **Наблюдение:** На модели `whisper-tiny` в тесте Common Voice получен WER = **201.24%**.
- **Причина:** Декомпозиция ошибок показала, что уровень вставок (**Insertion Rate = 118.08%**) превышает 100%! Декодер Whisper не может уверенно декодировать азербайджанские морфемы в условиях домашней реверберации и срывается в бесконечный генеративный луп:
  - *Референс:* `Orta məktəbi bitirdikdən sonra Ukraynanın Lvov şəhərində hərbi xidmətdə olmuşdur.`
  - *Предикшн Whisper-Tiny:* `Orta məkterdi bir dirdikten sonra ukranalını bu şərində şərində şərində şərində şərində... [448 токенов повторов]`
  - *Референс:* `Onların qalıqlarına Çin və Özbəkistan ərazilərində aşkarlanmışdır.`
  - *Предикшн Whisper-Tiny:* `O man qələ qələ qələ qələ qələ qələ... [повторы]`
- **Академическая ценность для статьи:**  
  *Draft paragraph for Paper:*
  > *"Contrary to high-resource languages where zero-shot errors are predominantly phonetic substitutions, our error decomposition on Azerbaijani reveals that small-footprint foundation models suffer from severe autoregressive repetition loops (Insertion Rate exceeding 118%). The model fails to emit the end-of-sequence token `<|endoftranscript|>`, continuously generating repetitive n-grams until reaching the architectural maximum context limit (448 tokens). This provides empirical evidence that zero-shot deployment of sub-100M parameter models is inherently unviable without parameter adaptation."*

---

### Инсайт 2.2: Акустический градиент галлюцинаций (Cross-Domain Gap)
- **Сравнение:**
  - Common Voice (домашний шум): WER = **201.24%** (Ins = 118.1%)
  - Google FLEURS (студия): WER = **145.63%** (Ins = 63.2%)
  - LocalDoc (видеолекции): WER = **120.91%** (Ins = 49.4%)
- **Вывод:** Качество студийной записи снижает галлюцинации почти в 2.5 раза (с 118% до 49%), однако даже в идеальной студии WER остается выше 100%. Это доказывает, что проблема носит не чисто акустический, а глубоко лингвистический характер (недостаток азербайджанских данных в pre-training корпусе Whisper).

---

### Инсайт 2.3: Лингвистический аудит фонемного инвентаря
- **Результаты аудита специфических букв азербайджанского алфавита на Whisper-Tiny:**
  - `ə` (самая высокочастотная буква азербайджанского языка): **0.0% accuracy**
  - `ğ`: **0.0% accuracy**
  - `ş`: **0.0% accuracy**
  - `ç`: **0.0% accuracy**
  - `ü`: **0.0% accuracy**
  - `ö`: **32.6% accuracy**
  - `ı`: **24.8% accuracy**
- **Вывод:** Модель полностью глуха к 5 из 7 фундаментальных гласных и согласных фонем азербайджанского языка. Модель пытается подменять их турецкими или общими латинскими символами.

---

### Инсайт 2.4: Инженерно-экономический баланс (RTF vs VRAM)
- На видеокарте потребительского уровня **NVIDIA GeForce RTX 5060 Ti (16 GB VRAM)** с использованием параллельного 8-поточного I/O и Tensor Cores TF32:
  - 3.24 часа студийного аудио FLEURS обрабатываются за **239 секунд** (менее 4 минут).
  - Реальный фактор времени (RTF) составил **0.0205**, что соответствует **48.8-кратному ускорению** относительно реального времени речи!
  - Это доказывает высочайшую практическую применимость локального развертывания для индустриальных задач в Азербайджане (банки, суды, университеты).

---

### Инсайт 2.5: Анализ ошибок технического лекционного домена (LocalDoc: IT-термины, англицизмы, цифры)
При детальном изучении первых 50 фраз из `predictions.csv` теста LocalDoc выявлены 4 критические закономерности:
1. **Фонетическая транслитерация англицизмов и IT-брендов:**
   - Модель не понимает контекст кода и транскрибирует английские термины на слух кириллизированной/турецкой фонетикой:
     - `Python` ➔ `paytın` / `paytan`
     - `Java` ➔ `Cava`
     - `Fortran` ➔ `Fartan`
     - `ABC və C dilləri` ➔ `eibisi və sidelərə` (буквенное английское произношение A-B-C)
     - `proqramı` ➔ `programı` (турецкая замена `q` на `g`)
2. **Словесное раскрытие числительных (Verbalization vs Digits):**
   - `1991-ci ilin` ➔ `Mindokguz dosan bilinzinin` (попытка модели развернуть дату словами, приводящая к каскаду ошибок WER).
   - `95%` ➔ `doksan beş faiz` (вместо процентов).
3. **Локальные всплески галлюцинаций даже в хорошем звуке:**
   - Строка 5: `...burada yazmaq mümkündür` ➔ `...burada yazmak mümkür mümkür mümkür... mümkün mümkün [100 повторов]`.
   - Даже на лекциях с петличным микрофоном конечное слово способно сорвать декодер в луп.

---

### Инсайт 2.6: Эволюция фонемного инвентаря: Скачок от Tiny (39M) к Base (74M)
При увеличении размера модели всего в 2 раза (с 39M до 74M параметров) произошел фундаментальный лингвистический фазовый переход:
- **Распознавание специфических букв азербайджанского алфавита (FLEURS):**
  - Буква **`ə`**: взлетела с **0.0%** до **42.8%**!
  - Буква **`ş`**: взлетела с **0.0%** до **75.0%**!
  - Буква **`ç`**: взлетела с **0.0%** до **54.8%**!
  - Буква **`ö`**: выросла с **32.6%** до **78.6%**!
  - Буква **`ü`**: взлетела с **0.0%** до **59.5%**!
  - Буква **`ı`**: выросла с **24.8%** до **66.5%**!
  - Буква **`ğ`**: осталась критически сложной (**4.6%**).
- **Снижение галлюцинаций:**
  - Common Voice: Ins Rate упал с **118.1%** до **27.2%** (в 4.3 раза!).
  - FLEURS: Ins Rate упал с **63.2%** до **15.3%** (в 4.1 раза!).
  - LocalDoc: Ins Rate упал с **49.4%** до **18.9%** (в 2.6 раза!).
- **Вывод для статьи:**  
  *Draft paragraph for Paper:*
  > *"Scaling the parameter count from 39M (Tiny) to 74M (Base) triggers a clear linguistic phase transition. While Tiny suffers from complete blindness to key Turkic graphemes (0% accuracy on 'ə', 'ş', 'ç', 'ü'), Base successfully recovers graphemic representations with accuracy reaching up to 78.6% for 'ö' and 75.0% for 'ş'. Simultaneously, autoregressive repetition loops drop by over 4-fold across all acoustic domains."*

---

## 3. Автоматически сгенерированные графики для статьи (в `results/plots/`)
1. **`fig1_wer_cross_domain.png` / `.pdf`**: Сравнение WER моделей по 3 доменам со столбиками ошибок.
2. **`fig2_error_decomposition.png` / `.pdf`**: Декомпозиция ошибок (Substitutions, Deletions, Insertions), демонстрирующая природу галлюцинаций.
3. **`fig3_efficiency_tradeoff.png` / `.pdf`**: Двумерная плоскость компромисса: точность (WER) против скорости (RTF).
4. **`fig4_phoneme_radar.png` / `.pdf`**: Радарная диаграмма точности специфических азербайджанских фонем (`ə`, `ı`, `ğ`, `ş`, `ç`, `ö`, `ü`).
