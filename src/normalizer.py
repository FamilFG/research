import unicodedata
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

# Официальный стандартный нормализатор OpenAI Whisper
_whisper_normalizer = BasicTextNormalizer(remove_diacritics=False)

def azerbaijani_casing(text: str) -> str:
    """
    Корректный перевод в нижний регистр для азербайджанского языка:
    - Заглавная 'I' (латинская заглавная без точки) -> 'ı' (строчная без точки)
    - Заглавная 'İ' (латинская заглавная с точкой)  -> 'i' (строчная с точкой)
    - Заглавная 'Ə' (латинская заглавная Schwa)    -> 'ə' (строчная Schwa)
    """
    if not isinstance(text, str):
        return ""
    text = text.replace("I", "ı")
    text = text.replace("İ", "i")
    text = text.replace("Ə", "ə")
    return text

def clean_text(text: str) -> str:
    """
    Официальная нормализация текста на базе OpenAI Whisper Normalizer:
    1. Unicode NFC нормализация.
    2. Тюркский lower-casing (сохраняет ı / i / ə).
    3. Замена кириллической 'ә' на азербайджанскую латиницу 'ə'.
    4. Стандартный Whisper BasicTextNormalizer (удаляет пунктуацию, схлопывает пробелы).
    """
    if not isinstance(text, str):
        return ""

    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # 2. Тюркский регистр
    text = azerbaijani_casing(text)

    # 3. Замена случайно попавшей кириллической 'ә'
    text = text.replace("\u04d9", "ə").replace("\u04d8", "ə")

    # 4. Официальный Whisper Normalizer (удаляет знаки препинания, скобки, лишние пробелы)
    text = _whisper_normalizer(text)

    return text

if __name__ == "__main__":
    import sys
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("🧪 ОФИЦИАЛЬНЫЙ WHISPER-НОРМАЛИЗАТОР ДЛЯ АЗЕРБАЙДЖАНСКОГО")
    print("=" * 60)

    test_examples = [
        "Dastanın ilk nüsxəsi nəsrlə yazılmışdır.",
        "Salam, Dünya! Necəsiz?",
        "İncəsənət və İqtisadiyyat",
        "IŞIQ VƏ İSTİLİK",
        "Bakı Teatrında — 2024-cü il!",
        "Ədəbiyyat və Mədəniyyət"
    ]

    for ex in test_examples:
        cleaned = clean_text(ex)
        print(f"До:    {ex}")
        print(f"После: {cleaned}\n")
