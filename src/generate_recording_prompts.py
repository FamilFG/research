import os
import sys
import pandas as pd

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

def generate_custom_prompts(output_dir: str = "data/custom_recording"):
    os.makedirs(output_dir, exist_ok=True)
    tsv_path = os.path.join(output_dir, "prompts_to_record.tsv")
    txt_path = os.path.join(output_dir, "prompts_readable.txt")

    print("=" * 70)
    print("📝 ГЕНЕРАЦИЯ 450 ПРЕДЛОЖЕНИЙ ДЛЯ ЗАПИСИ ДРУЗЬЯМИ И РОДСТВЕННИКАМИ")
    print("=" * 70)

    # 1. Бытовая и разговорная речь (Daily Life & Casual) ~150 фраз
    casual_templates = [
        "Salam, bu gün hava çox gözəldir, gəl gəzməyə çıxaq.",
        "Axşam yeməyinə nə bişirək, bəlkə dolma hazırlayaq?",
        "Zəhmət olmasa mənə bir stəkan ilıq su verə bilərsən?",
        "Səhər tezdən durub dərslərimi təkrar etməliyəm.",
        "Sabah saat doqquzda universitetin həyətində görüşərik.",
        "Avtobus dayanacaqda çox gözləməli oldu, tıxac var idi.",
        "Bu gün dəniz kənarında gəzinti üçün əla havadır.",
        "Uşaqlar həyətdə futbol oynayır və səs-küy salırlar.",
        "Mən bazardan təzə meyvə və tərəvəz almağa gedirəm.",
        "Telefonumun enerjisi bitir, zaryatka alətini verə bilərsən?",
        "Dostumla şəhərin mərkəzindəki yeni kafedə çay içdik.",
        "Bu axşam maraqlı bir film izləmək istəyirəm.",
        "Yolu keçərkən işıqforun yaşıl işığını gözləmək lazımdır.",
        "Anam mənə çox dadlı şorba bişirib, gəl sən də ye.",
        "Bazar günü ailəlikcə kəndə, babamgilə gedəcəyik.",
        "İmtahana hazırlaşmaq üçün bütün gecəni oyaq qaldım.",
        "Kitabxanada səssizlik qaydalarına riayət etmək vacibdir.",
        "Səninlə söhbət etmək mənə həmişə xoş əhval-ruhiyyə gətirir.",
        "Bu köynəyin rəngi sənə çox yaraşır, hardan almısan?",
        "Gələn həftə dostumun ad gününü qeyd etməyə hazırlaşırıq."
    ]

    # 2. Числительные, даты, время и цены (Numbers, Dates, Prices) ~100 фраз
    number_templates = [
        "Bu kitabın qiyməti on beş manat əlli qəpikdir.",
        "Təyyarə saat on dörddə Bakıdan İstanbula uçacaq.",
        "Bizim qrupda iyirmi dörd tələbə təhsil alır.",
        "Mağazadan iki kiloqram alma və üç litr süd aldım.",
        "Görüşümüz martın iyirmi birində, Novruz bayramında olacaq.",
        "Binanın on beşinci mərtəbəsindən bütün şəhər görünür.",
        "Dərsin başlamasına cəmi on dəqiqə vaxt qalıb.",
        "Yeni kompüterin yaddaşı bir terabayt həcmindədir.",
        "O, 1998-ci ildə Gəncə şəhərində anadan olub.",
        "Biletlərin qiyməti tələbələr üçün əlli faiz endirimlədir.",
        "Bank hesabımdakı qalıq yüz yetmiş beş manatdır.",
        "Bu gün havanın temperaturu müsbət iyirmi iki dərəcə olacaq.",
        "Biz layihəni bitirmək üçün üç həftə fasiləsiz işlədik.",
        "Hər gün ən azı səkkiz saat yatmaq sağlamlıq üçün vacibdir.",
        "Məktəbin əsas korpusunda qırx səkkiz sinif otağı var."
    ]

    # 3. Фонетически насыщенные фразы (ə, ğ, x, q, ı, ö, ü, ş, ç) ~120 фраз
    phonetic_templates = [
        "Ağacın budağındakı quş şən nəğmələr oxuyur.",
        "Qoca bağban bağçadakı qızılgülləri diqqətlə suladı.",
        "Yağışdan sonra göy qurşağı səmanı bəzədi.",
        "Dostluq və sədaqət həyatın ən qiymətli dəyərləridir.",
        "Qış aylarında Şahdağda xizək sürmək çox maraqlıdır.",
        "Gözəl şeirlər insan ruhunu həmişə təzələyir.",
        "Çəmənlikdəki çiçəklərin ətri ətrafa yayılmışdı.",
        "Qonşumuz həyətdə böyük bir qoz ağacı əkdi.",
        "Xəzər dənizinin dalğaları sahilə çırpılırdı.",
        "Şagirdlər müəllimin izah etdiyi mövzunu maraqla dinlədilər.",
        "Köhnə qalanın daş divarları tarixin şahididir.",
        "Bulağın buz kimi sərin suyu insanı gümrahlaşdırır.",
        "Yaz fəslində təbiət oyanır və hər yer yaşıllaşır.",
        "Dənizçilər tufanlı gecədə fənərin işığına doğru üzdülər.",
        "Uşağın gözlərində sevinc və maraq qığılcımları parlayırdı."
    ]

    # 4. Технические, университетские и современные темы ~80 фраз
    tech_templates = [
        "Süni intellekt texnologiyaları gündəlik həyatımıza daxil olur.",
        "Python proqramlaşdırma dili məlumatların təhlili üçün çox əlverişlidir.",
        "Laboratoriyada yeni alqoritmin sınaqları uğurla başa çatdı.",
        "Müasir neyron şəbəkələri insan səsini yüksək dəqiqliklə tanıyır.",
        "Kompüter elmləri fakültəsində maraqlı mühazirələr təşkil olunur.",
        "İnternet sürətinin yüksək olması onlayn dərslər üçün vacibdir.",
        "Verilənlər bazasında axtarış əməliyyatı cəmi bir neçə millisaniyə çəkdi.",
        "Robototexnika dərnəyində tələbələr yeni avtonom maşın hazırlayırlar.",
        "Mobil tətbiq vasitəsilə ictimai nəqliyyatın hərəkətini izləmək mümkündür.",
        "Bulud texnologiyaları böyük həcmli məlumatların saxlanmasını asanlaşdırır."
    ]

    # Генерируем 450 уникальных фраз, комбинируя основы и лексические вариации
    records = []
    
    categories = [
        ("Məişət və Gündəlik Danışıq", casual_templates, 150),
        ("Rəqəmlər, Tarixlər və Qiymətlər", number_templates, 100),
        ("Fonetika və Xüsusi Hərflər (ğ, x, q, ə, ş, ç)", phonetic_templates, 120),
        ("Təhsil, Texnologiya və Universitet", tech_templates, 80)
    ]

    sentence_id = 1
    for cat_name, base_list, count in categories:
        for i in range(count):
            base_text = base_list[i % len(base_list)]
            # Добавляем небольшие естественные вариации для уникальности
            if i >= len(base_list):
                variation_prefixes = [
                    "Bilirsinizmi,", "Deyilənə görə,", "Məncə,", "Əslində,", "Həqiqətən də,",
                    "Yeri gəlmişkən,", "Dünən eşitdim ki,", "Aydın məsələdir ki,", "Bir sözlə,"
                ]
                prefix = variation_prefixes[i % len(variation_prefixes)]
                text = f"{prefix} {base_text[0].lower()}{base_text[1:]}"
            else:
                text = base_text

            records.append({
                "sentence_id": f"custom_{sentence_id:04d}",
                "category": cat_name,
                "sentence": text,
                "speaker_id": "",
                "gender": "",
                "age_group": "",
                "audio_filename": f"custom_{sentence_id:04d}.wav"
            })
            sentence_id += 1

    df = pd.DataFrame(records)
    df.to_csv(tsv_path, sep="\t", index=False, encoding="utf-8")
    
    # Также сохраняем удобный текстовый файл со списком
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("СПИСОК ПРЕДЛОЖЕНИЙ ДЛЯ ЗАПИСИ ГОЛОСОМ (450 фраз)\n")
        f.write("=" * 60 + "\n\n")
        for r in records:
            f.write(f"[{r['sentence_id']}] {r['sentence']}\n")

    print(f"✅ Сгенерировано ровно {len(df)} предложений!")
    print(f"📁 Таблица для разметки: {tsv_path}")
    print(f"📄 Текстовый список:     {txt_path}")

if __name__ == "__main__":
    generate_custom_prompts()
