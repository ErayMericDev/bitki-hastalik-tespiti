"""
============================================================
ADIM 3a: BİLGİ TABANI + SEQ2SEQ EĞİTİM VERİSİ ÜRETİCİ
Gerçek tarım extension kaynaklarından derlenmiş bilgiler
============================================================

KAYNAK GÜVENİLİRLİĞİ:
Bu bilgi tabanındaki tüm öneriler aşağıdaki üniversite tarım
extension kaynaklarından derlenmiştir (uydurma DEĞİL):
  - University of Minnesota Extension
  - University of Wisconsin Vegetable Pathology
  - UC IPM (California)
  - University of Maine Cooperative Extension
  - Clemson University, Alabama Extension, Cornell
  - Purdue, LSU AgCenter, UMass Amherst

Her hastalık için: belirtiler + kültürel önlemler + kimyasal/biyolojik
mücadele bilgileri gerçek kaynaklardan alınmıştır.

Veri çeşitlendirme için şablon varyasyonu kullanılır (PlantVillageVQA'nın
kendisi de aynı yöntemi kullanmıştır) ama İÇERİK bilimsel kaynaklıdır.

Çıktı: seq2seq_egitim_verisi.json
"""

import json
import random

random.seed(42)

# ============================================================
# BİLGİ TABANI — gerçek tarım kaynaklarından derlenmiş
# ============================================================
# Her sınıf için:
#   - belirtiler: hastalığın görsel belirtileri
#   - patojen: hastalığa neden olan etken
#   - kulturel: kültürel mücadele yöntemleri (liste)
#   - kimyasal: kimyasal/biyolojik mücadele (liste)
#   - kosul: hastalığı tetikleyen çevre koşulları

BILGI_TABANI = {
    "Tomato_healthy": {
        "durum": "saglikli",
        "baslik": "Sağlıklı Domates",
        "belirtiler": [
            "Yapraklar canlı yeşil ve homojen görünümde",
            "Leke, sararma veya solgunluk belirtisi yok",
            "Doku yapısı düzgün ve sağlıklı",
        ],
        "oneriler": [
            "Mevcut sulama programınızı sürdürün",
            "Damla sulama ile yaprakları kuru tutun",
            "Düzenli olarak hastalık belirtisi açısından kontrol edin",
            "Bitkiler arası hava akışını koruyun",
            "Dengeli gübreleme uygulayın",
        ],
    },
    "Potato___healthy": {
        "durum": "saglikli",
        "baslik": "Sağlıklı Patates",
        "belirtiler": [
            "Yapraklar canlı yeşil renkte",
            "Yanıklık veya leke belirtisi bulunmuyor",
            "Bitki gelişimi normal seyrediyor",
        ],
        "oneriler": [
            "Mevcut bakım rutinine devam edin",
            "Düzenli tarla kontrolü yapın",
            "Yaprak ıslaklığını en aza indirin",
            "Dengeli azot ve potasyum takviyesi yapın",
        ],
    },
    "Pepper__bell___healthy": {
        "durum": "saglikli",
        "baslik": "Sağlıklı Biber",
        "belirtiler": [
            "Yapraklar parlak yeşil ve sağlam",
            "Bakteriyel leke veya başka hastalık belirtisi yok",
            "Bitki canlı ve dinç görünüyor",
        ],
        "oneriler": [
            "Mevcut yetiştirme koşullarını koruyun",
            "Düzenli sulama ve gübreleme yapın",
            "Hastalık önlemek için aletleri temiz tutun",
            "Bitkiler arası uygun mesafeyi koruyun",
        ],
    },
    "Tomato_Late_blight": {
        "durum": "hastalikli",
        "baslik": "Domates Mildiyösü (Geç Yanıklık)",
        "belirtiler": [
            "Yaprak ve gövdede su emmiş görünümlü düzensiz lekeler",
            "Lekeler zamanla kahverengiye döner",
            "Nemli koşullarda lekelerin altında beyaz küf gelişimi",
            "Bitki dondan etkilenmiş gibi görünür",
        ],
        "oneriler": [
            "Enfekte yaprakları derhal kesip imha edin (gömün veya yakın)",
            "Çiçeklenmeden itibaren koruyucu fungisit uygulayın",
            "Damla sulama kullanın, yaprakları kuru tutun",
            "Sabah erken sulayın ki yapraklar güneşte kurusun",
            "Bitkileri kazık veya kafesle destekleyip hava akışı sağlayın",
            "Bitkiler arası mesafeyi açın, havalandırmayı artırın",
            "Domates, patates ve biber ekilmemiş alana 3-4 yıl rotasyonla ekin",
            "Sezon sonunda bitki artıklarını kaldırın",
        ],
        "kosul": "Serin ve nemli hava koşullarında hızla yayılır",
    },
    "Potato___Late_blight": {
        "durum": "hastalikli",
        "baslik": "Patates Mildiyösü (Geç Yanıklık)",
        "belirtiler": [
            "Yapraklarda su emmiş görünümlü koyu lekeler",
            "Lekeler hızla büyüyüp kahverengi-siyah nekroza döner",
            "Nemli havada yaprak altında beyaz sporlanma",
            "Yumrularda kahverengi-kırmızımsı çürüme",
        ],
        "oneriler": [
            "Enfekte bitkileri sökerek imha edin",
            "Hastalık başlamadan önce koruyucu fungisit uygulayın",
            "Yaprakları mümkün olduğunca kuru tutun",
            "Damla sulama veya sızdıran hortum kullanın",
            "Sabah sulayın, hava akışı için aralıklı dikim yapın",
            "Dirençli çeşitler tercih edin",
            "Sezon sonunda atık yığınlarını yönetin",
        ],
        "kosul": "Serin, ıslak hava hastalığı tetikler",
    },
    "Potato___Early_blight": {
        "durum": "hastalikli",
        "baslik": "Patates Erken Yanıklık",
        "belirtiler": [
            "Yaşlı yapraklarda iç içe halkalı (hedef tahtası) lekeler",
            "Yaprak sararması ve erken dökülme",
            "Lekeler etrafında sarı haleler",
        ],
        "oneriler": [
            "Chlorothalonil veya mancozeb içerikli fungisit uygulayın",
            "Enfekte bitki artıklarını temizleyip imha edin",
            "Bitkiyi stresten koruyun, dengeli sulama yapın",
            "Yeterli azot takviyesi yapın (azot eksikliği duyarlılığı artırır)",
            "Ürün rotasyonu uygulayın",
            "Dirençli çeşitler kullanın",
            "Hastalık ilk belirtide ilaçlamaya başlayın",
        ],
        "kosul": "Sıcak (28-30°C) ve nemli koşullar hastalığı hızlandırır",
    },
    "Tomato_Bacterial_spot": {
        "durum": "hastalikli",
        "baslik": "Domates Bakteriyel Leke",
        "belirtiler": [
            "Yaprak ve meyvede küçük, koyu, ıslak görünümlü lekeler",
            "Lekeler birleşerek yaprak dökülmesine yol açar",
            "Meyvede kabarcıklı, kabuklu lekeler",
        ],
        "oneriler": [
            "Sabit bakır içerikli ürünleri koruyucu olarak uygulayın",
            "Bakır etkisini artırmak için mancozeb ile tank karışımı yapın",
            "İlk belirtide başlayıp 7-10 günde bir tekrarlayın",
            "Hastalıksız sertifikalı tohum ve fide kullanın",
            "Bitkiler arası en az 60 cm mesafe bırakın",
            "Yapraklar ıslakken bitkilere dokunmayın, işlem yapmayın",
            "Aletleri dezenfekte edin, ürün rotasyonu uygulayın",
        ],
        "kosul": "Sıcak ve nemli koşullarda yayılır; bakır tek başına koruyucudur, tedavi edici değil",
    },
    "Pepper__bell___Bacterial_spot": {
        "durum": "hastalikli",
        "baslik": "Biber Bakteriyel Leke",
        "belirtiler": [
            "Yapraklarda küçük, koyu, su emmiş lekeler",
            "Lekeler sararıp yaprak dökülmesine neden olur",
            "Meyvede kabuklu, çatlaklı lekeler",
        ],
        "oneriler": [
            "Sabit bakır bazlı bakterisitler uygulayın",
            "Bakırı mancozeb ile karıştırarak etkinliği artırın",
            "Fideleme sonrası koruyucu ilaçlamaya başlayın, haftalık tekrarlayın",
            "Sertifikalı, hastalıksız tohum kullanın",
            "Bitkiler arası en az 45 cm mesafe bırakın",
            "Islak bitkilerle çalışmaktan kaçının",
            "Bakteri dayanıklılığına karşı ürün rotasyonu yapın",
        ],
        "kosul": "Sıcak, nemli iklimlerde en yaygın biber hastalığıdır",
    },
    "Tomato_Leaf_Mold": {
        "durum": "hastalikli",
        "baslik": "Domates Yaprak Küfü",
        "belirtiler": [
            "Yaprak üst yüzeyinde soluk yeşil-sarı lekeler",
            "Yaprak alt yüzeyinde zeytin yeşili-kahverengi kadifemsi küf",
            "Yaşlı yapraklardan başlayıp yukarı doğru yayılır",
            "Şiddetli durumda yaprak ölümü ve dökülme",
        ],
        "oneriler": [
            "Nispi nemi %85'in altında tutun (en kritik önlem)",
            "Sürekli fanlarla hava sirkülasyonu sağlayın",
            "Üstten sulamadan kaçının, yaprakları kuru tutun",
            "Sera sıcaklığını gece düşmesini engelleyecek şekilde ayarlayın",
            "Enfekte yaprakları budayıp poşetleyerek imha edin",
            "Budama ve uygun aralıkla hava akışını artırın",
            "Dirençli çeşitler kullanın (Santa Fe, Legend gibi)",
            "Gerekirse azoxystrobin/difenoconazole veya chlorothalonil uygulayın",
        ],
        "kosul": "Yüksek nem (>%85) ve 21-24°C sıcaklıkta hızla gelişir; özellikle sera/tünel sorunudur",
    },
}


# ============================================================
# METİN ÜRETİM ŞABLONLARI (çeşitlendirme için)
# ============================================================

# Girdi varyasyonları — model farklı ifadelerle sorulsa da öğrensin
GIRDI_SABLONLARI = [
    "hastalik: {etiket}",
    "tespit: {etiket} bitki: {bitki}",
    "durum: {etiket}",
    "teshis: {etiket} sinif: {durum}",
]

# Açılış cümlesi varyasyonları
ACILIS_SAGLIKLI = [
    "Bitkiniz sağlıklı görünüyor.",
    "Analiz sonuçları olumlu.",
    "Bitkide herhangi bir hastalık belirtisi tespit edilmedi.",
    "Yaprak sağlıklı bir görünüm sergiliyor.",
]

ACILIS_HASTALIKLI = [
    "{baslik} tespit edildi.",
    "Bitkinizde {baslik} belirtileri görülüyor.",
    "Analiz {baslik} işaret ediyor.",
    "{baslik} hastalığı saptandı.",
]


def uret_oneri_metni(etiket, bilgi):
    """Bir hastalık için çeşitlendirilmiş öneri metni üret."""
    parts = []

    if bilgi["durum"] == "saglikli":
        parts.append(random.choice(ACILIS_SAGLIKLI))
        # 2-3 öneri seç
        secilen = random.sample(bilgi["oneriler"], min(3, len(bilgi["oneriler"])))
        parts.extend(secilen)
    else:
        acilis = random.choice(ACILIS_HASTALIKLI).format(baslik=bilgi["baslik"])
        parts.append(acilis)
        # 1 belirti cümlesi ekle (bazen)
        if random.random() > 0.4:
            parts.append(random.choice(bilgi["belirtiler"]) + ".")
        # 3-4 öneri seç
        secilen = random.sample(bilgi["oneriler"], min(4, len(bilgi["oneriler"])))
        parts.extend(secilen)

    return " ".join(parts)


def uret_girdi_metni(etiket, bilgi):
    """Bir hastalık için girdi metni üret."""
    bitki = etiket.split("_")[0].lower()
    sablon = random.choice(GIRDI_SABLONLARI)
    return sablon.format(etiket=etiket, bitki=bitki, durum=bilgi["durum"])


# ============================================================
# ANA ÜRETİM
# ============================================================

def main():
    print("=" * 60)
    print("SEQ2SEQ EĞİTİM VERİSİ ÜRETİLİYOR")
    print("Gerçek tarım kaynaklarından derlenmiş bilgi tabanı")
    print("=" * 60)

    ORNEK_PER_SINIF = 350  # Her hastalık için kaç örnek

    dataset = []

    for etiket, bilgi in BILGI_TABANI.items():
        print(f"   [{bilgi['baslik']}] {ORNEK_PER_SINIF} örnek üretiliyor...")
        for _ in range(ORNEK_PER_SINIF):
            girdi = uret_girdi_metni(etiket, bilgi)
            cikti = uret_oneri_metni(etiket, bilgi)
            dataset.append({
                "input": girdi,
                "output": cikti,
                "etiket": etiket,
            })

    random.shuffle(dataset)

    print(f"\n   Toplam üretilen örnek: {len(dataset)}")

    # Kaydet
    with open("seq2seq_egitim_verisi.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    # Bilgi tabanını da ayrıca kaydet (birleşik sistemde kullanılacak)
    with open("bilgi_tabani.json", "w", encoding="utf-8") as f:
        json.dump(BILGI_TABANI, f, ensure_ascii=False, indent=2)

    print(f"\n   Çıktılar:")
    print(f"   - seq2seq_egitim_verisi.json ({len(dataset)} örnek)")
    print(f"   - bilgi_tabani.json (referans için)")

    # Örnek göster
    print("\n   --- ÖRNEK VERİLER ---")
    for i in range(4):
        print(f"\n   [Örnek {i+1}]")
        print(f"   GİRDİ : {dataset[i]['input']}")
        print(f"   ÇIKTI : {dataset[i]['output']}")

    print("\n" + "=" * 60)
    print("TAMAMLANDI! Sıradaki adım: seq2seq modelini bu veriyle eğit.")
    print("=" * 60)


if __name__ == "__main__":
    main()