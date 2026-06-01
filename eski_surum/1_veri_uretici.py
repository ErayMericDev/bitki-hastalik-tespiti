"""
============================================================
ADIM 1: VERİ ÜRETİCİ
Bitki Bakım Önerisi Seq2Seq Modeli İçin Eğitim Verisi
============================================================

Bu script, seq2seq modelini eğitmek için sentetik veri seti üretir.
Her örnek şunları içerir:
  - Girdi (input)  : Bitkinin durumu + özellik değerleri (yapılandırılmış metin)
  - Çıktı (output) : Doğal dilde bakım önerisi metni

Çıktı: training_data.json
"""

import json
import random
import numpy as np
from itertools import product

random.seed(42)
np.random.seed(42)

# ============================================================
# ÖNERİ ŞABLONLARI
# Her durum için birden fazla varyasyon — modelin ezberlememesi için
# ============================================================

OPENING_PHRASES = {
    "saglikli": [
        "Bitkinizin durumu gayet iyi görünüyor.",
        "Yaprak analizi olumlu sonuçlar veriyor.",
        "Bitkiniz sağlıklı bir görünüm sergiliyor.",
        "Genel bitki sağlığı tatmin edici düzeyde.",
        "Analiz sonuçları normal aralıkta.",
    ],
    "su_eksikligi": [
        "Bitkinizde su eksikliği belirtileri tespit edildi.",
        "Yapraklarda kuruma ve sararma gözlemleniyor.",
        "Bitkiniz susuz kalmış görünüyor.",
        "Toprak nem oranının düştüğü anlaşılıyor.",
        "Yaprak turgoru azalmış, sulama gerekli.",
    ],
    "besin_eksikligi": [
        "Bitkinizde besin eksikliği işaretleri var.",
        "Yapraklardaki sararma azot eksikliğine işaret ediyor.",
        "Bitki yeterli besin alamıyor görünüyor.",
        "Toprak besin değerlerinin düşük olduğu anlaşılıyor.",
        "Klorofil üretiminde azalma tespit edildi.",
    ],
    "hastalikli": [
        "Bitkinizde hastalık belirtileri tespit edildi.",
        "Yapraklarda mantar veya bakteriyel enfeksiyon olabilir.",
        "Lekeler ve renk bozuklukları hastalığa işaret ediyor.",
        "Bitkide patojen enfeksiyonu şüphesi var.",
        "Yapraklarda anormal lekelenme gözlemleniyor.",
    ],
}

# Özellik değerlerine göre detay cümleleri
DETAIL_BY_FEATURE = {
    "high_yellow": [
        "Sararma oranı dikkat çekici düzeyde.",
        "Sarı renk yapraklarda baskın hale gelmiş.",
        "Klorofil kaybı belirgin şekilde gözleniyor.",
    ],
    "high_brown": [
        "Kahverengileşme yaprak dokusunda yayılmış.",
        "Yapraklarda nekrotik bölgeler oluşmuş.",
        "Kahverengi alanlar geniş bir yer kaplıyor.",
    ],
    "high_dark": [
        "Koyu lekeler yaprak yüzeyinde belirgin.",
        "Siyah benekler hastalık şüphesini güçlendiriyor.",
        "Yaprak üzerinde koyu noktalar yayılmış.",
    ],
    "low_green": [
        "Yeşil pigment oranı normal değerin altında.",
        "Yapraklarda yeterli klorofil bulunmuyor.",
        "Bitki fotosenteziniden tam verim alamıyor.",
    ],
    "high_green": [
        "Yeşil oran yüksek ve sağlıklı.",
        "Klorofil seviyesi ideal aralıkta.",
        "Yaprak rengi canlı ve homojen.",
    ],
    "high_contrast": [
        "Doku analizi düzensizlikler gösteriyor.",
        "Yaprak yüzeyinde lekeli yapı tespit edildi.",
    ],
}

# Aksiyon önerileri
ACTIONS = {
    "saglikli": [
        ["Mevcut sulama programınızı sürdürün.", "Düzenli kontrolü ihmal etmeyin.",
         "Aylık genel bakım yeterli olacaktır."],
        ["Bitkiyi aynı koşullarda tutmaya devam edin.", "Haftalık ışık dengesini koruyun."],
        ["Olağan bakım rutini yeterli görünüyor.", "Değişiklik yapmaya gerek yok."],
        ["Bitkiyi gözlemlemeye devam edin.", "Mevcut konum uygun görünüyor."],
    ],
    "su_eksikligi": [
        ["Hemen ılık suyla sulama yapın.", "Toprağı kontrol edip 2-3 günde bir sulayın.",
         "Bitkiyi doğrudan güneşten uzaklaştırın."],
        ["Sulama sıklığını artırın.", "Yaprakları sisleyici ile nemlendirin.",
         "Toprak nemini parmak testiyle kontrol edin."],
        ["Acil olarak yeterli su verin.", "Su tutma kapasitesi yüksek toprak kullanın.",
         "Saksı altında biriken suyu boşaltmayın."],
        ["Drenajı kontrol edin ve düzenli sulayın.", "Sıcak saatlerde sulamadan kaçının."],
    ],
    "besin_eksikligi": [
        ["Azot içerikli sıvı gübre uygulayın.", "Toprak pH değerini ölçün (6.0-7.0 ideal).",
         "Ayda iki kez gübreleme yapın."],
        ["Dengeli NPK gübresi kullanın.", "Demir şelatı eklemek faydalı olabilir.",
         "Toprak değiştirmeyi düşünün."],
        ["Organik gübre veya kompost ekleyin.", "Mineral takviyesi uygulayın.",
         "Yaprak gübresi püskürtün."],
        ["Toprak analizi yaptırmanızı tavsiye ederim.", "Kalsiyum ve magnezyum eklenebilir."],
    ],
    "hastalikli": [
        ["Etkilenen yaprakları hemen kesip atın.", "Geniş spektrumlu fungisit uygulayın.",
         "Bitkiyi diğerlerinden izole edin."],
        ["Havalandırmayı artırın ve nemi düşürün.", "Bakırlı ilaçlama yapın.",
         "Kullandığınız makasları dezenfekte edin."],
        ["Hastalıklı kısımları temizleyin.", "Düzenli neem yağı uygulaması yapın.",
         "Yaprak ıslatmaktan kaçının."],
        ["Acil müdahale gerekiyor.", "Profesyonel bir bahçıvana danışın.",
         "Sulama saatini sabaha alın."],
    ],
}


# ============================================================
# ÖZELLİK ÜRETİMİ
# Her durum için gerçekçi özellik değerleri üret
# ============================================================

def generate_features(condition):
    """Verilen duruma göre gerçekçi özellik değerleri üret."""
    if condition == "saglikli":
        return {
            "yesil": round(random.uniform(0.60, 0.85), 2),
            "sari": round(random.uniform(0.00, 0.10), 2),
            "kahverengi": round(random.uniform(0.00, 0.05), 2),
            "koyu_leke": round(random.uniform(0.00, 0.03), 2),
            "kontrast": round(random.uniform(20, 60), 1),
            "homojenlik": round(random.uniform(0.50, 0.80), 2),
        }
    elif condition == "su_eksikligi":
        return {
            "yesil": round(random.uniform(0.25, 0.45), 2),
            "sari": round(random.uniform(0.25, 0.50), 2),
            "kahverengi": round(random.uniform(0.05, 0.15), 2),
            "koyu_leke": round(random.uniform(0.00, 0.05), 2),
            "kontrast": round(random.uniform(40, 80), 1),
            "homojenlik": round(random.uniform(0.30, 0.55), 2),
        }
    elif condition == "besin_eksikligi":
        return {
            "yesil": round(random.uniform(0.30, 0.50), 2),
            "sari": round(random.uniform(0.20, 0.40), 2),
            "kahverengi": round(random.uniform(0.00, 0.10), 2),
            "koyu_leke": round(random.uniform(0.00, 0.05), 2),
            "kontrast": round(random.uniform(30, 60), 1),
            "homojenlik": round(random.uniform(0.40, 0.65), 2),
        }
    elif condition == "hastalikli":
        return {
            "yesil": round(random.uniform(0.30, 0.55), 2),
            "sari": round(random.uniform(0.05, 0.20), 2),
            "kahverengi": round(random.uniform(0.15, 0.35), 2),
            "koyu_leke": round(random.uniform(0.10, 0.25), 2),
            "kontrast": round(random.uniform(80, 150), 1),
            "homojenlik": round(random.uniform(0.20, 0.45), 2),
        }


def features_to_input_string(condition, features):
    """Özellikleri model girdisi haline getir."""
    return (f"durum: {condition} "
            f"yesil: {features['yesil']} "
            f"sari: {features['sari']} "
            f"kahverengi: {features['kahverengi']} "
            f"koyu: {features['koyu_leke']} "
            f"kontrast: {features['kontrast']} "
            f"homojen: {features['homojenlik']}")


# ============================================================
# ÖNERİ METNİ ÜRETİMİ
# Özelliklere göre uyarlanmış metin üret
# ============================================================

def generate_recommendation(condition, features):
    """Duruma ve özelliklere göre öneri metni üret."""
    parts = []

    # 1) Açılış cümlesi
    parts.append(random.choice(OPENING_PHRASES[condition]))

    # 2) Özellik bazlı detay cümlesi (rastgele 1-2 tane)
    detail_keys = []
    if features["sari"] > 0.20:
        detail_keys.append("high_yellow")
    if features["kahverengi"] > 0.15:
        detail_keys.append("high_brown")
    if features["koyu_leke"] > 0.10:
        detail_keys.append("high_dark")
    if features["yesil"] < 0.40:
        detail_keys.append("low_green")
    if features["yesil"] > 0.60:
        detail_keys.append("high_green")
    if features["kontrast"] > 80:
        detail_keys.append("high_contrast")

    if detail_keys:
        n_details = min(len(detail_keys), random.choice([1, 1, 2]))
        for key in random.sample(detail_keys, n_details):
            parts.append(random.choice(DETAIL_BY_FEATURE[key]))

    # 3) Aksiyon önerileri
    action_set = random.choice(ACTIONS[condition])
    parts.extend(action_set)

    return " ".join(parts)


# ============================================================
# ANA VERİ ÜRETİMİ
# ============================================================

def generate_dataset(n_per_condition=600):
    """Eğitim veri setini üret."""
    conditions = ["saglikli", "su_eksikligi", "besin_eksikligi", "hastalikli"]
    dataset = []

    print(f"Veri seti üretiliyor... (Her sınıftan {n_per_condition} örnek)")

    for condition in conditions:
        for i in range(n_per_condition):
            features = generate_features(condition)
            input_str = features_to_input_string(condition, features)
            output_str = generate_recommendation(condition, features)

            dataset.append({
                "input": input_str,
                "output": output_str,
                "condition": condition,
                "features": features,
            })

    random.shuffle(dataset)

    print(f"Toplam {len(dataset)} örnek üretildi.")

    # Örnek göster
    print("\n--- ÖRNEK VERİLER ---")
    for i in range(3):
        print(f"\n[Örnek {i + 1}]")
        print(f"INPUT : {dataset[i]['input']}")
        print(f"OUTPUT: {dataset[i]['output']}")

    return dataset


if __name__ == "__main__":
    print("=" * 60)
    print("BITKI BAKIM ÖNERİSİ VERİ ÜRETİCİSİ")
    print("=" * 60)

    dataset = generate_dataset(n_per_condition=600)

    # Kaydet
    with open("training_data.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nVeri seti kaydedildi: training_data.json ({len(dataset)} örnek)")
    print("Sıradaki adım: train_seq2seq.py dosyasını çalıştırın.")