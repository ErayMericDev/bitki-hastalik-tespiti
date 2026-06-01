# 🌿 Bitki Hastalık Tespiti ve Bakım Önerisi Sistemi

Görüntü işleme, makine öğrenmesi ve derin öğrenmeyi birleştiren uçtan uca bir yapay zeka sistemi. Bir bitki yaprağı fotoğrafından hastalığı tespit eder ve doğal dilde bilimsel temelli bakım önerisi üretir.



---

## 📋 Özet

Sistem üç aşamadan oluşur:

1. **Görüntü İşleme** — HSV renk analizi, yaprak segmentasyonu ve GLCM doku analizi ile fotoğraftan sayısal özellikler çıkarılır.
2. **Random Forest Sınıflandırıcı** — Gerçek PlantVillage görüntülerinden öğrenir, hastalığı tespit eder (test doğruluğu: **%81.8**).
3. **Seq2Seq Transformer** — Gerçek üniversite tarım kaynaklarından derlenmiş bilgilerle eğitilmiş, sıfırdan inşa edilmiş bir Encoder-Decoder modeli; doğal dilde bakım önerisi üretir.

İki model de **gerçek veriden öğrenir**; her ikisi de görmediği test verisinde değerlendirilerek gerçek genelleme yaptıkları doğrulanmıştır.

---

## 🏗️ Sistem Mimarisi

```
Bitki Yaprağı Fotoğrafı
        ↓
Görüntü İşleme (HSV + Segmentasyon + GLCM)
        ↓
Özellik Vektörü (yeşil%, sarı%, kahverengi%, koyu%, kontrast, homojenlik, enerji, korelasyon)
        ↓
Random Forest Sınıflandırıcı (9 hastalık sınıfı, %81.8 test doğruluğu)
        ↓
Tespit edilen hastalık
        ↓
Seq2Seq Transformer (gerçek tarım bilgisi)
        ↓
Doğal Dilde Bakım Önerisi
```

---

## 📂 Proje Yapısı

| Dosya | Açıklama |
|-------|----------|
| `1_ozellik_cikar.py` | PlantVillage görüntülerinden HSV + GLCM özellikleri çıkarır |
| `2_siniflandirici_egit.py` | Random Forest sınıflandırıcıyı eğitir ve değerlendirir |
| `3_bilgi_tabani_uret.py` | Gerçek tarım kaynaklarından seq2seq eğitim verisi üretir |
| `4_seq2seq_egit.py` | Seq2Seq Transformer modelini eğitir |
| `5_birlesik_sistem.py` | Tüm modülleri birleştiren final sistem |

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
```bash
pip install torch numpy opencv-python matplotlib scikit-learn scikit-image pandas
```

### Veri Seti
[PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) veri setini indirip proje kök dizinine `PlantVillage/` klasörü olarak çıkarın.

### Çalıştırma Sırası
```bash
python 1_ozellik_cikar.py        # Görüntülerden özellik çıkar
python 2_siniflandirici_egit.py  # Random Forest eğit
python 3_bilgi_tabani_uret.py    # Seq2seq verisi üret
python 4_seq2seq_egit.py         # Seq2seq eğit
python 5_birlesik_sistem.py      # Sistemi çalıştır
```

---

## 🔬 Kullanılan Sınıflar (9 adet)

Domates, patates ve biber bitkilerine ait sağlıklı ve hastalıklı sınıflar:

- Biber: Sağlıklı, Bakteriyel Leke
- Patates: Sağlıklı, Erken Yanıklık, Mildiyö (Geç Yanıklık)
- Domates: Sağlıklı, Mildiyö, Bakteriyel Leke, Yaprak Küfü

---

## 📊 Sonuçlar

| Model | Metrik | Değer |
|-------|--------|-------|
| Random Forest | Test doğruluğu | %81.8 |
| Random Forest | 5-katlı çapraz doğrulama | %79.6 |
| Seq2Seq | Doğrulama kaybı (40 epoch) | 0.261 |

---

## 🛠️ Kullanılan Teknolojiler

- **Python** — ana geliştirme dili
- **OpenCV** — görüntü işleme
- **scikit-image** — GLCM doku analizi
- **scikit-learn** — Random Forest
- **PyTorch** — Seq2Seq Transformer

---

## 📚 Veri Kaynakları

- **Görüntüler:** PlantVillage veri seti
- **Bakım önerileri:** University of Minnesota Extension, University of Wisconsin Vegetable Pathology, UC IPM, Clemson University, Cornell University ve diğer üniversite tarım extension kaynakları

---

## 🔮 Gelecek Çalışmalar

- CNN tabanlı (EfficientNet, MobileNetV3) sınıflandırıcıya geçiş
- TensorFlow Lite dönüşümü ve Flutter mobil uygulama entegrasyonu
- Daha fazla bitki ve hastalık sınıfı eklenmesi
- Bilgi tabanının genişletilmesi

---

## 📝 Not

Görüntüler kontrollü ortamda çekildiği için gerçek tarla koşullarında performans değişebilir. Bakım önerileri bilgilendirme amaçlıdır; kritik durumlarda bir ziraat uzmanına danışılması önerilir.
