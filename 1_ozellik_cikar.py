"""
============================================================
ADIM 1: GERÇEK GÖRÜNTÜLERDEN ÖZELLİK ÇIKARMA
PlantVillage gerçek veri seti → özellik tablosu
============================================================

Bu script:
  1. Seçilen 9 sınıfın klasörlerini tarar
  2. Her görüntüden HSV + GLCM özellikleri çıkarır (senin görüntü işlemen)
  3. Tüm özellikleri + gerçek etiketleri bir CSV tablosuna kaydeder

Bu tablo, ADIM 2'de Random Forest'ı eğitmek için kullanılacak.

Çıktı: ozellik_tablosu.csv
"""

import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# AYARLAR
# ============================================================

# PlantVillage klasörünün yolu (proje içindeki konum)
DATASET_PATH = "PlantVillage"

# Kullanacağımız 9 sınıf (dengeli: 3 bitki, sağlıklı + hastalıklı karışık)
SECILEN_SINIFLAR = [
    "Pepper__bell___healthy",
    "Pepper__bell___Bacterial_spot",
    "Potato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Leaf_Mold",
]

# Her sınıftan kaç görüntü işlenecek (hız için sınır - None = hepsi)
GORUNTU_PER_SINIF = 200


# ============================================================
# ÖZELLİK ÇIKARMA (senin görüntü işleme modülünün aynısı)
# ============================================================

class FeatureExtractor:
    def __init__(self):
        self.color_ranges = {
            "yesil": (np.array([30, 40, 40]), np.array([90, 255, 255])),
            "sari": (np.array([20, 60, 80]), np.array([32, 255, 255])),
            "kahverengi": (np.array([0, 60, 40]), np.array([20, 255, 220])),
            "koyu_leke": (np.array([0, 0, 0]), np.array([180, 255, 70])),
        }

    def extract(self, image_path):
        """Bir görüntüden tüm özellikleri çıkar."""
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        # Segmentasyon
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([5, 20, 20]), np.array([170, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask_clean = np.zeros_like(mask)
            cv2.drawContours(mask_clean, [largest], -1, 255, -1)
            mask = mask_clean

        total_pixels = max(cv2.countNonZero(mask), 1)

        # Renk oranları
        features = {}
        for name, (lower, upper) in self.color_ranges.items():
            color_mask = cv2.inRange(hsv, lower, upper)
            color_mask = cv2.bitwise_and(color_mask, mask)
            features[name] = cv2.countNonZero(color_mask) / total_pixels

        # GLCM doku özellikleri
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
        gray_q = (gray_masked // 4).astype(np.uint8)
        glcm = graycomatrix(gray_q, distances=[1],
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=64, symmetric=True, normed=True)
        features["kontrast"] = float(graycoprops(glcm, "contrast").mean())
        features["homojenlik"] = float(graycoprops(glcm, "homogeneity").mean())
        features["enerji"] = float(graycoprops(glcm, "energy").mean())
        features["korelasyon"] = float(graycoprops(glcm, "correlation").mean())

        return features


# ============================================================
# ANA İŞLEM
# ============================================================

def main():
    print("=" * 60)
    print("GERÇEK GÖRÜNTÜLERDEN ÖZELLİK ÇIKARMA")
    print("=" * 60)

    if not os.path.isdir(DATASET_PATH):
        print(f"\n   HATA: '{DATASET_PATH}' klasörü bulunamadı!")
        print(f"   PlantVillage klasörünü proje dizinine çıkardığından emin ol.")
        return

    extractor = FeatureExtractor()
    all_rows = []

    print(f"\n   {len(SECILEN_SINIFLAR)} sınıf işlenecek")
    print(f"   Her sınıftan max {GORUNTU_PER_SINIF} görüntü\n")

    for sinif in SECILEN_SINIFLAR:
        sinif_yolu = os.path.join(DATASET_PATH, sinif)
        if not os.path.isdir(sinif_yolu):
            print(f"   ATLANIYOR (bulunamadı): {sinif}")
            continue

        goruntular = [f for f in os.listdir(sinif_yolu)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        if GORUNTU_PER_SINIF:
            goruntular = goruntular[:GORUNTU_PER_SINIF]

        print(f"   [{sinif}] {len(goruntular)} görüntü işleniyor...", end=" ", flush=True)

        basarili = 0
        for img_name in goruntular:
            img_path = os.path.join(sinif_yolu, img_name)
            features = extractor.extract(img_path)
            if features is not None:
                features["etiket"] = sinif
                all_rows.append(features)
                basarili += 1

        print(f"OK ({basarili})")

    # Tabloya dönüştür
    df = pd.DataFrame(all_rows)

    print(f"\n   Toplam işlenen görüntü: {len(df)}")
    print(f"   Özellik sayısı: {len(df.columns) - 1}")
    print(f"\n   Sınıf dağılımı:")
    print(df["etiket"].value_counts().to_string())

    # Kaydet
    df.to_csv("ozellik_tablosu.csv", index=False, encoding="utf-8")

    print("\n" + "=" * 60)
    print("TAMAMLANDI!")
    print("=" * 60)
    print(f"   Çıktı: ozellik_tablosu.csv ({len(df)} satır)")
    print("\n   Örnek satırlar:")
    print(df.head().to_string())
    print("\n   Sıradaki adım: 2_siniflandirici_egit.py")


if __name__ == "__main__":
    main()