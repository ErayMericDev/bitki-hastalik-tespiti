"""
============================================================
ADIM 2: SINIFLANDIRICIYI EĞİT (GERÇEK ÖĞRENME)
Random Forest — özellik tablosundan ÖĞRENİR
============================================================

ÖNEMLİ: Bu modelde ELLE YAZILMIŞ HİÇBİR KURAL YOK.
Model, gerçek görüntülerden çıkarılan özellikler ile
gerçek hastalık etiketleri arasındaki ilişkiyi KENDİSİ öğrenir.

"Gerçekten öğreniyor mu?" sorusunun cevabı burada:
  - Veri %80 eğitim / %20 test olarak bölünür
  - Model SADECE eğitim verisini görür
  - Test verisi (görmediği örnekler) ile doğruluk ölçülür
  - Yüksek test doğruluğu = gerçek öğrenmenin kanıtı

Çıktı: siniflandirici_model.pkl + grafikler
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score)
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)


def main():
    print("=" * 60)
    print("RANDOM FOREST SINIFLANDIRICI EĞİTİMİ")
    print("Gerçek Veriden Öğrenen Model (Kural Yok!)")
    print("=" * 60)

    # 1) Özellik tablosunu yükle
    print("\n[1/6] Özellik tablosu yükleniyor...")
    df = pd.read_csv("ozellik_tablosu.csv")
    print(f"   Toplam örnek: {len(df)}")
    print(f"   Özellikler: {[c for c in df.columns if c != 'etiket']}")

    # 2) Özellikleri (X) ve etiketleri (y) ayır
    feature_cols = [c for c in df.columns if c != "etiket"]
    X = df[feature_cols].values
    y_text = df["etiket"].values

    # Etiketleri sayıya çevir
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    class_names = label_encoder.classes_
    print(f"   Sınıf sayısı: {len(class_names)}")

    # 3) Eğitim/Test ayrımı (%80 / %20) — KRİTİK ADIM
    print("\n[2/6] Veri eğitim/test olarak bölünüyor...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Eğitim seti: {len(X_train)} örnek (model bunları görecek)")
    print(f"   Test seti  : {len(X_test)} örnek (model bunları GÖRMEYECEK)")

    # 4) Ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5) Random Forest eğit
    print("\n[3/6] Random Forest eğitiliyor...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)
    print("   Eğitim tamamlandı!")

    # 6) DEĞERLENDİRME — gerçekten öğrendi mi?
    print("\n[4/6] Model değerlendiriliyor...")

    # Eğitim doğruluğu
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    # Test doğruluğu (GÖRMEDİĞİ veri!)
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

    # Çapraz doğrulama (daha güvenilir ölçüm)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    print(f"\n   {'='*50}")
    print(f"   SONUÇLAR:")
    print(f"   {'='*50}")
    print(f"   Eğitim doğruluğu     : {train_acc:.4f} (%{train_acc*100:.1f})")
    print(f"   TEST doğruluğu       : {test_acc:.4f} (%{test_acc*100:.1f})  <-- ÖNEMLİ")
    print(f"   Çapraz doğrulama (CV): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"   {'='*50}")
    print(f"\n   YORUM: Test doğruluğu, modelin HİÇ GÖRMEDİĞİ {len(X_test)}")
    print(f"   örnek üzerindeki başarısıdır. Bu, ezber DEĞİL gerçek")
    print(f"   öğrenmenin kanıtıdır.")

    # Detaylı sınıf bazlı rapor
    print("\n[5/6] Sınıf bazlı performans:")
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred,
                                target_names=[c[:20] for c in class_names],
                                zero_division=0))

    # 7) Görselleştirme
    print("\n[6/6] Grafikler oluşturuluyor...")

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(11, 9))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=[c[:15] for c in class_names])
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
    ax.set_title(f"Karışıklık Matrisi (Test Seti)\nTest Doğruluğu: %{test_acc*100:.1f}",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("rf_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   Kaydedildi: rf_confusion_matrix.png")

    # Özellik önem sıralaması
    fig, ax = plt.subplots(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    ax.barh(range(len(feature_cols)),
            importances[indices], color="#4CAF50", alpha=0.8)
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels([feature_cols[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Önem Derecesi")
    ax.set_title("Hangi Özellikler Daha Önemli? (Model Kendi Buldu)",
                 fontweight="bold")
    for i, v in enumerate(importances[indices]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("rf_ozellik_onem.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("   Kaydedildi: rf_ozellik_onem.png")

    # 8) Modeli kaydet
    with open("siniflandirici_model.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "label_encoder": label_encoder,
            "feature_cols": feature_cols,
        }, f)

    print("\n" + "=" * 60)
    print("TAMAMLANDI!")
    print("=" * 60)
    print("   Çıktılar:")
    print("   - siniflandirici_model.pkl  (eğitilmiş model)")
    print("   - rf_confusion_matrix.png   (karışıklık matrisi)")
    print("   - rf_ozellik_onem.png       (özellik önemleri)")
    print(f"\n   >>> TEST DOĞRULUĞU: %{test_acc*100:.1f} <<<")
    print("\n   Bu değeri hocana söyleyebilirsin: model görmediği")
    print(f"   {len(X_test)} örneğin %{test_acc*100:.1f}'ini doğru bildi.")


if __name__ == "__main__":
    main()