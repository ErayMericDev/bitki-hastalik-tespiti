"""
============================================================
ADIM 5: BİRLEŞİK SİSTEM (FİNAL)
Görüntü → Özellik → Random Forest → Seq2Seq → Öneri
============================================================

TAM İŞ AKIŞI:
  1. Kullanıcı bitki yaprağı fotoğrafı verir
  2. Görüntü işleme ile özellikler çıkarılır (HSV + GLCM)
  3. Random Forest hastalığı tahmin eder (GERÇEK veriden öğrenmiş)
  4. Tahmin seq2seq modeline verilir
  5. Seq2Seq gerçek tarım bilgisine dayalı öneri üretir

Bu sistem tamamen GERÇEK VERİDEN öğrenen iki model içerir:
  - Random Forest: PlantVillage gerçek görüntülerinden
  - Seq2Seq: Üniversite tarım extension kaynaklarından

Gereken dosyalar (önceki adımlardan):
  - siniflandirici_model.pkl   (ADIM 2)
  - seq2seq_model.pt           (ADIM 4)
  - tokenizers.pkl             (ADIM 4)

Gereken: pip install torch numpy opencv-python matplotlib scikit-image scikit-learn
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
import math
import re
from skimage.feature import graycomatrix, graycoprops
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# SEQ2SEQ MİMARİSİ (yükleme için gerekli - eğitimdekiyle aynı)
# ============================================================

class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.PAD, self.SOS, self.EOS, self.UNK = 0, 1, 2, 3

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r"([.,!?():\-])", r" \1 ", text)
        return text.split()

    def encode(self, text, max_len=None, add_special=True):
        tokens = self._tokenize(text)
        ids = [self.word2idx.get(t, self.UNK) for t in tokens]
        if add_special:
            ids = [self.SOS] + ids + [self.EOS]
        if max_len:
            if len(ids) > max_len:
                ids = ids[:max_len - 1] + [self.EOS]
            else:
                ids = ids + [self.PAD] * (max_len - len(ids))
        return ids

    def decode(self, ids, skip_special=True):
        words = []
        for i in ids:
            if i == self.EOS:
                break
            if skip_special and i in [self.PAD, self.SOS, self.UNK]:
                continue
            words.append(self.idx2word.get(i, "<UNK>"))
        return " ".join(words)

    @property
    def vocab_size(self):
        return len(self.word2idx)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=128, nhead=4, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=256,
                 dropout=0.1, max_len=200):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src, pad_idx=0):
        return (src == pad_idx)

    def make_tgt_mask(self, tgt, pad_idx=0):
        tgt_len = tgt.size(1)
        causal = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device),
                            diagonal=1).bool()
        return causal, (tgt == pad_idx)

    @torch.no_grad()
    def generate(self, src, max_len, sos_idx, eos_idx, pad_idx=0):
        self.eval()
        device = src.device
        bs = src.size(0)
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        src_pad = self.make_src_mask(src, pad_idx)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad)
        tgt = torch.full((bs, 1), sos_idx, dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
            causal, _ = self.make_tgt_mask(tgt, pad_idx)
            out = self.transformer.decoder(tgt_emb, memory, tgt_mask=causal,
                                           memory_key_padding_mask=src_pad)
            logits = self.fc_out(out[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
            if (next_token == eos_idx).all():
                break
        return tgt


# ============================================================
# 1. GÖRÜNTÜ İŞLEME MODÜLÜ
# ============================================================

class ImageAnalyzer:
    def __init__(self):
        self.color_ranges = {
            "yesil": (np.array([30, 40, 40]), np.array([90, 255, 255])),
            "sari": (np.array([20, 60, 80]), np.array([32, 255, 255])),
            "kahverengi": (np.array([0, 60, 40]), np.array([20, 255, 220])),
            "koyu_leke": (np.array([0, 0, 0]), np.array([180, 255, 70])),
        }

    def extract_features(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

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

        total = max(cv2.countNonZero(mask), 1)
        features = {}
        for name, (lo, hi) in self.color_ranges.items():
            cm = cv2.bitwise_and(cv2.inRange(hsv, lo, hi), mask)
            features[name] = cv2.countNonZero(cm) / total

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_m = cv2.bitwise_and(gray, gray, mask=mask)
        gray_q = (gray_m // 4).astype(np.uint8)
        glcm = graycomatrix(gray_q, distances=[1],
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=64, symmetric=True, normed=True)
        features["kontrast"] = float(graycoprops(glcm, "contrast").mean())
        features["homojenlik"] = float(graycoprops(glcm, "homogeneity").mean())
        features["enerji"] = float(graycoprops(glcm, "energy").mean())
        features["korelasyon"] = float(graycoprops(glcm, "correlation").mean())

        return features, img, mask


# ============================================================
# 2. RANDOM FOREST SINIFLANDIRICI (yükle)
# ============================================================

class DiseaseClassifier:
    def __init__(self, model_path="siniflandirici_model.pkl"):
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.label_encoder = data["label_encoder"]
        self.feature_cols = data["feature_cols"]
        print(f"   Random Forest yüklendi ({len(self.label_encoder.classes_)} sınıf)")

    def predict(self, features):
        x = np.array([[features[c] for c in self.feature_cols]])
        x_scaled = self.scaler.transform(x)
        pred = self.model.predict(x_scaled)[0]
        proba = self.model.predict_proba(x_scaled)[0]
        etiket = self.label_encoder.inverse_transform([pred])[0]
        guven = proba[pred]
        return etiket, guven


# ============================================================
# 3. SEQ2SEQ ÖNERİ ÜRETİCİ (yükle)
# ============================================================

class RecommendationGenerator:
    def __init__(self, model_path="seq2seq_model.pt", tok_path="tokenizers.pkl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(tok_path, "rb") as f:
            toks = pickle.load(f)
        self.src_tok = toks["src"]
        self.tgt_tok = toks["tgt"]
        ckpt = torch.load(model_path, map_location=self.device)
        self.src_max = ckpt["src_max"]
        self.tgt_max = ckpt["tgt_max"]
        self.model = Seq2SeqTransformer(**ckpt["model_config"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"   Seq2Seq yüklendi (cihaz: {self.device})")

    def generate(self, etiket):
        girdi = f"hastalik: {etiket}"
        src_ids = self.src_tok.encode(girdi, max_len=self.src_max)
        src = torch.tensor([src_ids], dtype=torch.long).to(self.device)
        out = self.model.generate(src, max_len=self.tgt_max,
                                  sos_idx=self.tgt_tok.SOS, eos_idx=self.tgt_tok.EOS)
        text = self.tgt_tok.decode(out[0].cpu().tolist())
        return self._postprocess(text)

    def _postprocess(self, text):
        text = re.sub(r"\s+([.,!?:])", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        # Cümle başlarını büyük harf
        parts = re.split(r"([.!?]\s+)", text)
        result = []
        for p in parts:
            if p and p[0].isalpha():
                p = p[0].upper() + p[1:]
            result.append(p)
        text = "".join(result)
        if text and not text.endswith("."):
            text += "."
        return text


# ============================================================
# BİRLEŞİK SİSTEM
# ============================================================

# Hastalık etiketlerinin Türkçe okunabilir karşılıkları
ETIKET_TR = {
    "Pepper__bell___Bacterial_spot": "Biber - Bakteriyel Leke",
    "Pepper__bell___healthy": "Biber - Sağlıklı",
    "Potato___Early_blight": "Patates - Erken Yanıklık",
    "Potato___Late_blight": "Patates - Mildiyö",
    "Potato___healthy": "Patates - Sağlıklı",
    "Tomato_Bacterial_spot": "Domates - Bakteriyel Leke",
    "Tomato_Late_blight": "Domates - Mildiyö",
    "Tomato_Leaf_Mold": "Domates - Yaprak Küfü",
    "Tomato_healthy": "Domates - Sağlıklı",
}


class PlantCareAI:
    def __init__(self):
        print("=" * 60)
        print("🌿 BİTKİ BAKIM AI SİSTEMİ — YÜKLENİYOR")
        print("=" * 60)
        self.analyzer = ImageAnalyzer()
        print("   Görüntü analiz modülü hazır")
        self.classifier = DiseaseClassifier()
        self.generator = RecommendationGenerator()
        print("   Sistem hazır!\n")

    def analyze(self, image_path):
        print("=" * 60)
        print(f"🔍 ANALİZ: {image_path}")
        print("=" * 60)

        # 1. Görüntü işleme
        print("\n[1/3] Görüntü işleniyor...")
        features, img, mask = self.analyzer.extract_features(image_path)
        print("   Özellikler:")
        for k, v in features.items():
            if k in ["yesil", "sari", "kahverengi", "koyu_leke"]:
                print(f"      {k:12s}: %{v*100:.1f}")
            else:
                print(f"      {k:12s}: {v:.3f}")

        # 2. Random Forest sınıflandırma
        print("\n[2/3] Hastalık tespiti (Random Forest)...")
        etiket, guven = self.classifier.predict(features)
        etiket_tr = ETIKET_TR.get(etiket, etiket)
        print(f"   Tespit: {etiket_tr}")
        print(f"   Güven : %{guven*100:.1f}")

        # 3. Seq2Seq öneri
        print("\n[3/3] Bakım önerisi üretiliyor (Seq2Seq)...")
        oneri = self.generator.generate(etiket)
        print(f"\n💡 ÖNERİ:")
        print(f"   {oneri}")

        # Görselleştir
        self._visualize(img, mask, features, etiket_tr, guven, oneri, image_path)

        return {"etiket": etiket, "guven": guven, "oneri": oneri}

    def _visualize(self, img, mask, features, etiket_tr, guven, oneri, path):
        fig = plt.figure(figsize=(15, 9))
        gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 0.9])

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img); ax1.set_title("Orijinal", fontweight="bold"); ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(mask, cmap="gray"); ax2.set_title("Segmentasyon", fontweight="bold"); ax2.axis("off")

        ax3 = fig.add_subplot(gs[0, 2])
        seg = cv2.bitwise_and(img, img, mask=mask)
        ax3.imshow(seg); ax3.set_title("Yaprak", fontweight="bold"); ax3.axis("off")

        # Renk dağılımı
        ax4 = fig.add_subplot(gs[1, 0])
        renkler = {k: features[k]*100 for k in ["yesil", "sari", "kahverengi", "koyu_leke"]}
        ax4.bar(renkler.keys(), renkler.values(),
                color=["#4CAF50", "#FFEB3B", "#795548", "#212121"], alpha=0.8)
        ax4.set_title("Renk Dağılımı (%)", fontweight="bold")
        ax4.tick_params(axis="x", rotation=20)

        # Doku
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.bar(["Kontrast", "Homojen.", "Enerji"],
                [features["kontrast"], features["homojenlik"]*100, features["enerji"]*100],
                color=["#2196F3", "#9C27B0", "#00BCD4"], alpha=0.8)
        ax5.set_title("Doku (GLCM)", fontweight="bold")

        # Tahmin
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")
        renk = "#4CAF50" if "Sağlıklı" in etiket_tr else "#F44336"
        ax6.text(0.5, 0.7, etiket_tr, ha="center", va="center",
                 fontsize=13, fontweight="bold", color=renk, wrap=True)
        ax6.text(0.5, 0.35, f"Güven: %{guven*100:.1f}", ha="center", va="center", fontsize=11)
        ax6.set_title("Tespit (Random Forest)", fontweight="bold")

        # Öneri
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis("off")
        ax7.text(0.02, 0.5, f"🤖 AI BAKIM ÖNERİSİ (Seq2Seq):\n\n{oneri}",
                 ha="left", va="center", fontsize=10, wrap=True,
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="#E8F5E9",
                           edgecolor="#4CAF50", linewidth=2))

        plt.tight_layout()
        plt.savefig("final_analiz.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("\n   Grafik kaydedildi: final_analiz.png")


# ============================================================
# ANA PROGRAM
# ============================================================

def main():
    system = PlantCareAI()

    print("=" * 60)
    print("Test edilecek görüntü yolunu girin")
    print("(örn: test_hastalikli2.jpg veya PlantVillage/Tomato_healthy/xxx.jpg)")
    print("=" * 60)

    while True:
        path = input("\nGörüntü yolu (çıkmak için 'q'): ").strip()
        if path.lower() == "q":
            print("Çıkılıyor...")
            break
        if not os.path.exists(path):
            print(f"   HATA: Dosya bulunamadı — {path}")
            continue
        try:
            system.analyze(path)
        except Exception as e:
            print(f"   HATA: {e}")


if __name__ == "__main__":
    main()