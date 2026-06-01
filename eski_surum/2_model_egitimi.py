"""
============================================================
ADIM 2: SEQ2SEQ TRANSFORMER MODELİ EĞİTİMİ
LLM Benzeri Küçük Bir Model — Sıfırdan Eğitilir
============================================================

Mimari:
  - Encoder-Decoder Transformer (PyTorch)
  - Kelime tabanlı tokenizer (sıfırdan inşa edilir)
  - Yaklaşık 500K parametre (eğitilebilir küçük model)
  - Pozisyonel encoding, multi-head attention, feed-forward
  - Teacher forcing ile eğitim
  - Greedy decoding ile inference

Girdi : "durum: su_eksikligi yesil: 0.35 sari: 0.40 ..."
Çıktı : "Bitkinizde su eksikliği belirtileri tespit edildi..."

Gereken kütüphaneler:
  pip install torch numpy matplotlib tqdm
"""

import json
import math
import re
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# TOKENIZER
# Kendi yazdığımız basit kelime tabanlı tokenizer
# ============================================================

class SimpleTokenizer:
    """Kelime tabanlı tokenizer. Özel tokenlar: PAD, SOS, EOS, UNK."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        # Özel tokenlar
        self.PAD = 0
        self.SOS = 1  # Start of sequence
        self.EOS = 2  # End of sequence
        self.UNK = 3  # Unknown

    def _tokenize(self, text):
        """Metni token'lara böl."""
        # Sayıları ve kelimeleri ayır
        text = text.lower()
        # Noktalama işaretlerini ayır
        text = re.sub(r"([.,!?():\-])", r" \1 ", text)
        # Sayıları ayrı bir token olarak tut
        tokens = text.split()
        return tokens

    def fit(self, texts, min_freq=1):
        """Kelime dağarcığını oluştur."""
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))

        # Özel tokenları ekle
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

        # Frekans filtresi
        for word, freq in counter.most_common():
            if freq >= min_freq:
                self.word2idx[word] = len(self.word2idx)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        print(f"   Dağarcık boyutu: {len(self.word2idx)}")
        return self

    def encode(self, text, max_len=None, add_special=True):
        """Metni indislere çevir."""
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
        """İndisleri metne çevir."""
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


# ============================================================
# VERİ SETİ SINIFI
# ============================================================

class PlantCareDataset(Dataset):
    def __init__(self, data, src_tokenizer, tgt_tokenizer, src_max_len, tgt_max_len):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src = self.src_tokenizer.encode(item["input"], max_len=self.src_max_len)
        tgt = self.tgt_tokenizer.encode(item["output"], max_len=self.tgt_max_len)

        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


# ============================================================
# POZİSYONEL ENCODING
# ============================================================

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


# ============================================================
# SEQ2SEQ TRANSFORMER MODELİ
# ============================================================

class Seq2SeqTransformer(nn.Module):
    """
    Encoder-Decoder Transformer (LLM benzeri mimari).

    Parametreler:
      d_model : embedding boyutu
      nhead   : attention head sayısı
      num_encoder_layers / num_decoder_layers
      dim_feedforward : feed-forward boyutu
    """

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=128, nhead=4,
                 num_encoder_layers=2, num_decoder_layers=2,
                 dim_feedforward=256, dropout=0.1,
                 max_len=200):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # Çıktı katmanı
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src, pad_idx=0):
        """Encoder için padding maskesi."""
        return (src == pad_idx)

    def make_tgt_mask(self, tgt, pad_idx=0):
        """Decoder için padding + causal mask."""
        tgt_len = tgt.size(1)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device),
                                 diagonal=1).bool()
        pad_mask = (tgt == pad_idx)
        return causal_mask, pad_mask

    def forward(self, src, tgt):
        # Embedding + pozisyonel encoding
        src_emb = self.dropout(self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model)))
        tgt_emb = self.dropout(self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))

        # Maskeler
        src_pad_mask = self.make_src_mask(src)
        causal_mask, tgt_pad_mask = self.make_tgt_mask(tgt)

        # Transformer
        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=causal_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )

        return self.fc_out(output)

    @torch.no_grad()
    def generate(self, src, max_len, sos_idx, eos_idx, pad_idx=0):
        """Greedy decoding ile metin üret."""
        self.eval()
        device = src.device
        batch_size = src.size(0)

        # Encoder kısmı
        src_emb = self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        src_pad_mask = self.make_src_mask(src, pad_idx)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_pad_mask)

        # Decoder — adım adım üret
        tgt = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_emb = self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
            causal_mask, _ = self.make_tgt_mask(tgt, pad_idx)

            out = self.transformer.decoder(
                tgt_emb, memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=src_pad_mask,
            )

            logits = self.fc_out(out[:, -1, :])
            next_token = logits.argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)

            if (next_token == eos_idx).all():
                break

        return tgt


# ============================================================
# EĞİTİM DÖNGÜSÜ
# ============================================================

def train_model(model, train_loader, val_loader, n_epochs, lr, device):
    """Modeli eğit."""
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD'i yok say
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                     factor=0.5, patience=2)

    history = {"train_loss": [], "val_loss": []}

    print("\n" + "=" * 60)
    print("MODEL EĞİTİMİ BAŞLIYOR")
    print("=" * 60)
    print(f"Toplam parametre: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Eğitilebilir parametre: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Cihaz: {device}")
    print()

    for epoch in range(n_epochs):
        # --- Eğitim ---
        model.train()
        train_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]  # son token hariç
            tgt_output = tgt[:, 1:]  # ilk token hariç (shifted)

            optimizer.zero_grad()
            output = model(src, tgt_input)

            loss = criterion(output.reshape(-1, output.size(-1)),
                             tgt_output.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Doğrulama ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                output = model(src, tgt_input)
                loss = criterion(output.reshape(-1, output.size(-1)),
                                 tgt_output.reshape(-1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"   Epoch {epoch + 1:3d}/{n_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.5f}")

    return history


# ============================================================
# DEĞERLENDİRME
# ============================================================

def evaluate_samples(model, dataset, src_tok, tgt_tok, device, n_samples=5):
    """Birkaç örnek üzerinde inference yap."""
    print("\n" + "=" * 60)
    print("ÖRNEK ÜRETİMLER")
    print("=" * 60)

    indices = random.sample(range(len(dataset)), n_samples)
    for i, idx in enumerate(indices):
        src, tgt = dataset[idx]
        src = src.unsqueeze(0).to(device)

        generated = model.generate(src, max_len=100,
                                   sos_idx=tgt_tok.SOS,
                                   eos_idx=tgt_tok.EOS)

        input_text = src_tok.decode(src[0].cpu().tolist())
        true_text = tgt_tok.decode(tgt.tolist())
        gen_text = tgt_tok.decode(generated[0].cpu().tolist())

        print(f"\n[Örnek {i + 1}]")
        print(f"GİRDİ    : {input_text}")
        print(f"GERÇEK   : {true_text}")
        print(f"ÜRETİLEN : {gen_text}")
        print("-" * 60)


def plot_training_history(history):
    """Eğitim grafiklerini çiz."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["train_loss"], label="Eğitim Kaybı", linewidth=2, color="#2196F3")
    ax.plot(history["val_loss"], label="Doğrulama Kaybı", linewidth=2, color="#FF9800")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Seq2Seq Transformer — Eğitim Süreci", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("egitim_grafigi.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nGrafik kaydedildi: egitim_grafigi.png")


# ============================================================
# ANA PROGRAM
# ============================================================

def main():
    print("=" * 60)
    print("SEQ2SEQ TRANSFORMER EĞİTİMİ")
    print("Bitki Bakım Önerisi Üretici (LLM Benzeri Model)")
    print("=" * 60)

    # Cihaz seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nCihaz: {device}")

    # 1) Veriyi yükle
    print("\n[1/5] Veri yükleniyor...")
    with open("training_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"   Toplam örnek: {len(data)}")

    # Train/val ayır
    random.shuffle(data)
    split = int(len(data) * 0.9)
    train_data = data[:split]
    val_data = data[split:]
    print(f"   Eğitim: {len(train_data)} | Doğrulama: {len(val_data)}")

    # 2) Tokenizer'ları oluştur
    print("\n[2/5] Tokenizer oluşturuluyor...")
    src_tok = SimpleTokenizer()
    tgt_tok = SimpleTokenizer()

    src_texts = [d["input"] for d in train_data]
    tgt_texts = [d["output"] for d in train_data]

    print("   Girdi tokenizer:")
    src_tok.fit(src_texts)
    print("   Çıktı tokenizer:")
    tgt_tok.fit(tgt_texts)

    # Maksimum uzunlukları hesapla
    src_max = max(len(src_tok._tokenize(t)) for t in src_texts) + 2
    tgt_max = max(len(tgt_tok._tokenize(t)) for t in tgt_texts) + 2
    print(f"   Girdi max uzunluk: {src_max}")
    print(f"   Çıktı max uzunluk: {tgt_max}")

    # 3) Dataset ve DataLoader
    print("\n[3/5] DataLoader hazırlanıyor...")
    train_ds = PlantCareDataset(train_data, src_tok, tgt_tok, src_max, tgt_max)
    val_ds = PlantCareDataset(val_data, src_tok, tgt_tok, src_max, tgt_max)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # 4) Modeli oluştur
    print("\n[4/5] Model oluşturuluyor...")
    model = Seq2SeqTransformer(
        src_vocab_size=src_tok.vocab_size,
        tgt_vocab_size=tgt_tok.vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=max(src_max, tgt_max) + 10,
    ).to(device)

    # 5) Eğit
    print("\n[5/5] Eğitim başlıyor...")
    history = train_model(model, train_loader, val_loader,
                          n_epochs=30, lr=5e-4, device=device)

    # Görselleştirme
    plot_training_history(history)

    # Örnek üretimler
    evaluate_samples(model, val_ds, src_tok, tgt_tok, device, n_samples=5)

    # Modeli kaydet
    print("\nModel kaydediliyor...")
    torch.save({
        "model_state": model.state_dict(),
        "model_config": {
            "src_vocab_size": src_tok.vocab_size,
            "tgt_vocab_size": tgt_tok.vocab_size,
            "d_model": 128,
            "nhead": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 256,
            "max_len": max(src_max, tgt_max) + 10,
        },
        "src_max": src_max,
        "tgt_max": tgt_max,
    }, "../seq2seq_model.pt")

    with open("../tokenizers.pkl", "wb") as f:
        pickle.dump({"src": src_tok, "tgt": tgt_tok}, f)

    print("Kayıtlı dosyalar:")
    print("   - seq2seq_model.pt   (model ağırlıkları)")
    print("   - tokenizers.pkl     (tokenizer'lar)")
    print("   - egitim_grafigi.png (eğitim grafiği)")
    print("\nSıradaki adım: 3_birlesik_sistem.py dosyasını çalıştırın.")


if __name__ == "__main__":
    main()