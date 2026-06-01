"""
============================================================
ADIM 3b: SEQ2SEQ MODELİNİ GERÇEK VERİYLE EĞİT
============================================================

Bu script, daha önce kurduğumuz Transformer mimarisinin AYNISINI kullanır.
Tek fark: Artık sentetik veri yerine GERÇEK TARIM KAYNAKLARINDAN
derlenmiş veriyle (seq2seq_egitim_verisi.json) eğitiliyor.

Mimari değişmedi çünkü zaten çalışıyordu; sadece beslediğimiz veri
bilimsel kaynaklı hale geldi.

Gereken: pip install torch numpy matplotlib

Çıktı: seq2seq_model.pt, tokenizers.pkl, egitim_grafigi.png
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
# TOKENIZER (öncekiyle aynı)
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

    def fit(self, texts, min_freq=1):
        counter = Counter()
        for text in texts:
            counter.update(self._tokenize(text))
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        for word, freq in counter.most_common():
            if freq >= min_freq:
                self.word2idx[word] = len(self.word2idx)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        print(f"   Dağarcık boyutu: {len(self.word2idx)}")
        return self

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


# ============================================================
# DATASET
# ============================================================

class PlantCareDataset(Dataset):
    def __init__(self, data, src_tok, tgt_tok, src_max, tgt_max):
        self.data = data
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok
        self.src_max = src_max
        self.tgt_max = tgt_max

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src = self.src_tok.encode(item["input"], max_len=self.src_max)
        tgt = self.tgt_tok.encode(item["output"], max_len=self.tgt_max)
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


# ============================================================
# POZİSYONEL ENCODING + TRANSFORMER (öncekiyle aynı)
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
        pad = (tgt == pad_idx)
        return causal, pad

    def forward(self, src, tgt):
        src_emb = self.dropout(self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model)))
        tgt_emb = self.dropout(self.pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))
        src_pad = self.make_src_mask(src)
        causal, tgt_pad = self.make_tgt_mask(tgt)
        out = self.transformer(src_emb, tgt_emb, tgt_mask=causal,
                               src_key_padding_mask=src_pad,
                               tgt_key_padding_mask=tgt_pad,
                               memory_key_padding_mask=src_pad)
        return self.fc_out(out)

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
# EĞİTİM
# ============================================================

def train_model(model, train_loader, val_loader, n_epochs, lr, device):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                      factor=0.5, patience=2)
    history = {"train_loss": [], "val_loss": []}

    print("\n" + "=" * 60)
    print("MODEL EĞİTİMİ BAŞLIYOR")
    print("=" * 60)
    print(f"Toplam parametre: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Cihaz: {device}\n")

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            optimizer.zero_grad()
            output = model(src, tgt_in)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                output = model(src, tgt_in)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_out.reshape(-1))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"   Epoch {epoch+1:3d}/{n_epochs} | Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.5f}")

    return history


def evaluate_samples(model, dataset, src_tok, tgt_tok, device, n=5):
    print("\n" + "=" * 60)
    print("ÖRNEK ÜRETİMLER (görmediği veriden)")
    print("=" * 60)
    indices = random.sample(range(len(dataset)), n)
    for i, idx in enumerate(indices):
        src, tgt = dataset[idx]
        src = src.unsqueeze(0).to(device)
        gen = model.generate(src, max_len=80, sos_idx=tgt_tok.SOS, eos_idx=tgt_tok.EOS)
        print(f"\n[Örnek {i+1}]")
        print(f"GİRDİ    : {src_tok.decode(src[0].cpu().tolist())}")
        print(f"GERÇEK   : {tgt_tok.decode(tgt.tolist())}")
        print(f"ÜRETİLEN : {tgt_tok.decode(gen[0].cpu().tolist())}")
        print("-" * 60)


def plot_history(history):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["train_loss"], label="Eğitim", linewidth=2, color="#2196F3")
    ax.plot(history["val_loss"], label="Doğrulama", linewidth=2, color="#FF9800")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Seq2Seq Eğitim Süreci (Gerçek Veri)", fontweight="bold")
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
    print("SEQ2SEQ TRANSFORMER — GERÇEK VERİYLE EĞİTİM")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    # 1) Gerçek veriyi yükle
    print("\n[1/5] Gerçek veri yükleniyor...")
    with open("seq2seq_egitim_verisi.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"   Toplam örnek: {len(data)}")

    random.shuffle(data)
    split = int(len(data) * 0.9)
    train_data, val_data = data[:split], data[split:]
    print(f"   Eğitim: {len(train_data)} | Doğrulama: {len(val_data)}")

    # 2) Tokenizer
    print("\n[2/5] Tokenizer oluşturuluyor...")
    src_tok, tgt_tok = SimpleTokenizer(), SimpleTokenizer()
    src_texts = [d["input"] for d in train_data]
    tgt_texts = [d["output"] for d in train_data]
    print("   Girdi:"); src_tok.fit(src_texts)
    print("   Çıktı:"); tgt_tok.fit(tgt_texts)

    src_max = max(len(src_tok._tokenize(t)) for t in src_texts) + 2
    tgt_max = max(len(tgt_tok._tokenize(t)) for t in tgt_texts) + 2
    print(f"   Girdi max: {src_max}, Çıktı max: {tgt_max}")

    # 3) DataLoader
    print("\n[3/5] DataLoader...")
    train_ds = PlantCareDataset(train_data, src_tok, tgt_tok, src_max, tgt_max)
    val_ds = PlantCareDataset(val_data, src_tok, tgt_tok, src_max, tgt_max)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # 4) Model
    print("\n[4/5] Model oluşturuluyor...")
    model = Seq2SeqTransformer(
        src_vocab_size=src_tok.vocab_size,
        tgt_vocab_size=tgt_tok.vocab_size,
        d_model=128, nhead=4,
        num_encoder_layers=2, num_decoder_layers=2,
        dim_feedforward=256, dropout=0.1,
        max_len=max(src_max, tgt_max) + 10,
    ).to(device)

    # 5) Eğit
    print("\n[5/5] Eğitim başlıyor...")
    history = train_model(model, train_loader, val_loader,
                          n_epochs=40, lr=5e-4, device=device)

    plot_history(history)
    evaluate_samples(model, val_ds, src_tok, tgt_tok, device, n=5)

    # Kaydet
    print("\nModel kaydediliyor...")
    torch.save({
        "model_state": model.state_dict(),
        "model_config": {
            "src_vocab_size": src_tok.vocab_size,
            "tgt_vocab_size": tgt_tok.vocab_size,
            "d_model": 128, "nhead": 4,
            "num_encoder_layers": 2, "num_decoder_layers": 2,
            "dim_feedforward": 256, "max_len": max(src_max, tgt_max) + 10,
        },
        "src_max": src_max, "tgt_max": tgt_max,
    }, "seq2seq_model.pt")

    with open("tokenizers.pkl", "wb") as f:
        pickle.dump({"src": src_tok, "tgt": tgt_tok}, f)

    print("Kaydedildi: seq2seq_model.pt, tokenizers.pkl, egitim_grafigi.png")
    print("\nSıradaki adım: birleşik sistem (4_birlesik_sistem.py)")


if __name__ == "__main__":
    main()