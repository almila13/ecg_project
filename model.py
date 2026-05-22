"""
model.py
--------
5 Blokluk Derin Öğrenme Modeli: ECG Sinyal Sınıflandırması

Mimari Akış:
    Ham EKG (12 kanal, 1000 zaman adımı)
        │
        ▼
    [BLOK 1] CNN Encoder     → Yerel özellik çıkarma
        │
        ▼
    [BLOK 2] LSTM            → Zamansal bağımlılık modelleme
        │
        ▼
    [BLOK 3] Attention        → Önemli zaman adımlarına odaklanma
        │
        ▼
    [BLOK 4] Autoencoder      → Sıkıştırılmış temsil öğrenme
        │
        ▼
    [BLOK 5] Classifier       → Nihai sınıf tahmini (5 sınıf)

Her blok ders notlarıyla birebir örtüşür:
    - CNN  → Week 6 (sparse connectivity, parameter sharing, Conv1d)
    - LSTM → Week 10 (forget/input/output gate, vanishing gradient çözümü)
    - AE   → Week 8 (bottleneck, reconstruction loss, undercomplete AE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# BLOK 1 — CNN Encoder
# ═══════════════════════════════════════════════════════════════════════════════
class CNNEncoder(nn.Module):
    """
    1D Convolutional Neural Network bloğu.

    Neden CNN?
        EKG sinyali zaman boyutunda yerel örüntüler içerir (QRS kompleksi,
        P dalgası gibi). CNN'in parameter sharing özelliği sayesinde bu
        örüntüler sinyalin her yerinde tespit edilebilir.

    Mimari: 3 katmanlı Conv1d + BatchNorm + ReLU + MaxPool
        Giriş : (batch, 12, 1000)   → 12 kanal, 1000 zaman adımı
        Çıkış : (batch, 128, 125)   → 128 özellik kanalı, 125 zaman adımı
    """
    def __init__(self, in_channels=12):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),  # yerel özellik
            nn.BatchNorm1d(32),        # eğitimi kararlı hale getirir
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 1000 → 500
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 500 → 250
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # 250 → 125
        )

        self.dropout = nn.Dropout(p=0.2)  # regularization: overfitting'i azaltır

    def forward(self, x):
        # x: (batch, 12, 1000)
        x = self.block1(x)  # → (batch, 32,  500)
        x = self.block2(x)  # → (batch, 64,  250)
        x = self.block3(x)  # → (batch, 128, 125)
        x = self.dropout(x)
        return x             # (batch, 128, 125)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOK 2 — LSTM
# ═══════════════════════════════════════════════════════════════════════════════
class LSTMBlock(nn.Module):
    """
    Long Short-Term Memory bloğu.

    Neden LSTM?
        CNN yerel örüntüleri yakalar ama uzun vadeli bağımlılıkları
        unutur. LSTM'in forget/input/output gate mekanizması sayesinde
        "R dalgasından 200 adım sonra gelen T dalgası" gibi uzak
        ilişkiler öğrenilebilir.

    Mimari:
        Giriş : (batch, 125, 128)  — CNN çıktısı (seq_len=125, features=128)
        Çıkış : (batch, 125, 256)  — her zaman adımı için gizli durum
    """
    def __init__(self, input_size=128, hidden_size=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,          # (batch, seq, feature) formatı
            dropout=0.3,               # katmanlar arası dropout
            bidirectional=False        # tek yönlü: gerçek zamanlı uyumlu
        )
        self.layer_norm = nn.LayerNorm(hidden_size)  # eğitim kararlılığı

    def forward(self, x):
        # CNN çıktısı: (batch, 128, 125) → LSTM beklentisi: (batch, 125, 128)
        x = x.permute(0, 2, 1)              # kanal ve zaman boyutunu değiştir
        out, _ = self.lstm(x)               # (batch, 125, 256)
        out = self.layer_norm(out)
        return out                           # (batch, 125, 256)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOK 3 — Attention Mekanizması
# ═══════════════════════════════════════════════════════════════════════════════
class AttentionBlock(nn.Module):
    """
    Additive (Bahdanau) Attention.

    Neden Attention?
        Tüm 125 zaman adımı eşit önemde değildir. QRS kompleksinin
        olduğu adımlar arritmiya teşhisi için çok daha önemlidir.
        Attention, her adıma bir ağırlık öğrenerek modelin önemli
        bölgelere odaklanmasını sağlar.

    Mimari:
        Giriş : (batch, 125, 256)  — LSTM çıktısı
        Çıkış : (batch, 256)       — ağırlıklı ortalama (tek vektör)
    """
    def __init__(self, hidden_size=256):
        super().__init__()
        # Her zaman adımının önemini hesaplayan küçük sinir ağı
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, 125, 256)
        scores  = self.attention(x)           # (batch, 125, 1)
        weights = F.softmax(scores, dim=1)    # zaman boyutunda normalize
        # Ağırlıklı toplam → tek bir bağlam vektörü
        context = (x * weights).sum(dim=1)   # (batch, 256)
        return context, weights               # context: (batch, 256)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOK 4 — Autoencoder (Bottleneck)
# ═══════════════════════════════════════════════════════════════════════════════
class AutoencoderBlock(nn.Module):
    """
    Undercomplete Autoencoder (Dar Şişe Boynu).

    Neden Autoencoder?
        Ders notuna göre: "Constraint is the feature-learning mechanism."
        256 boyutlu attention çıktısını 64'e sıkıştırıp tekrar 256'ya
        açarak modeli veriyi daha kompakt temsil etmeye zorlarız.
        Bu, gereksiz gürültüyü atar ve genelleme kabiliyetini artırır.

    Kayıp fonksiyonu:
        Ana sınıflandırma kaybına ek olarak reconstruction loss eklenir:
        total_loss = classification_loss + λ * reconstruction_loss

    Mimari:
        Encoder: 256 → 128 → 64    (sıkıştırma)
        Decoder: 64  → 128 → 256   (yeniden açma)
    """
    def __init__(self, input_dim=256, bottleneck_dim=64):
        super().__init__()

        # Encoder: sıkıştır
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, bottleneck_dim),
            nn.ReLU()
        )

        # Decoder: yeniden aç (reconstruction için)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Tanh()   # çıktıyı [-1,1] aralığına sınırlar
        )

    def forward(self, x):
        code          = self.encoder(x)    # (batch, 64)  ← sıkıştırılmış temsil
        reconstruction = self.decoder(code) # (batch, 256) ← yeniden oluşturma
        return code, reconstruction


# ═══════════════════════════════════════════════════════════════════════════════
# BLOK 5 — Classifier (Sınıflandırıcı)
# ═══════════════════════════════════════════════════════════════════════════════
class ClassifierBlock(nn.Module):
    """
    Tam bağlantılı (Fully Connected) sınıflandırıcı.

    Bottleneck'ten gelen 64 boyutlu vektörü n_classes'a (5 sınıf) eşler.

    Mimari: 64 → 32 → n_classes
    """
    def __init__(self, input_dim=64, n_classes=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)   # softmax train.py'de CrossEntropyLoss içinde
        )

    def forward(self, x):
        return self.fc(x)   # (batch, n_classes)


# ═══════════════════════════════════════════════════════════════════════════════
# TAM MODEL — 5 Bloğu Bir Araya Getirir
# ═══════════════════════════════════════════════════════════════════════════════
class ECGNet(nn.Module):
    """
    ECGNet: 5 Blokluk Derin EKG Sınıflandırma Ağı

    Blok Seçim Gerekçeleri (ders notlarına göre):
        [1] CNN  : "Grid-like topology" → EKG zaman serisi için ideal (Ch.9)
        [2] LSTM : "Addition replaces multiplication" → uzun bağımlılık (Week 10)
        [3] Att. : Önemli zaman dilimlerine odaklanma (Week 10, Slide 26)
        [4] AE   : "Constraint is feature-learning mechanism" (Week 8)
        [5] FC   : Nihai sınıflandırma kafası

    Kayıp Fonksiyonu:
        L_total = L_classification + λ * L_reconstruction
        λ = 0.1 (reconstruction'ın fazla baskın olmaması için)
    """
    def __init__(self, n_classes=5, ae_lambda=0.1):
        super().__init__()
        self.ae_lambda = ae_lambda  # reconstruction loss ağırlığı

        self.cnn        = CNNEncoder(in_channels=12)
        self.lstm       = LSTMBlock(input_size=128, hidden_size=256, num_layers=2)
        self.attention  = AttentionBlock(hidden_size=256)
        self.autoencoder = AutoencoderBlock(input_dim=256, bottleneck_dim=64)
        self.classifier = ClassifierBlock(input_dim=64, n_classes=n_classes)

    def forward(self, x):
        """
        x: (batch, 12, 1000) — 12 kanallı EKG sinyali
        """
        # [1] CNN → yerel özellikler
        cnn_out    = self.cnn(x)                        # (batch, 128, 125)

        # [2] LSTM → zamansal bağımlılıklar
        lstm_out   = self.lstm(cnn_out)                 # (batch, 125, 256)

        # [3] Attention → önemli adımları ağırlıklandır
        context, attn_weights = self.attention(lstm_out) # (batch, 256)

        # [4] Autoencoder → sıkıştırılmış temsil + reconstruction
        code, recon = self.autoencoder(context)         # code: (batch, 64)

        # [5] Classifier → sınıf logit'leri
        logits = self.classifier(code)                  # (batch, n_classes)

        return logits, recon, context, attn_weights

    def compute_loss(self, logits, labels, recon, context):
        """
        Toplam kayıp:
            L_total = CrossEntropy(logits, labels) + λ * MSE(recon, context)
        """
        cls_loss   = F.cross_entropy(logits, labels)
        recon_loss = F.mse_loss(recon, context.detach())  # detach: gradient ayrımı
        total_loss = cls_loss + self.ae_lambda * recon_loss
        return total_loss, cls_loss, recon_loss


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation Study için Sadeleştirilmiş Modeller
# ═══════════════════════════════════════════════════════════════════════════════
class ECGNet_NoCNN(nn.Module):
    """Ablation: CNN bloğu kaldırıldı → LSTM doğrudan ham sinyali alır"""
    def __init__(self, n_classes=5):
        super().__init__()
        # Ham sinyal: (batch, 12, 1000) → (batch, 1000, 12)
        self.lstm       = LSTMBlock(input_size=12, hidden_size=256, num_layers=2)
        self.attention  = AttentionBlock(hidden_size=256)
        self.autoencoder = AutoencoderBlock(input_dim=256, bottleneck_dim=64)
        self.classifier = ClassifierBlock(input_dim=64, n_classes=n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)          # (batch, 1000, 12)
        lstm_out = self.lstm.lstm(x)[0]  # direkt LSTM
        lstm_out = self.lstm.layer_norm(lstm_out)
        context, _ = self.attention(lstm_out)
        code, recon = self.autoencoder(context)
        logits = self.classifier(code)
        return logits, recon, context, None

    def compute_loss(self, logits, labels, recon, context):
        cls_loss   = F.cross_entropy(logits, labels)
        recon_loss = F.mse_loss(recon, context.detach())
        return cls_loss + 0.1 * recon_loss, cls_loss, recon_loss


class ECGNet_NoLSTM(nn.Module):
    """Ablation: LSTM bloğu kaldırıldı → CNN çıktısı global avg pool ile düzleştirilir"""
    def __init__(self, n_classes=5):
        super().__init__()
        self.cnn        = CNNEncoder(in_channels=12)
        self.attention  = AttentionBlock(hidden_size=128)
        self.autoencoder = AutoencoderBlock(input_dim=128, bottleneck_dim=64)
        self.classifier = ClassifierBlock(input_dim=64, n_classes=n_classes)

    def forward(self, x):
        cnn_out = self.cnn(x)               # (batch, 128, 125)
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch, 125, 128)
        context, _ = self.attention(cnn_out)
        code, recon = self.autoencoder(context)
        logits = self.classifier(code)
        return logits, recon, context, None

    def compute_loss(self, logits, labels, recon, context):
        cls_loss   = F.cross_entropy(logits, labels)
        recon_loss = F.mse_loss(recon, context.detach())
        return cls_loss + 0.1 * recon_loss, cls_loss, recon_loss


class ECGNet_NoAE(nn.Module):
    """Ablation: Autoencoder bloğu kaldırıldı → Attention çıktısı doğrudan classifier'a gider"""
    def __init__(self, n_classes=5):
        super().__init__()
        self.cnn        = CNNEncoder(in_channels=12)
        self.lstm       = LSTMBlock(input_size=128, hidden_size=256, num_layers=2)
        self.attention  = AttentionBlock(hidden_size=256)
        self.classifier = ClassifierBlock(input_dim=256, n_classes=n_classes)

    def forward(self, x):
        cnn_out  = self.cnn(x)
        lstm_out = self.lstm(cnn_out)
        context, _ = self.attention(lstm_out)
        logits = self.classifier(context)
        return logits, None, context, None

    def compute_loss(self, logits, labels, recon, context):
        return F.cross_entropy(logits, labels), F.cross_entropy(logits, labels), torch.tensor(0.0)


class ECGNet_NoAttention(nn.Module):
    """Ablation: Attention bloğu kaldırıldı → LSTM'in son adımı kullanılır"""
    def __init__(self, n_classes=5):
        super().__init__()
        self.cnn        = CNNEncoder(in_channels=12)
        self.lstm       = LSTMBlock(input_size=128, hidden_size=256, num_layers=2)
        self.autoencoder = AutoencoderBlock(input_dim=256, bottleneck_dim=64)
        self.classifier = ClassifierBlock(input_dim=64, n_classes=n_classes)

    def forward(self, x):
        cnn_out  = self.cnn(x)
        lstm_out = self.lstm(cnn_out)
        context  = lstm_out[:, -1, :]       # son zaman adımını al
        code, recon = self.autoencoder(context)
        logits = self.classifier(code)
        return logits, recon, context, None

    def compute_loss(self, logits, labels, recon, context):
        cls_loss   = F.cross_entropy(logits, labels)
        recon_loss = F.mse_loss(recon, context.detach())
        return cls_loss + 0.1 * recon_loss, cls_loss, recon_loss


# ═══════════════════════════════════════════════════════════════════════════════
# Model Özeti
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    model = ECGNet(n_classes=5)
    # Parametre sayısını hesapla
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam parametre   : {total_params:,}")
    print(f"Eğitilebilir param : {trainable:,}")

    # Sahte veri ile ileri geçiş testi
    dummy = torch.randn(4, 12, 1000)    # 4 örnek, 12 kanal, 1000 adım
    logits, recon, context, weights = model(dummy)
    print(f"\nGiriş  : {dummy.shape}")
    print(f"Logits : {logits.shape}   ← (batch, 5 sınıf)")
    print(f"Recon  : {recon.shape}    ← (batch, 256)")
    print(f"Code   : {context.shape} ← (batch, 256)")
    print(f"Attn   : {weights.shape}  ← (batch, 125, 1)")
