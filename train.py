"""
train.py
--------
Eğitim döngüsü, değerlendirme ve ablation study.

Kullanım:
    python3 train.py

Ne yapar:
    1. Tam modeli (ECGNet, 5 blok) eğitir
    2. Her epoch sonunda validation accuracy hesaplar
    3. En iyi modeli kaydeder
    4. Ablation study: her bloğu kaldırarak karşılaştırır
    5. Sonuçları görselleştirir ve kaydeder
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

from dataset import prepare_data
from model  import (ECGNet, ECGNet_NoCNN, ECGNet_NoLSTM,
                    ECGNet_NoAE, ECGNet_NoAttention)

# ── Hiperparametreler ─────────────────────────────────────────────────────────
# Bu değerler deneme-yanılma ve literatür rehberliğiyle seçildi.
# Detaylı açıklama README.md'de yer almaktadır.
CONFIG = {
    "epochs"       : 30,       # validation loss plateau'ya ulaşana kadar
    "lr"           : 1e-3,     # Adam optimizer için başlangıç öğrenme hızı
    "weight_decay" : 1e-4,     # L2 regularization katsayısı
    "ae_lambda"    : 0.1,      # reconstruction loss ağırlığı
    "patience"     : 7,        # early stopping: 7 epoch iyileşme olmazsa dur
    "clip_grad"    : 1.0,      # gradient clipping eşiği (Week 10, Slide 21)
    "ablation_epochs": 10,     # ablation modelleri için daha az epoch
}

# ── Cihaz Seçimi ──────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")    # MacBook M2 GPU
        print("✓ Apple MPS (GPU) kullanılıyor")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ CUDA GPU kullanılıyor")
    else:
        device = torch.device("cpu")
        print("✓ CPU kullanılıyor")
    return device


# ── Tek Epoch Eğitimi ─────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        logits, recon, context, _ = model(X)
        loss, cls_loss, recon_loss = model.compute_loss(logits, y, recon, context)

        loss.backward()

        # Gradient clipping: patlayan gradyanları önler (Week 10, Slide 21)
        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip_grad"])

        optimizer.step()

        total_loss += loss.item() * len(y)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total


# ── Değerlendirme ─────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits, recon, context, _ = model(X)
            loss, _, _ = model.compute_loss(logits, y, recon, context)

            total_loss += loss.item() * len(y)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y).sum().item()
            total      += len(y)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ── Tam Eğitim Döngüsü ────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, device, epochs, model_name="ECGNet"):
    """
    Adam optimizer + CosineAnnealingLR scheduler ile eğitim.

    Optimizer seçimi:
        Adam → adaptif öğrenme hızı, EKG gibi gürültülü veride hızlı yakınsama

    Scheduler:
        CosineAnnealingLR → öğrenme hızını yavaşça düşürür, keskin minimumlara
        sıkışmayı önler

    Early Stopping:
        Validation loss CONFIG['patience'] epoch boyunca iyileşmezse dur.
        Overfitting'i önler.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc  = 0.0
    patience_cnt  = 0
    best_state    = None

    print(f"\n{'='*55}")
    print(f"  Eğitim: {model_name}  ({epochs} epoch)")
    print(f"{'='*55}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss,   val_acc, _, _  = evaluate(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | "
              f"{elapsed:.1f}s")

        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG["patience"]:
                print(f"  ⚑ Early stopping: {CONFIG['patience']} epoch iyileşme yok.")
                break

    # En iyi ağırlıkları geri yükle
    if best_state:
        model.load_state_dict(best_state)

    print(f"\n  ✓ En iyi Val Accuracy: {best_val_acc:.4f}")
    return history, best_val_acc


# ── Görselleştirme ────────────────────────────────────────────────────────────
def plot_history(history, model_name, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="steelblue")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   color="tomato")
    ax1.set_title(f"{model_name} — Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Acc", color="steelblue")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc",   color="tomato")
    ax2.set_title(f"{model_name} — Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_training.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ Grafik kaydedildi: {path}")


def plot_ablation(results, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    names = list(results.keys())
    accs  = [results[n] for n in names]

    colors = ["#2ecc71" if n == "ECGNet (Tam)" else "#e74c3c" for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, accs, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Validation Accuracy")
    ax.set_title("Ablation Study: Her Bloğun Katkısı")
    ax.set_xlim(0, 1.0)
    ax.axvline(x=accs[0], color="#2ecc71", linestyle="--", alpha=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{acc:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    path = os.path.join(save_dir, "ablation_study.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ Ablation grafiği kaydedildi: {path}")


# ── Ablation Study ────────────────────────────────────────────────────────────
def run_ablation(train_loader, val_loader, n_classes, device):
    """
    Ablation study: her bloğu tek tek kaldır ve etkisini ölç.

    Ders notundan gerekçe (Week 10): her bileşenin katkısını anlamak
    için tek değişkenli deney yapılır.
    """
    print("\n" + "="*55)
    print("  ABLATION STUDY")
    print("="*55)

    variants = {
        "ECGNet (Tam)"       : ECGNet(n_classes),
        "- CNN Yok"          : ECGNet_NoCNN(n_classes),
        "- LSTM Yok"         : ECGNet_NoLSTM(n_classes),
        "- Attention Yok"    : ECGNet_NoAttention(n_classes),
        "- Autoencoder Yok"  : ECGNet_NoAE(n_classes),
    }

    ablation_results = {}
    for name, model in variants.items():
        model = model.to(device)
        _, best_acc = train_model(
            model, train_loader, val_loader, device,
            epochs=CONFIG["ablation_epochs"], model_name=name
        )
        ablation_results[name] = best_acc

    return ablation_results


# ── Ana Fonksiyon ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "★"*55)
    print("  ECGNet — PTB-XL EKG Sınıflandırma Projesi")
    print("★"*55 + "\n")

    device = get_device()

    # 1. Veriyi hazırla
    print("\n[1/4] Veri hazırlanıyor...")
    train_loader, val_loader, test_loader, n_classes, class_names = prepare_data()

    # 2. Tam modeli eğit
    print("\n[2/4] Tam model eğitiliyor...")
    model = ECGNet(n_classes=n_classes, ae_lambda=CONFIG["ae_lambda"]).to(device)
    history, _ = train_model(
        model, train_loader, val_loader, device,
        epochs=CONFIG["epochs"], model_name="ECGNet"
    )
    plot_history(history, "ECGNet")

    # Modeli kaydet
    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/ecgnet_best.pt")
    print("  ✓ Model kaydedildi: results/ecgnet_best.pt")

    # 3. Test seti değerlendirmesi
    print("\n[3/4] Test seti değerlendirmesi...")
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, device)
    print(f"\n  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}\n")
    print(classification_report(labels, preds, target_names=class_names))

    # 4. Ablation Study
    print("\n[4/4] Ablation study çalıştırılıyor...")
    ablation_results = run_ablation(train_loader, val_loader, n_classes, device)
    plot_ablation(ablation_results)

    # Sonuç tablosu
    print("\n" + "="*55)
    print("  ABLATION STUDY SONUÇLARI")
    print("="*55)
    for name, acc in ablation_results.items():
        marker = "◄ TAM MODEL" if name == "ECGNet (Tam)" else ""
        print(f"  {name:<25} → Acc: {acc:.4f}  {marker}")

    print("\n✓ Tüm sonuçlar 'results/' klasörüne kaydedildi.")
    print("✓ Proje tamamlandı!\n")


if __name__ == "__main__":
    main()
