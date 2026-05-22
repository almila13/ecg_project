# ECGNet: Multi-Block Deep Learning Architecture for ECG Arrhythmia Classification

**SWE012 – Deep Learning with Python | İstinye University**

---

## Abstract

We present **ECGNet**, a five-block deep learning architecture for automatic classification of cardiac arrhythmias from 12-lead ECG signals. The model integrates Convolutional Neural Networks (CNN), Long Short-Term Memory networks (LSTM), an Attention mechanism, an Autoencoder (AE), and a fully connected Classifier into a coherent end-to-end trainable pipeline. We evaluate the model on the **PTB-XL dataset** — a large-scale publicly available ECG dataset sourced from a peer-reviewed scientific publication — and conduct ablation studies to quantify the contribution of each architectural component.

---

## 1. Dataset

### 1.1 PTB-XL ECG Dataset

| Property | Value |
|---|---|
| Source | PhysioNet (Research Paper) |
| Citation | Wagner et al., *Scientific Data*, 2020 |
| DOI | 10.1038/s41597-020-0495-6 |
| URL | https://physionet.org/content/ptb-xl/1.0.3/ |
| Recordings | 21,837 EKG kaydı |
| Patients | 18,869 hasta |
| Channels | 12 standart EKG derivasyonu |
| Duration | Her kayıt 10 saniye (1000 örnek @ 100 Hz) |
| Labels | 5 süper-sınıf (superdiagnostic) |

### 1.2 Neden Bu Dataset?

PTB-XL seçimi üç temel gerekçeye dayanmaktadır:

1. **Gerçek klinik veri:** Reel hasta kayıtlarından oluşur; MNIST gibi yapay benchmark veri setlerinden farklı olarak klinik açıdan anlamlı bir problem içerir.
2. **Research paper kaynağı:** *Scientific Data* dergisinde yayımlanmış hakemli bir makaleden alınmıştır (+15 bonus puan kriteri).
3. **Mimari uygunluğu:** 12 kanallı zaman serisi verisi CNN + LSTM kombinasyonuna doğal bir şekilde uyar; bu da her bloğun varlığını mimari olarak gerekçelendirir.

### 1.3 Sınıf Dağılımı

| Sınıf | Açıklama | Kayıt Sayısı |
|---|---|---|
| NORM | Normal Ritim | ~9,528 |
| MI | Miyokard Enfarktüsü | ~5,486 |
| STTC | ST/T Değişikliği | ~5,250 |
| CD | İletim Bozukluğu | ~4,907 |
| HYP | Hipertrofi | ~2,655 |

### 1.4 Ön İşleme

1. **Örnekleme:** 500 Hz orijinal kayıtlardan 100 Hz versiyonu kullanılır (daha hızlı eğitim)
2. **Z-Score Normalizasyon:** Her kanal bağımsız olarak `μ=0, σ=1` olacak şekilde normalize edilir. Bu, farklı amplitude'lere sahip derivasyonların karşılaştırılabilir ölçeğe çekilmesini sağlar.
3. **Eksik Değer:** `np.nan_to_num` ile sıfır doldurma
4. **Train/Val/Test Bölümü:** PTB-XL'in resmi `strat_fold` sütunu kullanılır (fold 1–8: train, fold 9: val, fold 10: test). Bu, farklı çalışmalarla karşılaştırılabilirliği sağlar.

---

## 2. Model Mimarisi

### 2.1 Genel Bakış

```
Ham EKG Sinyali
(batch, 12, 1000)
        │
        ▼
┌───────────────────┐
│  [BLOK 1] CNN     │  Conv1d × 3, BatchNorm, ReLU, MaxPool
│  Encoder          │  Çıkış: (batch, 128, 125)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  [BLOK 2] LSTM    │  2 katmanlı, hidden=256, LayerNorm
│                   │  Çıkış: (batch, 125, 256)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  [BLOK 3]         │  Additive Attention (Bahdanau)
│  Attention        │  Çıkış: (batch, 256)
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  [BLOK 4]         │  Encoder: 256→128→64
│  Autoencoder      │  Decoder: 64→128→256
└─────────┬─────────┘
          │ code: (batch, 64)
          ▼
┌───────────────────┐
│  [BLOK 5]         │  FC: 64→32→5
│  Classifier       │  Çıkış: (batch, 5)
└───────────────────┘
```

### 2.2 Blok Gerekçeleri

#### Blok 1 — CNN Encoder

EKG sinyali 1000 zaman adımı boyunca yerel yapısal örüntüler barındırır: P dalgası, QRS kompleksi, T dalgası gibi. CNN'in **parametre paylaşımı (parameter sharing)** özelliği (Week 6 ders notu), aynı örüntü tespit filtresinin sinyalin her zaman adımında yeniden kullanılmasını sağlar. 3 kademeli `Conv1d` katmanı 1000 adımı 125 adıma indirirken özellik kanalı sayısını 12'den 128'e çıkarır. `MaxPool1d` işlemi öteleme değişmezliği (translation invariance) sağlar; küçük zaman kaymaları sınıflandırmayı etkilemez.

#### Blok 2 — LSTM

CNN yerel örüntüleri yakaladıktan sonra zamansal uzun vadeli bağımlılıklar hâlâ modellenmemiştir. Örneğin, "Q dalgasının ardından 200 adım sonra gelen anormal T dalgası MI belirtisi" gibi ilişkiler, klasik RNN'lerde vanishing gradient problemi nedeniyle öğrenilemez. LSTM'in **hücre durumu (cell state)** üzerinden `c_t = f_t ⊙ c_{t−1} + i_t ⊙ c̃_t` şeklindeki katkısal güncellemesi (Week 10 ders notu), gradyanların uzun mesafeler boyunca kaybolmadan akmasını sağlar. 2 katmanlı yapı hiyerarşik zamansal özellik öğrenimini destekler.

#### Blok 3 — Attention

125 zaman adımının tamamı eşit önemde değildir. Attention mekanizması, her adıma bir ağırlık skoru öğrenerek modelin EKG sinyalinin diagnostik açıdan kritik bölgelerine (aritmia oluşumunun gerçekleştiği zaman dilimleri) odaklanmasını sağlar. Bahdanau additive attention formülasyonu: `α_t = softmax(v^T tanh(W h_t))`. Bağlam vektörü `context = Σ α_t h_t` tüm diziyi tek bir vektörde özetler.

#### Blok 4 — Autoencoder

Week 8 ders notundaki temel ilke: *"The constraint is the feature-learning mechanism."* 256 boyutlu attention çıktısı 64 boyuta sıkıştırılarak model, sınıflandırma için en bilgilendirici özellikleri korumak zorunda bırakılır. Decoder, sıkıştırılmış kodu tekrar 256 boyuta açarak reconstruction yapabilir. Kayıp fonksiyonuna eklenen `L_reconstruction = MSE(recon, context)` terimi, bottleneck'in bilgi kaybını minimize eder ve regularization etkisi yaratır.

**Toplam kayıp fonksiyonu:**
```
L_total = L_classification + λ * L_reconstruction
        = CrossEntropy(logits, y) + 0.1 * MSE(recon, context)
```

#### Blok 5 — Classifier

Bottleneck'ten gelen 64 boyutlu yoğun temsil, iki tam bağlantılı katman (`64 → 32 → 5`) aracılığıyla 5 sınıf logit'ine dönüştürülür. Dropout (p=0.3) son regularization katmanı olarak overfitting'i engeller.

---

## 3. Hiperparametre Seçimi ve Regularization

### 3.1 Hiperparametre Tablosu

| Hiperparametre | Değer | Seçim Gerekçesi |
|---|---|---|
| Öğrenme Hızı | 1e-3 | Adam için standart başlangıç değeri; EKG gibi gürültülü veride hızlı yakınsama |
| Batch Size | 32 | Bellek/stabilite dengesi; büyük batch düz minima bulur |
| CNN Kernel | 7, 5, 3 | Büyükten küçüğe: önce geniş, sonra dar alıcı alan |
| LSTM Hidden | 256 | Yeterli kapasitede; 512 ile denendi, overfitting arttı |
| LSTM Katman | 2 | Hiyerarşik öğrenim; 3 katman marginal iyileşme, fazla süre |
| Bottleneck | 64 | 256'nın 1/4'ü; çok küçük (32) reconstruction kalitesini düşürdü |
| AE Lambda (λ) | 0.1 | Classification baskın kalmalı; 0.5 denendi, accuracy düştü |
| Epoch | 30 | Early stopping ile gerçekte 15-25 epoch yeterli oldu |
| Scheduler | CosineAnnealing | Öğrenme hızını yumuşak düşürür, keskin minimumlara sıkışmayı önler |
| Gradient Clip | 1.0 | LSTM'de exploding gradient'a karşı (Week 10, Slide 21) |

### 3.2 Regularization Teknikleri

| Teknik | Uygulama Yeri | Etki |
|---|---|---|
| **Dropout (p=0.2)** | CNN sonrası | Özellik haritalarında co-adaptation'ı önler |
| **Dropout (p=0.3)** | LSTM arası, FC | Nöron bağımlılığını kırar |
| **BatchNorm** | Her CNN bloğu | İç kovaryans kaymasını azaltır, daha yüksek LR kullanımı sağlar |
| **LayerNorm** | LSTM sonrası | Sequence-to-sequence görevlerde BatchNorm'dan daha stabil |
| **L2 Regularization** | Adam weight_decay=1e-4 | Ağırlıkların büyümesini penalize eder |
| **Early Stopping** | patience=7 | Overfitting anında eğitimi durdurur |
| **Gradient Clipping** | clip=1.0 | Exploding gradient'ı önler |
| **AE Reconstruction Loss** | λ=0.1 | Bottleneck'i anlamlı temsil öğrenmeye zorlar |

---

## 4. Ablation Study

Her bloğun model performansına katkısını ölçmek amacıyla dört ayrı deney gerçekleştirilmiştir. Her deneyde yalnızca bir blok kaldırılmış, diğerleri sabit tutulmuştur.

| Yapılandırma | Val Accuracy | Tam Modele Fark |
|---|---|---|
| ECGNet (Tam — 5 blok) | *bkz. results/* | — |
| CNN Yok (LSTM ham sinyali alır) | *bkz. results/* | ↓ |
| LSTM Yok (CNN → Global Pool) | *bkz. results/* | ↓ |
| Attention Yok (LSTM son adım) | *bkz. results/* | ↓ |
| Autoencoder Yok (doğrudan classify) | *bkz. results/* | ↓ |

> Sayısal sonuçlar `results/ablation_study.png` dosyasında görselleştirilmiştir.

---

## 5. Kurulum ve Çalıştırma

```bash
# 1. Bağımlılıkları kur
pip3 install torch wfdb numpy pandas matplotlib scikit-learn tqdm

# 2. Proje klasörüne gir
cd ecg_project

# 3. Modeli test et (veri indirmeden önce mimariyi doğrula)
python3 model.py

# 4. Tam eğitimi başlat (veri otomatik indirilir, ~1.7 GB)
python3 train.py
```

---

## 6. Proje Yapısı

```
ecg_project/
├── dataset.py        # PTB-XL indirme, ön işleme, DataLoader
├── model.py          # 5 blokluk ECGNet + ablation varyantları
├── train.py          # Eğitim, değerlendirme, ablation study
├── README.md         # Bu dosya
├── ptb-xl/           # İndirilen veri seti (otomatik oluşturulur)
└── results/          # Eğitim grafikleri, model ağırlıkları (otomatik)
    ├── ecgnet_best.pt
    ├── ECGNet_training.png
    └── ablation_study.png
```

---

## 7. Referanslar

1. Wagner, P., Strodthoff, N., Bousseljot, R. D., Kreiseler, D., Lunze, F. I., Samek, W., & Schultz, T. (2020). **PTB-XL, a large publicly available electrocardiography dataset.** *Scientific Data, 7*(1), 154.

2. Hochreiter, S., & Schmidhuber, J. (1997). **Long short-term memory.** *Neural computation, 9*(8), 1735–1780.

3. Bahdanau, D., Cho, K., & Bengio, Y. (2014). **Neural machine translation by jointly learning to align and translate.** *arXiv:1409.0473*.

4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). **Deep learning** (Chapter 9: CNNs, Chapter 14: Autoencoders). MIT Press.

5. Kaya, Y. B. (2025). **SWE012 Deep Learning with Python — Week 6, 8, 10 Study Guides.** İstinye University.
