# 🧠 Brain Tumor Segmentation Project

Bu proje, beyin MR görüntülerinde tümör segmentasyonu için kapsamlı bir pipeline sunmaktadır. U-Net derin öğrenme modeli ve web tabanlı kullanıcı arayüzü içerir.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Proje Yapısı](#-proje-yapısı)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Model Mimarisi](#-model-mimarisi)
- [Performans Metrikleri](#-performans-metrikleri)
- [Ekran Görüntüleri](#-ekran-görüntüleri)
- [Katkıda Bulunma](#-katkıda-bulunma)

## 🚀 Özellikler

### Makine Öğrenimi (ML)
- ✅ **U-Net Mimarisi**: Medikal görüntü segmentasyonu için optimize edilmiş encoder-decoder yapısı
- ✅ **Esnek Veri Pipeline**: PNG, JPG, TIFF, NIfTI formatları desteği
- ✅ **Data Augmentation**: Flip, rotate, zoom transformasyonları
- ✅ **Custom Loss Fonksiyonları**: BCE-Dice Loss, Dice Loss
- ✅ **Metrikler**: DICE Coefficient, IoU (Jaccard) Score
- ✅ **Model Export**: TFLite ve ONNX formatlarına dönüşüm
- ✅ **GPU Desteği**: CUDA/cuDNN ile hızlandırılmış eğitim

### Web Uygulaması
- ✅ **Drag & Drop Yükleme**: Kolay görüntü yükleme arayüzü
- ✅ **Gerçek Zamanlı Segmentasyon**: Anlık tümör tespiti
- ✅ **Overlay Görünümü**: Orijinal görüntü üzerinde maske gösterimi
- ✅ **Metrik Hesaplama**: DICE, IoU, Tümör Hacmi, Alan hesaplamaları
- ✅ **Sonuç İndirme**: Maske, overlay ve JSON rapor export
- ✅ **Responsive Tasarım**: Mobil uyumlu arayüz

## 📁 Proje Yapısı

```
BrSegmantion/
├── ml/                          # Makine öğrenimi modülü
│   ├── src/
│   │   ├── models/
│   │   │   └── unet.py          # U-Net model tanımı
│   │   ├── utils/
│   │   │   ├── data.py          # Veri pipeline
│   │   │   ├── losses.py        # Loss fonksiyonları
│   │   │   ├── metrics.py       # Metrik fonksiyonları
│   │   │   └── exporter.py      # Model export (TFLite, ONNX)
│   │   ├── train_unet.py        # Eğitim scripti
│   │   └── config.yaml          # Konfigürasyon
│   ├── data/                    # Veri klasörü
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── masks/
│   │   └── val/
│   │       ├── images/
│   │       └── masks/
│   ├── artifacts/               # Model çıktıları
│   └── prepare_dataset.py       # Veri hazırlama scripti
│
├── web/                         # Web uygulaması modülü
│   ├── app/
│   │   ├── main.py              # Flask backend
│   │   ├── templates/
│   │   │   └── index.html       # Ana sayfa
│   │   └── static/
│   │       ├── css/style.css    # Stil dosyaları
│   │       ├── js/app.js        # Frontend JavaScript
│   │       └── uploads/         # Yüklenen dosyalar
│   └── static/test_images/      # Test görüntüleri
│
├── requirements.txt             # Python bağımlılıkları
├── cudatools-and-cudn.txt       # CUDA kurulum komutu
└── README.md                    # Bu dosya
```

## 🔧 Kurulum

### 1. Ön Gereksinimler

- Python 3.8 veya üzeri
- NVIDIA GPU (opsiyonel, eğitim için önerilir)
- CUDA 11.2 ve cuDNN 8.1 (GPU kullanımı için)

### 2. Repository'yi Klonlayın

```bash
git clone https://github.com/yourusername/BrSegmantion.git
cd BrSegmantion
```

### 3. Sanal Ortam Oluşturun

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/MacOS
source .venv/bin/activate
```

### 4. Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### 5. GPU Desteği (Opsiyonel)

CUDA ve cuDNN için conda kullanarak:

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

## 💻 Kullanım

### Veri Hazırlama

```bash
cd ml
python prepare_dataset.py --source <veri_yolu> --target data --val-ratio 0.2
```

**Parametreler:**
- `--source`: Kaynak veri dizini (`images/` ve `masks/` içermeli)
- `--target`: Hedef dizin (varsayılan: `data`)
- `--val-ratio`: Validation oranı (varsayılan: 0.2)
- `--ext`: Dosya uzantıları (varsayılan: png)

### Model Eğitimi

```bash
cd ml
python -m src.train_unet --config src/config.yaml
```

Eğitim tamamlandığında çıktılar `artifacts/` klasörüne kaydedilir:
- `best.weights.h5`: En iyi model ağırlıkları
- `training_log.csv`: Eğitim logları
- `tensorboard/`: TensorBoard logları
- `model.tflite`: TFLite formatı (opsiyonel)

### Web Uygulaması

```bash
cd web
python -m app.main
```

Tarayıcıda http://localhost:5000 adresini açın.

## 🏗️ Model Mimarisi

### U-Net Yapısı

```
Input (256x256x1)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    ENCODER PATH                          │
├─────────────────────────────────────────────────────────┤
│  enc1: Conv(32) → BN → ReLU → Conv(32) → Pool           │
│  enc2: Conv(64) → BN → ReLU → Conv(64) → Pool           │
│  enc3: Conv(128) → BN → ReLU → Conv(128) → Pool         │
│  enc4: Conv(256) → BN → ReLU → Conv(256) → Pool         │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│               BOTTLENECK (512 filters)                   │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│                    DECODER PATH                          │
├─────────────────────────────────────────────────────────┤
│  dec1: UpConv(256) → Concat(enc4) → Conv(256)           │
│  dec2: UpConv(128) → Concat(enc3) → Conv(128)           │
│  dec3: UpConv(64) → Concat(enc2) → Conv(64)             │
│  dec4: UpConv(32) → Concat(enc1) → Conv(32)             │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
            Output Conv (1x1, Sigmoid)
                        │
                        ▼
              Output (256x256x1)
```

### Hiperparametreler (config.yaml)

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `img_size` | 256 | Görüntü boyutu |
| `batch_size` | 2 | Batch boyutu |
| `epochs` | 100 | Eğitim epoch sayısı |
| `learning_rate` | 0.0001 | Öğrenme oranı |
| `optimizer` | adam | Optimizer |
| `loss` | bce_dice | Loss fonksiyonu |
| `base_filters` | 32 | İlk katman filtre sayısı |
| `dropout` | 0.1 | Dropout oranı |

## 📊 Performans Metrikleri

### DICE Coefficient
```
DICE = (2 × |A ∩ B|) / (|A| + |B|)
```
- 0-1 arasında değer alır
- 1'e yakın değerler daha iyi segmentasyon gösterir

### IoU (Intersection over Union)
```
IoU = |A ∩ B| / |A ∪ B|
```
- Jaccard Index olarak da bilinir
- Segmentasyon kalitesinin standart ölçüsü

### Loss Fonksiyonları

**BCE-Dice Loss:**
```python
Loss = BCE(y_true, y_pred) + (1 - DICE(y_true, y_pred))
```

**Dice Loss:**
```python
Loss = 1 - DICE(y_true, y_pred)
```

## 🖼️ Desteklenen Formatlar

| Format | Uzantı | Açıklama |
|--------|--------|----------|
| PNG | .png | 8-bit/16-bit grayscale |
| JPEG | .jpg, .jpeg | 8-bit grayscale |
| TIFF | .tif, .tiff | 8-bit/16-bit |
| DICOM | .dcm | Medikal görüntü formatı |

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 📧 İletişim

Sorularınız için issue açabilir veya pull request gönderebilirsiniz.

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
