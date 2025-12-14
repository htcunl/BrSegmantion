# 🤖 ML Module - Brain Tumor Segmentation

Bu modül, beyin MR görüntülerinde tümör segmentasyonu için U-Net tabanlı derin öğrenme pipeline'ı içerir.

## 📋 İçindekiler

- [Dosya Yapısı](#-dosya-yapısı)
- [Kurulum](#-kurulum)
- [Veri Hazırlama](#-veri-hazırlama)
- [Model Eğitimi](#-model-eğitimi)
- [Konfigürasyon](#-konfigürasyon)
- [Model Mimarisi](#-model-mimarisi)
- [Utils Modülleri](#-utils-modülleri)
- [Model Export](#-model-export)
- [Sorun Giderme](#-sorun-giderme)

## 📁 Dosya Yapısı

```
ml/
├── src/
│   ├── __init__.py
│   ├── train_unet.py           # Ana eğitim scripti
│   ├── check_gpu.py            # GPU kontrol scripti
│   ├── config.yaml             # Eğitim konfigürasyonu
│   ├── inspect_h5.py           # H5 dosya inceleme
│   ├── prepare_brats.py        # BraTS dataset hazırlama
│   ├── prepare_h5_slices.py    # H5 slice hazırlama
│   ├── prepare_png_dataset.py  # PNG dataset hazırlama
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── unet.py             # U-Net model tanımı
│   │
│   └── utils/
│       ├── __init__.py
│       ├── data.py             # Veri pipeline (tf.data)
│       ├── losses.py           # Custom loss fonksiyonları
│       ├── metrics.py          # Custom metrikler
│       └── exporter.py         # Model export (TFLite, ONNX)
│
├── data/
│   ├── train/
│   │   ├── images/             # Eğitim görüntüleri
│   │   └── masks/              # Eğitim maskeleri
│   └── val/
│       ├── images/             # Validation görüntüleri
│       └── masks/              # Validation maskeleri
│
├── artifacts/                   # Model çıktıları
│   └── experiment_YYYYMMDD-HHMMSS/
│       ├── config.yaml         # Kullanılan konfigürasyon
│       ├── checkpoints/
│       │   └── best.weights.h5 # En iyi model ağırlıkları
│       ├── training_log.csv    # Eğitim logları
│       ├── tensorboard/        # TensorBoard logları
│       ├── model.tflite        # TFLite export
│       └── model.onnx          # ONNX export
│
├── prepare_dataset.py          # Dataset hazırlama scripti
├── install_tensorflow_gpu.md   # GPU kurulum rehberi
└── README.md
```

## 🔧 Kurulum

### 1. Sanal Ortam

```bash
cd ml
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r ../requirements.txt
```

### 2. GPU Desteği (Önerilir)

```bash
# Conda ile CUDA kurulumu
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

# GPU kontrolü
python -m src.check_gpu
```

Beklenen çıktı:
```
[OK] 1 GPU bulundu ve yapılandırıldı:
   GPU 0: /physical_device:GPU:0
```

## 📊 Veri Hazırlama

### Dataset Yapısı

Kaynak verinin aşağıdaki yapıda olması gerekir:

```
source_folder/
├── images/
│   ├── patient001_slice001.png
│   ├── patient001_slice002.png
│   └── ...
└── masks/
    ├── patient001_slice001.png  # Aynı isimle eşleşmeli
    ├── patient001_slice002.png
    └── ...
```

### prepare_dataset.py Kullanımı

```bash
python prepare_dataset.py --source <kaynak_yolu> --target data --val-ratio 0.2 --seed 42
```

**Parametreler:**

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `--source` | - | Kaynak veri dizini (zorunlu) |
| `--target` | `ml/data` | Hedef dizin |
| `--val-ratio` | `0.2` | Validation split oranı |
| `--seed` | `42` | Random seed |
| `--ext` | `png` | Dosya uzantıları (virgülle ayrılmış) |

**Örnek:**

```bash
python prepare_dataset.py \
    --source "C:\Users\tahak\Desktop\archive\brain_tumor" \
    --target data \
    --val-ratio 0.2 \
    --ext "png,jpg"
```

**Çıktı:**
```
Toplam çiftler: 1000 | Train: 800 | Val: 200
Kopyalama tamamlandı. Hedef: C:\Users\tahak\Desktop\BrSegmantion\ml\data
```

### BraTS Dataset Hazırlama

NIfTI formatındaki BraTS dataset için:

```bash
python -m src.prepare_brats --input <brats_path> --output data
```

## 🚀 Model Eğitimi

### Temel Kullanım

```bash
python -m src.train_unet --config src/config.yaml
```

### Eğitim Akışı

1. **GPU Kontrolü**: Mevcut GPU'lar tespit edilir ve memory growth aktifleştirilir
2. **Veri Yükleme**: `tf.data` pipeline ile verimli veri yükleme
3. **Model Oluşturma**: U-Net mimarisi build edilir
4. **Callback Hazırlığı**: 
   - ModelCheckpoint (en iyi modeli kaydet)
   - CSVLogger (eğitim logları)
   - TensorBoard (görselleştirme)
   - ReduceLROnPlateau (learning rate scheduling)
   - EarlyStopping (erken durdurma)
5. **Eğitim**: Belirtilen epoch sayısı kadar eğitim
6. **Export**: TFLite ve/veya ONNX formatına dönüşüm

### TensorBoard ile İzleme

```bash
tensorboard --logdir artifacts
```

Tarayıcıda http://localhost:6006 açın.

## ⚙️ Konfigürasyon

### config.yaml Yapısı

```yaml
# Experiment tanımı
experiment_name: "unet_brain_tumor"
seed: 42

# Veri yolları
paths:
  train_images: "data/train/images"
  train_masks: "data/train/masks"
  val_images: "data/val/images"
  val_masks: "data/val/masks"
  artifacts_dir: "artifacts"

# Eğitim parametreleri
training:
  img_size: 256           # Görüntü boyutu (256x256)
  batch_size: 2           # Batch size (GPU memory'ye göre ayarlayın)
  epochs: 100             # Maksimum epoch sayısı
  learning_rate: 0.0001   # Başlangıç learning rate
  optimizer: "adam"       # Optimizer (adam, sgd, rmsprop)
  loss: "bce_dice"        # Loss fonksiyonu (bce_dice, dice, binary_crossentropy)
  metrics:
    - "dice"              # DICE coefficient
    - "iou"               # IoU (Jaccard) score

# Data augmentation
augmentation:
  random_flip: true       # Yatay/dikey flip
  random_rotate: false    # 90° rotasyonlar
  random_zoom: false      # Zoom in/out
  zoom_scales: [0.9, 1.1] # Zoom aralığı

# Model export
export:
  save_best_only: true    # Sadece en iyi modeli kaydet
  tflite: true            # TFLite export
  onnx: false             # ONNX export (tf2onnx gerektirir)
```

### Parametre Önerileri

| Senaryo | batch_size | epochs | learning_rate |
|---------|------------|--------|---------------|
| Hızlı test | 8-16 | 10-20 | 0.001 |
| Normal eğitim | 4-8 | 50-100 | 0.0001 |
| Fine-tuning | 2-4 | 20-50 | 0.00001 |
| Büyük dataset | 16-32 | 100+ | 0.0001 |

## 🏗️ Model Mimarisi

### U-Net Detayları

```python
def build_unet(
    input_shape: Tuple[int, int, int] = (256, 256, 1),
    num_classes: int = 1,
    base_filters: int = 32,
    dropout: float = 0.1,
) -> keras.Model:
```

**Encoder Bloğu:**
```
Input → Conv2D(3x3) → BatchNorm → ReLU → Conv2D(3x3) → BatchNorm → ReLU → MaxPool2D
```

**Decoder Bloğu:**
```
UpConv2D(2x2) → Concat(skip_connection) → Conv2D(3x3) → BatchNorm → ReLU → Conv2D(3x3) → BatchNorm → ReLU
```

**Model Özeti:**
```
Total params: ~1.9M (base_filters=32)
Trainable params: ~1.9M
Non-trainable params: 0
```

### Katman Detayları

| Katman | Filtre | Çıktı Boyutu |
|--------|--------|--------------|
| enc1 | 32 | 256x256x32 |
| enc2 | 64 | 128x128x64 |
| enc3 | 128 | 64x64x128 |
| enc4 | 256 | 32x32x256 |
| bottleneck | 512 | 16x16x512 |
| dec1 | 256 | 32x32x256 |
| dec2 | 128 | 64x64x128 |
| dec3 | 64 | 128x128x64 |
| dec4 | 32 | 256x256x32 |
| output | 1 | 256x256x1 |

## 🧰 Utils Modülleri

### data.py - Veri Pipeline

```python
def create_dataset(
    image_dir: str,
    mask_dir: str,
    img_size: int,
    batch_size: int,
    shuffle: bool = True,
    augment: bool = False,
    augmentation_config: dict = None,
    extensions: Iterable[str] = ("png", "jpg", "jpeg", "tif", "tiff"),
) -> tf.data.Dataset:
```

**Özellikler:**
- Dosya ismine göre otomatik image-mask eşleştirme
- Lazy loading ile memory-efficient veri yükleme
- Paralel veri işleme (`tf.data.AUTOTUNE`)
- Configurable augmentation

**Desteklenen Augmentasyonlar:**
- `random_flip`: Yatay ve dikey flip
- `random_rotate`: 90° rotasyonlar (0°, 90°, 180°, 270°)
- `random_zoom`: Scale faktörü ile zoom

### losses.py - Loss Fonksiyonları

```python
# Dice Loss
def dice_loss(y_true, y_pred, smooth=1e-6):
    """1 - DICE coefficient"""
    
# BCE + Dice Loss (önerilen)
def bce_dice_loss(y_true, y_pred):
    """Binary Cross Entropy + Dice Loss kombinasyonu"""
```

**Loss Seçimi:**
- `bce_dice`: Dengeli sonuçlar için (önerilen)
- `dice`: Sadece overlap optimizasyonu
- `binary_crossentropy`: Standart BCE

### metrics.py - Metrikler

```python
# DICE Coefficient
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """2 * |A ∩ B| / (|A| + |B|)"""
    
# IoU Score
def iou_score(y_true, y_pred, smooth=1e-6):
    """|A ∩ B| / |A ∪ B|"""
```

### exporter.py - Model Export

```python
# TensorFlow Lite
def export_tflite(model: tf.keras.Model, export_path: Path) -> Path:
    """Keras modelini TFLite formatına dönüştürür"""

# ONNX
def export_onnx(model: tf.keras.Model, export_path: Path, opset: int = 13) -> Path:
    """Keras modelini ONNX formatına dönüştürür (tf2onnx gerektirir)"""
```

## 📤 Model Export

### TFLite Export

Config'de aktifleştirin:
```yaml
export:
  tflite: true
```

veya manuel:
```python
from src.utils.exporter import export_tflite
export_tflite(model, Path("model.tflite"))
```

### ONNX Export

```bash
pip install tf2onnx
```

```yaml
export:
  onnx: true
```

## 🔍 Sorun Giderme

### GPU Algılanmıyor

```bash
# GPU kontrolü
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# CUDA versiyon kontrolü
nvcc --version
```

**Çözüm:** CUDA 11.2 ve cuDNN 8.1 kurulu olduğundan emin olun.

### Out of Memory (OOM)

**Çözüm:**
1. `batch_size` değerini düşürün (2 veya 4)
2. `img_size` değerini düşürün (128)
3. Memory growth aktif olduğundan emin olun

### Veri Eşleşme Hatası

```
ValueError: Eşleşen dosya bulunamadı
```

**Çözüm:** 
- Image ve mask dosya isimlerinin aynı olduğundan emin olun
- Uzantıların doğru belirtildiğini kontrol edin

### Düşük DICE Score

**Çözüm:**
1. Daha fazla data augmentation ekleyin
2. Learning rate'i düşürün
3. Daha fazla epoch eğitin
4. Dropout oranını ayarlayın

## 📚 Referanslar

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [BraTS Challenge](https://www.med.upenn.edu/cbica/brats/)

---

📝 Detaylı kullanım için `config.yaml` dosyasını ve ana README'yi inceleyin.
