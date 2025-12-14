# 🌐 Web Module - Brain Tumor Segmentation UI

Bu modül, beyin MR görüntülerinde tümör segmentasyonu için Flask tabanlı web arayüzü sunar.

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Dosya Yapısı](#-dosya-yapısı)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [API Dokümantasyonu](#-api-dokümantasyonu)
- [Frontend Yapısı](#-frontend-yapısı)
- [Konfigürasyon](#-konfigürasyon)
- [Geliştirme](#-geliştirme)

## 🚀 Özellikler

### Kullanıcı Arayüzü
- 📁 **Drag & Drop Dosya Yükleme**: Kolay görüntü yükleme
- 🖼️ **Çoklu Format Desteği**: PNG, JPG, DICOM
- 📂 **Data Klasörü Tarayıcı**: Train/Val görüntülerini direkt seç
- 🔍 **Görüntü Arama**: Dataset içinde arama
- 📱 **Responsive Tasarım**: Mobil uyumlu

### Segmentasyon
- 🧠 **Gerçek Zamanlı Tahmin**: Model yüklü ise anlık segmentasyon
- 🎨 **Overlay Görünümü**: Tümör maskesi orijinal görüntü üzerinde
- 📊 **Metrik Hesaplama**: DICE, IoU, Hacim, Alan
- 🔄 **Mock Mode**: Model olmadan test için

### Export
- 💾 **Maske İndirme**: PNG formatında
- 🖼️ **Overlay İndirme**: PNG formatında
- 📋 **JSON Rapor**: Tüm metriklerle birlikte

## 📁 Dosya Yapısı

```
web/
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── main.py                  # Ana Flask uygulaması (684 satır)
│   │
│   ├── templates/
│   │   └── index.html           # Ana sayfa template (HTML/Jinja2)
│   │
│   └── static/
│       ├── css/
│       │   └── style.css        # Stil dosyası (703 satır)
│       ├── js/
│       │   └── app.js           # Frontend JavaScript (557 satır)
│       └── uploads/             # Yüklenen dosyalar (geçici)
│
├── static/
│   └── test_images/             # Test görüntüleri
│       ├── test_brain_1.png
│       ├── test_brain_2.png
│       └── test_brain_3.png
│
├── create_test_images.py        # Test görüntüsü oluşturma
├── run.py                       # Uygulama başlatıcı
└── README.md
```

## 🔧 Kurulum

### 1. Bağımlılıklar

```bash
cd web
pip install flask opencv-python pillow numpy tensorflow
```

Veya ana requirements.txt'den:
```bash
pip install -r ../requirements.txt
```

### 2. Test Görüntüleri (Opsiyonel)

```bash
python create_test_images.py
```

### 3. Model Hazırlama

Eğitilmiş model dosyasını belirtin:
```python
# app/main.py içinde
MODEL_PATH = "../ml/artifacts/experiment_xxx/checkpoints/best.weights.h5"
```

veya TFLite:
```python
MODEL_PATH = "../ml/artifacts/experiment_xxx/model.tflite"
```

## 🚀 Kullanım

### Uygulamayı Başlatma

```bash
cd web
python -m app.main
```

veya:
```bash
python run.py
```

**Çıktı:**
```
[OK] Model başarıyla yüklendi: ../ml/artifacts/.../best.weights.h5
 * Running on http://127.0.0.1:5000
```

### Tarayıcıda Açın

http://localhost:5000

### Kullanım Adımları

1. **Görüntü Yükleme**:
   - Drag & drop ile dosya sürükleyin
   - Veya "Dosya Seç" butonuna tıklayın
   - Data klasöründen görüntü seçin
   - Test görüntülerinden birini kullanın

2. **Segmentasyon**:
   - "🚀 Segmentasyon Yap" butonuna tıklayın
   - İşlem süresi: ~1-3 saniye (GPU), ~5-10 saniye (CPU)

3. **Sonuçları İncele**:
   - Orijinal görüntü
   - Segmentasyon maskesi
   - Overlay görünümü
   - Performans metrikleri

4. **Export**:
   - Maske İndir (PNG)
   - Overlay İndir (PNG)
   - Rapor İndir (JSON)

## 📡 API Dokümantasyonu

### Endpoints

#### `GET /`
Ana sayfa (index.html)

**Response:** HTML

---

#### `POST /api/upload`
Görüntü yükle ve segmentasyon yap

**Request:**
```
Content-Type: multipart/form-data
Body: file (image file)
```

**Response (Success - 200):**
```json
{
  "success": true,
  "original": "data:image/png;base64,...",
  "mask": "data:image/png;base64,...",
  "overlay": "data:image/png;base64,...",
  "metrics": {
    "dice": 0.8543,
    "iou": 0.7456,
    "volume": 12543,
    "area": "12543 px²"
  },
  "filename": "brain_scan.png",
  "processing_time": 1.234
}
```

**Response (Error - 400/500):**
```json
{
  "success": false,
  "error": "Hata mesajı"
}
```

---

#### `GET /api/status`
Model ve sistem durumu

**Response:**
```json
{
  "model_loaded": true,
  "model_path": "/path/to/model.h5",
  "model_type": "keras",
  "tensorflow_version": "2.10.1",
  "gpu_available": true
}
```

---

#### `GET /api/data-images`
Data klasöründeki görüntüleri listele

**Query Parameters:**
- `dataset`: "train", "val", veya "all" (varsayılan: "all")

**Response:**
```json
{
  "success": true,
  "images": [
    {
      "name": "patient001_slice001.png",
      "path": "/static/data/train/images/patient001_slice001.png",
      "dataset": "train"
    }
  ],
  "total": 1000
}
```

---

#### `POST /api/segment-data-image`
Data klasöründeki görüntüyü segmente et

**Request:**
```json
{
  "image_path": "train/images/patient001_slice001.png"
}
```

**Response:** `/api/upload` ile aynı format

## 🎨 Frontend Yapısı

### index.html - Ana Sayfa

```html
<!-- Header -->
<header class="header">
    <h1>🧠 Beyin MR Tümör Segmentasyonu</h1>
</header>

<!-- Model Status -->
<div class="model-status">
    <span id="modelStatus">{{ model_status }}</span>
</div>

<!-- Upload Section -->
<section class="upload-section">
    <div class="upload-area" id="uploadArea">
        <!-- Drag & Drop alanı -->
    </div>
    <div class="data-images-section">
        <!-- Data klasörü tarayıcısı -->
    </div>
    <div class="test-images-section">
        <!-- Test görüntüleri -->
    </div>
</section>

<!-- Results Section -->
<section class="results-section" id="resultsSection">
    <div class="results-grid">
        <!-- Original, Mask, Overlay canvasları -->
    </div>
    <div class="metrics-section">
        <!-- DICE, IoU, Volume, Area -->
    </div>
    <div class="download-section">
        <!-- İndirme butonları -->
    </div>
</section>
```

### app.js - Frontend JavaScript

**State Management:**
```javascript
const state = {
    selectedFile: null,
    results: null,
    isProcessing: false,
    dataImages: [],
    filteredImages: [],
    currentPage: 1,
    imagesPerPage: 20
};
```

**Ana Fonksiyonlar:**

| Fonksiyon | Açıklama |
|-----------|----------|
| `handleFileSelect(e)` | Dosya seçimi işleme |
| `handleDrop(e)` | Drag & drop işleme |
| `processImage()` | Segmentasyon isteği gönder |
| `displayResults(data)` | Sonuçları göster |
| `loadDataImages()` | Data klasörünü yükle |
| `filterAndDisplayImages()` | Filtreleme ve sayfalama |
| `downloadImage(canvas, filename)` | Canvas'ı indir |
| `downloadReport()` | JSON rapor indir |

### style.css - Stiller

**CSS Değişkenleri:**
```css
:root {
    --primary-color: #2563eb;
    --secondary-color: #64748b;
    --success-color: #10b981;
    --danger-color: #ef4444;
    --warning-color: #f59e0b;
    --bg-primary: #ffffff;
    --bg-secondary: #f8fafc;
}
```

**Ana Bileşenler:**
- `.container`: Ana konteyner
- `.header`: Üst başlık
- `.upload-area`: Drag & drop alanı
- `.results-grid`: Sonuç grid'i
- `.metrics-section`: Metrik kartları
- `.btn`: Buton stilleri

## ⚙️ Konfigürasyon

### main.py Ayarları

```python
# Dosya limitleri
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Upload klasörü
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Model ayarları
MODEL_PATH = None  # Otomatik algıla veya manuel belirt
IMAGE_SIZE = (256, 256)  # Model input boyutu
```

### Model Yükleme Önceliği

1. Manuel belirtilen yol
2. En son artifacts klasöründeki model
3. Mock mode (model bulunamazsa)

### Desteklenen Model Formatları

| Format | Uzantı | Açıklama |
|--------|--------|----------|
| Keras | .keras | TensorFlow 2.x native format |
| H5 | .h5 | Legacy Keras format |
| TFLite | .tflite | Mobile/Edge deployment |

## 🛠️ Geliştirme

### Debug Mode

```python
# main.py sonunda
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
```

### Yeni Endpoint Ekleme

```python
@app.route('/api/new-endpoint', methods=['POST'])
def new_endpoint():
    try:
        data = request.get_json()
        # İşlem yap
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
```

### Custom Metrik Ekleme

```python
# main.py içinde calculate_metrics fonksiyonuna ekle
def calculate_metrics(mask, original=None):
    metrics = {
        "dice": float(dice_score),
        "iou": float(iou_score),
        "volume": int(tumor_pixels),
        "area": f"{tumor_pixels} px²",
        # Yeni metrik
        "custom_metric": calculate_custom(mask)
    }
    return metrics
```

### Frontend Özelleştirme

**Yeni Buton:**
```html
<button class="btn btn-secondary" id="newBtn">Yeni İşlem</button>
```

```javascript
document.getElementById('newBtn').addEventListener('click', () => {
    // İşlem yap
});
```

**Stil Değişikliği:**
```css
:root {
    --primary-color: #your-color;
}
```

## 🔍 Sorun Giderme

### Model Yüklenmiyor

**Kontrol:**
```python
python -c "from app.main import load_model; load_model('path/to/model.h5')"
```

**Çözüm:**
- Model dosyasının var olduğundan emin olun
- TensorFlow versiyonunu kontrol edin
- Custom objects tanımlı mı kontrol edin

### CORS Hatası

```python
from flask_cors import CORS
CORS(app)
```

### Büyük Dosya Hatası

```python
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
```

### Yavaş İşleme

**Çözümler:**
1. GPU kullanımını aktifleştirin
2. Batch processing ekleyin
3. Model'i TFLite'a dönüştürün
4. Image size'ı küçültün

## 📚 Teknolojiler

| Teknoloji | Versiyon | Kullanım |
|-----------|----------|----------|
| Flask | 2.3+ | Web framework |
| TensorFlow | 2.10+ | Model inference |
| OpenCV | 4.7+ | Görüntü işleme |
| Pillow | 10.0+ | Görüntü I/O |
| NumPy | 1.23+ | Numerik işlemler |

---

📝 Ana proje README'si için üst dizine bakın.
