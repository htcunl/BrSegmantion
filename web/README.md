# Web Uygulaması

Web tabanlı Beyin MR Tümör Segmentasyonu uygulaması.

## Özellikler

- 🖼️ MR görüntüsü yükleme (PNG, JPG, DICOM)
- 🧠 U-Net modeli ile otomatik segmentasyon
- 🎨 Overlay görünümü
- 📊 DICE ve IoU metrikleri
- 💾 Sonuçları indirme

## Kurulum

```bash
# 1. Gerekli kütüphaneleri yükle
pip install -r requirements.txt

# 2. Uygulamayı başlat
python -m app.main
```

## Kullanım

1. Tarayıcıda http://localhost:5000 açın
2. MR görüntüsünü yükleyin
3. "Segmentasyon Yap" butonuna tıklayın
4. Sonuçları görüntüleyin ve indirin

## Yapı

```
web/
├── app/
│   ├── main.py              # Flask backend
│   ├── templates/
│   │   └── index.html       # HTML arayüzü
│   └── static/
│       ├── css/
│       │   └── style.css    # Stil dosyası
│       ├── js/
│       │   └── app.js       # Frontend JavaScript
│       └── uploads/         # Yüklenen dosyalar
├── requirements.txt         # Python bağımlılıkları
└── README.md               # Bu dosya
```

## API Endpoints

### POST /api/upload
MR görüntüsü yükle ve segmentasyon yap
- Parameter: `file` (multipart form data)
- Response: `{original, mask, overlay, filename}`

### GET /api/model-info
Model bilgileri
- Response: `{loaded, path, input_shape, output_shape, parameters}`

### POST /api/metrics
DICE ve IoU metrikleri (optional)
- Response: `{dice, iou}`

## Model

Eğitilmiş U-Net modeli otomatik olarak `/ml/artifacts/` dizininden yüklenir.
Model bulunamazsa hata mesajı gösterilir.

## Sistem Gereksinimleri

- Python 3.10+
- TensorFlow 2.15+
- GPU (önerilen, CPU'da daha yavaş çalışır)

## Notlar

- Görüntüler otomatik olarak 256x256 boyutuna yeniden boyutlandırılır
- Segmentasyon 0.5 threshold ile yapılır
- Overlay %40 opaklıkla gösterilir
