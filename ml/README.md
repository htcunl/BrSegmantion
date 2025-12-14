Prepare dataset and run training

Quick steps to prepare your PNG dataset (assuming you placed the attached folder at `C:\Users\Lenovo\Desktop\archive (1)`):

PowerShell example:

```powershell
# copy/split images into `ml/data` (train/val)
python .\ml\prepare_dataset.py --source "C:\Users\Lenovo\Desktop\archive (1)" --target "ml/data" --val-ratio 0.2

# then run training (uses `ml/src/config.yaml`)
python -m ml.src.train_unet --config ml/src/config.yaml
```

Notes:
- The script expects `images/` and `masks/` folders under the `--source` path.
- Masks are matched by filename (same name or same stem with different extension).
- Default extension is `png` but you can pass `--ext "png,jpg"` if needed.
## ML Bileşeni

TensorFlow/Keras tabanlı U-Net eğitimi için temel akış aşağıdaki gibidir:

1. `data/` klasörüne BraTS veya benzer bir beyin MR + tümör maskesi dataseti yerleştirilir (NIfTI veya PNG slice desteklenebilir).
2. `src/config.yaml` içinde eğitim parametreleri (dosya yolları, hiperparametreler) düzenlenir.
3. `python -m src.train_unet --config src/config.yaml` komutu ile eğitim başlatılır.
4. Model çıktıları `artifacts/` klasöründe saklanır (ağırlıklar, eğitim logları, TFLite/ONNX dönüşümleri).

### Dosya/Dizin Yapısı

```
ml/
├─ data/          # Ham veri (git'e eklenmez)
├─ notebooks/     # Keşif/deneme defterleri
├─ src/
│  ├─ models/     # U-Net tanımları
│  ├─ utils/      # Veri hazırlama yardımcıları
│  ├─ train_unet.py
│  └─ config.yaml
└─ requirements.txt
```

### Sanal Ortam & Kurulum

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows PowerShell
pip install -r requirements.txt
```

### Sonraki Adımlar

- `src/train_unet.py` içinde veri yükleyici ve eğitim döngüsü tamamlanacak.
- Eğitim sonrası modeli `.tflite`/`.onnx` formatına dönüştüren script eklenecek.
- Notebooks klasöründe veri keşfi için örnek defter paylaşılacak.

