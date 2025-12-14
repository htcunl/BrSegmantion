# TensorFlow GPU Kurulumu

TensorFlow'un GPU versiyonunu kurmak için:

## Yöntem 1: TensorFlow'u yeniden kur (Önerilen)

```powershell
cd "C:\Users\Lenovo\OneDrive\Desktop\BrSegmantion\ml"
.\.venv\Scripts\activate

# Mevcut TensorFlow'u kaldır
pip uninstall tensorflow -y

# GPU desteğiyle yeniden kur
pip install tensorflow[and-cuda]
```

## Yöntem 2: Eğer Yöntem 1 çalışmazsa

Bazı NVIDIA paketleri bulunamazsa, TensorFlow'un GPU versiyonunu manuel olarak kurmayı deneyebilirsin:

```powershell
pip install tensorflow-gpu
```

**Not:** TensorFlow 2.x'ten itibaren `tensorflow-gpu` paketi artık ayrı değil, ama bazı durumlarda çalışabilir.

## Kontrol

Kurulumdan sonra:

```powershell
python -m src.check_gpu
```

Bu komut GPU'nun görünüp görünmediğini gösterecek.

