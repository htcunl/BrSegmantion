"""
Test görüntüleri oluştur
"""
import os
import numpy as np
from PIL import Image
import cv2

# Test klasörü oluştur
test_dir = os.path.join(os.path.dirname(__file__), 'static', 'test_images')
os.makedirs(test_dir, exist_ok=True)

# 1. Gürültülü gri görüntü (MR benzeri)
for i in range(3):
    # Temel görüntü oluştur
    img = np.random.randint(50, 150, (256, 256), dtype=np.uint8)
    
    # Gaussian blur
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Contrast artır
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Daire şeklinde tümör ekle (merkez etrafında)
    center = (128, 128)
    radius = 30 + np.random.randint(-10, 10)
    cv2.circle(img, center, radius, 200, -1)
    cv2.circle(img, center, radius, 180, 2)
    
    # Kaydet
    Image.fromarray(img).save(os.path.join(test_dir, f'test_brain_{i+1}.png'))
    print(f"✓ test_brain_{i+1}.png oluşturuldu")

print(f"\nTest görüntüleri kaydedildi: {test_dir}")
