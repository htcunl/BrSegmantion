#!/usr/bin/env python
"""
Beyin MR Tümör Segmentasyonu Web Uygulaması
Başlatma scripti
"""

import os
import sys
from pathlib import Path

# İlgili dizinleri ekle
web_dir = Path(__file__).parent
sys.path.insert(0, str(web_dir))

# Flask uygulamasını çalıştır
if __name__ == '__main__':
    os.chdir(web_dir)
    from app.main import app
    
    # Geliştirme ortamında çalıştır
    print("=" * 60)
    print("Beyin MR Tümör Segmentasyonu Web Uygulaması")
    print("=" * 60)
    print("\nAnahtarlamalar:")
    print("  - Sayfayı ziyaret edin: http://localhost:5000")
    print("  - Debug modu: Açık")
    print("  - Host: 0.0.0.0")
    print("  - Port: 5000")
    print("\nÇıkmak için Ctrl+C tuşlarına basın\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
