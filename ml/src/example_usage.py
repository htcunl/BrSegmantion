
import sys
from pathlib import Path

# Script'in bulunduğu dizini path'e ekle
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent.parent.parent))

from ml.src.plot_training_history import plot_training_history

# Kullanım - gerçek bir history.json dosyasının yolunu belirtin
if __name__ == "__main__":
    # Mevcut bir history.json dosyasının yolu
    history_path = "ml/artifacts/unet_brain_tumor_20251205-163801/history.json"
    
    # Alternatif: kullanıcıdan al
    # history_path = input("History JSON dosyasının yolu: ")
    
    plot_training_history(history_path)

