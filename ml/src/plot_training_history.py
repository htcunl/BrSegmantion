

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt


def find_best_epochs(history: Dict) -> Dict[str, Tuple[int, float]]:
    """
    En iyi epoch'ları bulur.
    
    Returns:
        Dict: Her metrik için (epoch_index, value) tuple'ı
    """
    best_epochs = {}
    
    # En düşük validation loss
    if 'val_loss' in history and len(history['val_loss']) > 0:
        best_val_loss_idx = min(range(len(history['val_loss'])), key=lambda i: history['val_loss'][i])
        best_epochs['val_loss'] = (best_val_loss_idx + 1, history['val_loss'][best_val_loss_idx])
    
    # En yüksek validation dice coefficient
    if 'val_dice_coefficient' in history and len(history['val_dice_coefficient']) > 0:
        best_val_dice_idx = max(range(len(history['val_dice_coefficient'])), key=lambda i: history['val_dice_coefficient'][i])
        best_epochs['val_dice_coefficient'] = (best_val_dice_idx + 1, history['val_dice_coefficient'][best_val_dice_idx])
    
    # En yüksek validation IoU
    if 'val_iou_score' in history and len(history['val_iou_score']) > 0:
        best_val_iou_idx = max(range(len(history['val_iou_score'])), key=lambda i: history['val_iou_score'][i])
        best_epochs['val_iou_score'] = (best_val_iou_idx + 1, history['val_iou_score'][best_val_iou_idx])
    
    return best_epochs


def print_epoch_table(history: Dict, best_epochs: Dict[str, Tuple[int, float]]):
    """Her epoch'un değerlerini tablo formatında yazdırır."""
    if 'loss' not in history or len(history['loss']) == 0:
        return
    
    num_epochs = len(history['loss'])
    
    print("\n" + "="*120)
    print("EPOCH BAZINDA DETAYLI SONUÇLAR")
    print("="*120)
    
    # Başlık satırı
    header = ["Epoch"]
    if 'loss' in history:
        header.append("Train Loss")
    if 'val_loss' in history:
        header.append("Val Loss")
    if 'dice_coefficient' in history:
        header.append("Train Dice")
    if 'val_dice_coefficient' in history:
        header.append("Val Dice")
    if 'iou_score' in history:
        header.append("Train IoU")
    if 'val_iou_score' in history:
        header.append("Val IoU")
    if 'lr' in history:
        header.append("LR")
    
    print(f"{'Epoch':<8} ", end="")
    for h in header[1:]:
        print(f"{h:<12} ", end="")
    print()
    print("-"*120)
    
    # Her epoch için değerleri yazdır
    for epoch in range(1, num_epochs + 1):
        marker = ""
        
        # En iyi val_dice_coefficient epoch'unu işaretle (model seçimi bunu kullanıyor)
        if 'val_dice_coefficient' in best_epochs:
            best_epoch, _ = best_epochs['val_dice_coefficient']
            if epoch == best_epoch:
                marker = "*"
        
        print(f"{epoch}{marker:<7} ", end="")
        
        if 'loss' in history:
            val = history['loss'][epoch-1]
            print(f"{val:<12.6f} ", end="")
        if 'val_loss' in history:
            val = history['val_loss'][epoch-1]
            print(f"{val:<12.6f} ", end="")
        if 'dice_coefficient' in history:
            val = history['dice_coefficient'][epoch-1]
            print(f"{val:<12.6f} ", end="")
        if 'val_dice_coefficient' in history:
            val = history['val_dice_coefficient'][epoch-1]
            print(f"{val:<12.6f} ", end="")
        if 'iou_score' in history:
            val = history['iou_score'][epoch-1]
            print(f"{val:<12.6f} ", end="")
        if 'val_iou_score' in history:
            val = history['val_iou_score'][epoch-1]
            print(f"{val:<12.6f} ", end="")
        if 'lr' in history:
            val = history['lr'][epoch-1]
            print(f"{val:<12.8f} ", end="")
        
        print()
    
    print("="*120)
    print("* = En iyi epoch (val_dice_coefficient'a gore)")
    print()


def print_summary(history: Dict, best_epochs: Dict[str, Tuple[int, float]]):
    """Özet istatistikleri yazdırır."""
    if 'loss' not in history or len(history['loss']) == 0:
        return
    
    num_epochs = len(history['loss'])
    
    print("\n" + "="*80)
    print("ÖZET İSTATİSTİKLER")
    print("="*80)
    
    print(f"\nToplam Epoch Sayısı: {num_epochs}")
    print(f"Son Epoch: {num_epochs}")
    
    # En iyi epoch bilgileri
    print("\nEN IYI SONUCLAR:")
    print("-"*80)
    
    if 'val_dice_coefficient' in best_epochs:
        epoch, value = best_epochs['val_dice_coefficient']
        print(f"  En Iyi Val Dice Coefficient: {value:.6f} (Epoch {epoch})")
        if 'dice_coefficient' in history:
            train_val = history['dice_coefficient'][epoch-1]
            print(f"    -> Bu epoch'ta Train Dice: {train_val:.6f}")
        if 'val_loss' in history:
            loss_val = history['val_loss'][epoch-1]
            print(f"    -> Bu epoch'ta Val Loss: {loss_val:.6f}")
    
    if 'val_iou_score' in best_epochs:
        epoch, value = best_epochs['val_iou_score']
        print(f"  En Iyi Val IoU Score: {value:.6f} (Epoch {epoch})")
    
    if 'val_loss' in best_epochs:
        epoch, value = best_epochs['val_loss']
        print(f"  En Dusuk Val Loss: {value:.6f} (Epoch {epoch})")
    
    # Son epoch değerleri
    print("\nSON EPOCH DEGERLERI:")
    print("-"*80)
    if 'loss' in history:
        print(f"  Train Loss: {history['loss'][-1]:.6f}")
    if 'val_loss' in history:
        print(f"  Val Loss: {history['val_loss'][-1]:.6f}")
    if 'dice_coefficient' in history:
        print(f"  Train Dice: {history['dice_coefficient'][-1]:.6f}")
    if 'val_dice_coefficient' in history:
        print(f"  Val Dice: {history['val_dice_coefficient'][-1]:.6f}")
    if 'iou_score' in history:
        print(f"  Train IoU: {history['iou_score'][-1]:.6f}")
    if 'val_iou_score' in history:
        print(f"  Val IoU: {history['val_iou_score'][-1]:.6f}")
    
    # İyileşme oranları
    print("\nIYILESME ORANLARI (Ilk vs Son Epoch):")
    print("-"*80)
    if 'loss' in history and len(history['loss']) > 1:
        improvement = ((history['loss'][0] - history['loss'][-1]) / history['loss'][0]) * 100
        print(f"  Train Loss: {improvement:.2f}% iyileşme")
    
    if 'val_loss' in history and len(history['val_loss']) > 1:
        improvement = ((history['val_loss'][0] - history['val_loss'][-1]) / history['val_loss'][0]) * 100
        print(f"  Val Loss: {improvement:.2f}% iyileşme")
    
    if 'dice_coefficient' in history and len(history['dice_coefficient']) > 1:
        improvement = ((history['dice_coefficient'][-1] - history['dice_coefficient'][0]) / (1 - history['dice_coefficient'][0])) * 100 if history['dice_coefficient'][0] < 1 else 0
        print(f"  Train Dice: {improvement:.2f}% iyileşme")
    
    if 'val_dice_coefficient' in history and len(history['val_dice_coefficient']) > 1:
        improvement = ((history['val_dice_coefficient'][-1] - history['val_dice_coefficient'][0]) / (1 - history['val_dice_coefficient'][0])) * 100 if history['val_dice_coefficient'][0] < 1 else 0
        print(f"  Val Dice: {improvement:.2f}% iyileşme")
    
    print("="*80)
    print()


def plot_training_history(json_file_path: str, output_path: str = None, show_table: bool = True):
    """
    Eğitim geçmişini JSON dosyasından okuyup görselleştirir.

    Args:
        json_file_path: History JSON dosyasının yolu
        output_path: Çıktı grafiğinin kaydedileceği yol (opsiyonel)
        show_table: Epoch tablosunu yazdırıp yazdırmayacağı (varsayılan: True)
    """
    # 1. JSON verisini dosyadan oku
    json_path = Path(json_file_path)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"Hata: '{json_file_path}' dosyası bulunamadı.")
        return
    except json.JSONDecodeError as e:
        print(f"Hata: JSON dosyası okunamadı: {e}")
        return

    # Epoch sayısını belirle (uzunluktan)
    if 'loss' not in history or len(history['loss']) == 0:
        print("Hata: History dosyasında geçerli veri bulunamadı.")
        return
    
    epochs = list(range(1, len(history['loss']) + 1))
    
    # En iyi epoch'ları bul
    best_epochs = find_best_epochs(history)
    
    # Özet ve tablo yazdır
    print_summary(history, best_epochs)
    if show_table:
        print_epoch_table(history, best_epochs)

    # 2. Grafik Alanını Oluştur (2 satır, 2 sütun)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Eğitim Performansı', fontsize=16)

    # --- Grafik 1: Loss (Kayıp) ---
    if 'loss' in history and 'val_loss' in history:
        axs[0, 0].plot(epochs, history['loss'], label='Training Loss', color='blue', linewidth=2)
        axs[0, 0].plot(epochs, history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
        
        # En iyi val_loss noktasını işaretle
        if 'val_loss' in best_epochs:
            best_epoch, best_value = best_epochs['val_loss']
            axs[0, 0].scatter([best_epoch], [best_value], color='red', s=200, zorder=5, 
                            marker='*', edgecolors='black', linewidths=1.5, label=f'Best (Epoch {best_epoch})')
        
        axs[0, 0].set_title('Loss (Kayıp)')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss Değeri')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)

    # --- Grafik 2: Dice Coefficient ---
    if 'dice_coefficient' in history and 'val_dice_coefficient' in history:
        axs[0, 1].plot(epochs, history['dice_coefficient'], label='Train Dice', color='green', linewidth=2)
        axs[0, 1].plot(epochs, history['val_dice_coefficient'], label='Val Dice', color='red', linewidth=2)
        
        # En iyi val_dice_coefficient noktasını işaretle
        if 'val_dice_coefficient' in best_epochs:
            best_epoch, best_value = best_epochs['val_dice_coefficient']
            axs[0, 1].scatter([best_epoch], [best_value], color='gold', s=200, zorder=5, 
                            marker='*', edgecolors='black', linewidths=1.5, label=f'Best (Epoch {best_epoch})')
        
        axs[0, 1].set_title('Dice Coefficient')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Skor')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].set_ylim([0, 1])  # Dice score 0-1 arasında

    # --- Grafik 3: IoU Score ---
    if 'iou_score' in history and 'val_iou_score' in history:
        axs[1, 0].plot(epochs, history['iou_score'], label='Train IoU', color='purple', linewidth=2)
        axs[1, 0].plot(epochs, history['val_iou_score'], label='Val IoU', color='brown', linewidth=2)
        
        # En iyi val_iou_score noktasını işaretle
        if 'val_iou_score' in best_epochs:
            best_epoch, best_value = best_epochs['val_iou_score']
            axs[1, 0].scatter([best_epoch], [best_value], color='gold', s=200, zorder=5, 
                            marker='*', edgecolors='black', linewidths=1.5, label=f'Best (Epoch {best_epoch})')
        
        axs[1, 0].set_title('IoU Score (Jaccard Index)')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Skor')
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].set_ylim([0, 1])  # IoU score 0-1 arasında

    # --- Grafik 4: Learning Rate (Öğrenme Oranı) ---
    if 'lr' in history:
        axs[1, 1].plot(epochs, history['lr'], label='Learning Rate', color='black', linestyle='--', linewidth=2)
        axs[1, 1].set_title('Learning Rate Değişimi')
        axs[1, 1].set_xlabel('Epoch')
        axs[1, 1].set_ylabel('LR')
        axs[1, 1].set_yscale('log')  # Logaritmik ölçek (değişimleri daha iyi görmek için)
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3, which="both")

    # Düzeni sıkılaştır (yazıların üst üste binmesini engeller)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Çıktı yolu belirlenmemişse, JSON dosyasının bulunduğu dizine kaydet
    if output_path is None:
        output_path = json_path.parent / 'training_history_plots.png'
    else:
        output_path = Path(output_path)
    
    # Grafikleri bir dosyaya kaydet
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Grafikler kaydedildi: {output_path}")
    
    # Göster
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Eğitim geçmişini (training history) görselleştirir"
    )
    parser.add_argument(
        '--json',
        type=str,
        required=True,
        help='History JSON dosyasının yolu'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Çıktı grafiğinin kaydedileceği yol (opsiyonel)'
    )
    parser.add_argument(
        '--no-table',
        action='store_true',
        help='Epoch tablosunu yazdırma'
    )
    
    args = parser.parse_args()
    plot_training_history(args.json, args.output, show_table=not args.no_table)


if __name__ == "__main__":
    main()

