from __future__ import annotations

"""
PNG formatındaki images ve masks klasörlerindeki dosyaları
train/val olarak böler ve ml/data klasörüne kopyalar.

Kullanım:
    python -m src.prepare_png_dataset --data_dir "C:/Users/Lenovo/Desktop/archive (1)" --out_dir "ml/data" --train_ratio 0.9
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PNG images/masks -> train/val split")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="images ve masks klasörlerinin bulunduğu ana klasör (örn: archive (1))",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Çıkış data klasörü (ml/data)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train/val oranı (0-1 arası)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def find_matching_pairs(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    """Images ve masks klasörlerindeki eşleşen PNG dosyalarını bulur."""
    image_files = sorted(images_dir.glob("*.png"))
    mask_files = sorted(masks_dir.glob("*.png"))

    # Dosya isimlerine göre eşleştir
    image_dict = {img.name: img for img in image_files}
    mask_dict = {mask.name: mask for mask in mask_files}

    # Her iki klasörde de bulunan dosyaları bul
    common_names = set(image_dict.keys()) & set(mask_dict.keys())

    pairs = [(image_dict[name], mask_dict[name]) for name in sorted(common_names)]

    if len(pairs) == 0:
        raise ValueError(
            f"Eşleşen dosya bulunamadı. "
            f"Images: {len(image_files)}, Masks: {len(mask_files)}"
        )

    return pairs


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"

    if not images_dir.exists():
        raise SystemExit(f"images klasörü bulunamadı: {images_dir}")
    if not masks_dir.exists():
        raise SystemExit(f"masks klasörü bulunamadı: {masks_dir}")

    train_img_dir = out_dir / "train" / "images"
    train_mask_dir = out_dir / "train" / "masks"
    val_img_dir = out_dir / "val" / "images"
    val_mask_dir = out_dir / "val" / "masks"

    for d in (train_img_dir, train_mask_dir, val_img_dir, val_mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Eşleşen dosya çiftlerini bul
    pairs = find_matching_pairs(images_dir, masks_dir)
    print(f"Toplam {len(pairs)} eşleşen dosya çifti bulundu.")

    # Random seed ayarla ve shuffle yap
    random.seed(args.seed)
    random.shuffle(pairs)

    # Train/val split
    n_train = int(len(pairs) * args.train_ratio)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    print(f"Train: {len(train_pairs)} çift")
    print(f"Val:   {len(val_pairs)} çift")

    # Train dosyalarını kopyala
    for img_path, mask_path in tqdm(train_pairs, desc="Train kopyalanıyor"):
        shutil.copy2(img_path, train_img_dir / img_path.name)
        shutil.copy2(mask_path, train_mask_dir / mask_path.name)

    # Val dosyalarını kopyala
    for img_path, mask_path in tqdm(val_pairs, desc="Val kopyalanıyor"):
        shutil.copy2(img_path, val_img_dir / img_path.name)
        shutil.copy2(mask_path, val_mask_dir / mask_path.name)

    print(f"\nTamamlandı!")
    print(f"Train images: {len(list(train_img_dir.glob('*.png')))}")
    print(f"Train masks:   {len(list(train_mask_dir.glob('*.png')))}")
    print(f"Val images:   {len(list(val_img_dir.glob('*.png')))}")
    print(f"Val masks:    {len(list(val_mask_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()



