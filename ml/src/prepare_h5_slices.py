from __future__ import annotations

"""
BraTS2020_training_data/content/data içindeki .h5 slice dosyalarını
`ml/data/train/images` ve `ml/data/train/masks` klasörlerine PNG olarak dönüştürür.

Her .h5 dosyasında:
- `image`: (H, W, 4) float64
- `mask`:  (H, W, 3) uint8

Biz:
- image kanal ekseninde normalize edip [0, 255] aralığında tek bir gri görüntü üretiyoruz
- mask için kanallar üzerinde OR alıp (tümör var/yok) binary maske oluşturuyoruz
"""

import argparse
import random
from pathlib import Path
from typing import List

import h5py
import numpy as np
import imageio.v2 as imageio
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=".h5 slice -> images/masks PNG dönüştürücü")
    parser.add_argument("--data_dir", type=str, required=True, help="volume_XX_slice_YY.h5 dosyalarının olduğu klasör")
    parser.add_argument("--out_dir", type=str, required=True, help="Çıkış data klasörü (ml/data)")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train/val oranı (0-1 arası)")
    return parser.parse_args()


def list_h5_files(data_dir: Path) -> List[Path]:
    return sorted(data_dir.glob("*.h5"))


def load_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        img = np.array(f["image"], dtype=np.float32)  # (H, W, 4)
        mask = np.array(f["mask"], dtype=np.uint8)    # (H, W, 3)
    return img, mask


def preprocess_image(img: np.ndarray) -> np.ndarray:
    # (H, W, C) -> tek kanal gri görüntü
    # kanallar üzerinde ortalama alıp normalize ediyoruz
    if img.ndim == 3:
        img = img.mean(axis=-1)
    # normalize to 0-1
    img = img - img.min()
    denom = img.max() - img.min() + 1e-8
    img = img / denom
    img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
    return img_uint8


def preprocess_mask(mask: np.ndarray) -> np.ndarray:
    # (H, W, C) -> binary (0/255)
    if mask.ndim == 3:
        mask_bin = (mask.sum(axis=-1) > 0).astype(np.uint8)
    else:
        mask_bin = (mask > 0).astype(np.uint8)
    mask_uint8 = (mask_bin * 255).astype(np.uint8)
    return mask_uint8


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    if not data_dir.exists():
        raise SystemExit(f"data_dir bulunamadı: {data_dir}")

    train_img_dir = out_dir / "train" / "images"
    train_mask_dir = out_dir / "train" / "masks"
    val_img_dir = out_dir / "val" / "images"
    val_mask_dir = out_dir / "val" / "masks"

    for d in (train_img_dir, train_mask_dir, val_img_dir, val_mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    files = list_h5_files(data_dir)
    if not files:
        raise SystemExit(f"{data_dir} içinde .h5 dosyası bulunamadı.")

    # shuffle for train/val split
    random.shuffle(files)
    n_train = int(len(files) * args.train_ratio)

    for idx, path in enumerate(tqdm(files, desc="H5 slices")):
        img, mask = load_h5(path)
        img_p = preprocess_image(img)
        mask_p = preprocess_mask(mask)

        is_train = idx < n_train
        img_out_dir = train_img_dir if is_train else val_img_dir
        mask_out_dir = train_mask_dir if is_train else val_mask_dir

        stem = path.stem  # volume_XX_slice_YY
        img_name = f"{stem}.png"
        mask_name = f"{stem}.png"

        imageio.imwrite(img_out_dir / img_name, img_p)
        imageio.imwrite(mask_out_dir / mask_name, mask_p)

    print(f"Toplam .h5 sayısı: {len(files)}")
    print(f"Train images: {len(list(train_img_dir.glob('*.png')))}")
    print(f"Val images:   {len(list(val_img_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()


