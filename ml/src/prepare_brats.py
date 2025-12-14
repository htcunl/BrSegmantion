from __future__ import annotations

"""
BraTS2020 3D NIfTI dosyalarından 2D slice'lar üretip
`ml/data/train/images` ve `ml/data/train/masks` klasörlerine kaydeder.

Örnek kullanım (Windows PowerShell'de tek satır):

    python -m src.prepare_brats --raw_dir "PATH_TO_BraTS2020_training_data" --out_dir "PATH_TO_ml_data" --train_ratio 0.9
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm
import imageio.v2 as imageio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BraTS2020 -> 2D slice (images/masks) dönüştürücü")
    parser.add_argument("--raw_dir", type=str, required=True, help="BraTS2020_training_data klasör yolu")
    parser.add_argument("--out_dir", type=str, required=True, help="Çıkış data klasörü (ml/data)")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train/val oranı (0-1)")
    parser.add_argument("--modality", type=str, default="t1ce", help="Kullanılacak modalite (t1, t1ce, t2, flair)")
    parser.add_argument("--step", type=int, default=2, help="Her kaç slice'ta bir örnek alınacak")
    parser.add_argument("--min_tumor_pixels", type=int, default=50, help="Maskede min. tümör piksel eşiği")
    return parser.parse_args()


def find_case_dirs(raw_dir: Path) -> List[Path]:
    return sorted([p for p in raw_dir.iterdir() if p.is_dir()])


def load_nifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = img.get_fdata()
    return data.astype(np.float32)


def normalize_image(volume: np.ndarray) -> np.ndarray:
    v = volume
    v = (v - np.mean(v)) / (np.std(v) + 1e-8)
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    return v


def volume_to_slices(
    image_vol: np.ndarray,
    mask_vol: np.ndarray,
    step: int,
    min_tumor_pixels: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Z ekseninde gezen 2D slice listesi döndürür."""
    assert image_vol.shape == mask_vol.shape
    h, w, d = image_vol.shape
    slices: List[Tuple[np.ndarray, np.ndarray]] = []
    for z in range(0, d, step):
        img_slice = image_vol[:, :, z]
        mask_slice = mask_vol[:, :, z]
        # tümörü olmayan slice'ları at
        if np.sum(mask_slice > 0) < min_tumor_pixels:
            continue
        slices.append((img_slice, mask_slice))
    return slices


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    train_img_dir = out_dir / "train" / "images"
    train_mask_dir = out_dir / "train" / "masks"
    val_img_dir = out_dir / "val" / "images"
    val_mask_dir = out_dir / "val" / "masks"

    for d in (train_img_dir, train_mask_dir, val_img_dir, val_mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    case_dirs = find_case_dirs(raw_dir)
    n_train = int(len(case_dirs) * args.train_ratio)

    global_idx = 0
    for i, case_dir in enumerate(tqdm(case_dirs, desc="Cases")):
        case_name = case_dir.name
        # BraTS isimlendirmesi: BraTS20_Training_XXX_modality.nii.gz
        img_path = case_dir / f"{case_name}_{args.modality}.nii.gz"
        seg_path = case_dir / f"{case_name}_seg.nii.gz"
        if not img_path.exists() or not seg_path.exists():
            # farklı isimlendirme varsa atla
            continue

        image_vol = load_nifti(img_path)
        mask_vol = load_nifti(seg_path)

        # BraTS mask: 0,1,2,4 sınıfları içerir; biz tümör var/yok için binary yapıyoruz
        mask_vol = (mask_vol > 0).astype(np.float32)

        image_vol = normalize_image(image_vol)

        slices = volume_to_slices(
            image_vol=image_vol,
            mask_vol=mask_vol,
            step=args.step,
            min_tumor_pixels=args.min_tumor_pixels,
        )

        is_train = i < n_train
        img_out_dir = train_img_dir if is_train else val_img_dir
        mask_out_dir = train_mask_dir if is_train else val_mask_dir

        for img_slice, mask_slice in slices:
            img_uint8 = (img_slice * 255).astype(np.uint8)
            mask_uint8 = (mask_slice * 255).astype(np.uint8)

            img_name = f"{case_name}_slice{global_idx:05d}.png"
            mask_name = f"{case_name}_slice{global_idx:05d}.png"

            imageio.imwrite(img_out_dir / img_name, img_uint8)
            imageio.imwrite(mask_out_dir / mask_name, mask_uint8)
            global_idx += 1

    print(f"Toplam slice sayısı: {global_idx}")
    print(f"Train images: {len(list(train_img_dir.glob('*.png')))}")
    print(f"Val images:   {len(list(val_img_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()


