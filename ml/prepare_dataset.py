from __future__ import annotations

import argparse
import random
from pathlib import Path
import shutil
from typing import List, Tuple


def collect_pairs(image_dir: Path, mask_dir: Path, extensions: Tuple[str, ...]) -> List[Tuple[Path, Path]]:
    images = []
    for ext in extensions:
        images.extend(sorted(image_dir.rglob(f"*.{ext}")))

    pairs = []
    for img in images:
        mask = mask_dir / img.name
        if mask.exists():
            pairs.append((img, mask))
        else:
            # try matching by stem with common extensions
            found = None
            for ext in extensions:
                candidate = mask_dir / f"{img.stem}.{ext}"
                if candidate.exists():
                    found = candidate
                    break
            if found:
                pairs.append((img, found))
            else:
                print(f"Uyarı: Mask eşleşmedi: {img.name}")
    return pairs


def split_pairs(pairs: List[Tuple[Path, Path]], val_ratio: float, seed: int) -> Tuple[List, List]:
    random.seed(seed)
    indices = list(range(len(pairs)))
    random.shuffle(indices)
    split_at = int(len(indices) * (1 - val_ratio))
    train_idx = indices[:split_at]
    val_idx = indices[split_at:]
    train = [pairs[i] for i in train_idx]
    val = [pairs[i] for i in val_idx]
    return train, val


def ensure_dirs(base: Path):
    for sub in ("train/images", "train/masks", "val/images", "val/masks"):
        p = base / sub
        p.mkdir(parents=True, exist_ok=True)


def copy_pairs(pairs: List[Tuple[Path, Path]], out_img_dir: Path, out_mask_dir: Path):
    for img, mask in pairs:
        shutil.copy2(img, out_img_dir / img.name)
        shutil.copy2(mask, out_mask_dir / mask.name)


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for U-Net training (PNG images/masks)")
    parser.add_argument("--source", type=str, default=r"C:\\Users\\tahak\\Desktop\\BrSegmantion\\ml\\data", help="Path to folder that contains `images/` and `masks/`")
    parser.add_argument("--target", type=str, default="ml/data", help="Target dataset base directory inside project")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--ext", type=str, default="png", help="Image extension to look for (default: png)")
    args = parser.parse_args()

    src = Path(args.source).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"Kaynak bulunamadı: {src}")

    image_dir = src / "images"
    mask_dir = src / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise SystemExit("Kaynak dizinde `images/` veya `masks/` klasörü yok. Lütfen yapıyı kontrol edin.")

    target_base = Path(args.target)
    ensure_dirs(target_base)

    exts = tuple(e.strip().lower() for e in args.ext.split(",") if e.strip())
    pairs = collect_pairs(image_dir, mask_dir, exts)
    if len(pairs) == 0:
        raise SystemExit("Eşleşen görüntü-mask çifti bulunamadı. Uzantıları ve isimlendirmeyi kontrol edin.")

    train, val = split_pairs(pairs, args.val_ratio, args.seed)

    print(f"Toplam çiftler: {len(pairs)} | Train: {len(train)} | Val: {len(val)}")

    copy_pairs(train, target_base / "train" / "images", target_base / "train" / "masks")
    copy_pairs(val, target_base / "val" / "images", target_base / "val" / "masks")

    print(f"Kopyalama tamamlandı. Hedef: {target_base.resolve()}")
    print("Öneri: `python -m ml.src.train_unet --config ml/src/config.yaml` komutuyla eğitimi başlatın.")


if __name__ == "__main__":
    main()
