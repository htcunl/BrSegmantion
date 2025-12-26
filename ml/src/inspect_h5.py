from __future__ import annotations


import argparse
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=".h5 dosya yapısını incele")
    parser.add_argument("--path", type=str, required=True, help=".h5 dosyasının tam yolu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5_path = Path(args.path)
    if not h5_path.exists():
        raise SystemExit(f"Dosya bulunamadı: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        print("Dosya:", h5_path)
        print("Keys:", list(f.keys()))
        for k in f.keys():
            d = f[k]
            try:
                print(f"{k}: shape={getattr(d, 'shape', None)}, dtype={getattr(d, 'dtype', None)}")
            except Exception as e:  # pragma: no cover - sadece debug için
                print(f"{k}: hata -> {e}")


if __name__ == "__main__":
    main()


