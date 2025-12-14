from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import tensorflow as tf


def _collect_image_mask_pairs(image_dir: Path, mask_dir: Path, extensions: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Dosya isimlerine göre eşleşen image-mask çiftlerini bulur."""
    image_dict = {}
    mask_dict = {}

    for ext in extensions:
        for img_path in image_dir.rglob(f"*.{ext}"):
            image_dict[img_path.name] = str(img_path)
        for mask_path in mask_dir.rglob(f"*.{ext}"):
            mask_dict[mask_path.name] = str(mask_path)

    # Her iki klasörde de bulunan dosyaları bul
    common_names = sorted(set(image_dict.keys()) & set(mask_dict.keys()))

    if len(common_names) == 0:
        raise ValueError(
            f"Eşleşen dosya bulunamadı. "
            f"Images: {len(image_dict)}, Masks: {len(mask_dict)}. "
            f"Dosya isimlerinin aynı olduğundan emin olun."
        )

    image_paths = [image_dict[name] for name in common_names]
    mask_paths = [mask_dict[name] for name in common_names]

    return image_paths, mask_paths


def _load_image(path: tf.Tensor, img_size: int, channels: int = 1, is_mask: bool = False) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(img_bytes, channels=channels, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=(img_size, img_size), method="bilinear")
    if is_mask:
        image = tf.where(image > 0.5, 1.0, 0.0)
    return image


def _augment(image: tf.Tensor, mask: tf.Tensor, config: dict) -> Tuple[tf.Tensor, tf.Tensor]:
    if config.get("random_flip", False):
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)

    if config.get("random_rotate", False):
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k)
        mask = tf.image.rot90(mask, k)

    if config.get("random_zoom", False):
        scales = list(config.get("zoom_scales", (0.9, 1.1)))
        scale = tf.random.uniform([], min(scales), max(scales))
        target_size = tf.cast(tf.shape(image)[0:2], tf.float32)
        new_size = tf.cast(target_size * scale, tf.int32)
        image = tf.image.resize(image, new_size)
        mask = tf.image.resize(mask, new_size, method="nearest")
        image = tf.image.resize_with_crop_or_pad(image, tf.shape(mask)[0], tf.shape(mask)[1])
        mask = tf.image.resize_with_crop_or_pad(mask, tf.shape(image)[0], tf.shape(image)[1])
    return image, mask


def create_dataset(
    image_dir: str,
    mask_dir: str,
    img_size: int,
    batch_size: int,
    shuffle: bool = True,
    augment: bool = False,
    augmentation_config: dict | None = None,
    extensions: Iterable[str] | None = None,
) -> tf.data.Dataset:
    """
    Dizinlerdeki görüntü/mask çiftlerinden tf.data pipeline oluşturur.
    """
    extensions = tuple(extensions or ("png", "jpg", "jpeg", "tif", "tiff"))
    print(f"Veri toplanıyor: image_dir='{image_dir}', mask_dir='{mask_dir}', extensions={extensions}")
    image_paths, mask_paths = _collect_image_mask_pairs(Path(image_dir), Path(mask_dir), extensions)

    # Eğer dizin boşsa daha açıklayıcı hata ver
    if len(image_paths) == 0:
        raise ValueError(
            f"Veri bulunamadı: image_dir='{image_dir}', mask_dir='{mask_dir}'.\n"
            "Lütfen config içindeki yolları ve veri dizinini kontrol edin.\n"
            "(Örnek doğru yol: 'ml/data/train/images' göre proje kökünden çalıştırıyorsanız)"
        )

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    def _load(image_path: tf.Tensor, mask_path: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = _load_image(image_path, img_size, channels=1, is_mask=False)
        mask = _load_image(mask_path, img_size, channels=1, is_mask=True)
        if augment:
            image, mask = _augment(image, mask, augmentation_config or {})
        return image, mask

    dataset = dataset.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

