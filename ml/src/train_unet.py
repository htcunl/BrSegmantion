from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
import yaml


from .models.unet import build_unet
from .utils import data as data_utils
from .utils import exporter
from .utils import losses as loss_utils
from .utils import metrics as metric_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="U-Net Tümör Segmentasyonu Eğitimi")
    parser.add_argument("--config", type=str, required=True, help="YAML konfigürasyon dosyası")
    return parser.parse_args()


def setup_gpu():
    """GPU kullanımını yapılandırır ve kontrol eder."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[OK] {len(gpus)} GPU bulundu ve yapılandırildi:")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            return True
        except RuntimeError as e:
            print(f"[WARNING] GPU yapılandırma hatası: {e}")
            return False
    else:
        print("[WARNING] GPU bulunamadı, CPU kullanılacak.")
        return False


def set_seeds(seed: int):
    tf.random.set_seed(seed)
    np.random.seed(seed)


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_callbacks(run_dir: Path) -> List[tf.keras.callbacks.Callback]:
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    callbacks: List[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoints_dir / "best.weights.h5"),
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "training_log.csv")),
        tf.keras.callbacks.TensorBoard(log_dir=str(run_dir / "tensorboard")),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_dice_coefficient",
            mode="max",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coefficient",
            patience=10,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    return callbacks


def resolve_loss(loss_name: str):
    if loss_name.lower() == "bce_dice":
        return loss_utils.bce_dice_loss
    if loss_name.lower() == "dice":
        return loss_utils.dice_loss
    return tf.keras.losses.get(loss_name)


def resolve_metrics(metric_names: List[str]):
    resolved = []
    for name in metric_names:
        lname = name.lower()
        if lname == "dice":
            resolved.append(metric_utils.dice_coefficient)
        elif lname in ("iou", "jaccard"):
            resolved.append(metric_utils.iou_score)
        else:
            resolved.append(tf.keras.metrics.get(name))
    return resolved


def main():
    # GPU kullanımını yapılandır
    has_gpu = setup_gpu()

    # Mixed precision'ı GPU varsa etkinleştir (performans için)

    
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    seed = config.get("seed", 42)
    set_seeds(seed)

    paths_cfg = config["paths"]
    training_cfg = config["training"]
    augmentation_cfg = config.get("augmentation", {})
    export_cfg = config.get("export", {})

    artifacts_dir = Path(paths_cfg.get("artifacts_dir", "artifacts"))
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = artifacts_dir / f"{config.get('experiment_name', 'experiment')}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, run_dir / "config.yaml")

    train_ds = data_utils.create_dataset(
        image_dir=paths_cfg["train_images"],
        mask_dir=paths_cfg["train_masks"],
        img_size=training_cfg["img_size"],
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        augment=True,
        augmentation_config=augmentation_cfg,
    )

    val_ds = None
    if Path(paths_cfg["val_images"]).exists() and Path(paths_cfg["val_masks"]).exists():
        val_ds = data_utils.create_dataset(
            image_dir=paths_cfg["val_images"],
            mask_dir=paths_cfg["val_masks"],
            img_size=training_cfg["img_size"],
            batch_size=training_cfg["batch_size"],
            shuffle=False,
            augment=False,
        )

    input_shape = (training_cfg["img_size"], training_cfg["img_size"], training_cfg.get("channels", 1))
    model = build_unet(
        input_shape=input_shape,
        num_classes=training_cfg.get("num_classes", 1),
        base_filters=training_cfg.get("base_filters", 32),
        dropout=training_cfg.get("dropout", 0.1),
    )

    optimizer_name = training_cfg.get("optimizer", "adam").lower()
    learning_rate = training_cfg.get("learning_rate", 3e-4)

    optimizer = tf.keras.optimizers.get({"class_name": optimizer_name, "config": {"learning_rate": learning_rate}})
    loss_fn = resolve_loss(training_cfg.get("loss", "bce_dice"))
    metrics = resolve_metrics(training_cfg.get("metrics", ["dice"]))

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    callbacks = prepare_callbacks(run_dir)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=training_cfg["epochs"],
        callbacks=callbacks,
    )

    # History'yi JSON'a çevirirken numpy/tensorflow tiplerini Python native tiplerine çevir
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(v) if hasattr(v, 'item') else v for v in values]
    (run_dir / "history.json").write_text(json.dumps(history_dict, indent=2), encoding="utf-8")
    saved_model_path = run_dir / "model.keras"
    model.save(saved_model_path)

    if export_cfg.get("tflite", False):
        tflite_path = run_dir / "model.tflite"
        exporter.export_tflite(model, tflite_path)

    if export_cfg.get("onnx", False):
        onnx_path = run_dir / "model.onnx"
        try:
            exporter.export_onnx(model, onnx_path)
        except RuntimeError as err:
            print(f"ONNX dönüşümü atlandı: {err}")

    print(f"Eğitim tamamlandı. Çıktılar: {run_dir}")


if __name__ == "__main__":
    main()

