from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """İki adet Conv-BN-ReLU bloğu."""
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)

    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def encoder_block(x: tf.Tensor, filters: int, name: str, dropout: float = 0.0) -> Tuple[tf.Tensor, tf.Tensor]:
    c = conv_block(x, filters, name=f"{name}_conv")
    if dropout > 0.0:
        c = layers.Dropout(dropout, name=f"{name}_drop")(c)
    p = layers.MaxPool2D(pool_size=(2, 2), strides=2, name=f"{name}_pool")(c)
    return c, p


def decoder_block(x: tf.Tensor, skip: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same", name=f"{name}_up")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    x = conv_block(x, filters, name=f"{name}_conv")
    return x


def build_unet(
    input_shape: Tuple[int, int, int] = (256, 256, 1),
    num_classes: int = 1,
    base_filters: int = 32,
    dropout: float = 0.1,
) -> keras.Model:
    """
    Standart U-Net mimarisi döndürür.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    s1, p1 = encoder_block(inputs, base_filters, name="enc1", dropout=dropout)
    s2, p2 = encoder_block(p1, base_filters * 2, name="enc2", dropout=dropout)
    s3, p3 = encoder_block(p2, base_filters * 4, name="enc3", dropout=dropout)
    s4, p4 = encoder_block(p3, base_filters * 8, name="enc4", dropout=dropout)

    b = conv_block(p4, base_filters * 16, name="bottleneck")

    d1 = decoder_block(b, s4, base_filters * 8, name="dec1")
    d2 = decoder_block(d1, s3, base_filters * 4, name="dec2")
    d3 = decoder_block(d2, s2, base_filters * 2, name="dec3")
    d4 = decoder_block(d3, s1, base_filters, name="dec4")

    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Conv2D(num_classes, kernel_size=1, padding="same", activation=activation, name="output_mask")(d4)

    model = keras.Model(inputs, outputs, name="unet_brain_tumor")
    return model

