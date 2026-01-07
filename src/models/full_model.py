import tensorflow as tf
from tensorflow.keras import layers, models
from src.models.base_model import build_efficientnet_base


def build_foodvision_model(
    input_shape=(224, 224, 3),
    num_classes=101,
    dropout_rate=0.3
):
    """
    Full model:
    - EfficientNetB0 backbone (frozen)
    - GlobalAveragePooling
    - Dropout
    - Dense softmax head for Food-101
    """
    inputs = layers.Input(shape=input_shape)

    base_model = build_efficientnet_base(input_shape=(224, 224, 3), weights="imagenet")

    x = base_model(inputs, training=False)  # keep BN layers in inference mode while frozen

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="FoodVision101_EfficientNetB0")
    return model
