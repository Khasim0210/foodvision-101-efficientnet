import tensorflow as tf

def build_efficientnet_base(input_shape=(224, 224, 3), weights="imagenet"):
    """
    weights:
      - "imagenet" (default): uses pretrained weights (best, needs internet + SSL)
      - None: random initialization (for local testing only)
    """
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=input_shape
    )
    base_model.trainable = False
    return base_model
