import tensorflow as tf
import tensorflow_datasets as tfds
from src.data.pipeline import build_pipeline

val_raw = tfds.load("food101", split="validation", as_supervised=True)
val_ds = build_pipeline(val_raw, batch_size=32, training=False)

for x, y in val_ds.take(1):
    print("x:", x.shape, x.dtype, tf.reduce_min(x).numpy(), tf.reduce_max(x).numpy())
    print("y:", y.shape, y.dtype)
    print("first labels:", y[:10].numpy())
