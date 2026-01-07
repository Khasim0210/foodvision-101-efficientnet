import tensorflow_datasets as tfds
from src.data.pipeline import build_pipeline

if __name__ == "__main__":
    train_ds = tfds.load("food101", split="train", as_supervised=True)
    val_ds = tfds.load("food101", split="validation", as_supervised=True)

    train_pipe = build_pipeline(train_ds, batch_size=32, training=True)
    val_pipe = build_pipeline(val_ds, batch_size=32, training=False)

    # Take 1 batch and confirm shapes
    for images, labels in train_pipe.take(1):
        print("Train batch images:", images.shape, images.dtype)
        print("Train batch labels:", labels.shape, labels.dtype)

    for images, labels in val_pipe.take(1):
        print("Val batch images:", images.shape, images.dtype)
        print("Val batch labels:", labels.shape, labels.dtype)
