import tensorflow as tf
import tensorflow_datasets as tfds


def load_food101_dataset():
    """
    Loads the Food-101 dataset using TensorFlow Datasets (TFDS).

    Food-101 provides only:
    - train
    - validation
    """
    (train_ds, val_ds), info = tfds.load(
        "food101",
        split=["train", "validation"],
        as_supervised=True,
        with_info=True
    )

    return train_ds, val_ds, info


if __name__ == "__main__":
    train_ds, val_ds, info = load_food101_dataset()

    print("Number of classes:", info.features["label"].num_classes)
    print("First 5 class names:", info.features["label"].names[:5])
    print("Training samples:", tf.data.experimental.cardinality(train_ds))
    print("Validation samples:", tf.data.experimental.cardinality(val_ds))
