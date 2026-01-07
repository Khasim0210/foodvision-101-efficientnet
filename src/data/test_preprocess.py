import tensorflow_datasets as tfds
from src.data.preprocess import preprocess_image

if __name__ == "__main__":
    (train_ds,), info = tfds.load(
        "food101",
        split=["train"],
        as_supervised=True,
        with_info=True
    )

    for image, label in train_ds.take(1):
        x, y = preprocess_image(image, label)

        print("Original dtype:", image.dtype, "Original shape:", image.shape)
        print("Processed dtype:", x.dtype, "Processed shape:", x.shape)
        print("Label id:", int(y.numpy()), "Class name:", info.features["label"].names[int(y.numpy())])
