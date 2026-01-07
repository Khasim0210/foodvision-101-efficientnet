import tensorflow as tf
from src.data.preprocess import preprocess_image

def build_pipeline(ds: tf.data.Dataset, batch_size: int, training: bool):
    """
    Builds an optimized tf.data pipeline.

    Args:
      ds: raw dataset yielding (image, label)
      batch_size: number of samples per batch
      training: whether this pipeline is for training (enables shuffle)

    Returns:
      optimized tf.data.Dataset
    """
    if training:
        # Shuffle improves generalization. Buffer should be reasonably large.
        ds = ds.shuffle(buffer_size=10_000, reshuffle_each_iteration=True)

    # Apply preprocessing in parallel for speed
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch to utilize GPU efficiently
    ds = ds.batch(batch_size)

    # Prefetch overlaps CPU preprocessing with GPU training
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
