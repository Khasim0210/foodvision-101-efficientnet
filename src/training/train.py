import tensorflow as tf
import tensorflow_datasets as tfds

from src.data.pipeline import build_pipeline
from src.models.full_model import build_foodvision_model


BATCH_SIZE = 32
EPOCHS_STAGE_1 = 5   # warm-up (head training)
EPOCHS_STAGE_2 = 5   # fine-tuning
DRY_RUN = False       # True for local sanity check, set False in Colab
DRY_RUN_STEPS = 1    # 1 = minimal sanity check


def get_datasets():
    """
    What we are doing:
    - Load Food-101 from TFDS (train + validation)
    - Build optimized tf.data pipelines (shuffle/batch/prefetch)

    Why:
    - Keeps data loading consistent and reusable
    - Ensures the pipeline is GPU-friendly later in Colab
    """
    train_raw = tfds.load("food101", split="train", as_supervised=True)
    val_raw = tfds.load("food101", split="validation", as_supervised=True)

    train_ds = build_pipeline(train_raw, batch_size=BATCH_SIZE, training=True)
    val_ds = build_pipeline(val_raw, batch_size=BATCH_SIZE, training=False)

    return train_ds, val_ds


def _fit_kwargs():
    """
    For DRY_RUN:
    - We only run a single step to confirm everything works without heavy training.
    """
    if DRY_RUN:
        return {"steps_per_epoch": DRY_RUN_STEPS, "validation_steps": DRY_RUN_STEPS}
    return {}


def stage1_train_head(model, train_ds, val_ds):
    """
    Stage 1:
    - Train only the classification head (backbone frozen)
    Why:
    - Stabilizes learning and prevents destroying pretrained features
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            "accuracy",
            # IMPORTANT: sparse labels => use SparseTopKCategoricalAccuracy
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=2, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/stage1_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE_1,
        callbacks=callbacks,
        **_fit_kwargs(),
    )
    return history


def stage2_finetune(model, train_ds, val_ds, unfreeze_layers=20):
    """
    Stage 2:
    - Unfreeze last 'unfreeze_layers' of the backbone
    - Fine-tune with very small LR

    Why:
    - Lets model adapt to Food-101 textures/patterns without over-updating weights
    """
    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True

    # Freeze all but last N layers
    for layer in backbone.layers[:-unfreeze_layers]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=2, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/stage2_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE_2,
        callbacks=callbacks,
        **_fit_kwargs(),
    )
    return history


if __name__ == "__main__":
    # Local note:
    # If your Mac SSL blocks ImageNet weights download, your model builder may be using weights=None.
    # In Colab, switch back to weights="imagenet" for real transfer learning.
    model = build_foodvision_model()

    train_ds, val_ds = get_datasets()

    print("Stage 1: Training classification head...")
    stage1_train_head(model, train_ds, val_ds)

    print("Stage 2: Fine-tuning last layers...")
    stage2_finetune(model, train_ds, val_ds, unfreeze_layers=20)

    print("Training complete.")
