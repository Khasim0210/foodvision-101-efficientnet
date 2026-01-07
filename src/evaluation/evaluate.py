# src/evaluation/evaluate.py

import tensorflow as tf
import tensorflow_datasets as tfds

from src.data.pipeline import build_pipeline

BATCH_SIZE = 32
MODEL_PATH = "models/stage2_best.keras"


def debug_forward_pass(model: tf.keras.Model, val_ds: tf.data.Dataset) -> None:
    """Run one batch through the model and print sanity checks."""
    print("\n===== DEBUG CHECK =====")
    print("Model output shape:", model.output_shape)

    for x_batch, y_batch in val_ds.take(1):
        print("Input batch shape:", x_batch.shape)
        print("Label batch shape:", y_batch.shape)
        print("Sample labels:", y_batch[:10].numpy())

        preds = model(x_batch, training=False)
        print("Predictions shape:", preds.shape)
        print("Pred min:", float(tf.reduce_min(preds)))
        print("Pred max:", float(tf.reduce_max(preds)))
        print("Sum of first prediction row:", float(tf.reduce_sum(preds[0])))

    print("\nINTERPRETATION:")
    print("- If sum ≈ 1.0 → model outputs SOFTMAX probabilities")
    print("- If sum ≠ 1.0 → model outputs LOGITS")


def main():
    print("\n===== STEP 11: MODEL EVALUATION & DEBUG =====")

    # 1) Load validation dataset
    print("\nLoading Food-101 validation dataset...")
    val_raw = tfds.load("food101", split="validation", as_supervised=True)
    val_ds = build_pipeline(val_raw, batch_size=BATCH_SIZE, training=False)
    print("Validation dataset loaded.")

    # 2) Load model without compile state
    print(f"\nLoading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")

    # 3) Compile explicitly with metrics
    print("\nCompiling model with explicit metrics...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
        ],
        run_eagerly=False,
    )
    print("Model compiled.")

    # 4) Warmup to build internal metric state (optional but safe)
    _ = model.evaluate(val_ds.take(1), verbose=0)

    # 5) Debug one forward pass
    debug_forward_pass(model, val_ds)

    # 6) Full evaluation (use return_dict=True so keys are stable)
    print("\n===== FULL VALIDATION EVALUATION =====")
    results = model.evaluate(val_ds, verbose=1, return_dict=True)

    # 7) Print results cleanly
    print("\n===== FINAL EVALUATION RESULTS =====")
    for k, v in results.items():
        try:
            print(f"{k}: {float(v):.4f}")
        except Exception:
            print(f"{k}: {v}")

    print("\n===== EVALUATION COMPLETE =====\n")


if __name__ == "__main__":
    main()
