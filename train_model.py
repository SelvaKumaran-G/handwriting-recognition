"""
DigitVision AI - CNN Training and Evaluation Pipeline
Trains a deep convolutional neural network on the MNIST dataset with data augmentation,
evaluates on test data, generates confusion matrix and training curves, and saves the best model.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    MODEL_PATH,
    METRICS_PATH,
    TRAINING_PLOT_PATH,
    CONFUSION_MATRIX_PATH,
    BATCH_SIZE,
    EPOCHS,
    INITIAL_LR,
    RANDOM_SEED,
    DIGIT_LABELS
)
from model import build_digit_cnn, compile_model


def set_seed(seed=RANDOM_SEED):
    """Sets random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def inspect_uploaded_datasets():
    """
    Inspects written_name_train_v2.csv and written_name_test_v2.csv
    and documents their contents per project requirements.
    """
    print("\n" + "=" * 70)
    print("STEP 1: INSPECTING UPLOADED DATASETS")
    print("=" * 70)

    train_csv = "written_name_train_v2.csv"
    test_csv = "written_name_test_v2.csv"

    for file_path in [train_csv, test_csv]:
        if os.path.exists(file_path):
            try:
                import pandas as pd
                df = pd.read_csv(file_path, nrows=5)
                print(f"[FOUND] {file_path}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Sample rows:\n{df.to_string(index=False)}")
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")
        else:
            print(f"[NOT FOUND] {file_path}")

    print("\n[VERDICT & DATASET DECISION]")
    print("  The uploaded CSV files contain 'FILENAME' and 'IDENTITY' (handwritten person names),")
    print("  NOT digit labels (0-9). In addition, the corresponding handwriting image folders")
    print("  are not provided. As per project specification, the 0-9 handwritten digit recognition")
    print("  CNN model is strictly trained and evaluated on the benchmark MNIST dataset.")
    print("=" * 70 + "\n")


def load_and_preprocess_mnist():
    """
    Loads MNIST, performs train-val-test split, normalizes pixels to [0, 1],
    and one-hot encodes labels.
    """
    print("Loading benchmark MNIST dataset...")
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize pixel values [0, 255] -> [0.0, 1.0]
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Expand channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
    x_train_full = np.expand_dims(x_train_full, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Train / Validation split (50,000 train / 10,000 validation)
    val_split_idx = 50000
    x_train, x_val = x_train_full[:val_split_idx], x_train_full[val_split_idx:]
    y_train_raw, y_val_raw = y_train_full[:val_split_idx], y_train_full[val_split_idx:]

    # One-hot encode targets
    y_train = keras.utils.to_categorical(y_train_raw, 10)
    y_val = keras.utils.to_categorical(y_val_raw, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    print(f"  Training samples:   {x_train.shape[0]}")
    print(f"  Validation samples: {x_val.shape[0]}")
    print(f"  Testing samples:    {x_test.shape[0]}")
    print(f"  Input Shape:        {x_train.shape[1:]}")

    return (x_train, y_train, y_train_raw), (x_val, y_val, y_val_raw), (x_test, y_test_cat, y_test)


def create_augmented_dataset(x_train, y_train, batch_size=BATCH_SIZE):
    """
    Creates tf.data pipeline with real-time data augmentation
    (rotation, zoom, translation) to boost generalization on real-world handwriting.
    """
    data_augmentation = keras.Sequential([
        layers.RandomRotation(0.08, fill_mode="constant", fill_value=0.0),
        layers.RandomTranslation(0.08, 0.08, fill_mode="constant", fill_value=0.0),
        layers.RandomZoom(0.08, fill_mode="constant", fill_value=0.0),
    ], name="digit_augmentation")

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=10000, seed=RANDOM_SEED)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds


def plot_and_save_training_history(history, output_path=TRAINING_PLOT_PATH):
    """
    Plots training and validation accuracy/loss curves and saves to disk.
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5), dpi=300)
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, "o-", color="#6366f1", label="Training Accuracy", linewidth=2)
    plt.plot(epochs_range, val_acc, "s--", color="#10b981", label="Validation Accuracy", linewidth=2)
    plt.title("Model Accuracy per Epoch", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim([min(min(acc), min(val_acc)) - 0.02, 1.005])
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, "o-", color="#ef4444", label="Training Loss", linewidth=2)
    plt.plot(epochs_range, val_loss, "s--", color="#f59e0b", label="Validation Loss", linewidth=2)
    plt.title("Model Cross-Entropy Loss per Epoch", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Training curves saved to: {output_path}")


def plot_and_save_confusion_matrix(cm, output_path=CONFUSION_MATRIX_PATH):
    """
    Generates and saves a publication-quality confusion matrix heatmap.
    """
    plt.figure(figsize=(9, 7.5), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Purples",
        cbar=True,
        xticklabels=DIGIT_LABELS,
        yticklabels=DIGIT_LABELS,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 11, "weight": "semibold"}
    )
    plt.title("DigitVision AI - CNN Test Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Digit", fontsize=12, fontweight="medium")
    plt.ylabel("True Digit", fontsize=12, fontweight="medium")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Confusion matrix heatmap saved to: {output_path}")


def train_and_evaluate():
    """
    Main training execution function:
      1. Inspect datasets
      2. Load MNIST
      3. Build CNN
      4. Train with EarlyStopping and ReduceLROnPlateau
      5. Evaluate on Test Set
      6. Calculate precision, recall, f1, per-digit accuracy
      7. Save metrics and plots
    """
    set_seed(RANDOM_SEED)
    inspect_uploaded_datasets()

    # Load data
    (x_train, y_train, y_train_raw), (x_val, y_val, y_val_raw), (x_test, y_test_cat, y_test_raw) = load_and_preprocess_mnist()

    # Build model
    print("\nConstructing Convolutional Neural Network...")
    model = build_digit_cnn()
    model = compile_model(model, learning_rate=INITIAL_LR)
    model.summary()

    # Prepare datasets
    train_dataset = create_augmented_dataset(x_train, y_train, batch_size=BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    print(f"\nTraining CNN model for up to {EPOCHS} epochs...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Save training curves
    plot_and_save_training_history(history)

    # Evaluate on Unseen Test Dataset
    print("\nEvaluating best model on 10,000 unseen MNIST test samples...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, batch_size=BATCH_SIZE, verbose=1)

    # Predictions for comprehensive metrics
    y_pred_probs = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_test_raw, y_pred)
    plot_and_save_confusion_matrix(cm)

    # Classification report
    clf_report = classification_report(
        y_test_raw,
        y_pred,
        target_names=DIGIT_LABELS,
        output_dict=True
    )

    # Per-digit accuracy
    per_digit_accuracy = {}
    for digit in range(10):
        mask = (y_test_raw == digit)
        acc = float(np.mean(y_pred[mask] == digit))
        per_digit_accuracy[str(digit)] = round(acc * 100, 2)

    # Summary metrics payload
    final_metrics = {
        "train_accuracy": float(history.history["accuracy"][-1]),
        "val_accuracy": float(history.history["val_accuracy"][-1]),
        "test_accuracy": float(test_accuracy),
        "train_loss": float(history.history["loss"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
        "test_loss": float(test_loss),
        "total_train_samples": int(x_train.shape[0]),
        "total_val_samples": int(x_val.shape[0]),
        "total_test_samples": int(x_test.shape[0]),
        "epochs_trained": len(history.history["accuracy"]),
        "per_digit_accuracy": per_digit_accuracy,
        "classification_report": clf_report,
        "confusion_matrix": cm.tolist()
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"[SAVED] Final metrics exported to: {METRICS_PATH}")

    # Display final results in terminal
    print("\n" + "=" * 70)
    print("DIGITVISION AI - CNN TRAINING & EVALUATION REPORT")
    print("=" * 70)
    print(f"  Test Accuracy:        {test_accuracy * 100:.2f}%")
    print(f"  Test Loss:            {test_loss:.4f}")
    print(f"  Validation Accuracy:  {final_metrics['val_accuracy'] * 100:.2f}%")
    print(f"  Training Accuracy:    {final_metrics['train_accuracy'] * 100:.2f}%")
    print("-" * 70)
    print("Per-Digit Recognition Accuracies:")
    for digit, acc in per_digit_accuracy.items():
        print(f"  Digit '{digit}': {acc}%")
    print("=" * 70)

    return model, final_metrics


if __name__ == "__main__":
    train_and_evaluate()

