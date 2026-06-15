"""
AI-Driven Smart Urban Farming Resource Optimization
CNN-based Plant Disease & Water Stress Detection Model

Architecture: MobileNetV2 (Transfer Learning) fine-tuned on PlantVillage dataset
This script: Trains or loads the model, exposes a FastAPI inference server.

PlantVillage Dataset: 54,305 images across 38 plant disease categories
"""

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ─── Configuration ─────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 38  # PlantVillage has 38 classes (plant + disease combinations)
BASE_DIR = Path(__file__).resolve().parent
MODEL_SAVE_PATH = str(BASE_DIR / "models" / "plant_disease_model.h5")
CLASS_INDICES_PATH = str(BASE_DIR / "models" / "class_indices.json")
METRICS_PATH = str(BASE_DIR / "models" / "evaluation_metrics.json")
DATA_DIR = str(BASE_DIR / "data" / "PlantVillage")

# ─── PlantVillage Class Labels ──────────────────────────────────────────────
CLASS_LABELS = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca", "Grape___Leaf_blight", "Grape___healthy",
    "Orange___Haunglongbing", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper___Bacterial_spot", "Pepper___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

HEALTHY_CLASSES = {label for label in CLASS_LABELS if "healthy" in label}


def build_model(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    Build MobileNetV2-based transfer learning model for plant disease classification.
    MobileNetV2 is chosen for its efficiency and accuracy on mobile/edge devices.
    """
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    # Freeze base layers initially; unfreeze top layers for fine-tuning
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def get_data_generators(data_dir: str):
    """Create train/validation data generators with augmentation."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    return train_gen, val_gen


def train_model(data_dir: str = DATA_DIR):
    """Full training pipeline with callbacks."""
    os.makedirs(str(BASE_DIR / "models"), exist_ok=True)

    print("📦 Loading dataset...")
    train_gen, val_gen = get_data_generators(data_dir)

    print(f"🏗️  Building model (classes: {train_gen.num_classes})...")
    model = build_model(train_gen.num_classes)

    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    ]

    print("🚀 Starting training (Phase 1: Feature Extraction)...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Phase 2: Fine-tuning - unfreeze top layers of base model
    print("🔧 Fine-tuning top layers...")
    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=callbacks
    )

    with open(CLASS_INDICES_PATH, "w", encoding="utf-8") as handle:
        json.dump(train_gen.class_indices, handle, indent=2, sort_keys=True)

    metrics = evaluate_model(model, val_gen)
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"✅ Model saved to {MODEL_SAVE_PATH}")
    return model, history


def evaluate_model(model: tf.keras.Model, val_gen) -> dict:
    """Compute validation metrics and a confusion matrix from the validation generator."""
    y_true = val_gen.classes
    y_prob = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    num_classes = int(val_gen.num_classes)

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, predicted_label in zip(y_true, y_pred):
        confusion[int(true_label), int(predicted_label)] += 1

    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for class_index in range(num_classes):
        true_positive = int(confusion[class_index, class_index])
        false_positive = int(confusion[:, class_index].sum() - true_positive)
        false_negative = int(confusion[class_index, :].sum() - true_positive)

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(float(np.mean(precision_scores)), 4) if precision_scores else 0.0,
        "recall_macro": round(float(np.mean(recall_scores)), 4) if recall_scores else 0.0,
        "f1_macro": round(float(np.mean(f1_scores)), 4) if f1_scores else 0.0,
        "confusion_matrix": confusion.tolist(),
    }


def load_model_for_inference():
    """Load saved model or create a fresh one if not found."""
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"✅ Loading model from {MODEL_SAVE_PATH}")
        return tf.keras.models.load_model(MODEL_SAVE_PATH)
    else:
        print("⚠️  Saved model not found. Using untrained model (for demo only).")
        print("    To train: run `python model.py train`")
        return build_model()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        if not os.path.exists(DATA_DIR):
            print(f"❌ Dataset not found at {DATA_DIR}")
            print("   Download PlantVillage dataset and place it at: ml-model/data/PlantVillage/")
            print("   Dataset URL: https://www.kaggle.com/datasets/emmarex/plantdisease")
            sys.exit(1)
        train_model()
    else:
        print("Usage: python model.py train")
        print("       Then start the API server: python server.py")
