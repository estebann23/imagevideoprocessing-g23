import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from imports import X_train, y_train, X_test, test_ids

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
N_FOLDS    = 5      # number of folds
EPOCHS     = 50     # max epochs per fold, EarlyStopping cuts short
BATCH_SIZE = 64


# Model builder
# We need a function to build the model fresh for each fold
# because each fold trains a completely independent model
def build_model():
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


#  K-Fold Training
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# This will store the test predictions from each fold
# Shape: (N_FOLDS, n_test_images, 10) — 10 probabilities per image per fold
all_test_predictions = np.zeros((N_FOLDS, len(X_test), 10))
fold_val_accuracies  = []

print(f"Starting {N_FOLDS}-Fold Cross Validation...")
print("="*55)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\nFold {fold_idx + 1}/{N_FOLDS}")
    print(f"  Train size: {len(train_idx)} | Val size: {len(val_idx)}")

    # split data for this fold
    X_tr  = X_train[train_idx]
    y_tr  = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    # Augmentation (fit only on this fold's training data)
    datagen = ImageDataGenerator(
        rotation_range=15,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
    )
    datagen.fit(X_tr)

    # Build a fresh model for this fold
    model = build_model()

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                          patience=4, verbose=1),
    ]

    # Train
    history = model.fit(
        datagen.flow(X_tr, y_tr, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate this fold
    best_val_acc = max(history.history['val_accuracy'])
    fold_val_accuracies.append(best_val_acc)
    print(f"  Fold {fold_idx + 1} best val accuracy: {best_val_acc*100:.2f}%")

    #Save this fold's test predictions
    all_test_predictions[fold_idx] = model.predict(X_test)

    #  Save this fold's model
    model.save(OUTPUT_DIR / f"kfold_model_fold{fold_idx + 1}.h5")


# Final Results
print("\n" + "="*55)
print("K-Fold Results:")
for i, acc in enumerate(fold_val_accuracies):
    print(f"  Fold {i+1}: {acc*100:.2f}%")
print(f"  Mean accuracy : {np.mean(fold_val_accuracies)*100:.2f}%")
print(f"  Std deviation : {np.std(fold_val_accuracies)*100:.2f}%")
print("="*55)

# Save summary
summary = {
    "fold_accuracies": [float(a) for a in fold_val_accuracies],
    "mean_accuracy":   float(np.mean(fold_val_accuracies)),
    "std_accuracy":    float(np.std(fold_val_accuracies)),
}
with open(OUTPUT_DIR / "kfold_summary.json", "w") as f:
    json.dump(summary, f, indent=2)


# Ensemble: average predictions across all folds
# This is the key benefit of K-Fold — instead of one model's prediction,
# we average 5 models that each saw different subsets of the data
print("\nEnsembling predictions across all folds...")
ensemble_predictions_prob = np.mean(all_test_predictions, axis=0)  # average across folds
predictions = np.argmax(ensemble_predictions_prob, axis=-1)

submission = pd.DataFrame({
    "Id": test_ids,
    "Category": predictions,
})
submission_path = OUTPUT_DIR / "kfold_submission.csv"
submission.to_csv(submission_path, index=False)
print(f"kfold_submission.csv saved successfully.")
print(f"Expected accuracy: ~{np.mean(fold_val_accuracies)*100:.2f}%")