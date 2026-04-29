import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from imports import X_train, y_train, X_test, test_ids

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data Augmentation
print("Setting up data augmentation...")
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
#add validation set split here so we can evaluate on it later
val_split = int(len(X_train) * 0.8)
X_tr, X_val = X_train[:val_split], X_train[val_split:]
y_tr, y_val = y_train[:val_split], y_train[val_split:]

datagen.fit(X_tr)  # fit on training data only

# Model
print("Building improved CNN")
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Classifier
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# LR Logger Callback
lr_history = []
class LrLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        except Exception:
            lr = None
        lr_history.append(lr)

lr_logger = LrLogger()

# Callbacks
checkpoint_path = OUTPUT_DIR / "best_improved_cnn.h5"
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1),
    ModelCheckpoint(str(checkpoint_path), monitor='val_accuracy', save_best_only=True, verbose=1),
    lr_logger,
]

# Training
print("Training improved model")

history = model.fit(
    datagen.flow(X_tr, y_tr, batch_size=64),
    validation_data=(X_val, y_val),
    epochs=50,
    callbacks=callbacks,
)

# Save history and lr list
hist_df = pd.DataFrame(history.history)
hist_df['epoch'] = hist_df.index + 1
hist_df['lr'] = lr_history
history_csv = OUTPUT_DIR / "training_history_improved.csv"
history_json = OUTPUT_DIR / "training_history_improved.json"
hist_df.to_csv(history_csv, index=False)
with open(history_json, "w") as f:
    json.dump(history.history, f, indent=2)

# Summary file
best_epoch_idx = int(hist_df['val_accuracy'].idxmax())
summary = {
    "best_epoch": int(best_epoch_idx + 1),
    "best_val_accuracy": float(hist_df.loc[best_epoch_idx, 'val_accuracy']),
    "best_val_loss": float(hist_df.loc[best_epoch_idx, 'val_loss']),
    "epochs_run": len(history.epoch),
    "history_csv": str(history_csv.name),
    "best_model": str(checkpoint_path.name),
}
with open(OUTPUT_DIR / "training_summary_improved.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"Epochs run: {len(history.epoch)}")
print(f"Best epoch: {summary['best_epoch']}  best_val_accuracy: {summary['best_val_accuracy']:.4f}")

# Save final model also
final_model_path = OUTPUT_DIR / "final_improved_cnn.h5"
model.save(final_model_path)
print(f"Saved best model to: {checkpoint_path}")
print(f"Saved final model to: {final_model_path}")

# Prediction & Submission
print("Predicting test data...")
predictions_prob = model.predict(X_test)
predictions = np.argmax(predictions_prob, axis=-1)

submission = pd.DataFrame({
    "Id": test_ids,
    "Category": predictions,
})
submission_path = OUTPUT_DIR / "cnn_improved1_submission.csv"
submission.to_csv(submission_path, index=False)
print(f"{submission_path.name} saved successfully.")
