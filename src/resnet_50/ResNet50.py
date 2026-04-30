import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imports import X_train, y_train, X_test, test_ids


# Prepare images for ResNet50
TARGET_SIZE = 64

def prepare_for_resnet(X):
    X_rgb = np.repeat(X, 3, axis=-1)
    X_resized = tf.image.resize(X_rgb, [TARGET_SIZE, TARGET_SIZE]).numpy()
    return X_resized.astype(np.float32)

print("Preparing images for ResNet50...")
X_train_resnet = prepare_for_resnet(X_train)
X_test_resnet  = prepare_for_resnet(X_test)
print(f"X_train shape: {X_train_resnet.shape}")
print(f"X_test shape:  {X_test_resnet.shape}")


# Validation split
val_split = int(len(X_train_resnet) * 0.8)
X_tr, X_val = X_train_resnet[:val_split], X_train_resnet[val_split:]
y_tr, y_val = y_train[:val_split],        y_train[val_split:]


# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
datagen.fit(X_tr)


# Build model
print("Loading ResNet50 pretrained weights...")
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(TARGET_SIZE, TARGET_SIZE, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
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


# Phase 1: Train only the top layers
print("\nPhase 1: Training top layers only (base model frozen)...")
callbacks_phase1 = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, verbose=1)
]

history_phase1 = model.fit(
    datagen.flow(X_tr, y_tr, batch_size=64),
    validation_data=(X_val, y_val),
    epochs=50,
    callbacks=callbacks_phase1,
)
phase1_best = max(history_phase1.history['val_accuracy'])
print(f"\nPhase 1 best val accuracy: {phase1_best*100:.2f}%")


# Phase 2: Fine-tune — unfreeze top layers of ResNet
print("\nPhase 2: Fine-tuning top layers of ResNet50...")
base_model.trainable = True

# Only unfreeze the last 30 layers, keep earlier layers frozen
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),  # ← was 15, patience to stop
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1),              # ← was 15, give room before reducing lr
]

history_phase2 = model.fit(
    datagen.flow(X_tr, y_tr, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=50,          # ← increased from 20 to 50, EarlyStopping will cut it short
    callbacks=callbacks_phase2,
)
phase2_best = max(history_phase2.history['val_accuracy'])
print(f"\nPhase 2 best val accuracy: {phase2_best*100:.2f}%")


#  Final summary
print("\n" + "="*45)
print(f"{'Phase 1 (frozen base)':<30} {phase1_best*100:>8.2f}%")
print(f"{'Phase 2 (fine-tuned)':<30} {phase2_best*100:>8.2f}%")
print("="*45)


#Prediction & Submission
print("\nPredicting test data")
predictions_prob = model.predict(X_test_resnet)
predictions = np.argmax(predictions_prob, axis=-1)

submission = pd.DataFrame({
    "Id": test_ids,
    "Category": predictions,
})
submission.to_csv("resnet50_submission.csv", index=False)
print("resnet50_submission.csv saved successfully.")