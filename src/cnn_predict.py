import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from imports import X_train, y_train, X_test, test_ids



print("Building Convolutional Neural Network")
model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)), 
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'), 
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

print("Training model...")
model.fit(X_train, y_train, epochs= 10, validation_split=0.2)

print("Predicting test data...")
predictions_prob = model.predict(X_test)
predictions = np.argmax(predictions_prob, axis=-1)

submission = pd.DataFrame({
    "Id": test_ids,
    "Category": predictions,
})
submission.to_csv("cnn_submission.csv", index=False)
print("cnn_submission.csv saved successfully.")