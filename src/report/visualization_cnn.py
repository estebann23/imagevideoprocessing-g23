# generate_report.py
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# Paths
CNN_DIR = Path(__file__).resolve().parent.parent / "CNN"  # go up to src, then into CNN
best_model_path = CNN_DIR / "best_cnn.h5"
history_csv = CNN_DIR / "training_history.csv"

# Load data (imports.py)
from imports import X_train, y_train

# Recreate the same validation split as training script
val_split = int(len(X_train) * 0.8)
X_tr, X_val = X_train[:val_split], X_train[val_split:]
y_tr, y_val = y_train[:val_split], y_train[val_split:]

# Load model
model = load_model(best_model_path)
print(f"Loaded model: {best_model_path.name}")

# Predict on validation set
pred_probs = model.predict(X_val)
y_pred = np.argmax(pred_probs, axis=-1)

# Classification report
report = classification_report(y_val, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(CNN_DIR / "classification_report_val.csv")
print("Saved classification_report_val.csv")

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
cm_df = pd.DataFrame(cm)
cm_df.to_csv(CNN_DIR / "confusion_matrix_val.csv", index=False)
print("Saved confusion_matrix_val.csv")

# Plot confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion matrix (validation)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(CNN_DIR / "confusion_matrix_val.png", dpi=150)
plt.close()
print("Saved confusion_matrix_val.png")

# Plot training curves if history available
if history_csv.exists():
    hist_df = pd.read_csv(history_csv)
    epochs = hist_df['epoch']

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, hist_df['accuracy'], label='train_acc')
    plt.plot(epochs, hist_df['val_accuracy'], label='val_acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CNN_DIR / "accuracy_plot.png", dpi=150)
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(epochs, hist_df['loss'], label='train_loss')
    plt.plot(epochs, hist_df['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CNN_DIR / "loss_plot.png", dpi=150)
    plt.close()

    print("Saved accuracy_plot.png and loss_plot.png")

# Save a short JSON summary with key metrics
summary = {
    "val_accuracy_at_best": float(report_df.loc['accuracy', 'precision']) if 'accuracy' in report_df.index else None,
    "num_validation_samples": len(y_val),
}
with open(CNN_DIR / "report_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved report_summary.json")
