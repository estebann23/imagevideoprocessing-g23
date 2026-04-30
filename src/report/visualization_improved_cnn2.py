import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from imports import X_train, y_train

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_DIR   = Path(__file__).resolve().parent.parent / "CNN" / "training_values"
MODEL_DIR = Path(__file__).resolve().parent.parent / "CNN" / "output_saved_models"

hist_df = pd.read_csv(CSV_DIR / "training_history_cnn_v3.csv")

val_split = int(len(X_train) * 0.8)
X_val = X_train[val_split:]
y_val = y_train[val_split:]

print("Loading best CNN v2 model")
model = tf.keras.models.load_model(MODEL_DIR / "best_cnn_v3.h5")

print("Running predictions on validation set")
pred_probs = model.predict(X_val, verbose=0)
pred_labels = np.argmax(pred_probs, axis=-1)

#Accuracy Plot
print("Generating accuracy plot")
plt.figure(figsize=(10, 6))
plt.plot(hist_df['epoch'], hist_df['accuracy'],     label='train_acc', color='steelblue', linewidth=2)
plt.plot(hist_df['epoch'], hist_df['val_accuracy'], label='val_acc',   color='orange',    linewidth=2)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('CNN v2 — Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_plot_cnn_v2.png", dpi=150)
plt.close()
print("  Saved: accuracy_plot_cnn_v2.png")


# Loss plot
print("Generating loss plot")
plt.figure(figsize=(10, 6))
plt.plot(hist_df['epoch'], hist_df['loss'],     label='train_loss', color='steelblue', linewidth=2)
plt.plot(hist_df['epoch'], hist_df['val_loss'], label='val_loss',   color='orange',    linewidth=2)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('CNN v2 — Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_plot_cnn_v2.png", dpi=150)
plt.close()
print("  Saved: loss_plot_cnn_v2.png")


# confusion matrix
print("Generating confusion matrix")
cm = confusion_matrix(y_val, pred_labels)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix — CNN v2 (validation)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix_cnn_v2.png", dpi=150)
plt.close()
print("  Saved: confusion_matrix_cnn_v2.png")

# Save confusion matrix as CSV too
cm_df = pd.DataFrame(cm, index=range(10), columns=range(10))
cm_df.to_csv(OUTPUT_DIR / "confusion_matrix_cnn_v2.csv")


# classification
print("Generating classification report")
report = classification_report(y_val, pred_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(OUTPUT_DIR / "classification_report_cnn_v2.csv")
print(classification_report(y_val, pred_labels))


# missclassified images
print("Finding misclassified images")
wrong_idx = np.where(pred_labels != y_val)[0]
print(f"  Total misclassifications: {len(wrong_idx)} out of {len(y_val)}")

if len(wrong_idx) == 0:
    print("  Perfect score — no misclassifications to show!")
else:
    # Show all misclassified images (likely just 1-2 at 99.97%)
    n_wrong = len(wrong_idx)
    fig, axes = plt.subplots(1, n_wrong, figsize=(4 * n_wrong, 4))

    # handle case of only 1 misclassification
    if n_wrong == 1:
        axes = [axes]

    for i, idx in enumerate(wrong_idx):
        img = X_val[idx].squeeze()
        true_label  = y_val[idx]
        pred_label  = pred_labels[idx]
        confidence  = pred_probs[idx][pred_label] * 100

        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(
            f"True: {true_label}\nPredicted: {pred_label}\nConfidence: {confidence:.1f}%",
            fontsize=12, color='red'
        )
        axes[i].axis('off')

    fig.suptitle(f'CNN v2 — All Misclassified Images ({n_wrong} total)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "misclassified_cnn_v2.png", dpi=150)
    plt.close()
    print(f"  Saved: misclassified_cnn_v2.png")

    # Print details of each misclassification
    print("\n  Misclassification details:")
    print("  " + "="*45)
    for idx in wrong_idx:
        true_label = y_val[idx]
        pred_label = pred_labels[idx]
        confidence = pred_probs[idx][pred_label] * 100
        print(f"  Image #{idx}: True={true_label}, Predicted={pred_label}, Confidence={confidence:.1f}%")
    print("  " + "="*45)


# summary
best_val_acc = hist_df['val_accuracy'].max()
best_epoch   = hist_df['val_accuracy'].idxmax() + 1

summary = {
    "best_epoch":        int(best_epoch),
    "best_val_accuracy": float(best_val_acc),
    "total_val_samples": int(len(y_val)),
    "misclassified":     int(len(wrong_idx)),
    "correct":           int(len(y_val) - len(wrong_idx)),
}
with open(OUTPUT_DIR / "report_summary_cnn_v2.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*45}")
print(f"Best validation accuracy : {best_val_acc*100:.4f}%")
print(f"Best epoch               : {best_epoch}")
print(f"Misclassified            : {len(wrong_idx)} / {len(y_val)}")
print(f"{'='*45}")
print("All visualizations saved successfully.")