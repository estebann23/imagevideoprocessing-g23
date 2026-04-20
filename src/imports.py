import numpy as np
import pandas as pd
from load_kaggle import path

from PIL import Image
def load_image(path) -> np.ndarray:
    img = Image.open(path).convert("L").resize((28, 28))
    return np.array(img, dtype=np.float32).flatten() / 255.0


# ---------
# train data

train_df = pd.read_csv(path / "train.csv")
X_train_list, y_train_list = [], []

train_image_dir = path / "train"
stem_to_path = {
    p.stem: p
    for p in train_image_dir.rglob("*.png")
}
print(f"Train images found: {len(stem_to_path)}")

for _, row in train_df.iterrows():
    img_id  = str(row["Id"])
    label   = int(row["Category"])
    img_path = stem_to_path.get(img_id)

    if img_path is None:
        missing_train += 1
        continue

    X_train_list.append(load_image(img_path))
    y_train_list.append(label)

X_train = np.array(X_train_list, dtype=np.float32)
y_train = np.array(y_train_list, dtype=np.int32)
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")


# -----------
# test data

test_df = pd.read_csv(path / "test.csv")

test_image_dir = path / "test"
test_stem_to_path = {
    p.stem: p
    for p in test_image_dir.iterdir()
}
print(f"Test images found: {len(test_stem_to_path)}")

X_test_list, test_ids = [], []

for _, row in test_df.iterrows():
    img_id   = str(row["Id"])
    img_path = test_stem_to_path.get(img_id)

    X_test_list.append(load_image(img_path))
    test_ids.append(img_id)

X_test   = np.array(X_test_list, dtype=np.float32)
test_ids = np.array(test_ids)
print(f"X_test: {X_test.shape}, test_ids: {len(test_ids)}")