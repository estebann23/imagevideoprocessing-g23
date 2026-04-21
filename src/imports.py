from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from pathlib import Path
from load_kaggle import path

DATA_PATH = Path(path)

from PIL import Image
def load_image(img_path: Path) -> np.ndarray:
    img = Image.open(img_path).convert("L").resize((28, 28))
    return np.array(img, dtype=np.float32).reshape(28, 28, 1) / 255.0

# ---------
# train data

train_df = pd.read_csv(DATA_PATH / "train.csv")
X_train_list, y_train_list = [], []

train_image_dir = DATA_PATH / "train"
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

test_image_dir = DATA_PATH /"test"/"test"

test_paths = [p for p in test_image_dir.iterdir() if p.is_file() and p.suffix == ".png"]
print(f"Test images found: {len(test_paths)}")
 
# Load in parallel — same approach as training
with ThreadPoolExecutor() as executor:
    test_images = list(executor.map(load_image, test_paths))
 
X_test   = np.array(test_images, dtype=np.float32)
test_ids = np.array([p.stem for p in test_paths])
print(f"X_test: {X_test.shape}, test_ids: {len(test_ids)}")