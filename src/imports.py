import kagglehub
import os
import numpy as np
import pandas as pd

KAGGLE_API_TOKEN="KGAT_6716038ad78e3e6d9f96899d00f51e87"

username="estebanna"
key="156af646e15dabadbf85a3eb37f35c36"

os.environ["KAGGLE_USERNAME"] = username
os.environ["KAGGLE_KEY"] = key


path = kagglehub.competition_download('iivp-2026-challenge')

print("Path to competition files:", path)

def load_from_image_folder(path):
    from PIL import Image
 
    training_data = pd.read_csv(path / "train.csv")
    id_col, label_col = "Id", "Category"
 
    def read_images(df, folder, has_label=True):
        imgs, ids, labels = [], [], []
        for _, row in df.iterrows():
            img_path = path / folder / f"{row[id_col]}.png"
            img = Image.open(img_path).convert("L").resize((28, 28))
            imgs.append(np.array(img).flatten())
            ids.append(row[id_col])
            if has_label:
                labels.append(row[label_col])
        return np.array(imgs), ids, np.array(labels) if has_label else None
    test_df = pd.read_csv(path / "test.csv")
    X_train, train_ids, y_train = read_images(training_data, "train")
    X_test,  test_ids,  _       = read_images(test_df,      "test", has_label=False)
 
    return X_train, y_train, X_test, np.array(test_ids)

X_train, y_train, X_test, test_ids = load_from_image_folder(path)
print("Test data shape:", X_test.shape)

X_train = X_train.astype(np.float32) / 255.0
X_test  = X_test.astype(np.float32)  / 255.0