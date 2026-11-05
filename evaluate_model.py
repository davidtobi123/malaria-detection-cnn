# ============================================================
# MALARIA DETECTION - EVALUATE THE SAVED MODEL
# Reproduces the paper's results with malaria_cnn_model.h5:
#   1) the original 2,000-image test split used in script.py
#   2) the 17,558 images that script.py never loaded
# Run from the project folder:  python evaluate_model.py
# ============================================================
import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

DATASET_PATH = "cell_images"
IMG_SIZE = 64
MAX_IMAGES_PER_CLASS = 5000          # same limit as script.py

def list_pngs(folder):
    # Windows lists files in alphabetical order (ignoring case); sort the same way
    # so the result is identical on every computer.
    return sorted([f for f in os.listdir(folder) if f.endswith('.png')], key=str.upper)

def load(folder, names):
    images = []
    for name in names:
        img = cv2.imread(os.path.join(folder, name))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.array(images, dtype=np.float32) / 255.0

def report(title, y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print("\n" + "=" * 55)
    print(f"  {title}  (n = {len(y_true):,})")
    print("=" * 55)
    print(f"  TN = {tn:,}   FP = {fp:,}   FN = {fn:,}   TP = {tp:,}")
    print(f"  Accuracy:    {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"  Precision:   {precision_score(y_true, y_pred)*100:.2f}%")
    print(f"  Recall:      {recall_score(y_true, y_pred)*100:.2f}%")
    print(f"  F1-Score:    {f1_score(y_true, y_pred)*100:.2f}%")
    print(f"  Specificity: {tn/(tn+fp)*100:.2f}%")
    print(f"  AUC-ROC:     {roc_auc_score(y_true, y_prob):.4f}")

par_dir = os.path.join(DATASET_PATH, 'Parasitized')
uni_dir = os.path.join(DATASET_PATH, 'Uninfected')
par_names, uni_names = list_pngs(par_dir), list_pngs(uni_dir)
print(f"Parasitized images: {len(par_names):,}   Uninfected images: {len(uni_names):,}")

model = tf.keras.models.load_model('malaria_cnn_model.h5', compile=False)

# 1) Original test split (same steps and seed as script.py)
X = np.concatenate([load(par_dir, par_names[:MAX_IMAGES_PER_CLASS]),
                    load(uni_dir, uni_names[:MAX_IMAGES_PER_CLASS])])
y = np.array([1] * MAX_IMAGES_PER_CLASS + [0] * MAX_IMAGES_PER_CLASS)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
del X
report("ORIGINAL TEST SPLIT", y_test, model.predict(X_test, verbose=0, batch_size=256).ravel())

# 2) Images never loaded by script.py
X_new = np.concatenate([load(par_dir, par_names[MAX_IMAGES_PER_CLASS:]),
                        load(uni_dir, uni_names[MAX_IMAGES_PER_CLASS:])])
y_new = np.array([1] * len(par_names[MAX_IMAGES_PER_CLASS:]) + [0] * len(uni_names[MAX_IMAGES_PER_CLASS:]))
report("UNUSED IMAGES", y_new, model.predict(X_new, verbose=0, batch_size=256).ravel())
