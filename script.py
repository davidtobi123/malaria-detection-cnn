# ============================================================
# MALARIA DETECTION USING CNN - MEMORY EFFICIENT VERSION
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Conv2D, MaxPooling2D, 
                                     Flatten, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   MALARIA DETECTION SYSTEM - TRAINING SCRIPT")
print("=" * 60)
print(f"\nTensorFlow Version: {tf.__version__}")

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_PATH = "cell_images"
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20
MAX_IMAGES_PER_CLASS = 5000  # Limit images to save memory

# ============================================================
# STEP 1: LOAD DATASET (LIMITED FOR MEMORY)
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: LOADING DATASET")
print("=" * 60)

PARASITIZED_DIR = os.path.join(DATASET_PATH, 'Parasitized')
UNINFECTED_DIR = os.path.join(DATASET_PATH, 'Uninfected')

if not os.path.exists(PARASITIZED_DIR):
    print(f"ERROR: Cannot find {PARASITIZED_DIR}")
    exit()

def load_images(data_dir, label, max_images):
    images = []
    labels = []
    files = [f for f in os.listdir(data_dir) if f.endswith('.png')][:max_images]
    total = len(files)
    
    for idx, img_name in enumerate(files):
        try:
            img_path = os.path.join(data_dir, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(label)
            
            if (idx + 1) % 1000 == 0:
                print(f"   Loaded {idx + 1}/{total} images...")
        except:
            continue
    
    return images, labels

print(f"Loading Parasitized images (max {MAX_IMAGES_PER_CLASS})...")
parasitized_images, parasitized_labels = load_images(PARASITIZED_DIR, 1, MAX_IMAGES_PER_CLASS)

print(f"Loading Uninfected images (max {MAX_IMAGES_PER_CLASS})...")
uninfected_images, uninfected_labels = load_images(UNINFECTED_DIR, 0, MAX_IMAGES_PER_CLASS)

# Combine data
all_images = parasitized_images + uninfected_images
all_labels = parasitized_labels + uninfected_labels

print(f"\nTotal images loaded: {len(all_images)}")

# Convert to numpy and normalize
X = np.array(all_images, dtype=np.float32) / 255.0
y = np.array(all_labels)

# Free memory
del all_images, parasitized_images, uninfected_images
del all_labels, parasitized_labels, uninfected_labels

print(f"Image shape: {X[0].shape}")
print("Normalized pixel values to 0-1 range")

# ============================================================
# STEP 2: SPLIT DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: SPLITTING DATA")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} images (80%)")
print(f"Testing set: {len(X_test)} images (20%)")

# ============================================================
# STEP 3: BUILD CNN MODEL
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: BUILDING CNN MODEL")
print("=" * 60)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# STEP 4: TRAIN THE MODEL
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: TRAINING THE MODEL")
print("=" * 60)
print("This may take 5-15 minutes...\n")

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("\nTraining Complete!")

# ============================================================
# STEP 5: EVALUATE MODEL
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: EVALUATING MODEL")
print("=" * 60)

y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 50)
print("        PERFORMANCE METRICS")
print("=" * 50)
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1-Score:  {f1*100:.2f}%")
print("=" * 50)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Uninfected', 'Parasitized']))

# ============================================================
# STEP 6: SAVE VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: SAVING VISUALIZATIONS")
print("=" * 60)

if not os.path.exists('results'):
    os.makedirs('results')

# Training History
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['accuracy'], label='Training')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

axes[1].plot(history.history['loss'], label='Training')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plt.savefig('results/training_history.png')
plt.close()
print("Saved: results/training_history.png")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Uninfected', 'Parasitized'],
            yticklabels=['Uninfected', 'Parasitized'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('results/confusion_matrix.png')
plt.close()
print("Saved: results/confusion_matrix.png")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('results/roc_curve.png')
plt.close()
print("Saved: results/roc_curve.png")

# ============================================================
# STEP 7: SAVE MODEL
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: SAVING MODEL")
print("=" * 60)

model.save('malaria_cnn_model.h5')
print("Model saved as: malaria_cnn_model.h5")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("           PROJECT COMPLETE!")
print("=" * 60)
print(f"""
RESULTS:
  Accuracy:  {accuracy*100:.2f}%
  Precision: {precision*100:.2f}%
  Recall:    {recall*100:.2f}%
  F1-Score:  {f1*100:.2f}%
  AUC Score: {roc_auc:.4f}

FILES CREATED:
  - malaria_cnn_model.h5
  - results/training_history.png
  - results/confusion_matrix.png
  - results/roc_curve.png

NEXT: Run 'python app.py' for web interface
""")
print("=" * 60)