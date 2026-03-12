# ============================================================
#  CIFAR-10 — Neural Network (CNN)
#  Task    : Image Classification
#  Classes : Animal (bird, cat, deer, dog, frog, horse)
#            vs Non-Animal (airplane, automobile, ship, truck)
# ============================================================

# ── 1. ติดตั้ง library (รันครั้งแรกครั้งเดียว) ──────────────
# pip install tensorflow numpy matplotlib seaborn scikit-learn

# ── 2. Import ───────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")


# ── 3. โหลด CIFAR-10 Dataset ────────────────────────────────
print("\n" + "=" * 50)
print("📦 โหลด CIFAR-10 Dataset...")

(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = tf.keras.datasets.cifar10.load_data()

# ชื่อ class ทั้ง 10
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Train shape : {X_train_raw.shape}")
print(f"Test shape  : {X_test_raw.shape}")


# ── 4. Data Preparation ─────────────────────────────────────
print("\n" + "=" * 50)
print("🔧 เตรียมข้อมูล...")

# ── 4.1 ตรวจสอบความไม่สมบูรณ์ ──
# CIFAR-10 มีความไม่สมบูรณ์ดังนี้:
# 1) รูปขนาดเล็ก (32x32) → ต้อง normalize
# 2) จำนวนแต่ละ class เท่ากัน แต่หลัง relabel จะไม่สมดุล
# 3) ไม่มี label อันตราย/ไม่อันตราย → ต้อง relabel เอง
# 4) ต้อง augment เพราะรูปน้อยและเล็ก

# ── 4.2 Relabel เป็น Animal vs Non-Animal ──
# Animal      : bird=2, cat=3, deer=4, dog=5, frog=6, horse=7 → label 1
# Non-Animal  : airplane=0, automobile=1, ship=8, truck=9    → label 0
ANIMAL_CLASSES     = {2, 3, 4, 5, 6, 7}
NON_ANIMAL_CLASSES = {0, 1, 8, 9}

def relabel(y):
    return np.array([1 if int(label) in ANIMAL_CLASSES else 0 for label in y.flatten()])

y_train_binary = relabel(y_train_raw)
y_test_binary  = relabel(y_test_raw)

# ── 4.3 Normalize pixel 0-255 → 0.0-1.0 ──
X_train = X_train_raw.astype('float32') / 255.0
X_test  = X_test_raw.astype('float32')  / 255.0

# ── 4.4 ตรวจสอบ Class Distribution ──
animal_train     = np.sum(y_train_binary == 1)
non_animal_train = np.sum(y_train_binary == 0)
print(f"Train — Animal: {animal_train} | Non-Animal: {non_animal_train}")
print(f"Test  — Animal: {np.sum(y_test_binary==1)} | Non-Animal: {np.sum(y_test_binary==0)}")

# ── 4.5 Data Augmentation ──
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

print("✅ เตรียมข้อมูลเสร็จแล้ว!")


# ── 5. Visualize Dataset ─────────────────────────────────────
print("\n" + "=" * 50)
print("📊 Visualize Dataset...")

fig, axes = plt.subplots(3, 10, figsize=(18, 6))
fig.suptitle('CIFAR-10 — ตัวอย่างรูปภาพ (Animal=เขียว / Non-Animal=แดง)', fontsize=12)

for i in range(30):
    ax         = axes[i // 10][i % 10]
    ax.imshow(X_train_raw[i])
    label      = int(y_train_binary[i])
    class_name = CIFAR10_CLASSES[int(y_train_raw[i].flatten()[0])]
    color      = '#2ecc71' if label == 1 else '#e74c3c'
    ax.set_title(class_name, fontsize=7, color=color)
    ax.axis('off')

plt.tight_layout()
plt.savefig('cifar10_samples.png', dpi=150)
plt.show()
print("💾 บันทึกกราฟ → cifar10_samples.png")

# Class Distribution
plt.figure(figsize=(6, 4))
labels_name = ['Non-Animal', 'Animal']
counts      = [int(non_animal_train), int(animal_train)]
plt.bar(labels_name, counts, color=['#e74c3c', '#2ecc71'])
plt.title('Class Distribution หลัง Relabel')
plt.ylabel('จำนวนรูป')
for i, v in enumerate(counts):
    plt.text(i, v + 200, str(v), ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('cifar10_class_dist.png', dpi=150)
plt.show()
print("💾 บันทึกกราฟ → cifar10_class_dist.png")


# ── 6. สร้าง CNN Model ───────────────────────────────────────
print("\n" + "=" * 50)
print("🤖 สร้าง CNN Model...")

def build_cnn_model():
    model = models.Sequential([
        # Input + Augmentation
        layers.Input(shape=(32, 32, 3)),
        data_augmentation,

        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ], name="cifar10_cnn")
    return model

model = build_cnn_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ── 7. เทรนโมเดล ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("🏋️  เทรนโมเดล...")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                     patience=5, min_lr=1e-6, verbose=1)
]

history = model.fit(
    X_train, y_train_binary,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

print("✅ เทรนโมเดลเสร็จแล้ว!")


# ── 8. Plot Training History ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Training History — CIFAR-10 CNN', fontsize=13)

axes[0].plot(history.history['accuracy'],     label='Train Accuracy', color='#3498db')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',   color='#e74c3c')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train Loss', color='#3498db')
axes[1].plot(history.history['val_loss'], label='Val Loss',   color='#e74c3c')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cifar10_training_history.png', dpi=150)
plt.show()
print("💾 บันทึกกราฟ → cifar10_training_history.png")


# ── 9. ประเมินผล ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("📊 ผลการประเมิน:")

test_loss, test_acc = model.evaluate(X_test, y_test_binary, verbose=0)
print(f"\n✅ Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"   Test Loss    : {test_loss:.4f}")

y_pred_prob = model.predict(X_test, verbose=0)
y_pred      = (y_pred_prob > 0.5).astype(int).flatten()

print("\nClassification Report:")
print(classification_report(y_test_binary, y_pred,
      target_names=['Non-Animal', 'Animal']))

# Confusion Matrix
cm = confusion_matrix(y_test_binary, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Non-Animal', 'Animal'],
            yticklabels=['Non-Animal', 'Animal'])
plt.title('Confusion Matrix — CIFAR-10 CNN')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('cifar10_confusion_matrix.png', dpi=150)
plt.show()
print("💾 บันทึกกราฟ → cifar10_confusion_matrix.png")


# ── 10. ทดสอบด้วยรูปตัวอย่าง ─────────────────────────────────
print("\n" + "=" * 50)
print("🖼️  ทดสอบทำนายรูปตัวอย่าง:")

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle('ตัวอย่างการทำนาย (✅ ถูก / ❌ ผิด)', fontsize=13)

sample_idx = np.random.choice(len(X_test), 10, replace=False)
for i, idx in enumerate(sample_idx):
    ax       = axes[i // 5][i % 5]
    img      = X_test[idx]
    true_lbl = int(y_test_binary[idx])
    pred_lbl = int(y_pred[idx])
    prob     = float(y_pred_prob[idx].flatten()[0])
    correct  = "✅" if true_lbl == pred_lbl else "❌"

    ax.imshow(img)
    true_name = "Animal" if true_lbl == 1 else "Non-Animal"
    pred_name = "Animal" if pred_lbl == 1 else "Non-Animal"
    color     = '#2ecc71' if true_lbl == pred_lbl else '#e74c3c'
    ax.set_title(f"{correct} True: {true_name}\nPred: {pred_name} ({prob:.2f})",
                 fontsize=7, color=color)
    ax.axis('off')

plt.tight_layout()
plt.savefig('cifar10_predictions.png', dpi=150)
plt.show()
print("💾 บันทึกกราฟ → cifar10_predictions.png")


# ── 11. บันทึกโมเดล ──────────────────────────────────────────
print("\n" + "=" * 50)
print("💾 บันทึกโมเดล...")

model.save('cifar10_cnn_model.h5')
print("✅ บันทึกเสร็จแล้ว!")
print("   → cifar10_cnn_model.h5")
print("\n🎉 เสร็จสิ้นทุกขั้นตอน!")
