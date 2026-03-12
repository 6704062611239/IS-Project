import streamlit as st

def show():
    st.title("📄 Neural Network")
    st.markdown("### CIFAR-10 — Animal vs Non-Animal Classification")
    st.markdown("---")

    # ── 1. Dataset ──
    st.header("1. 📦 Dataset")
    st.markdown("""
    **ที่มา:** [CIFAR-10 — Built-in Dataset (TensorFlow/Keras)](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10)

    CIFAR-10 มีรูปภาพทั้งหมด **60,000 รูป** ขนาด 32×32 pixels แบ่งเป็น 10 class
    เราทำการ **Relabel** เป็น 2 กลุ่มคือ **Animal** และ **Non-Animal**
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.success("🐾 **Animal (Label = 1)**")
        st.markdown("""
        - ✈️ bird (นก)
        - 🐱 cat (แมว)
        - 🦌 deer (กวาง)
        - 🐶 dog (หมา)
        - 🐸 frog (กบ)
        - 🐴 horse (ม้า)
        """)
    with col2:
        st.error("🚗 **Non-Animal (Label = 0)**")
        st.markdown("""
        - ✈️ airplane (เครื่องบิน)
        - 🚗 automobile (รถยนต์)
        - 🚢 ship (เรือ)
        - 🚛 truck (รถบรรทุก)
        """)

    st.markdown("---")

    # ── 2. ความไม่สมบูรณ์ ──
    st.header("2. ⚠️ ความไม่สมบูรณ์ของข้อมูล")
    st.markdown("""
    | ปัญหา | วิธีแก้ |
    |---|---|
    | รูปขนาดเล็กมาก (32×32) | Normalize pixel 0–255 → 0.0–1.0 |
    | ไม่มี label Animal/Non-Animal | Relabel เองจาก 10 class เดิม |
    | Class ไม่สมดุลหลัง Relabel (Animal 30,000 vs Non-Animal 20,000) | ใช้ Class Weight ชดเชย |
    | รูปมีความหลากหลายมาก (มุม, แสง) | Data Augmentation |
    """)

    st.markdown("---")

    # ── 3. Data Preparation ──
    st.header("3. 🔧 การเตรียมข้อมูล (Data Preparation)")
    st.markdown("""
    1. **โหลด CIFAR-10** ผ่าน `tf.keras.datasets.cifar10.load_data()`
    2. **Relabel** 10 class → 2 class (Animal / Non-Animal)
    3. **Normalize** pixel values: หารด้วย 255.0
    4. **Data Augmentation** เพื่อเพิ่มความหลากหลาย:
        - Random Horizontal Flip
        - Random Rotation (±10%)
        - Random Zoom (±10%)
        - Random Contrast (±10%)
    5. **แบ่ง Train/Validation/Test** (64% / 16% / 20%)
    """)

    st.markdown("---")

    # ── 4. ทฤษฎี CNN ──
    st.header("4. 📚 ทฤษฎี Convolutional Neural Network (CNN)")
    st.markdown("""
    CNN เป็นโครงข่ายประสาทเทียมที่ออกแบบมาสำหรับการประมวลผลรูปภาพโดยเฉพาะ
    ประกอบด้วย Layer หลักๆ ดังนี้:

    - **Conv2D** — ตรวจจับ pattern ในรูปภาพ เช่น ขอบ, เส้น, รูปทรง
    - **BatchNormalization** — ทำให้การเทรนเสถียรขึ้น ลด internal covariate shift
    - **MaxPooling2D** — ลดขนาด feature map รักษา feature ที่สำคัญ
    - **Dropout** — สุ่มปิด neuron เพื่อป้องกัน overfitting
    - **Dense** — Fully connected layer สำหรับการจำแนกประเภท
    - **Sigmoid** — Output layer สำหรับ Binary Classification (0–1)
    """)

    st.markdown("""
    **โครงสร้าง CNN ที่ใช้:**
    ```
    Input (32×32×3)
         ↓ Data Augmentation
         ↓ Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
         ↓ Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
         ↓ Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)
         ↓ Flatten
         ↓ Dense(256) → BatchNorm → Dropout(0.5)
         ↓ Dense(1, sigmoid)
      Animal / Non-Animal
    ```
    """)

    st.markdown("---")

    # ── 5. ขั้นตอนการพัฒนา ──
    st.header("5. 🛠️ ขั้นตอนการพัฒนาโมเดล")
    st.markdown("""
    1. โหลด CIFAR-10 Dataset
    2. Relabel เป็น Animal / Non-Animal
    3. Normalize และทำ Data Augmentation
    4. ออกแบบโครงสร้าง CNN (3 Convolutional Blocks)
    5. Compile โมเดลด้วย Adam optimizer, Binary Crossentropy loss
    6. เทรนโมเดล (50 epochs, batch size 64)
    7. ใช้ EarlyStopping และ ReduceLROnPlateau
    8. ประเมินผลด้วย Accuracy, Precision, Recall, F1
    9. บันทึกโมเดลเป็น .h5
    """)

    st.markdown("---")

    # ── 6. อ้างอิง ──
    st.header("6. 📖 แหล่งอ้างอิง")
    st.markdown("""
    - CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
    - TensorFlow/Keras Documentation: https://www.tensorflow.org/api_docs/python/tf/keras
    - LeCun, Y., et al. (1998). Gradient-Based Learning Applied to Document Recognition.
    - Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. Technical Report.
    - Ioffe, S., & Szegedy, C. (2015). Batch Normalization. ICML 2015.
    """)
