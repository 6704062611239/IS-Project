import streamlit as st
import numpy as np
import os
from PIL import Image
import joblib

def train_and_save():
    from sklearn.neural_network import MLPClassifier
    import tensorflow as tf

    (X_train_raw, y_train_raw), _ = tf.keras.datasets.cifar10.load_data()

    ANIMAL_CLASSES = {2, 3, 4, 5, 6, 7}
    y_train = np.array([1 if int(l) in ANIMAL_CLASSES else 0 for l in y_train_raw.flatten()])

    # Flatten และ normalize
    X_train = X_train_raw.reshape(len(X_train_raw), -1).astype('float32') / 255.0

    # ใช้ subset เพื่อความเร็ว
    X_train = X_train[:10000]
    y_train = y_train[:10000]

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        max_iter=50,
        random_state=42,
        verbose=False
    )
    model.fit(X_train, y_train)
    joblib.dump(model, 'cifar10_mlp_model.pkl')
    return model


def show():
    st.title("🧪 ทดสอบ Neural Network")
    st.markdown("### CIFAR-10 — Animal vs Non-Animal Classification")
    st.markdown("---")

    model_path = "cifar10_mlp_model.pkl"

    if not os.path.exists(model_path):
        with st.spinner("⏳ กำลังเทรนโมเดล MLP กรุณารอสักครู่..."):
            model = train_and_save()
        st.success("✅ เทรนโมเดลเสร็จแล้ว!")
    else:
        model = joblib.load(model_path)
        st.success("✅ โหลดโมเดลสำเร็จ!")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 🖼️ อัพโหลดรูปภาพ")
        st.markdown("""
        **ตัวอย่างรูปที่ทดสอบได้:**
        - 🐾 สัตว์: นก, แมว, กวาง, หมา, กบ, ม้า
        - 🚗 ไม่ใช่สัตว์: เครื่องบิน, รถยนต์, เรือ, รถบรรทุก
        """)
        uploaded_file = st.file_uploader("เลือกรูปภาพ", type=["jpg", "jpeg", "png"])

    with col2:
        st.markdown("#### 📊 ผลการทำนาย")
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="รูปที่อัพโหลด", width=200)

            img_resized = image.resize((32, 32))
            img_array  = np.array(img_resized).astype("float32") / 255.0
            img_flat   = img_array.flatten().reshape(1, -1)

            with st.spinner("กำลังวิเคราะห์..."):
                prediction = model.predict(img_flat)[0]
                proba      = model.predict_proba(img_flat)[0]

            is_animal  = prediction == 1
            confidence = float(proba[1]) if is_animal else float(proba[0])

            if is_animal:
                st.success("## 🐾 สัตว์ (Animal)")
            else:
                st.error("## 🚗 ไม่ใช่สัตว์ (Non-Animal)")

            st.metric("ความมั่นใจ", f"{confidence*100:.2f}%")
            st.progress(float(confidence))

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🐾 Animal",     f"{proba[1]*100:.2f}%")
            with col_b:
                st.metric("🚗 Non-Animal", f"{proba[0]*100:.2f}%")
        else:
            st.info("👈 กรุณาอัพโหลดรูปภาพทางด้านซ้าย")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.success("**🐾 Animal (สัตว์)**\nbird • cat • deer • dog • frog • horse")
    with col2:
        st.error("**🚗 Non-Animal (ไม่ใช่สัตว์)**\nairplane • automobile • ship • truck")
