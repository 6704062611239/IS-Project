import streamlit as st
import numpy as np
import os
from PIL import Image

def show():
    st.title("🧪 ทดสอบ Neural Network")
    st.markdown("### CIFAR-10 — Animal vs Non-Animal Classification")
    st.markdown("---")

    # โหลดโมเดล
    model_path = "cifar10_cnn_model.h5"

    if not os.path.exists(model_path):
        st.error("❌ ไม่พบไฟล์โมเดล กรุณารัน `cifar10_neural_network.py` ก่อน")
        st.code("python cifar10_neural_network.py", language="bash")
        return

    # lazy import tensorflow
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)

    st.success("✅ โหลดโมเดลสำเร็จ!")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 🖼️ อัพโหลดรูปภาพ")
        st.markdown("""
        **ตัวอย่างรูปที่ทดสอบได้:**
        - 🐾 สัตว์: นก, แมว, กวาง, หมา, กบ, ม้า
        - 🚗 ไม่ใช่สัตว์: เครื่องบิน, รถยนต์, เรือ, รถบรรทุก
        """)

        uploaded_file = st.file_uploader(
            "เลือกรูปภาพ", type=["jpg", "jpeg", "png"],
            help="รองรับไฟล์ .jpg, .jpeg, .png"
        )

    with col2:
        st.markdown("#### 📊 ผลการทำนาย")

        if uploaded_file is not None:
            # แสดงรูปที่อัพโหลด
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="รูปที่อัพโหลด", width=200)

            # Preprocess
            img_resized = image.resize((32, 32))
            img_array   = np.array(img_resized).astype("float32") / 255.0
            img_input   = np.expand_dims(img_array, axis=0)

            # Predict
            with st.spinner("กำลังวิเคราะห์..."):
                prediction = model.predict(img_input, verbose=0)[0][0]

            is_animal   = prediction >= 0.5
            confidence  = prediction if is_animal else (1 - prediction)

            st.markdown("---")
            if is_animal:
                st.success(f"## 🐾 สัตว์ (Animal)")
            else:
                st.error(f"## 🚗 ไม่ใช่สัตว์ (Non-Animal)")

            st.metric("ความมั่นใจ", f"{confidence*100:.2f}%")
            st.progress(float(confidence))

            # แสดง probability ทั้งคู่
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🐾 Animal",     f"{prediction*100:.2f}%")
            with col_b:
                st.metric("🚗 Non-Animal", f"{(1-prediction)*100:.2f}%")

        else:
            st.info("👈 กรุณาอัพโหลดรูปภาพทางด้านซ้าย")

    st.markdown("---")

    # ── ตัวอย่าง class ที่จำแนกได้ ──
    st.markdown("### 📋 ประเภทที่โมเดลสามารถจำแนกได้")
    col1, col2 = st.columns(2)
    with col1:
        st.success("""
        **🐾 Animal (สัตว์)**
        bird • cat • deer • dog • frog • horse
        """)
    with col2:
        st.error("""
        **🚗 Non-Animal (ไม่ใช่สัตว์)**
        airplane • automobile • ship • truck
        """)

    st.info("💡 **หมายเหตุ:** โมเดลเทรนด้วยรูป 32×32 pixels ผลลัพธ์อาจแตกต่างกันสำหรับรูปที่มีหลาย object")
