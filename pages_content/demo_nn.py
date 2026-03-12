import streamlit as st
import numpy as np
import os
from PIL import Image

def train_and_save():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping

    (X_train_raw, y_train_raw), _ = tf.keras.datasets.cifar10.load_data()

    ANIMAL_CLASSES = {2, 3, 4, 5, 6, 7}
    def relabel(y):
        return np.array([1 if int(l) in ANIMAL_CLASSES else 0 for l in y.flatten()])

    y_train = relabel(y_train_raw)
    X_train = X_train_raw.astype('float32') / 255.0

    data_aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        data_aug,
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=64,
              validation_split=0.2,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
              verbose=0)
    model.save('cifar10_cnn_model.h5')
    return model


def show():
    st.title("🧪 ทดสอบ Neural Network")
    st.markdown("### CIFAR-10 — Animal vs Non-Animal Classification")
    st.markdown("---")

    model_path = "cifar10_cnn_model.h5"

    if not os.path.exists(model_path):
        with st.spinner("⏳ กำลังเทรนโมเดล CNN อาจใช้เวลา 5-10 นาที..."):
            import tensorflow as tf
            model = train_and_save()
        st.success("✅ เทรนโมเดลเสร็จแล้ว!")
    else:
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
        uploaded_file = st.file_uploader("เลือกรูปภาพ", type=["jpg", "jpeg", "png"])

    with col2:
        st.markdown("#### 📊 ผลการทำนาย")
        if uploaded_file is not None:
            image       = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="รูปที่อัพโหลด", width=200)

            img_resized = image.resize((32, 32))
            img_array   = np.array(img_resized).astype("float32") / 255.0
            img_input   = np.expand_dims(img_array, axis=0)

            with st.spinner("กำลังวิเคราะห์..."):
                prediction = float(model.predict(img_input, verbose=0)[0][0])

            is_animal  = prediction >= 0.5
            confidence = prediction if is_animal else (1 - prediction)

            if is_animal:
                st.success("## 🐾 สัตว์ (Animal)")
            else:
                st.error("## 🚗 ไม่ใช่สัตว์ (Non-Animal)")

            st.metric("ความมั่นใจ", f"{confidence*100:.2f}%")
            st.progress(float(confidence))

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("🐾 Animal",     f"{prediction*100:.2f}%")
            with col_b:
                st.metric("🚗 Non-Animal", f"{(1-prediction)*100:.2f}%")
        else:
            st.info("👈 กรุณาอัพโหลดรูปภาพทางด้านซ้าย")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.success("**🐾 Animal (สัตว์)**\nbird • cat • deer • dog • frog • horse")
    with col2:
        st.error("**🚗 Non-Animal (ไม่ใช่สัตว์)**\nairplane • automobile • ship • truck")
