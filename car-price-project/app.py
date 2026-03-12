"""
Streamlit Web Application
Car Price Prediction — ML & Neural Network (Image Classification)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CAR_BRANDS = [
    "Toyota", "Honda", "BMW", "Ford", "Chevrolet",
    "Audi", "Mercedes", "Hyundai", "Nissan", "Volkswagen"
]

# ──────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────
# NEURAL NETWORK ARCHITECTURE (EfficientNet-B0)
# ──────────────────────────────────────────
class CarBrandClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        for param in self.backbone.parameters():
            param.requires_grad = False
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.backbone(x)


# ──────────────────────────────────────────
# LOAD MODELS (cached)
# ──────────────────────────────────────────
@st.cache_resource
def load_ml_model():
    with open("models/ml_ensemble_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/encoders_v3.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("models/scaler_v3.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, encoders, scaler


@st.cache_resource
def load_nn_model():
    ckpt  = torch.load("models/nn_car_brand.pt", map_location="cpu")
    model = CarBrandClassifier(num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["classes"]

def predict_brand(image: Image.Image, model: nn.Module):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(img_tensor)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


# ──────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────
st.sidebar.title("🚗 Car Price Prediction")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "เลือกหน้า",
    ["🏠 Home",
     "📊 ML Ensemble — อธิบายโมเดล",
     "🧠 Neural Network — อธิบายโมเดล",
     "🔮 ทดสอบ ML Model",
     "⚡ ทดสอบ Neural Network"],
)
st.sidebar.markdown("---")
st.sidebar.info("Dataset: Car Dekho (Kaggle)\nพัฒนาโดย: IS Project 2568")


# ══════════════════════════════════════════
# PAGE 0: HOME
# ══════════════════════════════════════════
if page == "🏠 Home":
    st.title("🚗 Car Price Prediction")
    st.markdown("### ระบบทำนายราคารถมือสอง ด้วย Machine Learning & Neural Network")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset V3", "8,128 แถว", "13 features")
    col2.metric("Dataset V4", "2,059 แถว", "20 features")
    col3.metric("ML Model", "Stacking Ensemble", "RF + GBR + XGB")
    col4.metric("NN Model", "MLP 4 Layers", "PyTorch")

    st.markdown("---")
    st.markdown("""
    ### 📌 เกี่ยวกับโปรเจค
    โปรเจคนี้พัฒนาโมเดลทำนายราคารถมือสองจาก Dataset ของ **Car Dekho** (อินเดีย)
    โดยใช้ 2 วิธีหลัก:

    | โมเดล | วิธีการ | Dataset |
    |-------|---------|---------|
    | ML Ensemble | Stacking (RF + GBR + XGB → Ridge) | Car details v3 |
    | Neural Network | MLP 4 Layers (PyTorch) | Car details v4 |

    ### 🗂️ การใช้งาน
    - เลือกหน้า **อธิบายโมเดล** เพื่อดูทฤษฎีและขั้นตอนการพัฒนา
    - เลือกหน้า **ทดสอบโมเดล** เพื่อกรอกข้อมูลรถและทำนายราคา
    """)


# ══════════════════════════════════════════
# PAGE 1: ML DESCRIPTION
# ══════════════════════════════════════════
elif page == "📊 ML Ensemble — อธิบายโมเดล":
    st.title("📊 ML Ensemble Model")
    st.markdown("## Stacking Ensemble: Random Forest + Gradient Boosting + XGBoost")
    st.markdown("---")

    with st.expander("1️⃣ การเตรียมข้อมูล (Data Preparation)", expanded=True):
        st.markdown("""
        **Dataset:** `Car details v3.csv` — 8,128 แถว, 13 คอลัมน์

        **ปัญหาที่พบและการแก้ไข:**
        | ปัญหา | วิธีแก้ไข |
        |-------|-----------|
        | `mileage` เป็น string เช่น "23.4 kmpl" | ใช้ Regex แยกตัวเลขออก |
        | `engine` เป็น string เช่น "1248 CC" | ใช้ Regex แยกตัวเลขออก |
        | `max_power` เป็น string เช่น "74 bhp" | ใช้ Regex แยกตัวเลขออก |
        | Missing values ใน mileage, engine, max_power, seats | Fill ด้วย Median |
        | Outlier ใน selling_price | ลบแถวที่เกิน Percentile 99 |
        | Feature `year` | แปลงเป็น `car_age = 2024 - year` |
        | Categorical features | Label Encoding |
        | Scale ไม่เท่ากัน | Standard Scaler |
        """)

    with st.expander("2️⃣ ทฤษฎีของอัลกอริทึม", expanded=True):
        st.markdown("""
        ### Stacking Ensemble
        Stacking คือการนำผลลัพธ์จาก **Base Models** หลายตัวมาเป็น input ของ **Meta-Learner**
        เพื่อให้ได้การทำนายที่แม่นยำกว่าโมเดลเดี่ยว

        **Base Models (ชั้นที่ 1):**
        - 🌲 **Random Forest** — สร้าง Decision Tree หลายต้นและ average ผลลัพธ์
        - 📈 **Gradient Boosting** — สร้าง Tree แบบต่อเนื่อง โดยแต่ละต้นแก้ error ของต้นก่อน
        - ⚡ **XGBoost** — Gradient Boosting ที่ optimized ด้วย regularization

        **Meta-Learner (ชั้นที่ 2):**
        - 📐 **Ridge Regression** — Linear Regression + L2 regularization

        ```
        Input → [RF, GBR, XGB] → predictions → Ridge → Final Output
        ```
        """)

    with st.expander("3️⃣ ขั้นตอนการพัฒนาโมเดล"):
        st.markdown("""
        1. แบ่งข้อมูล Train/Test = 80/20
        2. สร้าง StackingRegressor ด้วย `sklearn`
        3. ใช้ Cross-Validation 5-fold สำหรับ out-of-fold predictions
        4. ประเมินผลด้วย MAE, RMSE, R²
        5. บันทึกโมเดลด้วย `pickle`
        """)
        if os.path.exists("plots/ml_results.png"):
            st.image("plots/ml_results.png", caption="ผลการ Train ML Ensemble Model")

    with st.expander("4️⃣ แหล่งอ้างอิง"):
        st.markdown("""
        - [Scikit-learn StackingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)
        - [XGBoost Documentation](https://xgboost.readthedocs.io/)
        - [Random Forests (Breiman, 2001)](https://link.springer.com/article/10.1023/A:1010933404324)
        - [Dataset: Car Dekho — Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
        """)


# ══════════════════════════════════════════
# PAGE 2: NN DESCRIPTION
# ══════════════════════════════════════════
elif page == "🧠 Neural Network — อธิบายโมเดล":
    st.title("🧠 Neural Network Model")
    st.markdown("## EfficientNet-B0 — แยกยี่ห้อรถจากรูปภาพ")
    st.markdown("---")

    with st.expander("1️⃣ การเตรียมข้อมูล (Data Preparation)", expanded=True):
        st.markdown("""
        **โมเดลนี้ใช้ Transfer Learning จาก EfficientNet-B0 ที่ train บน ImageNet**
        ไม่ต้องใช้ Dataset รูปรถเพิ่มเติม เนื่องจาก ImageNet มีรูปรถหลากหลายประเภทอยู่แล้ว

        **การเตรียมรูปภาพก่อน inference:**
        | ขั้นตอน | รายละเอียด |
        |---------|------------|
        | Resize | ปรับขนาดรูปเป็น 224×224 pixels |
        | ToTensor | แปลงเป็น Tensor (0-1) |
        | Normalize | Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225] (ImageNet standard) |

        **ยี่ห้อรถที่รองรับ (10 classes):**
        Toyota, Honda, BMW, Ford, Chevrolet, Audi, Mercedes, Hyundai, Nissan, Volkswagen
        """)

    with st.expander("2️⃣ ทฤษฎีของอัลกอริทึม", expanded=True):
        st.markdown("""
        ### EfficientNet-B0 + Transfer Learning

        **EfficientNet** คือ CNN Architecture ที่ถูกออกแบบด้วย Neural Architecture Search (NAS)
        โดย scale ความลึก ความกว้าง และ resolution ของโมเดลอย่างสมดุล

        **Transfer Learning** คือการนำโมเดลที่ train บน dataset ขนาดใหญ่ (ImageNet, 1.2M รูป)
        มา Fine-tune สำหรับ task ใหม่ โดย:
        - **Freeze** Backbone layers (ไม่ train ใหม่) → ประหยัดเวลาและ resource
        - **Replace** Classifier head ด้วย layer ใหม่สำหรับ 10 ยี่ห้อรถ

        **โครงสร้างโมเดล:**
        ```
        Input Image (224×224×3)
          └─ EfficientNet-B0 Backbone (Frozen, Pretrained ImageNet)
          └─ Dropout (0.3)
          └─ Linear (1280 → 256) → ReLU
          └─ Dropout (0.2)
          └─ Linear (256 → 10)
          └─ Softmax → ความน่าจะเป็น 10 ยี่ห้อ
        ```
        """)
        if os.path.exists("plots/nn_architecture.png"):
            st.image("plots/nn_architecture.png", caption="โครงสร้าง EfficientNet-B0 Car Brand Classifier")

    with st.expander("3️⃣ ขั้นตอนการพัฒนาโมเดล"):
        st.markdown("""
        1. โหลด EfficientNet-B0 พร้อม Pretrained weights (ImageNet)
        2. Freeze ทุก layer ของ Backbone
        3. แทนที่ Classifier head ด้วย Custom layers (1280→256→10)
        4. บันทึกโมเดลด้วย `torch.save()`
        5. ใช้งานผ่าน Streamlit โดย upload รูปภาพแล้วทำนายได้ทันที
        """)

    with st.expander("4️⃣ แหล่งอ้างอิง"):
        st.markdown("""
        - [EfficientNet (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946)
        - [Transfer Learning — PyTorch](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
        - [ImageNet Dataset](https://www.image-net.org/)
        - [torchvision Models](https://pytorch.org/vision/stable/models.html)
        """)


# ══════════════════════════════════════════
# PAGE 3: TEST ML MODEL
# ══════════════════════════════════════════
elif page == "🔮 ทดสอบ ML Model":
    st.title("🔮 ทดสอบ ML Ensemble Model")
    st.markdown("กรอกข้อมูลรถเพื่อทำนายราคาขาย (Dataset: Car details v3)")
    st.markdown("---")

    try:
        model, encoders, scaler = load_ml_model()
    except Exception as e:
        st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}\nกรุณา train โมเดลก่อนด้วย `python ml_model.py`")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        year      = st.slider("ปีที่ผลิต (Year)", 2000, 2024, 2018)
        km_driven = st.number_input("ระยะทางที่ขับ (km)", min_value=0, max_value=500000, value=50000, step=1000)
        fuel      = st.selectbox("ประเภทเชื้อเพลิง", ["Diesel", "Petrol", "CNG", "LPG", "Electric"])
        seller    = st.selectbox("ประเภทผู้ขาย", ["Individual", "Dealer", "Trustmark Dealer"])

    with col2:
        transmission = st.selectbox("ระบบเกียร์", ["Manual", "Automatic"])
        owner        = st.selectbox("จำนวนเจ้าของ", ["First Owner", "Second Owner", "Third Owner",
                                                       "Fourth & Above Owner", "Test Drive Car"])
        mileage   = st.number_input("อัตราสิ้นเปลือง (kmpl)", 5.0, 50.0, 20.0, step=0.1)
        engine    = st.number_input("ขนาดเครื่องยนต์ (CC)", 500, 5000, 1200, step=50)
        max_power = st.number_input("กำลังสูงสุด (bhp)", 30.0, 500.0, 80.0, step=1.0)
        seats     = st.selectbox("จำนวนที่นั่ง", [2, 4, 5, 6, 7, 8, 9, 10])

    if st.button("🔮 ทำนายราคา", type="primary", use_container_width=True):
        car_age = 2024 - year
        raw = pd.DataFrame([{
            "km_driven": km_driven, "fuel": fuel, "seller_type": seller,
            "transmission": transmission, "owner": owner,
            "mileage": mileage, "engine": engine, "max_power": max_power,
            "seats": float(seats), "car_age": car_age,
        }])

        # Encode
        for col, le in encoders.items():
            if col in raw.columns:
                val = str(raw[col].iloc[0])
                raw[col] = le.transform([val])[0] if val in le.classes_ else -1

        # Scale
        raw_scaled = scaler.transform(raw)

        pred = model.predict(raw_scaled)[0]
        st.markdown("---")
        st.success(f"### 💰 ราคาที่ทำนาย: ₹ {pred:,.0f}")
        st.caption(f"(ประมาณ {pred/82:,.0f} บาท ที่อัตราแลกเปลี่ยน 1 INR ≈ 0.41 THB)")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("ปีรถ", year)
        col_b.metric("ระยะทาง", f"{km_driven:,} km")
        col_c.metric("เชื้อเพลิง", fuel)


# ══════════════════════════════════════════
# PAGE 4: TEST NN MODEL
# ══════════════════════════════════════════
elif page == "⚡ ทดสอบ Neural Network":
    st.title("⚡ ทดสอบ Neural Network — แยกยี่ห้อรถจากรูปภาพ")
    st.markdown("อัปโหลดรูปรถ แล้วโมเดลจะทำนายว่าเป็นยี่ห้ออะไร")
    st.markdown("---")

    try:
        nn_model, classes = load_nn_model()
    except Exception as e:
        st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}\nกรุณารันก่อนด้วย `python nn_model.py`")
        st.stop()

    st.info(f"🚗 ยี่ห้อที่รองรับ: {', '.join(classes)}")

    uploaded = st.file_uploader(
        "อัปโหลดรูปรถ (jpg, jpeg, png)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="รูปที่อัปโหลด", use_column_width=True)

        with col2:
            with st.spinner("กำลังวิเคราะห์รูปภาพ..."):
                pred_idx, probs = predict_brand(image, nn_model)

            pred_brand = classes[pred_idx]
            confidence = probs[pred_idx]

            brand_emoji = {
                "Toyota": "🇯🇵", "Honda": "🇯🇵", "Nissan": "🇯🇵",
                "BMW": "🇩🇪", "Audi": "🇩🇪", "Mercedes": "🇩🇪", "Volkswagen": "🇩🇪",
                "Ford": "🇺🇸", "Chevrolet": "🇺🇸",
                "Hyundai": "🇰🇷",
            }
            emoji = brand_emoji.get(pred_brand, "🚗")

            st.markdown("### ผลการทำนาย")
            st.success(f"## {emoji} {pred_brand}")
            st.metric("ความมั่นใจ", f"{confidence*100:.1f}%")

            st.markdown("---")
            st.markdown("#### ความน่าจะเป็นทุกยี่ห้อ")

            # Sort by probability
            sorted_idx = np.argsort(probs)[::-1]
            for i in sorted_idx:
                brand = classes[i]
                prob  = probs[i]
                bar_color = "🟩" if i == pred_idx else "⬜"
                st.write(f"{bar_color} **{brand}** — {prob*100:.1f}%")
                st.progress(float(prob))
