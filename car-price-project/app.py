"""
Streamlit Web Application
Car Price Prediction — ML & Neural Network
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
import torch
import torch.nn as nn
from PIL import Image

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
# NEURAL NETWORK ARCHITECTURE (must match nn_model.py)
# ──────────────────────────────────────────
class CarPriceNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)


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
    ckpt = torch.load("models/nn_model.pt", map_location="cpu")
    model = CarPriceNN(input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with open("models/encoders_v4.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("models/scaler_v4.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, encoders, scaler, ckpt["feature_names"]


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
    st.markdown("## MLP Regressor ด้วย PyTorch")
    st.markdown("---")

    with st.expander("1️⃣ การเตรียมข้อมูล (Data Preparation)", expanded=True):
        st.markdown("""
        **Dataset:** `car details v4.csv` — 2,059 แถว, 20 คอลัมน์

        **ปัญหาที่พบและการแก้ไข:**
        | ปัญหา | วิธีแก้ไข |
        |-------|-----------|
        | `engine`, `max_power`, `max_torque` เป็น string | Regex แยกตัวเลข |
        | Missing values (numeric) | Fill ด้วย Median |
        | Missing values (categorical) | Fill ด้วย Mode |
        | Outlier ใน price | ลบแถวที่เกิน Percentile 99 |
        | Target (`price`) มี skewness สูง | Log Transform: `log1p(price)` |
        | Categorical features | Label Encoding |
        | Scale ไม่เท่ากัน | Standard Scaler |
        """)

    with st.expander("2️⃣ ทฤษฎีของอัลกอริทึม", expanded=True):
        st.markdown("""
        ### Multi-Layer Perceptron (MLP)
        MLP เป็น Feedforward Neural Network ที่ประกอบด้วย Layer หลายชั้น

        **โครงสร้างโมเดล:**
        ```
        Input (n features)
          └─ BatchNorm1d
          └─ Linear(256) → ReLU → Dropout(0.3)
          └─ Linear(128) → ReLU → Dropout(0.2)
          └─ Linear(64)  → ReLU
          └─ Linear(1)   → Output (log price)
        ```

        **เทคนิคที่ใช้:**
        - **BatchNorm** — Normalize input แต่ละ batch ลด internal covariate shift
        - **ReLU** — Activation function: `max(0, x)` แก้ปัญหา vanishing gradient
        - **Dropout** — Regularization โดยสุ่ม "ปิด" neurons บางส่วนระหว่าง training
        - **Huber Loss** — Loss function ที่ทนทานต่อ outliers มากกว่า MSE
        - **Adam Optimizer** — Adaptive learning rate ด้วย momentum
        - **ReduceLROnPlateau** — ลด learning rate เมื่อ val loss ไม่ลดลง
        """)

    with st.expander("3️⃣ ขั้นตอนการพัฒนาโมเดล"):
        st.markdown("""
        1. แบ่งข้อมูล Train/Test = 80/20
        2. สร้าง DataLoader (batch_size=64)
        3. Train 100 epochs พร้อม early stopping (บันทึก best model)
        4. ประเมินผลด้วย MAE, RMSE, R² (inverse log transform ก่อน)
        5. บันทึกโมเดลด้วย `torch.save()`
        """)
        if os.path.exists("plots/nn_results.png"):
            st.image("plots/nn_results.png", caption="ผลการ Train Neural Network Model")

    with st.expander("4️⃣ แหล่งอ้างอิง"):
        st.markdown("""
        - [PyTorch Documentation](https://pytorch.org/docs/)
        - [Batch Normalization (Ioffe & Szegedy, 2015)](https://arxiv.org/abs/1502.03167)
        - [Dropout (Srivastava et al., 2014)](https://jmlr.org/papers/v15/srivastava14a.html)
        - [Adam Optimizer (Kingma & Ba, 2014)](https://arxiv.org/abs/1412.6980)
        - [Dataset: Car Dekho — Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
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
    st.title("⚡ ทดสอบ Neural Network Model")
    st.markdown("กรอกข้อมูลรถเพื่อทำนายราคาขาย (Dataset: Car details v4)")
    st.markdown("---")

    try:
        nn_model, encoders, scaler, feature_names = load_nn_model()
    except Exception as e:
        st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}\nกรุณา train โมเดลก่อนด้วย `python nn_model.py`")
        st.stop()

    # Load original data for dropdown options
    df4_orig = pd.read_csv("data/car_details_v4.csv")
    df4_orig.columns = [c.lower().replace(" ", "_") for c in df4_orig.columns]

    col1, col2 = st.columns(2)
    with col1:
        year         = st.slider("ปีที่ผลิต (Year)", 2000, 2024, 2018)
        kilometer    = st.number_input("ระยะทางที่ขับ (km)", 0, 500000, 50000, step=1000)
        fuel_type    = st.selectbox("ประเภทเชื้อเพลิง", df4_orig["fuel_type"].dropna().unique().tolist())
        transmission = st.selectbox("ระบบเกียร์", df4_orig["transmission"].dropna().unique().tolist())
        owner        = st.selectbox("จำนวนเจ้าของ", df4_orig["owner"].dropna().unique().tolist())

    with col2:
        seller_type  = st.selectbox("ประเภทผู้ขาย", df4_orig["seller_type"].dropna().unique().tolist())
        engine       = st.number_input("ขนาดเครื่องยนต์ (CC)", 500.0, 6000.0, 1200.0, step=50.0)
        max_power    = st.number_input("กำลังสูงสุด (bhp)", 30.0, 500.0, 80.0, step=1.0)
        max_torque   = st.number_input("แรงบิดสูงสุด (Nm)", 50.0, 1000.0, 150.0, step=5.0)
        drivetrain   = st.selectbox("ระบบขับเคลื่อน", df4_orig["drivetrain"].dropna().unique().tolist())
        length       = st.number_input("ความยาว (mm)", 2000.0, 6000.0, 4000.0, step=10.0)
        width        = st.number_input("ความกว้าง (mm)", 1000.0, 3000.0, 1700.0, step=10.0)
        height       = st.number_input("ความสูง (mm)", 1000.0, 3000.0, 1500.0, step=10.0)
        seating      = st.selectbox("จำนวนที่นั่ง", [2, 4, 5, 6, 7, 8])
        fuel_tank    = st.number_input("ขนาดถังน้ำมัน (L)", 20.0, 100.0, 45.0, step=1.0)

    if st.button("⚡ ทำนายราคา", type="primary", use_container_width=True):
        car_age = 2024 - year
        raw = pd.DataFrame([{
            "kilometer": kilometer, "fuel_type": fuel_type,
            "transmission": transmission, "owner": owner,
            "seller_type": seller_type, "engine": engine,
            "max_power": max_power, "max_torque": max_torque,
            "drivetrain": drivetrain, "length": length,
            "width": width, "height": height,
            "seating_capacity": float(seating),
            "fuel_tank_capacity": fuel_tank, "car_age": car_age,
        }])

        # Ensure all feature columns present
        for col in feature_names:
            if col not in raw.columns:
                raw[col] = 0

        raw = raw[feature_names]

        for col, le in encoders.items():
            if col in raw.columns:
                val = str(raw[col].iloc[0])
                raw[col] = le.transform([val])[0] if val in le.classes_ else -1

        raw_scaled = scaler.transform(raw.values.astype(np.float32))
        X_tensor = torch.FloatTensor(raw_scaled)

        with torch.no_grad():
            log_pred = nn_model(X_tensor).item()
        pred = np.expm1(log_pred)

        st.markdown("---")
        st.success(f"### 💰 ราคาที่ทำนาย: ₹ {pred:,.0f}")
        st.caption(f"(ประมาณ {pred/82:,.0f} บาท ที่อัตราแลกเปลี่ยน 1 INR ≈ 0.41 THB)")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("ปีรถ", year)
        col_b.metric("ระยะทาง", f"{kilometer:,} km")
        col_c.metric("เชื้อเพลิง", fuel_type)
