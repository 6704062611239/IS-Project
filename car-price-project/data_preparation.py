"""
Data Preparation Script
Dataset: Car Price Prediction
Sources:
  - Car details v3.csv  (Kaggle - Car Dekho)
  - car details v4.csv  (Kaggle - Car Dekho)
"""

import pandas as pd
import numpy as np
import re
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ────────────────────────────────────────────────
# 1. LOAD DATA
# ────────────────────────────────────────────────
def load_data(base_dir="data"):
    df3 = pd.read_csv(os.path.join(base_dir, "car_details_v3.csv"))
    df4 = pd.read_csv(os.path.join(base_dir, "car_details_v4.csv"))
    print(f"[V3] shape: {df3.shape}")
    print(f"[V4] shape: {df4.shape}")
    return df3, df4


# ────────────────────────────────────────────────
# 2. CLEAN V3
# ────────────────────────────────────────────────
def clean_v3(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # แปลง mileage: "23.4 kmpl" / "17.3 km/kg" → float
    def parse_mileage(val):
        if pd.isna(val):
            return np.nan
        m = re.search(r"[\d.]+", str(val))
        return float(m.group()) if m else np.nan

    # แปลง engine: "1248 CC" → float
    def parse_cc(val):
        if pd.isna(val):
            return np.nan
        m = re.search(r"[\d.]+", str(val))
        return float(m.group()) if m else np.nan

    # แปลง max_power: "74 bhp" → float
    def parse_bhp(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip()
        if val in ("", "bhp"):
            return np.nan
        m = re.search(r"[\d.]+", val)
        return float(m.group()) if m else np.nan

    df["mileage"]   = df["mileage"].apply(parse_mileage)
    df["engine"]    = df["engine"].apply(parse_cc)
    df["max_power"] = df["max_power"].apply(parse_bhp)
    df["seats"]     = df["seats"].astype(float)

    # ลบ torque ออก (ข้อมูลซับซ้อน/ไม่สม่ำเสมอ)
    df.drop(columns=["torque"], inplace=True)

    # fill missing ด้วย median (numeric)
    for col in ["mileage", "engine", "max_power", "seats"]:
        df[col].fillna(df[col].median(), inplace=True)

    # ลบ outlier selling_price (top 1%)
    q99 = df["selling_price"].quantile(0.99)
    df  = df[df["selling_price"] <= q99]

    # car_age แทน year
    df["car_age"] = 2024 - df["year"]
    df.drop(columns=["year", "name"], inplace=True)

    print(f"[V3 cleaned] shape: {df.shape}, missing: {df.isnull().sum().sum()}")
    return df


# ────────────────────────────────────────────────
# 3. CLEAN V4
# ────────────────────────────────────────────────
def clean_v4(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    def parse_num(val):
        if pd.isna(val):
            return np.nan
        m = re.search(r"[\d.]+", str(val))
        return float(m.group()) if m else np.nan

    for col in ["engine", "max_power", "max_torque"]:
        df[col] = df[col].apply(parse_num)

    # fill missing ด้วย median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # fill missing categorical ด้วย mode
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # ลบ outlier price (top 1%)
    q99 = df["price"].quantile(0.99)
    df  = df[df["price"] <= q99]

    df["car_age"] = 2024 - df["year"]
    df.drop(columns=["year", "make", "model", "color", "location"], inplace=True)

    print(f"[V4 cleaned] shape: {df.shape}, missing: {df.isnull().sum().sum()}")
    return df


# ────────────────────────────────────────────────
# 4. ENCODE + SCALE
# ────────────────────────────────────────────────
def encode_and_scale(df: pd.DataFrame, target_col: str, encoders: dict = None, scaler=None, fit: bool = True):
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if fit:
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        scaler = StandardScaler()
        feature_cols = [c for c in df.columns if c != target_col]
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        for col in cat_cols:
            le = encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        feature_cols = [c for c in df.columns if c != target_col]
        df[feature_cols] = scaler.transform(df[feature_cols])

    X = df[[c for c in df.columns if c != target_col]]
    y = df[target_col]
    return X, y, encoders, scaler


# ────────────────────────────────────────────────
# 5. MAIN
# ────────────────────────────────────────────────
def prepare_v3(base_dir="data", save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    df3, _ = load_data(base_dir)
    df3_clean = clean_v3(df3)
    X, y, encoders, scaler = encode_and_scale(df3_clean, target_col="selling_price", fit=True)

    with open(os.path.join(save_dir, "encoders_v3.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(save_dir, "scaler_v3.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"[V3] X: {X.shape}, y: {y.shape}")
    print(f"Features: {list(X.columns)}")
    return X, y, encoders, scaler, df3_clean


def prepare_v4(base_dir="data", save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)
    _, df4 = load_data(base_dir)
    df4_clean = clean_v4(df4)
    X, y, encoders, scaler = encode_and_scale(df4_clean, target_col="price", fit=True)

    with open(os.path.join(save_dir, "encoders_v4.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(save_dir, "scaler_v4.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"[V4] X: {X.shape}, y: {y.shape}")
    print(f"Features: {list(X.columns)}")
    return X, y, encoders, scaler, df4_clean


if __name__ == "__main__":
    prepare_v3()
    prepare_v4()
    print("\n✅ Data preparation complete!")
