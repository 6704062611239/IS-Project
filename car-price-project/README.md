# 🚗 Car Price Prediction — IS Project 2568

ระบบทำนายราคารถมือสองด้วย Machine Learning และ Neural Network

## Dataset
| ไฟล์ | แถว | Features | แหล่งที่มา |
|------|-----|----------|------------|
| car_details_v3.csv | 8,128 | 13 | [Kaggle — Car Dekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) |
| car_details_v4.csv | 2,059 | 20 | [Kaggle — Car Dekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) |

## โครงสร้างโปรเจค
```
car-price-project/
├── data/
│   ├── car_details_v3.csv
│   └── car_details_v4.csv
├── models/              ← สร้างอัตโนมัติหลัง train
├── plots/               ← สร้างอัตโนมัติหลัง train
├── data_preparation.py  ← เตรียมข้อมูล
├── ml_model.py          ← ML Ensemble (RF + GBR + XGB → Ridge)
├── nn_model.py          ← Neural Network (PyTorch MLP)
├── app.py               ← Streamlit Web App
├── requirements.txt
└── README.md
```

## วิธีรัน

### 1. ติดตั้ง dependencies
```bash
pip install -r requirements.txt
```

### 2. Train โมเดล
```bash
python ml_model.py
python nn_model.py
```

### 3. รัน Web App
```bash
streamlit run app.py
```

## Deploy บน Streamlit Community Cloud
1. Push โค้ดและ dataset ขึ้น GitHub
2. ไปที่ [share.streamlit.io](https://share.streamlit.io)
3. เลือก repo และไฟล์ `app.py`
4. กด Deploy

> ⚠️ หมายเหตุ: ต้อง train โมเดลและ commit ไฟล์ใน `models/` ขึ้น GitHub ด้วย เพื่อให้ cloud โหลดได้
