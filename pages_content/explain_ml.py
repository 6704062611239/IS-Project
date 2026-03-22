import streamlit as st

def show():
    st.title(" Ensemble Machine Learning")
    st.markdown("### Titanic — Survival Prediction")
    st.markdown("---")

    # ── 1. Dataset ──
    st.header("1.  Dataset")
    st.markdown("""
    **ที่มา:** [Kaggle — Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)

    Dataset มีข้อมูลผู้โดยสารเรือ Titanic ทั้งหมด **891 แถว 12 คอลัมน์** โดยมีเป้าหมายคือ
    ทำนายว่าผู้โดยสารแต่ละคนรอดชีวิตหรือไม่ (Survived = 1 หรือ 0)
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Features หลัก**")
        st.markdown("""
        | Feature | คำอธิบาย |
        |---|---|
        | Pclass | ชั้นโดยสาร (1, 2, 3) |
        | Sex | เพศ |
        | Age | อายุ |
        | SibSp | จำนวนพี่น้อง/คู่สมรส |
        | Parch | จำนวนพ่อแม่/ลูก |
        | Fare | ราคาตั๋ว |
        | Embarked | ท่าเรือที่ขึ้น |
        | Cabin | หมายเลขห้องโดยสาร |
        """)
    with col2:
        st.markdown("**ความไม่สมบูรณ์ของข้อมูล**")
        st.error("🔴 Cabin — หายไปมากกว่า 70%")
        st.warning("🟡 Age — หายไปประมาณ 20%")
        st.warning("🟡 Embarked — หายไป 2 แถว")
        st.info("🔵 ไม่มี label ที่ชัดเจนสำหรับบาง feature")

    st.markdown("---")

    # ── 2. Data Preparation ──
    st.header("2.  การเตรียมข้อมูล (Data Preparation)")
    st.markdown("""
    **ขั้นตอนการเตรียมข้อมูล:**

    1. **Drop คอลัมน์ที่ไม่จำเป็น** — PassengerId, Name, Ticket
    2. **จัดการ Missing Values**
        - `Age` → เติมด้วยค่า **Median**
        - `Embarked` → เติมด้วยค่า **Mode**
        - `Cabin` → ดึงตัวอักษรแรก หรือใส่ 'Unknown'
    3. **Encode Categorical Features**
        - `Sex` → male=0, female=1
        - `Embarked` → S=0, C=1, Q=2
        - `Cabin` → Label Encoding
    4. **Feature Engineering**
        - `FamilySize` = SibSp + Parch + 1
        - `IsAlone` = 1 ถ้า FamilySize == 1
    5. **StandardScaler** — Normalize ข้อมูลตัวเลขก่อนเข้าโมเดล
    """)

    st.markdown("---")

    # ── 3. ทฤษฎี Ensemble / Stacking ──
    st.header("3.  ทฤษฎี Ensemble Learning — Stacking")
    st.markdown("""
    **Ensemble Learning** คือการนำโมเดลหลายตัวมารวมกัน เพื่อให้ผลลัพธ์ดีกว่าโมเดลเดี่ยว

    **Stacking** คือวิธี Ensemble ที่แบ่งเป็น 2 ชั้น:
    - **Layer 1 (Base Models):** โมเดลหลายตัวเรียนรู้จาก training data
    - **Layer 2 (Meta Model):** รับผลลัพธ์จาก Base Models มารวมกันและทำนายขั้นสุดท้าย
    """)

    st.markdown("""
    ```
    Input Features
         ↓
    ┌──────────────────────────────┐
    │  Layer 1 (Base Models)       │
    │  ├── Random Forest           │
    │  ├── SVM                     │  → predictions
    │  └── XGBoost                 │
    └──────────────────────────────┘
         ↓
    ┌──────────────────────────────┐
    │  Layer 2 (Meta Model)        │
    │  Logistic Regression         │  → Final Output
    └──────────────────────────────┘
         ↓
      รอดชีวิต / ไม่รอด
    ```
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ** Random Forest**
        - รวม Decision Tree หลายต้น
        - แต่ละต้นเรียนรู้จาก subset ของข้อมูล
        - โหวตผลลัพธ์เพื่อลด overfitting
        """)
    with col2:
        st.markdown("""
        ** XGBoost**
        - Gradient Boosting แบบ optimized
        - เรียนรู้จากข้อผิดพลาดของโมเดลก่อนหน้า
        - ประสิทธิภาพสูง เร็วกว่า Gradient Boosting ทั่วไป
        """)
    with col3:
        st.markdown("""
        ** SVM**
        - หา hyperplane ที่แบ่ง class ได้ดีที่สุด
        - ใช้ RBF kernel สำหรับข้อมูลที่ไม่เป็นเส้นตรง
        - เหมาะกับข้อมูลขนาดกลาง
        """)

    st.markdown("---")

    # ── 4. ขั้นตอนการพัฒนา ──
    st.header("4.  ขั้นตอนการพัฒนาโมเดล")
    st.markdown("""
    1. โหลดและสำรวจ Dataset (EDA)
    2. เตรียมข้อมูล (Data Preparation)
    3. แบ่ง Train/Test (80:20) พร้อม Stratify
    4. Scale features ด้วย StandardScaler
    5. สร้าง Base Models ทั้ง 3 ตัว
    6. สร้าง Stacking Classifier (cv=5)
    7. เทรนโมเดลและประเมินผลด้วย Accuracy, F1-Score
    8. เปรียบเทียบผลกับโมเดลเดี่ยว
    9. บันทึกโมเดลด้วย joblib
    """)

    st.markdown("---")

    # ── 5. อ้างอิง ──
    st.header("5.  แหล่งอ้างอิง")
    st.markdown("""
    - Kaggle Titanic Competition: https://www.kaggle.com/competitions/titanic
    - Scikit-learn StackingClassifier: https://scikit-learn.org/stable/modules/ensemble.html
    - XGBoost Documentation: https://xgboost.readthedocs.io/
    - Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.
    - Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
    """)
