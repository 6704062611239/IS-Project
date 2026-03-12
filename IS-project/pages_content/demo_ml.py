import streamlit as st
import numpy as np
import joblib
import os

def show():
    st.title("🧪 ทดสอบ Ensemble ML")
    st.markdown("### Titanic — ทำนายการรอดชีวิต")
    st.markdown("---")

    # โหลดโมเดล
    model_path  = "titanic_stacking_model.pkl"
    scaler_path = "titanic_scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("❌ ไม่พบไฟล์โมเดล กรุณารัน `titanic_ensemble.py` ก่อน")
        st.code("python titanic_ensemble.py", language="bash")
        return

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    st.success("✅ โหลดโมเดลสำเร็จ!")
    st.markdown("กรอกข้อมูลผู้โดยสารด้านล่าง แล้วกด **ทำนาย** ได้เลยครับ")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👤 ข้อมูลผู้โดยสาร")
        pclass   = st.selectbox("ชั้นโดยสาร (Pclass)", [1, 2, 3],
                                help="1=First Class, 2=Second Class, 3=Third Class")
        sex      = st.radio("เพศ (Sex)", ["ชาย", "หญิง"], horizontal=True)
        age      = st.slider("อายุ (Age)", 1, 80, 30)
        fare     = st.number_input("ราคาตั๋ว (Fare)", min_value=0.0,
                                   max_value=600.0, value=32.0, step=0.5)

    with col2:
        st.markdown("#### 🚢 ข้อมูลการเดินทาง")
        sibsp    = st.number_input("จำนวนพี่น้อง/คู่สมรส (SibSp)", 0, 8, 0)
        parch    = st.number_input("จำนวนพ่อแม่/ลูก (Parch)", 0, 6, 0)
        embarked = st.selectbox("ท่าเรือที่ขึ้น (Embarked)",
                                ["S — Southampton", "C — Cherbourg", "Q — Queenstown"])
        cabin    = st.selectbox("ประเภทห้อง (Cabin)",
                                ["Unknown", "A", "B", "C", "D", "E", "F", "G", "T"])

    st.markdown("---")

    if st.button("🔮 ทำนายการรอดชีวิต", use_container_width=True, type="primary"):
        # แปลงค่า input — ใช้ float() และ int() เพื่อป้องกัน numpy scalar error
        sex_val      = 0 if sex == "ชาย" else 1
        embarked_val = {"S — Southampton": 0, "C — Cherbourg": 1, "Q — Queenstown": 2}[embarked]
        cabin_map    = {"Unknown": -1, "A": 0, "B": 1, "C": 2, "D": 3,
                        "E": 4, "F": 5, "G": 6, "T": 7}
        cabin_val    = cabin_map[cabin]
        family_size  = int(sibsp) + int(parch) + 1
        is_alone     = 1 if family_size == 1 else 0

        features = np.array([[
            int(pclass),
            int(sex_val),
            float(age),
            int(sibsp),
            int(parch),
            float(fare),
            int(cabin_val),
            int(embarked_val),
            int(family_size),
            int(is_alone)
        ]], dtype=float)

        features_scaled = scaler.transform(features)
        prediction      = model.predict(features_scaled)[0]
        probability     = model.predict_proba(features_scaled)[0]

        st.markdown("---")
        st.markdown("### 📊 ผลการทำนาย")

        if prediction == 1:
            st.success(f"## ✅ รอดชีวิต (Survived)")
            st.metric("ความมั่นใจ", f"{probability[1]*100:.2f}%")
            st.balloons()
        else:
            st.error(f"## ❌ ไม่รอดชีวิต (Not Survived)")
            st.metric("ความมั่นใจ", f"{probability[0]*100:.2f}%")

        # Probability bar
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🟢 โอกาสรอดชีวิต",   f"{probability[1]*100:.2f}%")
        with col_b:
            st.metric("🔴 โอกาสไม่รอดชีวิต", f"{probability[0]*100:.2f}%")

        st.progress(float(probability[1]))

        # สรุปข้อมูลที่กรอก
        with st.expander("📋 ดูข้อมูลที่ใช้ทำนาย"):
            st.markdown(f"""
            | Feature | ค่า |
            |---|---|
            | Pclass | {pclass} |
            | Sex | {sex} |
            | Age | {age} |
            | SibSp | {sibsp} |
            | Parch | {parch} |
            | Fare | {fare} |
            | Embarked | {embarked} |
            | Cabin | {cabin} |
            | FamilySize | {family_size} |
            | IsAlone | {'ใช่' if is_alone else 'ไม่ใช่'} |
            """)
