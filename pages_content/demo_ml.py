import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd

def train_and_save():
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df  = pd.read_csv(url)

    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Cabin']      = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'Unknown')
    df['Sex']        = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked']   = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['Cabin']      = df['Cabin'].astype('category').cat.codes
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone']    = (df['FamilySize'] == 1).astype(int)

    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    base_models = [
        ('random_forest', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
        ('svm',           SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
        ('xgboost',       XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                        random_state=42, eval_metric='logloss'))
    ]
    model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )
    model.fit(X_train_scaled, y_train)
    joblib.dump(model,  'titanic_stacking_model.pkl')
    joblib.dump(scaler, 'titanic_scaler.pkl')
    return model, scaler


def show():
    st.title(" ทดสอบ Ensemble ML")
    st.markdown("### Titanic — ทำนายการรอดชีวิต")
    st.markdown("---")

    model_path  = "titanic_stacking_model.pkl"
    scaler_path = "titanic_scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        with st.spinner(" กำลังเทรนโมเดล กรุณารอสักครู่..."):
            model, scaler = train_and_save()
        st.success(" เทรนโมเดลเสร็จแล้ว!")
    else:
        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        st.success(" โหลดโมเดลสำเร็จ!")

    st.markdown("กรอกข้อมูลผู้โดยสารด้านล่าง แล้วกด **ทำนาย** ได้เลยครับ")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("####  ข้อมูลผู้โดยสาร")
        pclass = st.selectbox("ชั้นโดยสาร (Pclass)", [1, 2, 3])
        sex    = st.radio("เพศ (Sex)", ["ชาย", "หญิง"], horizontal=True)
        age    = st.slider("อายุ (Age)", 1, 80, 30)
        fare   = st.number_input("ราคาตั๋ว (Fare)", min_value=0.0, max_value=600.0, value=32.0, step=0.5)

    with col2:
        st.markdown("####  ข้อมูลการเดินทาง")
        sibsp    = st.number_input("จำนวนพี่น้อง/คู่สมรส (SibSp)", 0, 8, 0)
        parch    = st.number_input("จำนวนพ่อแม่/ลูก (Parch)", 0, 6, 0)
        embarked = st.selectbox("ท่าเรือที่ขึ้น (Embarked)",
                                ["S — Southampton", "C — Cherbourg", "Q — Queenstown"])
        cabin    = st.selectbox("ประเภทห้อง (Cabin)",
                                ["Unknown", "A", "B", "C", "D", "E", "F", "G", "T"])

    st.markdown("---")
    if st.button(" ทำนายการรอดชีวิต", use_container_width=True, type="primary"):
        sex_val      = 0 if sex == "ชาย" else 1
        embarked_val = {"S — Southampton": 0, "C — Cherbourg": 1, "Q — Queenstown": 2}[embarked]
        cabin_map    = {"Unknown": -1, "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "T": 7}
        cabin_val    = cabin_map[cabin]
        family_size  = int(sibsp) + int(parch) + 1
        is_alone     = 1 if family_size == 1 else 0

        features = np.array([[
            int(pclass), int(sex_val), float(age),
            int(sibsp),  int(parch),  float(fare),
            int(cabin_val), int(embarked_val),
            int(family_size), int(is_alone)
        ]], dtype=float)

        features_scaled = scaler.transform(features)
        prediction      = model.predict(features_scaled)[0]
        probability     = model.predict_proba(features_scaled)[0]

        st.markdown("###  ผลการทำนาย")
        if prediction == 1:
            st.success("##  รอดชีวิต (Survived)")
            st.balloons()
        else:
            st.error("##  ไม่รอดชีวิต (Not Survived)")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🟢 โอกาสรอดชีวิต",   f"{probability[1]*100:.2f}%")
        with col_b:
            st.metric("🔴 โอกาสไม่รอดชีวิต", f"{probability[0]*100:.2f}%")
        st.progress(float(probability[1]))
