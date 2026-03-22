import streamlit as st

def show():
    st.title(" IS Project ")
    st.subheader("Machine Learning & Neural Network")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ###  Ensemble ML
        **Dataset:** Titanic — Machine Learning from Disaster

        **โมเดล:** Stacking Classifier
        - Random Forest
        - SVM
        - XGBoost
        - Meta: Logistic Regression

        **Task:** ทำนายว่าผู้โดยสารรอดชีวิตหรือไม่
        """)


    with col2:
        st.markdown("""
        ###  Neural Network
        **Dataset:** CIFAR-10

        **โมเดล:** CNN (Convolutional Neural Network)
        - 3 Convolutional Blocks
        - Batch Normalization
        - Dropout
        - Data Augmentation

        **Task:** แยกรูปภาพ สัตว์ vs ไม่ใช่สัตว์
        """)


    st.markdown("---")
    st.markdown("###  ทดสอบโมเดล")
    col3, col4 = st.columns(2)
    with col3:
        st.info(" **ทดสอบ Ensemble ML**\nกรอกข้อมูลผู้โดยสาร Titanic → ทำนายการรอดชีวิต")
    with col4:
        st.info(" **ทดสอบ Neural Network**\nอัพโหลดรูปภาพ → ทำนายว่าเป็นสัตว์หรือไม่")
