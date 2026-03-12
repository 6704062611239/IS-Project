import streamlit as st

st.set_page_config(
    page_title="IS Project — ML & Neural Network",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🤖 IS Project 2568")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 เมนู")

page = st.sidebar.radio("เลือกหน้า", [
    "🏠 หน้าหลัก",
    "📄 อธิบาย Ensemble ML",
    "📄 อธิบาย Neural Network",
    "🧪 ทดสอบ Ensemble ML",
    "🧪 ทดสอบ Neural Network",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset ที่ใช้**")
st.sidebar.markdown("- 🚢 Titanic (Ensemble ML)")
st.sidebar.markdown("- 🖼️ CIFAR-10 (Neural Network)")

# ── Router ──
if page == "🏠 หน้าหลัก":
    from pages_content import home
    home.show()
elif page == "📄 อธิบาย Ensemble ML":
    from pages_content import explain_ml
    explain_ml.show()
elif page == "📄 อธิบาย Neural Network":
    from pages_content import explain_nn
    explain_nn.show()
elif page == "🧪 ทดสอบ Ensemble ML":
    from pages_content import demo_ml
    demo_ml.show()
elif page == "🧪 ทดสอบ Neural Network":
    from pages_content import demo_nn
    demo_nn.show()
