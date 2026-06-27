import streamlit as st
import pickle
import numpy as np

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
model = pickle.load(open("heart_disease_model.pkl", "rb"))

# ----------------------------
# Title
# ----------------------------
st.title("❤️ Heart Disease Prediction System")
st.markdown("### Machine Learning Based Heart Disease Prediction")
st.markdown("---")

st.info("Enter the patient's medical information and click **Predict**.")

# ----------------------------
# Input Fields
# ----------------------------

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=20,
        max_value=100,
        value=45
    )

    sex = st.selectbox(
        "Gender",
        ("Female", "Male")
    )

    cp = st.selectbox(
        "Chest Pain Type",
        (0, 1, 2, 3),
        help="""
0 = Typical Angina
1 = Atypical Angina
2 = Non-anginal Pain
3 = Asymptomatic
"""
    )

    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=60,
        max_value=220,
        value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        ("No", "Yes")
    )

with col2:

    oldpeak = st.number_input(
        "Oldpeak (ST Depression)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        (0, 1, 2)
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        (0, 1, 2, 3)
    )

    thal = st.selectbox(
        "Thal",
        (0, 1, 2, 3)
    )

# ----------------------------
# Convert Inputs
# ----------------------------

sex = 1 if sex == "Male" else 0
exang = 1 if exang == "Yes" else 0

input_data = np.array([[
    age,
    sex,
    cp,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
]])

# ----------------------------
# Prediction
# ----------------------------

st.markdown("---")

if st.button("Predict Heart Disease", use_container_width=True):

    prediction = model.predict(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
        st.write("The model predicts that the patient **may have heart disease**.")
    else:
        st.success("✅ No Heart Disease Detected")
        st.write("The model predicts that the patient **does not have heart disease**.")

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title("About")

st.sidebar.info(
    """
This application predicts the possibility of heart disease using a Logistic Regression model.

**Model**
- Logistic Regression

**Input Features**
- Age
- Gender
- Chest Pain Type
- Maximum Heart Rate
- Exercise Induced Angina
- Oldpeak
- Slope
- Number of Major Vessels
- Thal

Developed By : [Tejas Babar].
"""
)