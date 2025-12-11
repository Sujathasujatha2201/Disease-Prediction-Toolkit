import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(page_title="Heart Attack Risk Predictor", page_icon="❤️", layout="wide")

st.markdown("""
# ❤️ Heart Attack Risk Prediction App
Smart ML-powered prediction with easy dropdown inputs.
""")

# --- Load Model ---
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# --- Input Columns ---
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 1, 120, 45)
    sex = st.selectbox("Sex", ("Male", "Female"))
    cp = st.selectbox("Chest Pain Type", ("Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"))
    trestbps = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ("Yes", "No"))
    restecg = st.selectbox("Resting ECG", ("Normal", "ST-T Abnormality", "LV Hypertrophy"))

with col2:
    thalach = st.number_input("Max Heart Rate", 60, 250, 150)
    exang = st.selectbox("Exercise-Induced Angina", ("Yes", "No"))
    oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", ("Upsloping", "Flat", "Downsloping"))
    ca = st.selectbox("Major Vessels (0-3)", (0, 1, 2, 3))
    thal = st.selectbox("Thalassemia", ("Normal", "Fixed Defect", "Reversible Defect"))

# --- Encoding ---
sex = 1 if sex == "Male" else 0
cp = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
fbs = 1 if fbs == "Yes" else 0
restecg = ["Normal", "ST-T Abnormality", "LV Hypertrophy"].index(restecg)
exang = 1 if exang == "Yes" else 0
slope = ["Upsloping", "Flat", "Downsloping"].index(slope)
thal = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)

# --- Prediction Button ---
if st.button("Predict ❤️"):
    data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"🔴 High Risk of Heart Attack (Probability: {prob:.2f})")
    else:
        st.success(f"🟢 Low Risk of Heart Attack (Probability: {prob:.2f})")

    st.markdown("### 🩺 Health Suggestions")
    if prediction == 1:
        st.write("""
        - Reduce salt and oily food
        - Walk 30 minutes daily
        - Reduce stress
        - Regular health checkups
        - Consult cardiologist if symptoms appear
        """)
    else:
        st.write("Maintain your healthy lifestyle! ✔️")
