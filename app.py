import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Disease Prediction Toolkit",
    page_icon="❤️",
    layout="wide",
)

# ================= CSS =================
st.markdown("""
<style>
.title { font-size:36px; font-weight:bold; color:#e63946; text-align:center; }
.subtitle { font-size:20px; text-align:center; color:#457b9d; }
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown("<h1 class='title'>❤️ Disease Prediction Toolkit</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-based Healthcare Decision Support System</p>", unsafe_allow_html=True)
st.write("___")

# ================= SIDEBAR =================
st.sidebar.title("🔍 Navigation")

disease = st.sidebar.selectbox(
    "Select Disease",
    ["Heart Disease", "Diabetes", "Liver Disease"]
)

page = st.sidebar.radio("Go to", ["📝 Predict Risk", "📘 About Dataset"])

# ================= LOAD DATA =================
@st.cache_data
def load_heart_data():
    return pd.read_csv("heart.csv")

@st.cache_data
def load_diabetes_data():
    return pd.read_csv("diabetes.csv")

@st.cache_data
def load_liver_data():
    return pd.read_csv("indian_liver_patient.csv")  # make sure file is added

heart_df = load_heart_data()
diabetes_df = load_diabetes_data()
liver_df = load_liver_data()

# ================= MODELS =================
# Heart
X_h = heart_df.drop("target", axis=1)
y_h = heart_df["target"]
scaler_h = StandardScaler()
X_h_scaled = scaler_h.fit_transform(X_h)
Xh_train, Xh_test, yh_train, yh_test = train_test_split(
    X_h_scaled, y_h, test_size=0.2, random_state=42
)
heart_model = LogisticRegression()
heart_model.fit(Xh_train, yh_train)

# Diabetes
X_d = diabetes_df.drop("Outcome", axis=1)
y_d = diabetes_df["Outcome"]
scaler_d = StandardScaler()
X_d_scaled = scaler_d.fit_transform(X_d)
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    X_d_scaled, y_d, test_size=0.2, random_state=42
)
diabetes_model = LogisticRegression()
diabetes_model.fit(Xd_train, yd_train)

# Liver
X_l = liver_df.drop("Dataset", axis=1)
y_l = liver_df["Dataset"]  # 1= liver patient, 2= healthy
scaler_l = StandardScaler()
X_l_scaled = scaler_l.fit_transform(X_l)
Xl_train, Xl_test, yl_train, yl_test = train_test_split(
    X_l_scaled, y_l, test_size=0.2, random_state=42
)
liver_model = LogisticRegression(max_iter=500)
liver_model.fit(Xl_train, yl_train)

# =====================================================
# PAGE : PREDICT RISK
# =====================================================
if page == "📝 Predict Risk" and disease == "Heart Disease":
    st.header("❤️ Heart Disease Risk Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 10, 100, 40)
        trestbps = st.number_input("Resting BP", 80, 200, 120)
        chol = st.number_input("Cholesterol", 100, 500, 200)
    with col2:
        sex = st.selectbox("Sex", ["Female", "Male"])
        fbs = st.selectbox("Fasting Blood Sugar > 120?", ["No", "Yes"])
        thalach = st.number_input("Max Heart Rate", 60, 230, 150)
    with col3:
        exang = st.selectbox("Exercise-Induced Angina?", ["No", "Yes"])
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
        ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
    cp = st.selectbox("Chest Pain Type",
                      ["Typical Angina", "Atypical Angina",
                       "Non-anginal Pain", "Asymptomatic"])
    restecg = st.selectbox("Resting ECG",
                           ["Normal", "ST-T abnormality",
                            "Left ventricular hypertrophy"])
    slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
    thal = st.selectbox("Thal", ["Normal", "Fixed Defect", "Reversible Defect"])

    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0
    cp_map = {"Typical Angina":0,"Atypical Angina":1,"Non-anginal Pain":2,"Asymptomatic":3}
    restecg_map = {"Normal":0,"ST-T abnormality":1,"Left ventricular hypertrophy":2}
    slope_map = {"Upsloping":0,"Flat":1,"Downsloping":2}
    thal_map = {"Normal":1,"Fixed Defect":2,"Reversible Defect":3}

    user_data = np.array([[age, sex, cp_map[cp], trestbps, chol, fbs,
                            restecg_map[restecg], thalach, exang,
                            oldpeak, slope_map[slope], ca, thal_map[thal]]])
    scaled_input = scaler_h.transform(user_data)

    if st.button("🚀 Predict Heart Risk"):
        prob = heart_model.predict_proba(scaled_input)[0][1] * 100
        if prob < 30:
            st.success("🟢 Low Risk")
        elif prob < 70:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")
        st.info(f"Risk Probability: {prob:.2f}%")

# =====================================================
if page == "📝 Predict Risk" and disease == "Diabetes":
    st.header("🩺 Diabetes Risk Prediction")
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 50, 300, 120)
        bp = st.number_input("Blood Pressure", 40, 200, 70)
        skin = st.number_input("Skin Thickness", 0, 100, 20)
    with col2:
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 10, 100, 30)

    user_input = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    scaled_input = scaler_d.transform(user_input)

    if st.button("🚀 Predict Diabetes Risk"):
        prob = diabetes_model.predict_proba(scaled_input)[0][1] * 100
        if prob < 30:
            st.success("🟢 Low Risk")
        elif prob < 70:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")
        st.info(f"Risk Probability: {prob:.2f}%")

# =====================================================
if page == "📝 Predict Risk" and disease == "Liver Disease":
    st.header("🩺 Liver Disease Risk Prediction")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 100, 35)
        total_bilirubin = st.number_input("Total Bilirubin", 0.0, 10.0, 1.0)
        direct_bilirubin = st.number_input("Direct Bilirubin", 0.0, 5.0, 0.5)
        alkaline_phosphotase = st.number_input("Alkaline Phosphotase", 0, 300, 100)
        alamine_aminotransferase = st.number_input("Alamine Aminotransferase", 0, 300, 50)
    with col2:
        aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", 0, 300, 50)
        total_proteins = st.number_input("Total Proteins", 0.0, 10.0, 6.0)
        albumin = st.number_input("Albumin", 0.0, 10.0, 3.5)
        albumin_and_globulin_ratio = st.number_input("Albumin/Globulin Ratio", 0.0, 2.5, 1.0)
        gender = st.selectbox("Gender", ["Female", "Male"])

    gender = 1 if gender=="Male" else 0

    user_input = np.array([[age, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                            alamine_aminotransferase, aspartate_aminotransferase,
                            total_proteins, albumin, albumin_and_globulin_ratio, gender]])
    scaled_input = scaler_l.transform(user_input)

    if st.button("🚀 Predict Liver Risk"):
        prob = liver_model.predict_proba(scaled_input)[0][1] * 100
        if prob < 30:
            st.success("🟢 Low Risk")
        elif prob < 70:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")
        st.info(f"Risk Probability: {prob:.2f}%")

# =====================================================
if page == "📘 About Dataset":
    st.header("📘 Dataset Information")
    if disease == "Heart Disease":
        st.dataframe(heart_df)
    elif disease == "Diabetes":
        st.dataframe(diabetes_df)
    elif disease == "Liver Disease":
        st.dataframe(liver_df)
