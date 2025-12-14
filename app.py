import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Disease Prediction Toolkit",
    page_icon="❤️",
    layout="wide",
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.title {
    font-size: 36px;
    font-weight: bold;
    color: #e63946;
    text-align: center;
}
.subtitle {
    font-size: 20px;
    text-align: center;
    color: #457b9d;
}
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
    ["Heart Disease", "Diabetes (Coming Soon)", "Liver Disease (Coming Soon)"]
)

page = st.sidebar.radio("Go to", ["📝 Predict Risk", "📘 About Dataset"])

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

# ================= MODEL PREPARATION =================
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# ================= PAGE 1 : PREDICT RISK =================
if page == "📝 Predict Risk" and disease == "Heart Disease":

    st.header("📝 Enter Your Health Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 10, 100, 40)
        trestbps = st.number_input("Resting Blood Pressure (mm/Hg)", 80, 200, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 500, 200)

    with col2:
        sex = st.selectbox("Sex", ["Female", "Male"])
        fbs = st.selectbox("Fasting Blood Sugar > 120?", ["No", "Yes"])
        thalach = st.number_input("Max Heart Rate Achieved", 60, 230, 150)

    with col3:
        exang = st.selectbox("Exercise-Induced Angina?", ["No", "Yes"])
        oldpeak = st.number_input("Oldpeak (ECG Depression)", 0.0, 10.0, 1.0)
        ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])cp = st.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
    )

    restecg = st.selectbox(
        "Resting ECG Result",
        ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"]
    )

    slope = st.selectbox(
        "Slope of ST Segment",
        ["Upsloping", "Flat", "Downsloping"]
    )

    thal = st.selectbox(
        "Thalassemia Result",
        ["Normal", "Fixed Defect", "Reversible Defect"]
    )

    # ===== CONVERT TO NUMERIC =====
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }

    restecg_map = {
        "Normal": 0,
        "ST-T abnormality": 1,
        "Left ventricular hypertrophy": 2
    }

    slope_map = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }

    thal_map = {
        "Normal": 1,
        "Fixed Defect": 2,
        "Reversible Defect": 3
    }

    user_data = np.array([[
        age, sex, cp_map[cp], trestbps, chol, fbs,
        restecg_map[restecg], thalach, exang, oldpeak,
        slope_map[slope], ca, thal_map[thal]
    ]])

    scaled_input = scaler.transform(user_data)

    # ================= PREDICT BUTTON =================
    if st.button("🚀 Predict Risk"):

        probability = model.predict_proba(scaled_input)[0][1] * 100

        # ===== RISK LEVEL =====
        if probability < 30:
            risk = "Low Risk"
            st.success("🟢 Low Risk of Heart Attack")
        elif probability < 70:
            risk = "Medium Risk"
            st.warning("🟡 Medium Risk of Heart Attack")
        else:
            risk = "High Risk"
            st.error("🔴 High Risk of Heart Attack")

        st.info(f"Risk Probability: {probability:.2f}%")

        # ===== EXPLANATION =====
        st.subheader("🧠 Why this result?")
        if age > 50:
            st.write("- Age above 50 increases heart risk")
        if chol > 240:
            st.write("- High cholesterol level detected")
        if trestbps > 140:
            st.write("- Blood pressure is higher than normal")
        if exang == 1:
            st.write("- Exercise-induced chest pain observed")

        # ===== HEALTH SUGGESTIONS =====
        st.subheader("💡 Health Suggestions")

        if risk == "Low Risk":
            st.write("- Maintain a balanced diet")
            st.write("- Continue regular physical activity")

        elif risk == "Medium Risk":
            st.write("- Reduce salt and fatty food intake")
            st.write("- Exercise at least 30 minutes daily")
            st.write("- Monitor blood pressure and cholesterol")

 else:
            st.write("- Consult a cardiologist immediately")
            st.write("- Follow medical advice strictly")
            st.write("- Regular heart checkups are necessary")

# ================= COMING SOON PAGES =================
if page == "📝 Predict Risk" and disease != "Heart Disease":
    st.warning("🚧 This disease module will be added soon.")

# ================= PAGE 2 : ABOUT DATASET =================
if page == "📘 About Dataset":
    st.header("📘 Dataset Information")
    st.write("This dataset contains medical data used to predict heart disease risk.")
    st.dataframe(df)

    st.write("""
    ### Features Explained:
    - **age**: Age  
    - **sex**: Male/Female  
    - **cp**: Chest Pain Type  
    - **trestbps**: Resting Blood Pressure  
    - **chol**: Cholesterol  
    - **fbs**: Fasting Blood Sugar  
    - **restecg**: Resting ECG  
    - **thalach**: Max Heart Rate  
    - **exang**: Exercise-Induced Angina  
    - **oldpeak**: ECG Depression  
    - **slope**: ST Slope  
    - **ca**: Major Vessels  
    - **thal**: Thalassemia  
    """)
