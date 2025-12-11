
        }
        .card {
            padding: 15px;
            border-radius: 10px;
            background-color: #f1faee;
            box-shadow: 0px 0px 10px #e0e0e0;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ==== TITLE ====
st.markdown("<h1 class='title'>❤️ Heart Attack Risk Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>A Machine Learning-based Smart Health Assessment Tool</p>", unsafe_allow_html=True)

st.write("___")

# ==== SIDEBAR ====
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📝 Predict Risk", "📘 About Dataset", "📄 Project Info"])

# ==== LOAD DATA ====
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

# Prepare model only once
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# ===========================
#        PAGE 1 – HOME
# ===========================
if page == "🏠 Home":
    st.markdown("### 📊 Project Summary")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'>"
                    "<h4>💡 What This App Does</h4>"
                    "Predicts the possibility of a <b>heart attack</b> using a trained ML model."
                    "</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div class='card'>"
                    "<h4>🤖 ML Model Accuracy</h4>"
                    f"<h2 style='color:#e63946;'>{acc*100:.2f}%</h2>"
                    "</div>", unsafe_allow_html=True)

    st.markdown("### 🧠 Why Heart Attack Prediction?")
    st.write("""
    - Early detection can save lives  
    - Helps identify risk factors  
    - Supports doctors and health assessment  
    - Useful for personal health awareness  
    """)

    st.image("https://www.narayanahealth.org/sites/default/files/2022-07/Heart-Attack.jpg", use_column_width=True)

# ===========================
#   PAGE 2 – PREDICT RISK
# ===========================
if page == "📝 Predict Risk":
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
        ca = st.selectbox("Number of Major Vessels (0–3)", [0,1,2,3])

    cp = st.selectbox("Chest Pain Type", 
                      ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])

    restecg = st.selectbox("Resting ECG Result", 
                           ["Normal", "ST-T abnormality", "Left ventricular hypertrophy"])

    slope = st.selectbox("Slope of ST Segment", 
                         ["Upsloping", "Flat", "Downsloping"])

    thal = st.selectbox("Thalassemia Result", 
                        ["Normal", "Fixed Defect", "Reversible Defect"])

    # Convert to numeric
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    cp_map = {"Typical Angina":0, "Atypical Angina":1, "Non-anginal Pain":2, "Asymptomatic":3}
    restecg_map = {"Normal":0, "ST-T abnormality":1, "Left ventricular hypertrophy":2}
    slope_map = {"Upsloping":0, "Flat":1, "Downsloping":2}
    thal_map = {"Normal":1, "Fixed Defect":2, "Reversible Defect":3}

    user_data = np.array([[age, sex, cp_map[cp], trestbps, chol, fbs,
                           restecg_map[restecg], thalach, exang, oldpeak,
                           slope_map[slope], ca, thal_map[thal]]])

    scaled_input = scaler.transform(user_data)

    if st.button("🚀 Predict Risk"):
        result = model.predict(scaled_input)[0]

        if result == 1:
            st.error("⚠️ **High Risk of Heart Attack**")
            st.warning("Please consult a cardiologist immediately.")
        else:
            st.success("✅ **Low Risk of Heart Attack**")
            st.info("Maintain a healthy lifestyle!")

# ===========================
#   PAGE 3 – ABOUT DATASET
# ===========================
if page == "📘 About Dataset":
    st.header("📘 Dataset Information")
    st.write("This dataset contains medical data used to predict heart attack risk.")
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

# ===========================
#   PAGE 4 – PROJECT INFO
# ===========================
if page == "📄 Project Info":
    st.header("📄 Project Details")

    st.write("""
    ### 🏆 Why This Project Is Best?
    - Professional UI  
    - Machine Learning model  
    - User-friendly inputs  
    - Real-time prediction  
    - Clean structure  
    """)

    st.write("""
    ### 📚 Ideal For:
    - B.Tech / MCA / MSc Projects  
    - Hackathons  
    - Resume & Portfolio  
    - Internship Submission  
    """)

    st.success("If you want, I can also generate your **full project report**, PPT, UML diagrams & documentation.")
