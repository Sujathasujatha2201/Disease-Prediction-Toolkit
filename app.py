import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

st.title("❤️ Disease Prediction Toolkit")
st.write("This app predicts the presence of **Heart Disease** using Machine Learning.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Prepare data
X = df.drop("target", axis=1)
y = df["target"]

# Train model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
st.write(f"### Model Accuracy: **{acc*100:.2f}%**")

st.subheader("🔍 Enter Patient Details for Prediction")

# Creating input fields dynamically
user_data = []
for column in X.columns:
    value = st.number_input(f"{column}", value=float(df[column].mean()))
    user_data.append(value)

# Convert to array
input_data = np.array(user_data).reshape(1, -1)
input_scaled = scaler.transform(input_data)

# Predict Button
if st.button("Predict"):
    result = model.predict(input_scaled)[0]

    if result == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
