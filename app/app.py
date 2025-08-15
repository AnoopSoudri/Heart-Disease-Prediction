import os
import joblib
import streamlit as st
import numpy as np

# ---- Load Model & Scaler ----
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error(
        "❌ Model or scaler file is missing.\n\n"
        "Please run:\n```\npython -m src.train --config configs/config.yaml\n```"
        "to generate them."
    )
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ---- Streamlit App ----
st.title(" Heart Disease Prediction")

# Define input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Serum Cholestoral in mg/dl", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Resting ECG results (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise", value=1.0, format="%.1f")
slope = st.selectbox("Slope of the peak exercise ST segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (1 = normal; 2 = fixed defect; 3 = reversible defect)", [1, 2, 3])

if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                             exang, oldpeak, slope, ca, thal]])
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f" The patient is likely to have a heart disease ({prob:.2%} probability).")
    else:
        st.success(f" The patient is unlikely to have a heart disease ({prob:.2%} probability).")
