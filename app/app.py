import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# --- Custom CSS for a professional look ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 100%);
    }
    h1 {
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        border-bottom: 3px solid #e74c3c;
        padding-bottom: 15px;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #9ca3af;
        font-size: 16px;
        margin-bottom: 30px;
    }
    div[data-testid="stForm"] {
        background-color: #1a1f2e;
        padding: 30px;
        border-radius: 12px;
        border: 1px solid #2d3444;
    }
    label {
        color: #d1d5db !important;
        font-weight: 600 !important;
    }
    .stButton button {
        background-color: #e74c3c;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 10px 30px;
        border: none;
        width: 100%;
        transition: background-color 0.2s;
    }
    .stButton button:hover {
        background-color: #c0392b;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load model, scaler, and column structure ---
model = joblib.load('model/heart_disease_model.pkl')
scaler = joblib.load('model/scaler.pkl')
model_columns = joblib.load('model/model_columns.pkl')

st.title("❤️ Heart Disease Prediction")
st.markdown('<p class="subtitle">Enter patient clinical details below for an instant risk assessment.</p>', unsafe_allow_html=True)

# --- Input form ---
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
    exang = st.selectbox("Exercise-Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-4)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia Type", options=[0, 1, 2, 3])

    submitted = st.form_submit_button("Predict")

# --- Prediction logic ---
if submitted:
    input_dict = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    input_df = pd.DataFrame([input_dict])

    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_columns]

    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    input_encoded[numerical_cols] = scaler.transform(input_encoded[numerical_cols])

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ **High risk of heart disease** — probability: {probability:.1%}")
    else:
        st.success(f"✅ **Low risk of heart disease** — probability: {probability:.1%}")