import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 100%);
    }
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .hero {
        text-align: center;
        padding: 30px 10px 10px 10px;
    }
    .hero h1 {
        font-size: 42px;
        margin-bottom: 5px;
    }
    .hero p {
        color: #9ca3af;
        font-size: 17px;
        max-width: 600px;
        margin: 0 auto;
    }
    .section-card {
        background-color: #1a1f2e;
        padding: 25px 30px;
        border-radius: 12px;
        border: 1px solid #2d3444;
        margin-bottom: 20px;
    }
    label, .stSelectbox label, .stNumberInput label {
        color: #d1d5db !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    .stButton button {
        background-color: #e74c3c;
        color: white;
        font-weight: 700;
        font-size: 16px;
        border-radius: 8px;
        padding: 12px 40px;
        border: none;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #c0392b;
    }
    .risk-factor {
        background-color: #262d3d;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 4px solid #e74c3c;
        color: #e5e7eb;
    }
    .protective-factor {
        border-left: 4px solid #2ecc71;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load model files ---
model = joblib.load('model/heart_disease_model.pkl')
scaler = joblib.load('model/scaler.pkl')
model_columns = joblib.load('model/model_columns.pkl')

# --- Welcome / Hero section ---
st.markdown("""
    <div class="hero">
        <h1>❤️ Welcome to Heart Disease Predictor</h1>
        <p>Enter a patient's clinical details below to get an instant, AI-powered risk 
        assessment — along with a plain-language explanation of the key factors behind it.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Input form, laid out in columns to avoid heavy scrolling ---
with st.form("prediction_form"):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Patient Details")
    st.caption("You can type directly into any field, or use the +/- controls.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        ca = st.selectbox("Major Vessels Colored (0-4)", options=[0, 1, 2, 3, 4])

    with col2:
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        exang = st.selectbox("Exercise-Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2])
        thal = st.selectbox("Thalassemia Type", options=[0, 1, 2, 3])

    with col3:
        cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])

    st.markdown('</div>', unsafe_allow_html=True)
    submitted = st.form_submit_button("Get Prediction")

# --- Prediction + explanation logic ---
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

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ **High risk of heart disease** — estimated probability: {probability:.1%}")
    else:
        st.success(f"✅ **Low risk of heart disease** — estimated probability: {probability:.1%}")

    # --- Rule-based explanation of contributing factors ---
    st.markdown("#### Why this result?")
    st.caption("Based on the clinical factors most associated with heart disease in this dataset.")

    factors = []

    if cp in [1, 2]:
        factors.append(("risk", "Chest pain type reported is a type commonly associated with higher heart disease likelihood in this dataset."))
    if thalach > 150:
        factors.append(("risk", f"Max heart rate achieved ({thalach} bpm) is on the higher end, a pattern linked to positive cases in this dataset."))
    if exang == 1:
        factors.append(("protective", "No exercise-induced angina would typically lower risk — but this patient reported experiencing it, which raises concern."))
    if oldpeak > 2:
        factors.append(("risk", f"ST depression (oldpeak = {oldpeak}) is notably elevated, often associated with reduced blood flow during exercise."))
    if age > 55:
        factors.append(("risk", f"Age ({age}) is in a higher-risk bracket for cardiovascular conditions."))
    if chol > 240:
        factors.append(("risk", f"Cholesterol level ({chol} mg/dl) is above the commonly recommended threshold (240 mg/dl)."))
    if ca > 0:
        factors.append(("risk", f"{ca} major vessel(s) showed coloring on fluoroscopy, which is associated with increased risk."))

    if exang == 0:
        factors.append(("protective", "No exercise-induced angina was reported, which is generally a protective sign."))
    if oldpeak <= 1:
        factors.append(("protective", "ST depression is low, suggesting normal blood flow response during exercise."))
    if chol <= 200:
        factors.append(("protective", f"Cholesterol level ({chol} mg/dl) is within a healthy range."))

    if not factors:
        st.info("No single factor stands out strongly — the result reflects a combination of moderate values across all inputs.")
    else:
        for kind, text in factors:
            css_class = "risk-factor" if kind == "risk" else "risk-factor protective-factor"
            icon = "🔺" if kind == "risk" else "🟢"
            st.markdown(f'<div class="{css_class}">{icon} {text}</div>', unsafe_allow_html=True)

    st.caption("This explanation is based on general medical thresholds and patterns observed in the training "
               "data — it is not a substitute for professional medical diagnosis.")
    st.markdown('</div>', unsafe_allow_html=True)
