import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    @keyframes gradientMove {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes heartbeat {
        0%   { transform: scale(1); }
        14%  { transform: scale(1.2); }
        28%  { transform: scale(1); }
        42%  { transform: scale(1.2); }
        70%  { transform: scale(1); }
        100% { transform: scale(1); }
    }

    /* Lighter, softer animated gradient */
    .stApp {
        background: linear-gradient(-45deg, #dfe9ff, #e8dcff, #cfe0ff, #f3e8ff);
        background-size: 400% 400%;
        animation: gradientMove 20s ease infinite;
    }

    .stApp::before {
        content: "";
        position: fixed;
        top: -10%;
        left: -10%;
        width: 45%;
        height: 45%;
        background: radial-gradient(circle, rgba(106,63,181,0.10) 0%, rgba(106,63,181,0) 70%);
        border-radius: 50%;
        z-index: 0;
        pointer-events: none;
    }
    .stApp::after {
        content: "";
        position: fixed;
        bottom: -15%;
        right: -10%;
        width: 50%;
        height: 50%;
        background: radial-gradient(circle, rgba(58,123,213,0.12) 0%, rgba(58,123,213,0) 70%);
        border-radius: 50%;
        z-index: 0;
        pointer-events: none;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
        position: relative;
        z-index: 1;
    }

    /* Hero */
    .hero-wrap {
        text-align: center;
        padding: 20px 20px 10px 20px;
    }
    .heart-icon {
        display: inline-block;
        font-size: 44px;
        animation: heartbeat 1.6s ease-in-out infinite;
    }
    .hero-title {
        font-size: 46px !important;
        font-weight: 800 !important;
        color: #2b2350 !important;
        margin: 10px 0 14px 0 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .hero-subtitle {
        color: #4a4468 !important;
        font-size: 17px !important;
        max-width: 620px;
        margin: 0 auto !important;
        line-height: 1.6 !important;
    }

    /* Make the actual Streamlit form itself the card - avoids double boxes */
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.92) !important;
        padding: 32px 36px !important;
        border-radius: 18px !important;
        border: 1px solid rgba(255,255,255,0.7) !important;
        box-shadow: 0 15px 35px rgba(80, 60, 150, 0.15) !important;
        margin-bottom: 24px;
    }

    .card {
        background: rgba(255, 255, 255, 0.92);
        padding: 32px 36px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.7);
        margin-bottom: 24px;
        box-shadow: 0 15px 35px rgba(80, 60, 150, 0.15);
    }
    .card-title {
        color: #2b2350 !important;
        font-size: 24px !important;
        font-weight: 800 !important;
        margin-bottom: 4px !important;
    }
    .card-subtitle {
        color: #6b6480 !important;
        font-size: 14px !important;
        margin-bottom: 20px !important;
    }

    section.main label p {
        font-weight: 600 !important;
        font-size: 14px !important;
        color: #3a3560 !important;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #f4f2ff !important;
        color: #221c40 !important;
        border: 1.5px solid #c9c2f5 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #f4f2ff !important;
        border: 1.5px solid #c9c2f5 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] span {
        color: #221c40 !important;
        font-weight: 600 !important;
    }

    .stButton button {
        background: linear-gradient(135deg, #6a3fb5, #3a7bd5);
        color: white !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        border-radius: 10px !important;
        padding: 14px 40px !important;
        border: none !important;
        width: 100%;
        box-shadow: 0 6px 18px rgba(106, 63, 181, 0.35);
        transition: transform 0.15s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        background: linear-gradient(135deg, #5a2fa5, #2a6bc5);
    }

    /* Explicit high-contrast alert styling */
    div[data-testid="stAlertContentError"], div[data-testid="stAlertContentError"] * {
        color: #7a1a1a !important;
    }
    div[data-testid="stAlertContentSuccess"], div[data-testid="stAlertContentSuccess"] * {
        color: #145c32 !important;
    }
    div[data-testid="stNotification"] {
        border-radius: 10px !important;
    }

    .risk-factor {
        background-color: #fdeaea;
        padding: 14px 20px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 5px solid #d63031;
        color: #7a1a1a !important;
        font-size: 15px;
        font-weight: 600;
    }
    .protective-factor {
        background-color: #e7f9ee;
        border-left: 5px solid #27ae60;
        color: #145c32 !important;
    }
    .disclaimer {
        color: #6b6480 !important;
        font-size: 13px !important;
        margin-top: 18px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load model files ---
model = joblib.load('model/heart_disease_model.pkl')
scaler = joblib.load('model/scaler.pkl')
model_columns = joblib.load('model/model_columns.pkl')

# --- Hero section ---
st.markdown("""
    <div class="hero-wrap">
        <span class="heart-icon">🫀</span>
        <h1 class="hero-title">Heart Disease Predictor</h1>
        <p class="hero-subtitle">Enter a patient's clinical details below to get an instant risk 
        assessment, along with a plain-language explanation of the key factors behind the result.</p>
    </div>
""", unsafe_allow_html=True)

# --- Input form (the form container itself is styled as the card now) ---
with st.form("prediction_form"):
    st.markdown('<p class="card-title">Patient Details</p>', unsafe_allow_html=True)
    st.markdown('<p class="card-subtitle">Type directly into any field, or use the +/- controls.</p>', unsafe_allow_html=True)

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

    submitted = st.form_submit_button("Get Prediction")

# --- Prediction + explanation ---
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

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">Result</p>', unsafe_allow_html=True)

    if prediction == 1:
        st.error(f"⚠️ **High risk of heart disease** — estimated probability: {probability:.1%}")
    else:
        st.success(f"✅ **Low risk of heart disease** — estimated probability: {probability:.1%}")

    st.markdown('<p class="card-title" style="font-size:19px !important; margin-top:20px;">Why this result?</p>', unsafe_allow_html=True)
    st.markdown('<p class="card-subtitle">Based on the clinical factors most associated with heart disease in this dataset.</p>', unsafe_allow_html=True)

    factors = []
    if cp in [1, 2]:
        factors.append(("risk", "Chest pain type reported is a type commonly associated with higher heart disease likelihood in this dataset."))
    if thalach > 150:
        factors.append(("risk", f"Max heart rate achieved ({thalach} bpm) is on the higher end, a pattern linked to positive cases in this dataset."))
    if oldpeak > 2:
        factors.append(("risk", f"ST depression (oldpeak = {oldpeak}) is notably elevated, often associated with reduced blood flow during exercise."))
    if age > 55:
        factors.append(("risk", f"Age ({age}) is in a higher-risk bracket for cardiovascular conditions."))
    if chol > 240:
        factors.append(("risk", f"Cholesterol level ({chol} mg/dl) is above the commonly recommended threshold (240 mg/dl)."))
    if ca > 0:
        factors.append(("risk", f"{ca} major vessel(s) showed coloring on fluoroscopy, which is associated with increased risk."))
    if exang == 1:
        factors.append(("risk", "Exercise-induced angina was reported, which raises concern."))

    if exang == 0:
        factors.append(("protective", "No exercise-induced angina was reported, which is generally a protective sign."))
    if oldpeak <= 1:
        factors.append(("protective", "ST depression is low, suggesting a normal blood flow response during exercise."))
    if chol <= 200:
        factors.append(("protective", f"Cholesterol level ({chol} mg/dl) is within a healthy range."))
    if age <= 45:
        factors.append(("protective", f"Age ({age}) is in a lower-risk bracket for cardiovascular conditions."))

    if not factors:
        st.info("No single factor stands out strongly — the result reflects a combination of moderate values across all inputs.")
    else:
        for kind, text in factors:
            css_class = "risk-factor" if kind == "risk" else "risk-factor protective-factor"
            icon = "🔺" if kind == "risk" else "🟢"
            st.markdown(f'<div class="{css_class}">{icon} {text}</div>', unsafe_allow_html=True)

    st.markdown(
        '<p class="disclaimer">This explanation is based on general medical thresholds and patterns '
        'observed in the training data — it is not a substitute for professional medical diagnosis.</p>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
