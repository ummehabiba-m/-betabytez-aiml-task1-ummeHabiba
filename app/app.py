import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="wide")

# --- Custom CSS: strong, explicit contrast ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stApp {
        background: radial-gradient(circle at 20% 0%, #1c2333 0%, #0a0d14 55%);
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* Hero section */
    .hero-wrap {
        text-align: center;
        padding: 40px 20px 30px 20px;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(231, 76, 60, 0.15);
        color: #ff6b5b !important;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 18px;
        border: 1px solid rgba(231, 76, 60, 0.4);
    }
    .hero-title {
        font-size: 46px !important;
        font-weight: 800 !important;
        color: #f5f6fa !important;
        margin: 0 0 14px 0 !important;
        font-family: 'Segoe UI', sans-serif;
    }
    .hero-subtitle {
        color: #aab2c0 !important;
        font-size: 18px !important;
        max-width: 620px;
        margin: 0 auto !important;
        line-height: 1.6 !important;
    }

    /* Card sections */
    .card {
        background: #141a29;
        padding: 32px 36px;
        border-radius: 16px;
        border: 1px solid #262f45;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    .card-title {
        color: #ffffff !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        margin-bottom: 4px !important;
    }
    .card-subtitle {
        color: #8b93a7 !important;
        font-size: 14px !important;
        margin-bottom: 24px !important;
    }

    /* Force readable text/labels everywhere inside our app */
    section.main label, section.main .stMarkdown p, section.main span {
        color: #d7dbe4 !important;
    }
    section.main label p {
        font-weight: 600 !important;
        font-size: 14px !important;
        color: #d7dbe4 !important;
    }

    /* Inputs */
    div[data-baseweb="input"] input,
    div[data-baseweb="select"] div {
        color: #f5f6fa !important;
    }
    div[data-testid="stNumberInput"] input {
        background-color: #1c2436 !important;
        color: #f5f6fa !important;
        border: 1px solid #333e58 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #1c2436 !important;
        border: 1px solid #333e58 !important;
        border-radius: 8px !important;
    }

    /* Button */
    .stButton button {
        background: linear-gradient(135deg, #ff5a4e, #e63946);
        color: white !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        border-radius: 10px !important;
        padding: 14px 40px !important;
        border: none !important;
        width: 100%;
        box-shadow: 0 4px 14px rgba(230, 57, 70, 0.4);
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #e63946, #c0392b);
    }

    /* Result banners */
    div[data-testid="stAlert"] {
        border-radius: 10px !important;
        font-size: 16px !important;
    }

    /* Risk factor pills */
    .risk-factor {
        background-color: #241a24;
        padding: 14px 20px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #e63946;
        color: #f0e6e6 !important;
        font-size: 15px;
    }
    .protective-factor {
        background-color: #16241d;
        border-left: 4px solid #2ecc71;
        color: #e6f0ea !important;
    }
    .disclaimer {
        color: #7d8598 !important;
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
        <span class="hero-badge">AI-Powered Risk Assessment</span>
        <h1 class="hero-title">❤️ Heart Disease Predictor</h1>
        <p class="hero-subtitle">Enter a patient's details below to get an instant risk 
        assessment, along with a reason behind the result.</p>
    </div>
""", unsafe_allow_html=True)

# --- Input form ---
with st.form("prediction_form"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
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

    st.markdown('</div>', unsafe_allow_html=True)
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
