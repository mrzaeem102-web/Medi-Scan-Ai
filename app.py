import os
from pathlib import Path
from textwrap import dedent

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# MEDI-SCAN AI - STREAMLIT APPLICATION
# ============================================================

st.set_page_config(
    page_title="Medi-Scan AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# EMBEDDED CSS
# ============================================================

st.markdown(
    dedent(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #07111f;
            --panel: #0d1b2a;
            --panel-2: #102235;
            --border: rgba(148,163,184,.18);
            --text: #f8fafc;
            --muted: #94a3b8;
            --primary: #38bdf8;
            --primary-2: #22d3ee;
            --success: #34d399;
            --danger: #fb7185;
            --warning: #fbbf24;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 0%, rgba(56,189,248,.10), transparent 30%),
                radial-gradient(circle at 90% 10%, rgba(34,211,238,.08), transparent 28%),
                var(--bg);
            color: var(--text);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #07111f 0%, #0a1727 100%);
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1rem;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 8px;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] > label {
            border: 1px solid transparent;
            border-radius: 14px;
            padding: 10px 13px;
            margin: 0;
            cursor: pointer;
            transition:
                transform .20s ease,
                background .20s ease,
                border-color .20s ease,
                box-shadow .20s ease;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            transform: translateX(5px);
            background: rgba(56,189,248,.08);
            border-color: rgba(56,189,248,.20);
        }

        [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
            background: linear-gradient(
                90deg,
                rgba(56,189,248,.18),
                rgba(34,211,238,.06)
            );
            border-color: rgba(56,189,248,.38);
            box-shadow: 0 8px 28px rgba(0,0,0,.18);
        }

        [data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
            color: #ffffff !important;
            font-weight: 700;
        }

        .brand {
            padding: 8px 4px 22px 4px;
            animation: fadeInUp .45s ease both;
        }

        .brand-title {
            font-size: 25px;
            font-weight: 800;
            letter-spacing: -.7px;
            color: #f8fafc;
        }

        .brand-title span {
            color: var(--primary);
        }

        .brand-subtitle {
            color: var(--muted);
            font-size: 12px;
            margin-top: 3px;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 38px;
            border-radius: 26px;
            margin-bottom: 24px;
            background:
                linear-gradient(135deg, rgba(15,38,59,.96), rgba(7,25,42,.92));
            border: 1px solid rgba(56,189,248,.18);
            box-shadow: 0 20px 60px rgba(0,0,0,.22);
            animation: fadeInUp .45s ease both;
        }

        .hero:after {
            content: "";
            position: absolute;
            width: 240px;
            height: 240px;
            right: -80px;
            top: -100px;
            border-radius: 50%;
            background: rgba(34,211,238,.10);
            filter: blur(4px);
        }

        .hero-kicker {
            color: var(--primary);
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 1.5px;
            margin-bottom: 12px;
        }

        .hero h1 {
            margin: 0;
            font-size: clamp(32px, 5vw, 58px);
            line-height: 1.03;
            letter-spacing: -2px;
            color: #fff;
        }

        .hero h1 span {
            color: var(--primary);
        }

        .hero p {
            max-width: 780px;
            color: #b9c7d8;
            font-size: 16px;
            line-height: 1.7;
            margin: 17px 0 0 0;
        }

        .section-title {
            font-size: 26px;
            font-weight: 800;
            margin: 8px 0 6px;
            color: #fff;
        }

        .section-subtitle {
            color: var(--muted);
            margin-bottom: 20px;
        }

        .metric-card,
        .disease-card,
        .result-card,
        .status-card,
        .info-card {
            background: linear-gradient(145deg, rgba(15,31,48,.96), rgba(10,25,40,.96));
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 22px;
            box-shadow: 0 14px 40px rgba(0,0,0,.14);
            animation: fadeInUp .45s ease both;
        }

        .metric-label {
            color: var(--muted);
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .8px;
        }

        .metric-value {
            color: #fff;
            font-size: 29px;
            font-weight: 800;
            margin-top: 8px;
        }

        .metric-note {
            color: #7dd3fc;
            font-size: 12px;
            margin-top: 5px;
        }

        .disease-icon {
            font-size: 31px;
            margin-bottom: 10px;
        }

        .disease-title {
            font-size: 19px;
            font-weight: 800;
            color: #fff;
        }

        .disease-text {
            color: var(--muted);
            line-height: 1.6;
            font-size: 13px;
            min-height: 65px;
        }

        .pill {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: .4px;
        }

        .pill-ready {
            color: #6ee7b7;
            background: rgba(52,211,153,.10);
            border: 1px solid rgba(52,211,153,.25);
        }

        .pill-missing {
            color: #fda4af;
            background: rgba(251,113,133,.10);
            border: 1px solid rgba(251,113,133,.25);
        }

        .result-positive {
            border-color: rgba(251,113,133,.35);
            background: linear-gradient(145deg, rgba(71,24,38,.94), rgba(32,20,32,.96));
        }

        .result-negative {
            border-color: rgba(52,211,153,.30);
            background: linear-gradient(145deg, rgba(12,51,48,.94), rgba(10,31,39,.96));
        }

        .result-title {
            font-size: 25px;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .result-text {
            color: #cbd5e1;
            line-height: 1.65;
        }

        .small-note {
            color: #64748b;
            font-size: 11px;
            line-height: 1.5;
        }

        .footer {
            margin-top: 35px;
            padding: 18px 0;
            color: #64748b;
            text-align: center;
            font-size: 12px;
            border-top: 1px solid var(--border);
        }

        div[data-testid="stForm"] {
            background: rgba(13,27,42,.55);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 20px;
        }

        .stButton > button,
        .stFormSubmitButton > button {
            border-radius: 12px;
            border: 1px solid rgba(56,189,248,.30);
            background: linear-gradient(135deg, #0284c7, #0891b2);
            color: white;
            font-weight: 800;
            transition: transform .18s ease, box-shadow .18s ease;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(14,165,233,.20);
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        </style>
        """
    ),
    unsafe_allow_html=True,
)


# ============================================================
# PATHS / MODEL LOADING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

EXPECTED_FILES = {
    "Diabetes model": "xgb_diabetes.pkl",
    "Diabetes scaler": "scaler_diabetes.pkl",
    "Anemia model": "lgb_anemia.pkl",
    "Anemia scaler": "scaler_anemia.pkl",
    "Thyroid model": "xgb_thyroid.pkl",
    "Thyroid scaler": "scaler_thyroid.pkl",
    "Thyroid target encoder": "thyroid_target_encoder.pkl",
}


def find_model_file(filename):
    """Look in models/ first, then project root."""
    candidates = [
        MODEL_DIR / filename,
        BASE_DIR / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


@st.cache_resource(show_spinner=False)
def load_models():
    loaded = {}
    errors = {}

    mapping = {
        "model_diabetes": "xgb_diabetes.pkl",
        "scaler_diabetes": "scaler_diabetes.pkl",
        "model_anemia": "lgb_anemia.pkl",
        "scaler_anemia": "scaler_anemia.pkl",
        "model_thyroid": "xgb_thyroid.pkl",
        "scaler_thyroid": "scaler_thyroid.pkl",
        "thyroid_target_encoder": "thyroid_target_encoder.pkl",
    }

    for key, filename in mapping.items():
        path = find_model_file(filename)

        if path is None:
            errors[key] = f"File not found: {filename}"
            continue

        try:
            loaded[key] = joblib.load(path)
        except Exception as exc:
            errors[key] = f"{filename}: {exc}"

    return loaded, errors


models, model_errors = load_models()


def model_ready(model_key, scaler_key):
    return (
        model_key in models
        and scaler_key in models
    )


DIABETES_READY = model_ready(
    "model_diabetes",
    "scaler_diabetes",
)

ANEMIA_READY = model_ready(
    "model_anemia",
    "scaler_anemia",
)

THYROID_READY = (
    model_ready("model_thyroid", "scaler_thyroid")
    and "thyroid_target_encoder" in models
)


# ============================================================
# HELPERS
# ============================================================

def render_html(html):
    st.markdown(dedent(html), unsafe_allow_html=True)


def metric_card(label, value, note=""):
    render_html(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """
    )


def disease_card(icon, title, description, ready):
    status = (
        '<span class="pill pill-ready">MODEL READY</span>'
        if ready
        else
        '<span class="pill pill-missing">MODEL UNAVAILABLE</span>'
    )

    render_html(
        f"""
        <div class="disease-card">
            <div class="disease-icon">{icon}</div>
            <div class="disease-title">{title}</div>
            <div class="disease-text">{description}</div>
            {status}
        </div>
        """
    )


def show_result(title, prediction, probability=None):
    """
    Display a generic result without making unsupported medical claims.
    """
    text = str(prediction).strip()

    negative_words = {
        "0",
        "no",
        "negative",
        "normal",
        "not diabetic",
        "not anemia",
        "not anaemia",
        "not thyroid",
        "healthy",
    }

    positive_words = {
        "1",
        "yes",
        "positive",
        "diabetic",
        "anemia",
        "anaemia",
        "abnormal",
        "hypothyroid",
        "hyperthyroid",
    }

    normalized = text.lower()

    if normalized in negative_words:
        positive = False
    elif normalized in positive_words:
        positive = True
    else:
        positive = None

    if positive is True:
        card_class = "result-positive"
        heading = "Model indicates a positive result"
        icon = "⚠️"
    elif positive is False:
        card_class = "result-negative"
        heading = "Model indicates a negative result"
        icon = "✅"
    else:
        card_class = "result-card"
        heading = "Prediction completed"
        icon = "🧠"

    probability_html = ""

    if probability is not None:
        try:
            probability_html = (
                f'<div class="metric-note">'
                f'Model confidence: {float(probability) * 100:.1f}%'
                f'</div>'
            )
        except Exception:
            probability_html = ""

    render_html(
        f"""
        <div class="result-card {card_class}">
            <div class="result-title">{icon} {heading}</div>
            <div class="result-text">
                <strong>{title}</strong><br>
                Predicted class: <strong>{text}</strong>
            </div>
            {probability_html}
            <br>
            <div class="small-note">
                This prediction is generated by the machine-learning model
                and is not a medical diagnosis. Consult a qualified healthcare
                professional for clinical interpretation.
            </div>
        </div>
        """
    )


def get_binary_prediction(model, scaled_data):
    prediction = model.predict(scaled_data)[0]

    probability = None

    if hasattr(model, "predict_proba"):
        try:
            probability = float(
                np.max(model.predict_proba(scaled_data)[0])
            )
        except Exception:
            probability = None

    return prediction, probability


def decode_prediction(prediction, encoder=None):
    if encoder is None:
        return prediction

    try:
        return encoder.inverse_transform(
            np.asarray([prediction]).astype(int)
        )[0]
    except Exception:
        return prediction


def safe_numeric(value):
    try:
        return float(value)
    except Exception:
        return 0.0


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    render_html(
        """
        <div class="brand">
            <div class="brand-title">Medi-Scan <span>AI</span></div>
            <div class="brand-subtitle">AI-powered health analytics dashboard</div>
        </div>
        """
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🩸 Diabetes",
            "🧬 Anemia",
            "🦋 Thyroid",
            "📊 Model Status",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    total_ready = sum(
        [
            DIABETES_READY,
            ANEMIA_READY,
            THYROID_READY,
        ]
    )

    if total_ready == 3:
        status_text = "All models ready"
        status_class = "pill-ready"
    elif total_ready > 0:
        status_text = f"{total_ready}/3 models ready"
        status_class = "pill-missing"
    else:
        status_text = "Models unavailable"
        status_class = "pill-missing"

    render_html(
        f"""
        <div class="status-card">
            <div class="metric-label">System status</div>
            <br>
            <span class="pill {status_class}">{status_text}</span>
        </div>
        """
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    render_html(
        """
        <div class="hero">
            <div class="hero-kicker">AI HEALTH ANALYTICS • MULTI-MODEL</div>
            <h1>Smarter health insights with <span>AI.</span></h1>
            <p>
                Medi-Scan AI brings together machine-learning models for
                diabetes, anemia and thyroid screening in one clean dashboard.
                Enter patient measurements to generate a model prediction.
            </p>
        </div>
        """
    )

    st.markdown(
        '<div class="section-title">System overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">Live availability of your trained models.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Models ready",
            f"{total_ready}/3",
            "Loaded successfully",
        )

    with c2:
        metric_card(
            "Diabetes",
            "READY" if DIABETES_READY else "OFFLINE",
            "XGBoost",
        )

    with c3:
        metric_card(
            "Anemia",
            "READY" if ANEMIA_READY else "OFFLINE",
            "LightGBM",
        )

    with c4:
        metric_card(
            "Thyroid",
            "READY" if THYROID_READY else "OFFLINE",
            "XGBoost",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Screening modules</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">Choose a module from the sidebar to start a prediction.</div>',
        unsafe_allow_html=True,
    )

    d1, d2, d3 = st.columns(3)

    with d1:
        disease_card(
            "🩸",
            "Diabetes",
            "Uses glucose, blood pressure, BMI, age and other clinical measurements.",
            DIABETES_READY,
        )

    with d2:
        disease_card(
            "🧬",
            "Anemia",
            "Uses hemoglobin, MCH, MCHC, MCV and gender-based input features.",
            ANEMIA_READY,
        )

    with d3:
        disease_card(
            "🦋",
            "Thyroid",
            "Uses demographic, medication, symptom and thyroid laboratory inputs.",
            THYROID_READY,
        )


# ============================================================
# DIABETES
# ============================================================

elif page == "🩸 Diabetes":

    render_html(
        """
        <div class="hero">
            <div class="hero-kicker">DIABETES SCREENING</div>
            <h1>Diabetes <span>Prediction</span></h1>
            <p>
                Enter the eight measurements used by the trained diabetes
                model. The model returns its predicted class and, when
                available, a confidence estimate.
            </p>
        </div>
        """
    )

    if not DIABETES_READY:
        st.error(
            "The diabetes model is not available. "
            "Add xgb_diabetes.pkl and scaler_diabetes.pkl to your "
            "models/ folder."
        )
    else:
        with st.form("diabetes_form"):

            st.markdown("### Patient measurements")

            col1, col2 = st.columns(2)

            with col1:
                pregnancies = st.number_input(
                    "Pregnancies",
                    min_value=0,
                    max_value=30,
                    value=1,
                    step=1,
                )

                glucose = st.number_input(
                    "Glucose",
                    min_value=0.0,
                    max_value=400.0,
                    value=120.0,
                    step=1.0,
                )

                blood_pressure = st.number_input(
                    "Blood Pressure",
                    min_value=0.0,
                    max_value=250.0,
                    value=70.0,
                    step=1.0,
                )

                skin_thickness = st.number_input(
                    "Skin Thickness",
                    min_value=0.0,
                    max_value=150.0,
                    value=20.0,
                    step=1.0,
                )

            with col2:
                insulin = st.number_input(
                    "Insulin",
                    min_value=0.0,
                    max_value=1000.0,
                    value=80.0,
                    step=1.0,
                )

                bmi = st.number_input(
                    "BMI",
                    min_value=0.0,
                    max_value=100.0,
                    value=25.0,
                    step=0.1,
                )

                pedigree = st.number_input(
                    "Diabetes Pedigree Function",
                    min_value=0.0,
                    max_value=3.0,
                    value=0.47,
                    step=0.01,
                )

                age = st.number_input(
                    "Age",
                    min_value=1,
                    max_value=120,
                    value=33,
                    step=1,
                )

            submitted = st.form_submit_button(
                "🔍 Predict Diabetes",
                use_container_width=True,
            )

        if submitted:
            try:
                data = np.array(
                    [
                        pregnancies,
                        glucose,
                        blood_pressure,
                        skin_thickness,
                        insulin,
                        bmi,
                        pedigree,
                        age,
                    ],
                    dtype=float,
                ).reshape(1, -1)

                scaled = models["scaler_diabetes"].transform(data)

                prediction, probability = get_binary_prediction(
                    models["model_diabetes"],
                    scaled,
                )

                show_result(
                    "Diabetes model",
                    prediction,
                    probability,
                )

            except Exception as exc:
                st.error(
                    "Diabetes prediction failed. "
                    f"Model/scaler feature mismatch or invalid model file: {exc}"
                )


# ============================================================
# ANEMIA
# ============================================================

elif page == "🧬 Anemia":

    render_html(
        """
        <div class="hero">
            <div class="hero-kicker">ANEMIA SCREENING</div>
            <h1>Anemia <span>Prediction</span></h1>
            <p>
                Enter the laboratory values used by the trained anemia model.
            </p>
        </div>
        """
    )

    if not ANEMIA_READY:
        st.error(
            "The anemia model is not available. "
            "Add lgb_anemia.pkl and scaler_anemia.pkl to your models/ folder."
        )
    else:

        with st.form("anemia_form"):

            st.markdown("### Patient measurements")

            col1, col2 = st.columns(2)

            with col1:
                gender = st.selectbox(
                    "Gender",
                    ["Female", "Male"],
                )

                hemoglobin = st.number_input(
                    "Hemoglobin",
                    min_value=0.0,
                    max_value=30.0,
                    value=12.0,
                    step=0.1,
                )

                mch = st.number_input(
                    "MCH",
                    min_value=0.0,
                    max_value=60.0,
                    value=27.0,
                    step=0.1,
                )

            with col2:
                mchc = st.number_input(
                    "MCHC",
                    min_value=0.0,
                    max_value=60.0,
                    value=32.0,
                    step=0.1,
                )

                mcv = st.number_input(
                    "MCV",
                    min_value=0.0,
                    max_value=150.0,
                    value=85.0,
                    step=0.1,
                )

            submitted = st.form_submit_button(
                "🔍 Predict Anemia",
                use_container_width=True,
            )

        if submitted:
            try:
                gender_value = 1 if gender == "Male" else 0

                data = np.array(
                    [
                        gender_value,
                        hemoglobin,
                        mch,
                        mchc,
                        mcv,
                    ],
                    dtype=float,
                ).reshape(1, -1)

                scaled = models["scaler_anemia"].transform(data)

                prediction, probability = get_binary_prediction(
                    models["model_anemia"],
                    scaled,
                )

                show_result(
                    "Anemia model",
                    prediction,
                    probability,
                )

            except Exception as exc:
                st.error(
                    "Anemia prediction failed. "
                    f"Model/scaler feature mismatch or invalid model file: {exc}"
                )


# ============================================================
# THYROID
# ============================================================

elif page == "🦋 Thyroid":

    render_html(
        """
        <div class="hero">
            <div class="hero-kicker">THYROID SCREENING</div>
            <h1>Thyroid <span>Prediction</span></h1>
            <p>
                Enter the clinical, medication, symptom and laboratory
                features used by the trained thyroid model.
            </p>
        </div>
        """
    )

    if not THYROID_READY:
        st.error(
            "The thyroid model is not available. "
            "Add xgb_thyroid.pkl, scaler_thyroid.pkl and "
            "thyroid_target_encoder.pkl to your models/ folder."
        )
    else:

        with st.form("thyroid_form"):

            st.markdown("### Patient information")

            c1, c2, c3 = st.columns(3)

            with c1:
                age = st.number_input(
                    "Age",
                    min_value=1,
                    max_value=120,
                    value=35,
                    step=1,
                )

            with c2:
                sex = st.selectbox(
                    "Sex",
                    ["Female", "Male"],
                )

            with c3:
                sick = st.checkbox("Sick")

            st.markdown("### Medication and history")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                on_thyroxine = st.checkbox("On Thyroxine")

            with c2:
                query_on_thyroxine = st.checkbox(
                    "Query on Thyroxine"
                )

            with c3:
                on_antithyroid_medication = st.checkbox(
                    "On Antithyroid Medication"
                )

            with c4:
                pregnant = st.checkbox("Pregnant")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                thyroid_surgery = st.checkbox(
                    "Thyroid Surgery"
                )

            with c2:
                i131_treatment = st.checkbox(
                    "I-131 Treatment"
                )

            with c3:
                query_hypothyroid = st.checkbox(
                    "Query Hypothyroid"
                )

            with c4:
                query_hyperthyroid = st.checkbox(
                    "Query Hyperthyroid"
                )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                lithium = st.checkbox("Lithium")

            with c2:
                goitre = st.checkbox("Goitre")

            with c3:
                tumor = st.checkbox("Tumor")

            with c4:
                hypopituitary = st.checkbox(
                    "Hypopituitary"
                )

            psych = st.checkbox("Psychological condition")

            st.markdown("### Thyroid laboratory values")

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                tsh = st.number_input(
                    "TSH",
                    min_value=0.0,
                    value=2.0,
                    step=0.01,
                )

            with c2:
                t3 = st.number_input(
                    "T3",
                    min_value=0.0,
                    value=1.8,
                    step=0.01,
                )

            with c3:
                tt4 = st.number_input(
                    "TT4",
                    min_value=0.0,
                    value=100.0,
                    step=0.1,
                )

            with c4:
                t4u = st.number_input(
                    "T4U",
                    min_value=0.0,
                    value=1.0,
                    step=0.01,
                )

            with c5:
                fti = st.number_input(
                    "FTI",
                    min_value=0.0,
                    value=100.0,
                    step=0.1,
                )

            submitted = st.form_submit_button(
                "🔍 Predict Thyroid Class",
                use_container_width=True,
            )

        if submitted:
            try:
                sex_value = 1 if sex == "Male" else 0

                # IMPORTANT:
                # This order matches the 21-feature thyroid training
                # pipeline supplied earlier for train_models.py.
                data = np.array(
                    [
                        age,
                        sex_value,
                        int(on_thyroxine),
                        int(query_on_thyroxine),
                        int(on_antithyroid_medication),
                        int(sick),
                        int(pregnant),
                        int(thyroid_surgery),
                        int(i131_treatment),
                        int(query_hypothyroid),
                        int(query_hyperthyroid),
                        int(lithium),
                        int(goitre),
                        int(tumor),
                        int(hypopituitary),
                        int(psych),
                        tsh,
                        t3,
                        tt4,
                        t4u,
                        fti,
                    ],
                    dtype=float,
                ).reshape(1, -1)

                scaled = models["scaler_thyroid"].transform(data)

                raw_prediction, probability = get_binary_prediction(
                    models["model_thyroid"],
                    scaled,
                )

                decoded = decode_prediction(
                    raw_prediction,
                    models.get("thyroid_target_encoder"),
                )

                show_result(
                    "Thyroid model",
                    decoded,
                    probability,
                )

            except Exception as exc:
                st.error(
                    "Thyroid prediction failed. "
                    f"Model/scaler feature mismatch or invalid model file: {exc}"
                )


# ============================================================
# MODEL STATUS
# ============================================================

elif page == "📊 Model Status":

    render_html(
        """
        <div class="hero">
            <div class="hero-kicker">MODEL MONITORING</div>
            <h1>Model <span>Status</span></h1>
            <p>
                Check every model artifact individually. A missing scaler or
                encoder is reported separately instead of marking the whole
                system as failed.
            </p>
        </div>
        """
    )

    for display_name, filename in EXPECTED_FILES.items():

        path = find_model_file(filename)

        if path is not None and filename not in {
            str(error).split(":")[0]
            for error in model_errors.values()
        }:
            status = "READY"
            css = "pill-ready"
            detail = f"Found at {path.relative_to(BASE_DIR)}"
        else:
            status = "MISSING / ERROR"
            css = "pill-missing"

            matching_error = None

            for error in model_errors.values():
                if filename in error:
                    matching_error = error
                    break

            detail = matching_error or f"File not found: {filename}"

        render_html(
            f"""
            <div class="status-card">
                <strong>{display_name}</strong>
                <br><br>
                <span class="pill {css}">{status}</span>
                <br><br>
                <div class="small-note">{detail}</div>
            </div>
            """
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if model_errors:
        st.warning(
            "Some model artifacts could not be loaded. "
            "Check the messages above and confirm your models/ folder "
            "contains the files produced by the training notebook."
        )
    else:
        st.success(
            "All seven expected model artifacts loaded successfully."
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    render_html(
        """
        <div class="hero">
            <div class="hero-kicker">ABOUT MEDI-SCAN AI</div>
            <h1>Healthcare analytics, <span>simplified.</span></h1>
            <p>
                Medi-Scan AI is an educational machine-learning dashboard
                demonstrating how trained classification models can be
                integrated into an interactive Streamlit application.
            </p>
        </div>

        <div class="info-card">
            <div class="section-title">Models</div>
            <p class="result-text">
                • Diabetes — XGBoost<br>
                • Anemia — LightGBM<br>
                • Thyroid — XGBoost
            </p>
        </div>

        <br>

        <div class="info-card">
            <div class="section-title">Important notice</div>
            <p class="result-text">
                The predictions shown by this application are generated by
                machine-learning models for educational and demonstration
                purposes. They are not a substitute for professional medical
                evaluation, diagnosis or treatment.
            </p>
        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="footer">
        Medi-Scan AI • Machine Learning Health Analytics<br>
        Educational screening dashboard — not a medical diagnostic system.
    </div>
    """
)
