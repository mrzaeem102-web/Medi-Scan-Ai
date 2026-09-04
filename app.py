# ============================================================
# MEDI-SCAN AI
# Multi-Disease AI Assessment Dashboard
# Streamlit Application
# ============================================================

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Medi-Scan AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CUSTOM PROFESSIONAL UI / CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(37, 99, 235, 0.15),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(8, 145, 178, 0.12),
                transparent 28%
            ),
            #07111f;
        color: #e5edf7;
    }

    /* Sidebar */

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #081525 0%,
            #0b1728 100%
        );

        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }

    [data-testid="stSidebar"] * {
        color: #dbe7f5;
    }

    /* Main hero */

    .hero {
        padding: 35px 38px;
        border-radius: 25px;

        background:
            linear-gradient(
                135deg,
                rgba(15, 39, 67, 0.97),
                rgba(7, 25, 43, 0.96)
            );

        border: 1px solid rgba(96, 165, 250, 0.18);

        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.25);

        margin-bottom: 25px;
    }

    .hero-kicker {
        color: #67e8f9;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-title {
        color: #f8fbff;
        font-size: 46px;
        font-weight: 800;
        line-height: 1.05;
        margin: 0;
    }

    .hero-text {
        color: #9fb1c7;
        font-size: 15px;
        line-height: 1.7;
        max-width: 850px;
        margin-top: 15px;
    }

    /* Section headings */

    .section-title {
        color: #f1f5f9;
        font-size: 23px;
        font-weight: 800;
        margin-top: 28px;
        margin-bottom: 7px;
    }

    .section-subtitle {
        color: #8fa3ba;
        font-size: 13px;
        margin-bottom: 18px;
    }

    /* Metric cards */

    .metric-card {
        min-height: 125px;
        padding: 20px;

        border-radius: 18px;

        background:
            linear-gradient(
                145deg,
                rgba(15, 35, 57, 0.92),
                rgba(8, 24, 41, 0.92)
            );

        border: 1px solid rgba(148, 163, 184, 0.12);

        box-shadow:
            0 10px 35px rgba(0, 0, 0, 0.18);
    }

    .metric-label {
        color: #91a5bc;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: #f8fbff;
        font-size: 27px;
        font-weight: 800;
        margin-top: 9px;
    }

    /* Disease cards */

    .disease-card {
        padding: 24px;

        border-radius: 20px;

        background:
            linear-gradient(
                145deg,
                rgba(15, 35, 57, 0.94),
                rgba(8, 24, 41, 0.94)
            );

        border: 1px solid rgba(96, 165, 250, 0.13);

        box-shadow:
            0 12px 35px rgba(0, 0, 0, 0.18);

        min-height: 205px;
    }

    .disease-icon {
        font-size: 34px;
        margin-bottom: 8px;
    }

    .disease-name {
        color: #f8fbff;
        font-size: 20px;
        font-weight: 800;
    }

    .disease-description {
        color: #91a5bc;
        font-size: 13px;
        line-height: 1.6;
        margin-top: 9px;
    }

    /* Result cards */

    .result-card {
        padding: 25px;

        border-radius: 20px;

        background:
            linear-gradient(
                145deg,
                rgba(13, 31, 51, 0.97),
                rgba(7, 23, 39, 0.97)
            );

        border: 1px solid rgba(96, 165, 250, 0.16);

        margin-top: 20px;
    }

    .result-title {
        color: #9fb1c7;
        font-size: 12px;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .result-value {
        color: #f8fbff;
        font-size: 30px;
        font-weight: 800;
        margin-top: 7px;
    }

    /* Info box */

    .info-box {
        padding: 17px 20px;

        border-radius: 15px;

        background: rgba(37, 99, 235, 0.08);

        border: 1px solid rgba(96, 165, 250, 0.20);

        color: #b9cbe0;

        font-size: 13px;

        line-height: 1.6;

        margin-top: 20px;
    }

    .warning-box {
        padding: 17px 20px;

        border-radius: 15px;

        background: rgba(245, 158, 11, 0.08);

        border: 1px solid rgba(245, 158, 11, 0.20);

        color: #fbd38d;

        font-size: 13px;

        line-height: 1.6;

        margin-top: 20px;
    }

    /* Buttons */

    .stButton > button,
    .stFormSubmitButton > button {

        width: 100%;

        min-height: 48px;

        border-radius: 13px;

        border: 1px solid rgba(96, 165, 250, 0.25);

        background:
            linear-gradient(
                135deg,
                #2563eb,
                #0891b2
            );

        color: white;

        font-weight: 800;

        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {

        transform: translateY(-2px);

        box-shadow:
            0 12px 30px rgba(37, 99, 235, 0.25);
    }

    /* Inputs */

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {

        background-color: #0d1c2e !important;

        border-color:
            rgba(148, 163, 184, 0.18) !important;
    }

    /* Footer */

    .footer {
        text-align: center;

        color: #64748b;

        font-size: 11px;

        padding:
            40px 0 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def get_model_path(filename):
    """
    Returns the absolute path of a model/artifact
    stored beside app.py.
    """
    return BASE_DIR / filename


# ============================================================
# 4. LOAD MACHINE LEARNING MODELS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_models():

    model_files = {

        "scaler_thyroid":
            "scaler_thyroid.pkl",

        "scaler_diabetes":
            "scaler_diabetes.pkl",

        "scaler_anemia":
            "scaler_anemia.pkl",

        "thyroid_target_encoder":
            "thyroid_target_encoder.pkl",

        "model_thyroid":
            "xgb_thyroid.pkl",

        "model_diabetes":
            "xgb_diabetes.pkl",

        "model_anemia":
            "lgb_anemia.pkl",
    }

    models = {}
    errors = {}

    for key, filename in model_files.items():

        path = get_model_path(filename)

        if not path.exists():

            errors[key] = (
                f"Missing file: {filename}"
            )

            continue

        try:

            models[key] = joblib.load(path)

        except Exception as error:

            errors[key] = (
                f"Could not load {filename}: {error}"
            )

    return models, errors


models, model_errors = load_models()


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def metric_card(label, value):

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                {label}
            </div>

            <div class="metric-value">
                {value}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


def create_binary_prediction(
    model,
    scaler,
    features
):

    scaled_features = scaler.transform(
        features
    )

    prediction = model.predict(
        scaled_features
    )[0]

    prediction = int(prediction)

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = (
            model.predict_proba(
                scaled_features
            )[0]
        )

        if len(probabilities) > 1:

            probability = float(
                probabilities[1]
            )

        else:

            probability = float(
                probabilities[0]
            )

    else:

        probability = 0.0

    return prediction, probability


# ============================================================
# 6. THYROID PREPROCESSING
# ============================================================

def preprocess_thyroid(data):

    sex_value = (
        0
        if data["sex"].upper() == "F"
        else 1
    )

    features = [

        data["age"],

        sex_value,

        int(data["on_thyroxine"]),

        int(data["query_on_thyroxine"]),

        int(
            data[
                "on_antithyroid_medication"
            ]
        ),

        int(data["sick"]),

        int(data["pregnant"]),

        int(data["thyroid_surgery"]),

        int(data["I131_treatment"]),

        int(data["query_hypothyroid"]),

        int(data["query_hyperthyroid"]),

        int(data["lithium"]),

        int(data["goitre"]),

        int(data["tumor"]),

        int(data["hypopituitary"]),

        int(data["psych"]),

        data["TSH"],

        data["T3"],

        data["TT4"],

        data["T4U"],

        data["FTI"],

        0,
    ]

    return np.asarray(
        features,
        dtype=float
    ).reshape(1, -1)


# ============================================================
# 7. RESULT DISPLAY
# ============================================================

def show_result(
    title,
    prediction,
    probability
):

    st.markdown(
        f"""
        <div class="result-card">

            <div class="result-title">
                {title}
            </div>

            <div class="result-value">
                {prediction}
            </div>

            <div style="
                color:#91a5bc;
                margin-top:8px;
                font-size:13px;
            ">

                Model confidence:
                <strong>
                    {probability:.1%}
                </strong>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(
        min(
            max(probability, 0.0),
            1.0
        )
    )


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:25px;
            font-weight:800;
            color:#f8fbff;
        ">
            🩺 Medi-Scan AI
        </div>

        <div style="
            color:#91a5bc;
            font-size:12px;
            margin-top:5px;
        ">
            Multi-Disease AI Platform
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

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
    )

    st.divider()

    st.markdown(
        "**SYSTEM STATUS**"
    )

    if model_errors:

        st.error(
            "Some model files are unavailable."
        )

    else:

        st.success(
            "All models loaded successfully."
        )

    st.caption(
        "Educational / research use only."
    )


# ============================================================
# 9. MAIN HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-kicker">
            AI HEALTH ANALYTICS • MULTI-MODEL
        </div>

        <div class="hero-title">
            Medi-Scan AI
        </div>

        <div class="hero-text">

            Intelligent health-assessment dashboard
            powered by machine-learning models for
            Diabetes, Anemia, and Thyroid prediction.

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 10. DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'A unified AI dashboard for three predictive models.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        metric_card(
            "AI MODELS",
            "03"
        )

    with col2:

        metric_card(
            "DIABETES",
            "READY"
        )

    with col3:

        metric_card(
            "ANEMIA",
            "READY"
        )

    with col4:

        metric_card(
            "THYROID",
            "READY"
        )


    st.markdown(
        '<div class="section-title">'
        'Available Assessments'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Select a module from the sidebar to start an assessment.'
        '</div>',
        unsafe_allow_html=True,
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.markdown(
            """
            <div class="disease-card">

                <div class="disease-icon">
                    🩸
                </div>

                <div class="disease-name">
                    Diabetes
                </div>

                <div class="disease-description">

                    Analyze glucose, blood pressure,
                    BMI, insulin, age and other
                    diabetes-related measurements.

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    with c2:

        st.markdown(
            """
            <div class="disease-card">

                <div class="disease-icon">
                    🧬
                </div>

                <div class="disease-name">
                    Anemia
                </div>

                <div class="disease-description">

                    Analyze hemoglobin, MCH,
                    MCHC, MCV and gender-based
                    input features.

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    with c3:

        st.markdown(
            """
            <div class="disease-card">

                <div class="disease-icon">
                    🦋
                </div>

                <div class="disease-name">
                    Thyroid
                </div>

                <div class="disease-description">

                    Analyze thyroid-related
                    indicators including TSH,
                    T3, TT4, T4U and FTI.

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    st.markdown(
        """
        <div class="warning-box">

            ⚠️ <strong>Important:</strong>
            Medi-Scan AI is an educational and
            research demonstration. Model predictions
            can be incorrect and must not be treated
            as a medical diagnosis.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 11. DIABETES PAGE
# ============================================================

elif page == "🩸 Diabetes":

    st.markdown(
        '<div class="section-title">'
        'Diabetes Risk Assessment'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Enter the measurements required by the trained diabetes model.'
        '</div>',
        unsafe_allow_html=True,
    )


    with st.form("diabetes_form"):

        left, right = st.columns(2)


        with left:

            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0,
                max_value=30,
                value=2,
                step=1,
            )

            glucose = st.number_input(
                "Glucose",
                min_value=0.0,
                max_value=500.0,
                value=120.0,
            )

            blood_pressure = st.number_input(
                "Blood Pressure",
                min_value=0.0,
                max_value=300.0,
                value=70.0,
            )

            skin_thickness = st.number_input(
                "Skin Thickness",
                min_value=0.0,
                max_value=150.0,
                value=20.0,
            )


        with right:

            insulin = st.number_input(
                "Insulin",
                min_value=0.0,
                max_value=1000.0,
                value=79.0,
            )

            bmi = st.number_input(
                "BMI",
                min_value=0.0,
                max_value=100.0,
                value=25.5,
            )

            pedigree = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=5.0,
                value=0.52,
            )

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=33,
                step=1,
            )


        submit = st.form_submit_button(
            "🔍 Run Diabetes Assessment"
        )


    if submit:

        if (
            "model_diabetes"
            not in models
            or
            "scaler_diabetes"
            not in models
        ):

            st.error(
                "Diabetes model or scaler is missing."
            )

        else:

            try:

                features = np.array(
                    [[

                        pregnancies,
                        glucose,
                        blood_pressure,
                        skin_thickness,
                        insulin,
                        bmi,
                        pedigree,
                        age,

                    ]],
                    dtype=float,
                )


                prediction, probability = (
                    create_binary_prediction(
                        models["model_diabetes"],
                        models["scaler_diabetes"],
                        features,
                    )
                )


                if prediction == 1:

                    result = (
                        "Higher model-predicted risk"
                    )

                else:

                    result = (
                        "Lower model-predicted risk"
                    )


                show_result(
                    "Diabetes Model Output",
                    result,
                    probability,
                )


                if prediction
