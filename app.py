import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# ------------------------------------------------------------------
# 1. FastAPI App Initialization & CORS Configuration
# ------------------------------------------------------------------
app = FastAPI(
    title="Multi-Disease Assessment API",
    description="Backend API for Thyroid, Diabetes, and Anemia prediction models.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# 2. Load Pretrained Models & Artifacts
# ------------------------------------------------------------------
try:
    # Scalers & Encoders
    scaler_thyroid = joblib.load("scaler_thyroid.pkl")
    scaler_diabetes = joblib.load("scaler_diabetes.pkl")
    scaler_anemia = joblib.load("scaler_anemia.pkl")
    thyroid_target_encoder = joblib.load("thyroid_target_encoder.pkl")

    # ML Models (XGBoost / LightGBM)
    model_thyroid = joblib.load("xgb_thyroid.pkl")
    model_diabetes = joblib.load("xgb_diabetes.pkl")
    model_anemia = joblib.load("lgb_anemia.pkl")
    
    print("All ML artifacts loaded successfully!")
except Exception as e:
    print(f"Error loading model artifacts: {e}")

# ------------------------------------------------------------------
# 3. Request Schemas (Pydantic Models)
# ------------------------------------------------------------------
class DiabetesInput(BaseModel):
    Pregnancies: int = Field(..., example=2)
    Glucose: float = Field(..., example=120.0)
    BloodPressure: float = Field(..., example=70.0)
    SkinThickness: float = Field(..., example=20.0)
    Insulin: float = Field(..., example=79.0)
    BMI: float = Field(..., example=25.5)
    DiabetesPedigreeFunction: float = Field(..., example=0.52)
    Age: int = Field(..., example=33)

class AnemiaInput(BaseModel):
    Gender: int = Field(..., example=1, description="0 for Female, 1 for Male")
    Hemoglobin: float = Field(..., example=13.5)
    MCH: float = Field(..., example=27.0)
    MCHC: float = Field(..., example=32.0)
    MCV: float = Field(..., example=85.0)

class ThyroidInput(BaseModel):
    age: float = Field(..., example=45.0)
    sex: str = Field(..., example="F", description="'F' or 'M'")
    on_thyroxine: bool = Field(False)
    query_on_thyroxine: bool = Field(False)
    on_antithyroid_medication: bool = Field(False)
    sick: bool = Field(False)
    pregnant: bool = Field(False)
    thyroid_surgery: bool = Field(False)
    I131_treatment: bool = Field(False)
    query_hypothyroid: bool = Field(False)
    query_hyperthyroid: bool = Field(False)
    lithium: bool = Field(False)
    goitre: bool = Field(False)
    tumor: bool = Field(False)
    hypopituitary: bool = Field(False)
    psych: bool = Field(False)
    TSH: float = Field(..., example=2.5)
    T3: float = Field(..., example=1.8)
    TT4: float = Field(..., example=100.0)
    T4U: float = Field(..., example=1.0)
    FTI: float = Field(..., example=100.0)
    referral_source: str = Field("other", example="other")

class ComprehensiveInput(BaseModel):
    diabetes: Optional[DiabetesInput] = None
    anemia: Optional[AnemiaInput] = None
    thyroid: Optional[ThyroidInput] = None

# ------------------------------------------------------------------
# 4. Helper Functions for Feature Engineering
# ------------------------------------------------------------------
def preprocess_thyroid_features(data: ThyroidInput) -> np.ndarray:
    """Formats raw Thyroid JSON into feature array expected by scaler."""
    sex_val = 0 if data.sex.upper() == 'F' else 1
    
    # Static list matching Thyroid-Dataset feature vector ordering
    features = [
        data.age,
        sex_val,
        int(data.on_thyroxine),
        int(data.query_on_thyroxine),
        int(data.on_antithyroid_medication),
        int(data.sick),
        int(data.pregnant),
        int(data.thyroid_surgery),
        int(data.I131_treatment),
        int(data.query_hypothyroid),
        int(data.query_hyperthyroid),
        int(data.lithium),
        int(data.goitre),
        int(data.tumor),
        int(data.hypopituitary),
        int(data.psych),
        data.TSH,
        data.T3,
        data.TT4,
        data.T4U,
        data.FTI,
        0  # referral_source placeholder/encoded integer
    ]
    return np.array(features).reshape(1, -1)

# ------------------------------------------------------------------
# 5. API Endpoints
# ------------------------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "online", "message": "Multi-Disease AI Assessment Service Running"}

@app.post("/predict/diabetes")
def predict_diabetes(payload: DiabetesInput):
    features = np.array([[
        payload.Pregnancies, payload.Glucose, payload.BloodPressure,
        payload.SkinThickness, payload.Insulin, payload.BMI,
        payload.DiabetesPedigreeFunction, payload.Age
    ]])
    scaled_features = scaler_diabetes.transform(features)
    prob = float(model_diabetes.predict_proba(scaled_features)[0][1])
    prediction = int(prob >= 0.5)
    
    return {
        "disease": "Diabetes",
        "prediction": "Positive" if prediction == 1 else "Negative",
        "risk_probability": round(prob, 4)
    }

@app.post("/predict/anemia")
def predict_anemia(payload: AnemiaInput):
    features = np.array([[
        payload.Gender, payload.Hemoglobin, payload.MCH, payload.MCHC, payload.MCV
    ]])
    scaled_features = scaler_anemia.transform(features)
    prob = float(model_anemia.predict_proba(scaled_features)[0][1])
    prediction = int(prob >= 0.5)
    
    return {
        "disease": "Anemia",
        "prediction": "Anemic" if prediction == 1 else "Normal",
        "risk_probability": round(prob, 4)
    }

@app.post("/predict/thyroid")
def predict_thyroid(payload: ThyroidInput):
    raw_features = preprocess_thyroid_features(payload)
    scaled_features = scaler_thyroid.transform(raw_features)
    
    pred_idx = model_thyroid.predict(scaled_features)[0]
    class_name = thyroid_target_encoder.inverse_transform([pred_idx])[0]
    probabilities = model_thyroid.predict_proba(scaled_features)[0].tolist()
    
    return {
        "disease": "Thyroid Condition",
        "predicted_class": str(class_name),
        "class_index": int(pred_idx),
        "confidence": round(max(probabilities), 4)
    }

@app.post("/predict/all")
def predict_all(payload: ComprehensiveInput):
    """Executes predictions across all metrics provided in a single request."""
    results = {}
    
    if payload.diabetes:
        results["diabetes"] = predict_diabetes(payload.diabetes)
    if payload.anemia:
        results["anemia"] = predict_anemia(payload.anemia)
    if payload.thyroid:
        results["thyroid"] = predict_thyroid(payload.thyroid)
        
    if not results:
        raise HTTPException(status_code=400, detail="No diagnostic data provided.")
        
    return {"diagnostic_report": results}

# ------------------------------------------------------------------
# 6. Server Execution Command
# ------------------------------------------------------------------
# To run this server locally or on a cloud instance, execute:
# uvicorn app:app --host 0.0.0.0 --port 8000 --reload