# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model pipeline
pipeline = joblib.load("heart_model_pipeline.pkl")

app = FastAPI(title="Heart Disease Prediction API")

# Define input schema
class HeartInput(BaseModel):
    age: int
    gender: int
    BMI: float
    systolic_bp: int
    diastolic_bp: int
    cholesterol_total_mg_dl: float
    ldl_direct_mg_dl: float
    hdl_cholesterol_mg_dl: float
    triglycerides_mg_dl: float
    vldl_cholesterol_mg_dl: float
    cholesterol_hdl_chol_ratio: float
    blood_sugar_fasting_mg_dl: float
    glucose_post_prandial_mg_dl: float
    creatinine_mg_dl: float
    blood_urea_mg_dl: float

# Root endpoint
@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API Running"}

# Predict endpoint
@app.post("/predict")
def predict(input_data: HeartInput):
    df = pd.DataFrame([input_data.dict()])
    pred = pipeline.predict(df)[0]
    prob = pipeline.predict_proba(df)[0][1]  # probability of heart disease
    return {"heart_disease": int(pred), "risk_probability": round(prob*100, 2)}
