from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# ----------------------------
# Load trained pipeline
# ----------------------------
pipeline_path = os.path.join(os.path.dirname(__file__), "heart_model_pipeline.pkl")
pipeline = joblib.load(pipeline_path)

app = FastAPI(title="Heart Disease Prediction API")

# ----------------------------
# Input schema
# ----------------------------
class HeartInput(BaseModel):
    age: int
    gender: int
    BMI: float
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
    systolic_bp: int
    diastolic_bp: int

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API Running"}

@app.post("/predict")
def predict(input_data: HeartInput):
    # Correct feature order
    feature_order = [
        'age', 'gender', 'BMI', 'cholesterol_total_mg_dl', 'ldl_direct_mg_dl', 'hdl_cholesterol_mg_dl',
        'triglycerides_mg_dl', 'vldl_cholesterol_mg_dl', 'cholesterol_hdl_chol_ratio',
        'blood_sugar_fasting_mg_dl', 'glucose_post_prandial_mg_dl', 'creatinine_mg_dl',
        'blood_urea_mg_dl', 'systolic_bp', 'diastolic_bp'
    ]

    df = pd.DataFrame([input_data.dict()], columns=feature_order)

    # Match dtypes
    int_cols = ['age', 'gender', 'systolic_bp', 'diastolic_bp']
    float_cols = [c for c in df.columns if c not in int_cols]
    df[int_cols] = df[int_cols].astype(int)
    df[float_cols] = df[float_cols].astype(float)

    # Predict
    pred = pipeline.predict(df)[0]
    prob = pipeline.predict_proba(df)[0][1]

    return {"heart_disease": int(pred), "risk_probability": round(prob*100, 2)}
