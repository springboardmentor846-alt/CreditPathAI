from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="CreditPathAI - Loan Default Prediction API")

model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")


class LoanRequest(BaseModel):
    LoanAmount: float
    InterestRate: float
    DTIRatio: float
    CreditScore: int
    MonthsEmployed: int


@app.get("/")
def home():
    return {
        "message": "CreditPathAI API is running successfully"
    }


@app.post("/predict")
def predict(data: LoanRequest):

    features = np.array([[  
        data.LoanAmount,
        data.InterestRate,
        data.DTIRatio,
        data.CreditScore,
        data.MonthsEmployed
    ]])

    features_scaled = scaler.transform(features)
    probability = model.predict_proba(features_scaled)[0][1]

    if probability >= 0.75:
        risk = "Very High Risk"
        recommendation = "Reject loan or demand collateral"
    elif probability >= 0.55:
        risk = "High Risk"
        recommendation = "Approve with higher interest rate"
    elif probability >= 0.30:
        risk = "Moderate Risk"
        recommendation = "Approve with monitoring"
    else:
        risk = "Low Risk"
        recommendation = "Approve loan"

    return {
        "default_probability": round(float(probability), 4),
        "risk_level": risk,
        "recommendation": recommendation
    }
