from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# CORS for React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model matches your React form fields
class InputData(BaseModel):
    loan_amount: float
    rate_of_interest: float
    term: int
    LTV: float
    Upfront_charges: float
    Credit_Worthiness: str
    loan_type: str
    Security_Type: str
    loan_purpose: str
    open_credit: str
    business_or_commercial: str
    approv_in_adv: str
    Neg_ammortization: str

@app.get("/")
def home():
    return {"message": "CreditPathAI API running"}

@app.post("/predict")
def predict(data: InputData):
    # Convert input to DataFrame
    user_df = pd.DataFrame([data.dict()])

    # Derived features
    user_df["loan_interest_burden"] = user_df["loan_amount"] * user_df["rate_of_interest"]
    user_df["loan_term_pressure"] = user_df["loan_amount"] / (user_df["term"] + 1)
    user_df["high_ltv_flag"] = (user_df["LTV"] > 80).astype(int)
    user_df["negative_amort_flag"] = (user_df["Neg_ammortization"] == "Yes").astype(int)
    user_df["business_risk_flag"] = (user_df["business_or_commercial"] == "Yes").astype(int)

    # Load model and expected features
    model = joblib.load("loan_default_model.pkl")
    feature_cols = joblib.load("model_features.pkl")

    # Add missing columns
    for col in feature_cols:
        if col not in user_df.columns:
            user_df[col] = 0  # or np.nan depending on model training

    user_df = user_df[feature_cols]

    probability = model.predict_proba(user_df)[0][1]

    if probability < 0.3:
        risk = "Low Risk"
        action = "Send payment reminder via SMS/Email"
    elif probability < 0.6:
        risk = "Medium Risk"
        action = "Offer flexible EMI or short-term payment plan"
    else:
        risk = "High Risk"
        action = "Assign to recovery agent and initiate call"

    return {
        "default_probability": float(round(probability * 100, 2)),
        "risk_level": risk,
        "recommended_action": action
    }
