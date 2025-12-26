from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

# ======================================================
# LOAD MODELS
# ======================================================
logistic_model = joblib.load("models/logistic_model.pkl")
xgb_model = joblib.load("models/xgboost_model.pkl")
lgbm_model = joblib.load("models/lightgbm_model.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")

# ======================================================
# FASTAPI APP
# ======================================================
app = FastAPI(
    title="Credit Default Risk API",
    description="Multi-model risk scoring & recommendation engine",
    version="1.0.0"
)

# ======================================================
# CORS (for React)
# ======================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# INPUT SCHEMA
# ======================================================
class LoanApplication(BaseModel):
    purpose: str
    grade: str
    residentialstate: str
    homeownership: str
    loanamount: float
    interestrate: float
    monthlypayment: float
    annualincome: float
    dtiratio: float
    lengthcredithistory: int
    term_months: int

# ======================================================
# RECOMMENDATION ENGINE
# ======================================================
def recommend_action(prob: float):
    if prob >= 0.6:
        return "High Risk", "Reject or Manual Review", "High probability of default"
    elif prob >= 0.3:
        return "Medium Risk", "Approve with Conditions", "Moderate default risk"
    else:
        return "Low Risk", "Approve", "Low probability of default"

# ======================================================
# HEALTH CHECK
# ======================================================
@app.get("/")
def health():
    return {
        "status": "running",
        "models": ["Logistic Regression", "XGBoost", "LightGBM"]
    }

# ======================================================
# PREDICTION ENDPOINT
# ======================================================
@app.post("/predict")
def predict(data: LoanApplication):
    try:
        # -----------------------------
        # RAW INPUT (for Logistic Pipeline)
        # -----------------------------
        raw_df = pd.DataFrame([{
            "purpose": data.purpose,
            "grade": data.grade,
            "residentialstate": data.residentialstate,
            "homeownership": data.homeownership,
            "loanamount": data.loanamount,
            "interestrate": data.interestrate,
            "monthlypayment": data.monthlypayment,
            "annualincome": data.annualincome,
            "dtiratio": data.dtiratio,
            "lengthcredithistory": data.lengthcredithistory,
            "term_months": data.term_months,

            # static features (same as training)
            "loan_year": 2016,
            "loan_month": 6,
            "loan_dayofweek": 2,
            "isjointapplication": 0,
            "incomeverified": 1,
            "numtotalcreditlines": 12,
            "numopencreditlines": 8,
            "numopencreditlines1year": 4,
            "revolvingbalance": 15000,
            "revolvingutilizationrate": 70,
            "numderogatoryrec": 0,
            "numdelinquency2years": 1,
            "numchargeoff1year": 0,
            "numinquiries6mon": 2
        }])

        # -----------------------------
        # ONE-HOT ENCODED INPUT (XGB & LGBM)
        # -----------------------------
        encoded_df = pd.get_dummies(raw_df)
        encoded_df = encoded_df.reindex(columns=feature_cols, fill_value=0)

        # -----------------------------
        # MODEL PREDICTIONS
        # -----------------------------
        prob_logistic = float(logistic_model.predict_proba(raw_df)[0][1])
        prob_xgb = float(xgb_model.predict_proba(encoded_df)[0][1])
        prob_lgbm = float(lgbm_model.predict_proba(encoded_df)[0][1])

        # -----------------------------
        # FINAL DECISION (LightGBM)
        # -----------------------------
        risk, action, reason = recommend_action(prob_lgbm)

        return {
            "final_model": "LightGBM",
            "default_probability": round(prob_lgbm, 4),
            "risk_level": risk,
            "recommended_action": action,
            "reason": reason,
            "model_comparison": {
                "logistic_regression": round(prob_logistic, 4),
                "xgboost": round(prob_xgb, 4),
                "lightgbm": round(prob_lgbm, 4)
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
