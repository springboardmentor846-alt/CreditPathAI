from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import roc_curve, roc_auc_score

# Load test data
X_test = joblib.load("models/X_test.pkl")
y_test = joblib.load("models/y_test.pkl")

# Load models
log_model = joblib.load("models/logistic_regression_model.pkl")
xgb_model = joblib.load("models/xgboost_model.pkl")
lgbm_model = joblib.load("models/lightgbm_model.pkl")

# Load models & preprocessing artifacts
baseline_model = joblib.load("models/logistic_regression_model.pkl")  # Baseline
final_model = joblib.load("models/lightgbm_model.pkl")               # Final

scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

print("Loaded baseline model, final model, scaler, encoders, and features!")

# FastAPI app
app = FastAPI(title="Loan Default Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class LoanInput(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float

    Education: str
    EmploymentType: str
    MaritalStatus: str
    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str

# Prediction endpoint
@app.post("/predict")
def predict(data: LoanInput):

    # ---- Convert input to DataFrame ----
    df_input = pd.DataFrame([data.dict()])

    # ---- Feature Engineering (same as training) ----
    df_input["LoanToIncome"] = df_input["LoanAmount"] / df_input["Income"]

    df_input["MonthlyPayment"] = (
        (df_input["LoanAmount"] *
         (df_input["InterestRate"] / 100 / 12) *
         (1 + df_input["InterestRate"] / 100 / 12) ** df_input["LoanTerm"]) /
        ((1 + df_input["InterestRate"] / 100 / 12) ** df_input["LoanTerm"] - 1)
    )

    df_input["PaymentToIncome"] = df_input["MonthlyPayment"] / (df_input["Income"] / 12)
    df_input["CreditUtilization"] = df_input["LoanAmount"] / df_input["CreditScore"]

    # ---- Label Encoding ----
    for col, encoder in label_encoders.items():
        df_input[col + "_encoded"] = encoder.transform(df_input[col])

    # ---- Align with training features ----
    final_input = pd.DataFrame(columns=feature_columns)
    final_input.loc[0] = 0
    final_input.update(df_input)

    # ---- Scale input ----
    scaled_input = scaler.transform(final_input)

    # Predictions

    # Baseline model (Logistic Regression)
    base_pred = baseline_model.predict(scaled_input)[0]
    base_prob = baseline_model.predict_proba(scaled_input)[0][1]

    # Final model (LightGBM)
    final_pred = final_model.predict(scaled_input)[0]
    final_prob = final_model.predict_proba(scaled_input)[0][1]

    # Response
    return {
        "baseline_model": {
            "name": "Logistic Regression",
            "prediction": int(base_pred),
            "probability": round(float(base_prob), 4),
            "risk_level": "HIGH DEFAULT RISK" if base_pred == 1 else "LOW DEFAULT RISK"
        },
        "final_model": {
            "name": "LightGBM",
            "prediction": int(final_pred),
            "probability": round(float(final_prob), 4),
            "risk_level": "HIGH DEFAULT RISK" if final_pred == 1 else "LOW DEFAULT RISK"
        }
    }

@app.get("/roc-metrics")
def get_roc_metrics():

    models = {
        "Logistic Regression": log_model,
        "XGBoost": xgb_model,
        "LightGBM": lgbm_model,
    }

    roc_results = {}

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        roc_results[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(float(auc), 4),
        }

    return roc_results
