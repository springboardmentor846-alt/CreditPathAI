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
intermediate_model = joblib.load("models/xgboost_model.pkl")
final_model = joblib.load("models/lightgbm_model.pkl")               # Final

scaler = joblib.load("models/scaler.pkl")
ohe = joblib.load("models/onehot_encoder.pkl")
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
    CreditScore: int
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

    categorical_cols = [
    'Education', 'EmploymentType', 'MaritalStatus',
    'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
]

    encoded = ohe.transform(df_input[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(categorical_cols)
    )

    df_input = pd.concat(
        [df_input.drop(columns=categorical_cols), encoded_df],
        axis=1
    )

    # ---- Align with training features ----
    final_input = pd.DataFrame(
        data=np.zeros((1, len(feature_columns)), dtype=float),
        columns=feature_columns
    )
    for col in df_input.columns:
        if col in final_input.columns:
            final_input.at[0, col] = float(df_input.at[0, col])
    # ---- Scale input ----
    scaled_input = scaler.transform(final_input)

    # Predictions

    # Baseline model (Logistic Regression)
    base_pred = baseline_model.predict(scaled_input)[0]
    base_prob = baseline_model.predict_proba(scaled_input)[0][1]

    int_pred = intermediate_model.predict(final_input)[0]
    int_prob = intermediate_model.predict_proba(final_input)[0][1]

    # Final model (LightGBM) - uses NON-scaled
    final_pred = final_model.predict(final_input)[0]
    final_prob = final_model.predict_proba(final_input)[0][1]

    # Response
    return {
        "baseline_model": {
            "name": "Logistic Regression",
            "prediction": int(base_pred),
            "probability": round(float(base_prob), 4),
            "risk_level": (
                "LOW DEFAULT RISK" if base_prob < 0.33
                else "MEDIUM DEFAULT RISK" if base_prob < 0.66
                else "HIGH DEFAULT RISK"
            )
        },
        "intermediate_model": {
            "name": "XGBoost",
            "prediction": int(int_pred),
            "probability": round(float(int_prob), 4),
            "risk_level": (
                "LOW DEFAULT RISK" if int_prob < 0.33
                else "MEDIUM DEFAULT RISK" if int_prob < 0.66
                else "HIGH DEFAULT RISK"
            )

        },
        "final_model": {
            "name": "LightGBM",
            "prediction": int(final_pred),
            "probability": round(float(final_prob), 4),
            "risk_level": (
                "LOW DEFAULT RISK" if final_prob < 0.33
                else "MEDIUM DEFAULT RISK" if final_prob < 0.66
                else "HIGH DEFAULT RISK"
            )
        }
    }

@app.get("/roc-metrics")
def get_roc_metrics():
    models = {
        "Logistic Regression": (log_model, True),
        "XGBoost": (xgb_model, False),
        "LightGBM": (lgbm_model, False),
    }
    
    roc_results = {}
    for name, (model, needs_scaling) in models.items():
        if needs_scaling:
            X_test_processed = scaler.transform(X_test)
        else:
            X_test_processed = X_test
        
        y_prob = model.predict_proba(X_test_processed)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        
        roc_results[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(float(auc), 4),
        }
    return roc_results
