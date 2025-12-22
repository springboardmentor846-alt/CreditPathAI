
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI(title="CreditPathAI - Credit Default Prediction API")

# Load frozen model and scaler
model = joblib.load("creditpath_xgb.pkl")
scaler = joblib.load("creditpath_scaler.pkl")

# -------- Input Schema --------
class Borrower(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int

    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int

    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


# -------- Prediction Endpoint --------
@app.post("/predict")
def predict_default(data: Borrower):

    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Apply same scaling used during training
    scaled_input = scaler.transform(df)

    # Predict probability of default
    prob = model.predict_proba(scaled_input)[0][1]

    # Risk categorization logic
    if prob < 0.20:
        risk = "Low Risk"
        action = "Send gentle SMS reminder"
    elif prob < 0.40:
        risk = "Moderate Risk"
        action = "Call customer and confirm repayment date"
    elif prob < 0.60:
        risk = "High Risk"
        action = "Offer restructuring or part-payment plan"
    else:
        risk = "Very High Risk"
        action = "Escalate to field visit or legal notice"

    return {
        "default_probability": round(float(prob), 4),
        "risk_category": risk,
        "recommended_action": action
    }
