from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load trained model
model = joblib.load("best_model.pkl")

class InputData(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"message": "CreditPathAI API running"}

@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prob = model.predict_proba(X)[0][1]

    if prob < 0.3:
        risk = "Low Risk"
        action = "Send payment reminder via SMS/Email"
    elif prob < 0.6:
        risk = "Medium Risk"
        action = "Offer flexible EMI or short-term payment plan"
    else:
        risk = "High Risk"
        action = "Assign to recovery agent and initiate call"

    return {
        "default_probability": float(round(prob, 3)),
        "risk_level": risk,
        "recommended_action": action
    }
