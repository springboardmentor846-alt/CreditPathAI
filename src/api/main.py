from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd
from src.utils.risk_logic import risk_category
from src.utils.recovery_recommendations import recovery_action

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all
    allow_credentials=True,
    allow_methods=["*"],   # allow all HTTP methods
    allow_headers=["*"],   # allow all headers
)

model = joblib.load("src/models/xgboost_model.pkl")

class LoanData(BaseModel):
    income: float
    loan_amount: float
    credit_score: int
    ltv: float
    dtir1: float

@app.get("/")
def home():
    return {"status": "API running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict/")
def predict(data: LoanData):
    # TEMPORARY MOCK RESPONSE (for frontend + review)
    prob = 0.42
    category = risk_category(prob)
    action = recovery_action(category)

    return {
        "risk_category": category,
        "probability": prob,
        "recommendation": action
    }

if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)