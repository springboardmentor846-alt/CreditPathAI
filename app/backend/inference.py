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
    
    Age: int
    Income: int
    LoanAmount: int
    CreditScore: int
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float

    HasMortgage: str
    HasDependents: str
    HasCoSigner: str

    Education: str
    EmploymentType: str
    MaritalStatus: str
    LoanPurpose: str
    

@app.get("/")
def home():
    return {"message": "CreditPathAI API running"}

@app.post("/predict")
def predict(data: InputData):
    # Convert input to DataFrame
    user_df = pd.DataFrame([data.dict()])

    # Load model and expected features
    model = joblib.load("loan_default_pipeline.pkl")

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
