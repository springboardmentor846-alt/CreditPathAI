from fastapi import FastAPI
from pydantic import BaseModel
from src.recommendation.recommend import recommend_action
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="CreditPathAI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    LoanIncomeRatio: float

@app.get("/")
def health():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: LoanInput):
    features = [
        data.Age,
        data.Income,
        data.LoanAmount,
        data.CreditScore,
        data.MonthsEmployed,
        data.NumCreditLines,
        data.InterestRate,
        data.LoanTerm,
        data.DTIRatio,
        data.LoanIncomeRatio
    ]
    return recommend_action(features)
