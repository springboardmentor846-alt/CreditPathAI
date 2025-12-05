from fastapi import FastAPI
import pickle
import pandas as pd
from src.utils.risk_logic import risk_category
from src.utils.recommendation_engine import recommend_action

app = FastAPI(title="CreditPathAI Risk API")

# Load model
with open("src/models/advanced_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(data: dict):

    # Convert incoming JSON → DataFrame
    df = pd.DataFrame([data])

    # Predict probability
    prob = model.predict_proba(df)[0][1]

    # Determine risk category
    category = risk_category(prob)

    # Map category → recommendation
    advice = get_recommendation(category)

    return {
        "probability": float(prob),
        "risk_category": category,
        "recommendation": advice
    }
