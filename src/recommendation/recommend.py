import joblib
import pandas as pd
from src.recommendation.risk_rules import assign_risk
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.pkl")

model = joblib.load(MODEL_PATH)

def recommend_action(input_data):
    df = pd.DataFrame([input_data])
    prob = model.predict_proba(df)[0][1]

    risk, action = assign_risk(prob)

    return {
        "default_probability": round(prob, 3),
        "risk_level": risk,
        "recommended_action": action
    }
