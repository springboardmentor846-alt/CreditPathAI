import pandas as pd
import joblib
import os
import json
import uvicorn
import math
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --- 1. CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "home-credit-default-risk")

if not os.path.exists(DATA_DIR):
    DATA_DIR = r"C:/Users/R Sai Charan/OneDrive/Documents/CreditPathAI/home-credit-default-risk/"

MODEL_FILE = "final_xgboost_model.pkl"
FEATURES_FILE = "model_features.json"
RAW_DATA_FILE = "train_master_full.csv"
NORMALIZED_DATA_FILE = "train_ready_for_model.csv"

# --- 2. INITIALIZE APP ---
app = FastAPI(title="CreditPathAI Risk Engine", version="3.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. GLOBAL STATE ---
resources = {
    "model": None, "features": [], "df_raw": None, "df_norm": None
}

# --- 4. HELPER FUNCTIONS ---
def sanitize_float(value, decimals=2):
    """
    Converts NaN/Inf to 0 and rounds the number.
    """
    if value is None: return 0.0
    if isinstance(value, (float, np.floating)):
        if math.isnan(value) or math.isinf(value): return 0.0
    return round(float(value), decimals)

# --- 5. STARTUP EVENT ---
@app.on_event("startup")
def startup_event():
    print("🚀 STARTING CREDITPATH AI SERVER")
    
    # Load Model & Features
    resources["model"] = joblib.load(os.path.join(DATA_DIR, MODEL_FILE))
    with open(os.path.join(DATA_DIR, FEATURES_FILE), 'r') as f:
        resources["features"] = json.load(f)

    # Load Data (Fast Mode)
    raw_path = os.path.join(DATA_DIR, RAW_DATA_FILE)
    if os.path.exists(raw_path):
        # Loading 5000 rows for better coverage
        resources["df_raw"] = pd.read_csv(raw_path, nrows=5000) 
        resources["df_raw"].set_index('SK_ID_CURR', inplace=True)

    norm_path = os.path.join(DATA_DIR, NORMALIZED_DATA_FILE)
    if os.path.exists(norm_path):
        df = pd.read_csv(norm_path, nrows=5000)
        if 'SK_ID_CURR' in df.columns:
            df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)
            df.set_index('SK_ID_CURR', inplace=True)
            resources["df_norm"] = df
    
    print("✅ SERVER READY.")

# --- 6. LOGIC (UPDATED WITH REASONS) ---
def get_strategy(risk, context):
    """
    Returns (Action, Reason)
    """
    # 1. LOW RISK
    if risk < 0.40:
        return "No Action Required", "Customer is in Good Standing (Low Probability)"
    
    # 2. MEDIUM RISK
    elif 0.40 <= risk < 0.75:
        if context.get('INST_AVG_LATE', 0) > 5:
            days = round(context.get('INST_AVG_LATE', 0), 1)
            return "Digital Nudge", f"Habitual Late Payer (Avg {days} days late)"
        
        elif context.get('CREDIT_INCOME_PERCENT', 0) > 2.0:
            return "Send Financial Tips", "Loan is high compared to Income"
        
        else:
            return "Automated SMS", "Medium Risk Standard Procedure"
            
    # 3. HIGH RISK
    else:
        if context.get('CC_ATM_DRAWINGS', 0) > 0:
            return "URGENT: Human Call", "Financial Distress Flag (Cash Withdrawal Detected)"
        
        elif context.get('BUREAU_TOTAL_DEBT', 0) > 50000:
            debt = round(context.get('BUREAU_TOTAL_DEBT', 0), 0)
            return "OFFER: Restructuring", f"High External Debt (${debt})"
        
        else:
            return "Escalate to Senior Officer", "High Probability of Default"

# --- 7. ENDPOINTS ---
@app.get("/")
def home():
    return {"status": "online", "message": "Go to /predict/{id}"}

@app.get("/predict/{customer_id}")
def predict(customer_id: int):
    # Check System Health
    if resources["df_norm"] is None or customer_id not in resources["df_norm"].index:
        raise HTTPException(status_code=404, detail="Customer ID not found.")

    try:
        # 1. Get Data & Predict
        user_row = resources["df_norm"].loc[[customer_id]]
        
        # Align Columns
        X_input = pd.DataFrame(index=user_row.index)
        for col in resources["features"]:
            X_input[col] = user_row[col] if col in user_row.columns else 0.0
        X_input = X_input[resources["features"]]

        # Risk Score (0.429...)
        raw_risk = float(resources["model"].predict_proba(X_input)[:, 1][0])

        # 2. Get Context
        user_context = {}
        if resources["df_raw"] is not None and customer_id in resources["df_raw"].index:
            user_context = resources["df_raw"].loc[customer_id].to_dict()

        # 3. Get Strategy
        action_text, reason_text = get_strategy(raw_risk, user_context)

        # 4. Return Formatted Response
        return {
            "customer_id": customer_id,
            "prediction": {
                # CHANGE 1: Convert 0.429 -> 42.9 (Percentage)
                "default_probability": sanitize_float(raw_risk * 100, 1), 
                "risk_category": "High" if raw_risk >= 0.75 else "Medium" if raw_risk >= 0.40 else "Low"
            },
            "recommendation": {
                "action": action_text,
                "reason": reason_text
            },
            "context": {
                "income": sanitize_float(user_context.get('AMT_INCOME_TOTAL'), 0),
                
                # CHANGE 2: Convert 7.9 -> 8 (Integer/Single Digit)
                "late_days": int(sanitize_float(user_context.get('INST_AVG_LATE'), 0)), 
                
                "debt": sanitize_float(user_context.get('BUREAU_TOTAL_DEBT'), 0),
                "cash_withdrawals": sanitize_float(user_context.get('CC_ATM_DRAWINGS'), 0)
            }
        }

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)