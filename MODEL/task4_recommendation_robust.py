import pandas as pd
import numpy as np
import joblib
import json
import os

# --- CONFIGURATION ---
DATA_DIR = r"C:/Users/R Sai Charan/OneDrive/Documents/CreditPathAI/home-credit-default-risk/"
MODEL_FILE = "final_xgboost_model.pkl"
FEATURES_FILE = "model_features.json"

# We need BOTH files
RAW_FILE = "train_master_full.csv"             # For Context (Business Logic)
NORMALIZED_FILE = "train_ready_for_model.csv"  # For Scoring (Math)

def get_complex_strategy(row):
    """
    Advanced Logic: Uses Risk Score + 5 Behavioral Features
    """
    risk = row['PREDICTED_RISK']
    
    # --- LEVEL 1: LOW RISK (Safe) ---
    if risk < 0.40:
        return "No Action - Good Standing"

    # --- LEVEL 2: MEDIUM RISK (Warning Signs) ---
    elif 0.40 <= risk < 0.75:
        
        # Check 1: Are they habitual late payers?
        if row.get('INST_AVG_LATE', 0) > 5:
            return "Digital Nudge: 'Did you forget your payment?'"
        
        # Check 2: Are they over-leveraged? (Loan is too big for Income)
        elif row.get('CREDIT_INCOME_PERCENT', 0) > 2.0:
            return "Education: Send Financial Planning Tips"
            
        else:
            return "Automated SMS Reminder"

    # --- LEVEL 3: HIGH RISK (Critical) ---
    else:
        # Check 3: DESPERATION (Cash Withdrawals from Credit Card)
        # This is the biggest predictor of imminent default
        if row.get('CC_ATM_DRAWINGS', 0) > 0:
            return "URGENT: Human Call (Cash Withdrawal Flag)"
        
        # Check 4: EXTERNAL PRESSURE (High Debt at other banks)
        elif row.get('BUREAU_TOTAL_DEBT', 0) > 50000:
            return "OFFER: Restructuring Plan (Debt Consolidation)"
            
        # Check 5: REJECTION HISTORY (They tried to get loans and failed)
        elif row.get('PREV_TOTAL_REFUSALS', 0) > 1:
            return "High Risk: Monitor for Fraud / Bust-out"
            
        else:
            return "Escalate to Senior Collection Officer"

def run_recommendation_engine():
    print("--- STEP 4: RECOMMENDATION ENGINE ---")
    
    # 1. Load Resources
    print("Loading Model and Feature List...")
    if not os.path.exists(os.path.join(DATA_DIR, MODEL_FILE)):
        print("❌ Error: Model not found. Run Task 3 first.")
        return
        
    model = joblib.load(os.path.join(DATA_DIR, MODEL_FILE))
    
    with open(os.path.join(DATA_DIR, FEATURES_FILE), 'r') as f:
        feature_names = json.load(f)

    # 2. Load NORMALIZED Data (For Prediction)
    print("Loading Normalized Data for Scoring...")
    df_norm = pd.read_csv(os.path.join(DATA_DIR, NORMALIZED_FILE))
    
    # Ensure columns align exactly with training
    # Create a DataFrame with only the features used in training
    X_score = df_norm[feature_names]
    
    # 3. Generate Scores
    print("Predicting Risk Scores...")
    risk_scores = model.predict_proba(X_score)[:, 1]
    
    # 4. Load RAW Data (For Logic)
    print("Loading Raw Data for Context...")
    df_raw = pd.read_csv(os.path.join(DATA_DIR, RAW_FILE))
    
    # Attach the scores to the raw data
    df_raw['PREDICTED_RISK'] = risk_scores
    
    # 5. Apply the Advanced Strategy
    print("Applying Business Logic...")
    
    # Safety: Ensure columns exist (if some features weren't calculated, fill with 0)
    required_cols = ['INST_AVG_LATE', 'CREDIT_INCOME_PERCENT', 'CC_ATM_DRAWINGS', 'BUREAU_TOTAL_DEBT', 'PREV_TOTAL_REFUSALS']
    for col in required_cols:
        if col not in df_raw.columns:
            df_raw[col] = 0
            
    df_raw['STRATEGY'] = df_raw.apply(get_complex_strategy, axis=1)
    
    # 6. Save Report
    output_cols = ['SK_ID_CURR', 'PREDICTED_RISK', 'STRATEGY'] + required_cols
    output_path = os.path.join(DATA_DIR, "Final_Action_Plan.csv")
    
    df_raw[output_cols].to_csv(output_path, index=False)
    
    print("\n✅ PREDICTION & RECOMMENDATION COMPLETE.")
    print(f"📄 Report saved to: {output_path}")
    print("\nSAMPLE RESULTS:")
    print(df_raw[['SK_ID_CURR', 'PREDICTED_RISK', 'STRATEGY']].head(10))

if __name__ == "__main__":
    run_recommendation_engine()