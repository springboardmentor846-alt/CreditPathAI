import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import json
import os

# --- CONFIGURATION ---
DATA_DIR = r"C:/Users/R Sai Charan/OneDrive/Documents/CreditPathAI/home-credit-default-risk/"
INPUT_FILE = "train_ready_for_model.csv"  # The NORMALIZED file
MODEL_FILE = "final_xgboost_model.pkl"
FEATURES_FILE = "model_features.json"     # IMPORTANT: We save the list of column names

def train_robust_model():
    print("--- STEP 3: ROBUST MODEL TRAINING ---")
    
    # 1. Load Normalized Data
    path = os.path.join(DATA_DIR, INPUT_FILE)
    if not os.path.exists(path):
        print("❌ Error: Normalized data not found.")
        return
    
    print("Loading Data...")
    df = pd.read_csv(path)
    
    # 2. Separate Target & Features
    if 'TARGET' not in df.columns:
        print("❌ Error: TARGET column missing.")
        return
        
    y = df['TARGET']
    # Drop ID and Target to get pure features
    X = df.drop(columns=['TARGET', 'SK_ID_CURR']) 
    
    # SAVE FEATURE NAMES (Critical for Step 4 alignment)
    feature_names = list(X.columns)
    with open(os.path.join(DATA_DIR, FEATURES_FILE), 'w') as f:
        json.dump(feature_names, f)
    print(f"✅ Saved {len(feature_names)} feature names for future use.")

    # 3. Split Data
    print("Splitting Data (80% Train / 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Configure XGBoost
    print("Configuring XGBoost...")
    # Calculate ratio for imbalanced classes (Defaults are rare)
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    model = xgb.XGBClassifier(
        n_estimators=200,        # Number of trees
        learning_rate=0.05,      # Slower learning = more precision
        max_depth=5,             # Complexity of trees
        scale_pos_weight=ratio,  # Fixes Class Imbalance
        eval_metric='auc',
        n_jobs=-1                # Use all CPU cores
    )
    
    # 5. Train
    print("Training Model (This may take a minute)...")
    model.fit(X_train, y_train)
    
    # 6. Evaluate
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"\n🏆 FINAL VALIDATION SCORE (AUC): {auc:.4f}")
    print("-" * 30)
    
    # 7. Save Model
    save_path = os.path.join(DATA_DIR, MODEL_FILE)
    joblib.dump(model, save_path)
    print(f"💾 Model saved to: {save_path}")

if __name__ == "__main__":
    train_robust_model()