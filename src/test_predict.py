# src/test_predict.py
import joblib
import pandas as pd
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "logistic_baseline.pkl"
DATA_PATH = BASE / "data" / "prosper_engineered.csv"

# Load model
model = joblib.load(MODEL_PATH)
print("Loaded model:", MODEL_PATH)

# Load engineered data
df = pd.read_csv(DATA_PATH, low_memory=False)
print("Loaded engineered data shape:", df.shape)

# Columns to drop (same as modeling.py)
drop_cols = [
    "loanstatus", "loanstatus_clean", "target_default",
    "listingcreationdate", "closeddate",
    "loanoriginationdate", "datecreditpulled"
]

# Drop non-features
df_features = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# Select numeric columns only (modeling used numeric-only)
X_all_numeric = df_features.select_dtypes(include=[np.number]).fillna(0)

# Ensure no accidental 'target_default' column remains
if "target_default" in X_all_numeric.columns:
    X_all_numeric = X_all_numeric.drop(columns=["target_default"])

# Optional: if you saved a feature list, load that to ensure ordering (see alternative below)
# feat_path = BASE / "models" / "feature_list.txt"
# if feat_path.exists():
#     feat_cols = [c.strip() for c in open(feat_path).read().splitlines()]
#     X_all_numeric = X_all_numeric[feat_cols]

# Take small sample
X_sample = X_all_numeric.iloc[:5]
print("Using feature shape for prediction:", X_sample.shape)

# Predict
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_sample)[:, 1]
else:
    try:
        probs = model.decision_function(X_sample)
    except Exception:
        probs = model.predict(X_sample)

print("Predicted probabilities for sample rows:", probs)
