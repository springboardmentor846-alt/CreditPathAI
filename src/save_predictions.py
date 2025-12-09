# src/save_predictions.py
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "logistic_baseline.pkl"
DATA_PATH = BASE / "data" / "prosper_engineered.csv"
OUT_PATH = BASE / "data" / "prosper_predictions.csv"
TOP_N_PATH = BASE / "data" / "top_100_risky.csv"

# Load model & data
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH, low_memory=False)

# Recreate features same as test_predict (load feature list if exists)
feat_path = BASE / "models" / "feature_list.txt"
if feat_path.exists():
    with open(feat_path) as f:
        feat_cols = [r.strip() for r in f.readlines() if r.strip()]
    X = df.reindex(columns=feat_cols).fillna(0)
else:
    drop_cols = ["loanstatus", "loanstatus_clean", "target_default",
                 "listingcreationdate", "closeddate",
                 "loanoriginationdate", "datecreditpulled"]
    df_features = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    X = df_features.select_dtypes(include=[np.number]).fillna(0)

# Predict probs
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X)[:, 1]
else:
    probs = model.predict(X)

df['pred_prob'] = probs

# Attempt to create risk buckets using qcut with fallback logic
quantiles = [0, 0.6, 0.9, 0.98, 1.0]
labels = ['low', 'medium', 'high', 'very_high']

try:
    # primary attempt (allow dropping duplicate bins)
    df['risk_bucket'] = pd.qcut(df['pred_prob'], q=quantiles, labels=labels, duplicates='drop')
    # If qcut dropped bins, ensure labels match
    # pd.qcut with duplicates='drop' may produce fewer bins; we'll handle below if needed
    # Check number of unique buckets created
    unique_buckets = df['risk_bucket'].nunique(dropna=True)
    if unique_buckets < (len(labels)):
        # fallback to dynamic binning
        raise ValueError("qcut produced fewer bins than labels; falling back to dynamic binning.")
except Exception:
    # Fallback: compute unique quantile edges and adapt labels accordingly
    edges = np.unique(np.quantile(df['pred_prob'].values, quantiles))
    # If edges are fewer than 2, create simple binary split
    if len(edges) < 2:
        # All predictions identical (rare); fallback to binary
        df['risk_bucket'] = pd.cut(df['pred_prob'], bins=[-np.inf, np.median(df['pred_prob']), np.inf],
                                   labels=['low', 'high'])
    else:
        # Build labels dynamically to match number of bins
        nbins = len(edges) - 1
        # Build labels like low, medium, high, very_high but truncated/expanded as needed
        base_labels = ['low', 'medium', 'high', 'very_high', 'extreme']
        # choose first nbins labels
        dyn_labels = base_labels[:nbins]
        # Use pd.cut with these edges
        df['risk_bucket'] = pd.cut(df['pred_prob'], bins=edges, labels=dyn_labels, include_lowest=True)

# Create binary flag with business threshold (customize threshold as needed)
threshold = 0.10
df['pred_flag_threshold_0.10'] = (df['pred_prob'] >= threshold).astype(int)

# Save full predictions
df.to_csv(OUT_PATH, index=False)
print("Saved predictions to:", OUT_PATH)

# Save top 100 risky borrowers (sorted by probability desc)
df.sort_values("pred_prob", ascending=False).head(100).to_csv(TOP_N_PATH, index=False)
print("Saved top 100 risky borrowers to:", TOP_N_PATH)
