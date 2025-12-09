# src/run_shap.py
"""
FAST & ROBUST SHAP analysis for Prosper credit models.

- Uses saved feature list (models/feature_list.txt) if present, otherwise numeric-only fallback.
- Samples rows (default 3000) for speed and memory safety.
- Supports tree models (TreeExplainer) and non-tree models (fallback).
- Handles various shap_values shapes and outputs:
    - reports/shap_summary_bar.png
    - reports/shap_summary_beeswarm.png
    - reports/shap_feature_importance_top50.csv
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "rf_baseline.pkl"        # change to preferred model file if needed
DATA_PATH = BASE / "data" / "prosper_engineered.csv"
OUT_DIR = BASE / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading model and data...")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Engineered data not found at {DATA_PATH}")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH, low_memory=False)

# ---------------- LOAD FEATURES USED DURING TRAINING ----------------
feat_path = BASE / "models" / "feature_list.txt"
if feat_path.exists():
    print("Using saved feature list...")
    with open(feat_path) as f:
        feat_cols = [r.strip() for r in f.readlines() if r.strip()]
    X = df.reindex(columns=feat_cols).fillna(0)
else:
    print("Feature list not found. Using numeric-only fallback.")
    drop_cols = [
        "loanstatus", "loanstatus_clean", "target_default",
        "listingcreationdate", "closeddate", "loanoriginationdate", "datecreditpulled"
    ]
    df_features = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = df_features.select_dtypes(include=[np.number]).fillna(0)

print("Full dataset shape:", X.shape)

# ---------------- SPEEDUP: SAMPLE ONLY N ROWS ----------------
SAMPLE_SIZE = 3000    # change to 2000/5000 depending on speed/memory tradeoff
if len(X) > SAMPLE_SIZE:
    X = X.sample(SAMPLE_SIZE, random_state=42)
    print(f"Using a random sample of {SAMPLE_SIZE} rows for SHAP (speedup).")
else:
    print("Using full dataset for SHAP (small dataset).")

print("Final SHAP dataset shape:", X.shape)

# ---------------- SELECT EXPLAINER ----------------
is_tree = (
    hasattr(model, "feature_importances_")
    or model.__class__.__name__.lower().startswith("xgb")
    or "lgb" in model.__class__.__name__.lower()
)

print("Building SHAP explainer...")
if is_tree:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
else:
    print("Non-tree model detected — using generic Explainer (slower).")
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)

print("SHAP values computed!")

# ---------------- GLOBAL SUMMARY BAR PLOT ----------------
print("Saving SHAP summary bar plot...")
try:
    shap.summary_plot(shap_values, X, show=False, plot_type="bar")
    plt.savefig(OUT_DIR / "shap_summary_bar.png", bbox_inches="tight")
    plt.close()
except Exception as e:
    print("Warning: could not produce bar plot:", e)

# ---------------- BEESWARM PLOT ----------------
print("Saving SHAP beeswarm plot...")
try:
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(OUT_DIR / "shap_summary_beeswarm.png", bbox_inches="tight")
    plt.close()
except Exception as e:
    print("Warning: could not produce beeswarm plot:", e)

# ---------------- TOP FEATURES CSV (robust handling) ----------------
print("Computing and saving SHAP feature importances (robust)...")

# Convert shap_values to numpy array(s)
arr = np.asarray(shap_values)

# If original shap_values was a list, attempt to stack into ndarray with shape (n_classes, n_samples, n_features)
if isinstance(shap_values, list):
    try:
        arr = np.stack(shap_values, axis=0)
        print(f"Stacked shap_values list into array with shape {arr.shape}")
    except Exception:
        # Fallback: compute per-class mean abs importances then average
        per_class_importances = []
        for a in shap_values:
            a_arr = np.asarray(a)
            if a_arr.ndim == 2:  # (n_samples, n_features)
                per_class_importances.append(np.abs(a_arr).mean(axis=0))
        if len(per_class_importances) > 0:
            importances = np.mean(per_class_importances, axis=0)
            feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
            feat_imp.head(50).to_csv(OUT_DIR / "shap_feature_importance_top50.csv")
            print("Saved shap_feature_importance_top50.csv (from per-class fallback).")
            print("\nSHAP analysis completed!")
            print("Files saved in:", OUT_DIR)
            raise SystemExit(0)
        else:
            raise RuntimeError("Unable to process shap_values list into importances.")

# At this point arr is a numpy array. Determine which axis corresponds to features.
feat_axis = None
for i, dim in enumerate(arr.shape):
    if dim == X.shape[1]:
        feat_axis = i
        break

# Common conventions fallback
if feat_axis is None:
    if arr.ndim == 2 and arr.shape[1] == X.shape[1]:
        feat_axis = 1
    elif arr.ndim == 3 and arr.shape[2] == X.shape[1]:
        feat_axis = 2
    elif arr.ndim == 3 and arr.shape[1] == X.shape[1]:
        feat_axis = 1

if feat_axis is None:
    raise ValueError(f"Could not detect feature axis in shap_values with shape {arr.shape} for {X.shape[1]} features.")

# Average over all axes except the feature axis
axes_to_avg = tuple(i for i in range(arr.ndim) if i != feat_axis)
importances = np.abs(arr).mean(axis=axes_to_avg)

# Ensure 1D and correct length
importances = np.asarray(importances).reshape(-1)
if importances.shape[0] != X.shape[1]:
    raise ValueError(f"Computed importances length {importances.shape[0]} != num features {X.shape[1]}")

feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
feat_imp.head(50).to_csv(OUT_DIR / "shap_feature_importance_top50.csv")
print("Saved shap_feature_importance_top50.csv")

print("\nSHAP analysis completed!")
print("Files saved in:", OUT_DIR)
