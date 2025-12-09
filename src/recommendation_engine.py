import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

PRED_PATH = BASE / "data" / "prosper_predictions.csv"
SHAP_PATH = BASE / "reports" / "shap_feature_importance_top50.csv"
OUT_PATH = BASE / "data" / "prosper_with_actions.csv"

print("Loading prediction + SHAP data...")

df = pd.read_csv(PRED_PATH, low_memory=False)

# Load SHAP importance ranking (top global features)
shap_imp = pd.read_csv(SHAP_PATH, header=None)
shap_imp.columns = ["feature", "importance"]

top_feats = shap_imp["feature"].head(5).tolist()
print("\nTop SHAP features used for driver extraction:")
print(top_feats)

# -----------------------------------------
# 1) FAST ACTION RECOMMENDATION (vectorized)
# -----------------------------------------

action_map = {
    "very_high": "Escalate to collections - High touch agent call",
    "high": "Agent call + Offer repayment plan",
    "medium": "Personalized email + Follow-up reminder",
    "low": "SMS/Email reminder only"
}

df["recommended_action"] = df["risk_bucket"].map(action_map).fillna("No Action")

# --------------------------------------------------
# 2) SUPER-FAST RISK DRIVER EXTRACTION (vectorized)
# --------------------------------------------------

# Create empty column first
df["risk_drivers"] = ""

for feat in top_feats:
    if feat in df.columns:
        median_val = df[feat].median()

        # Condition where feature is risk-elevating
        mask = df[feat] > median_val

        # Append driver tag
        df.loc[mask, "risk_drivers"] += f"{feat}: high, "

# Clean trailing comma/space
df["risk_drivers"] = df["risk_drivers"].str.rstrip(", ")
df["risk_drivers"] = df["risk_drivers"].replace("", "N/A")

# Save final output
df.to_csv(OUT_PATH, index=False)

print("\nRecommendation Engine Completed FAST!")
print("Saved to:", OUT_PATH)
