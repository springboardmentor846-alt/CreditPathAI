# scripts/generate_feature_list.py
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "prosper_engineered.csv"
MODELS_DIR = BASE / "models"
OUT_PATH = MODELS_DIR / "feature_list.txt"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Engineered data not found at: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)

# Use same drop list used in modeling.py
drop_cols = [
    "loanstatus", "loanstatus_clean", "target_default",
    "listingcreationdate", "closeddate", "loanoriginationdate", "datecreditpulled"
]

# Remove those columns if present, then keep numeric columns (same logic as modeling pipeline)
df_features = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
feature_cols = df_features.select_dtypes(include=['number']).columns.tolist()

# Ensure models folder exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Save feature list (one column per line, order preserved)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for col in feature_cols:
        f.write(col + "\n")

print("Saved feature list to:", OUT_PATH)
print("Number of features:", len(feature_cols))
