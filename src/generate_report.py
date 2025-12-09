# src/generate_report.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

BASE = Path(__file__).resolve().parents[1]
PRED_PATH = BASE / "data" / "prosper_predictions.csv"
OUT_DIR = BASE / "reports"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(PRED_PATH, low_memory=False)
probs = df["pred_prob"].values
print("Rows:", len(df))

# Distribution
desc = df["pred_prob"].describe()
desc.to_csv(OUT_DIR / "pred_prob_summary.csv")
print("Saved pred_prob summary")

# Histogram
plt.figure(figsize=(8,4))
plt.hist(probs, bins=100)
plt.title("Predicted probability distribution")
plt.xlabel("pred_prob")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(OUT_DIR / "pred_prob_hist.png")
plt.close()

# Risk bucket counts
if "risk_bucket" in df.columns:
    df["risk_bucket"].value_counts().to_csv(OUT_DIR / "risk_bucket_counts.csv")
    print("Saved risk bucket counts")

# If ground truth exists, compute PR curve & threshold suggestions
if "target_default" in df.columns:
    y_true = df["target_default"].values
    precision, recall, thr = precision_recall_curve(y_true, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
    best_idx = np.nanargmax(f1_scores)
    best_thr = thr[best_idx] if best_idx < len(thr) else 0.5
    # Save top-thresholds
    pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "f1": f1_scores[:len(precision)]
    }).to_csv(OUT_DIR / "pr_curve_table.csv", index=False)
    with open(OUT_DIR / "recommended_threshold.txt", "w") as f:
        f.write(f"Best F1 threshold: {best_thr}\n")
    print("Saved PR curve and recommended threshold:", best_thr)

    # ROC AUC
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    print("ROC AUC:", roc_auc)
    pd.DataFrame({"fpr":fpr, "tpr":tpr}).to_csv(OUT_DIR / "roc_curve.csv", index=False)
else:
    print("No target_default in predictions — skip PR/ROC metrics.")

print("Report files saved in:", OUT_DIR)
