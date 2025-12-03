import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

DATA_PATH = "data/loan_data_clean.csv"
MODELS_DIR = "models"

df = pd.read_csv(DATA_PATH)

X = df.drop("Default", axis=1)
y = df["Default"]

print("\nLoading best model...")
model = joblib.load(f"{MODELS_DIR}/best_model.pkl")

print("Generating predictions...")
proba = model.predict_proba(X)[:, 1]
pred = model.predict(X)

fpr, tpr, _ = roc_curve(y, proba)
auc_val = roc_auc_score(y, proba)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
plt.plot([0,1],[0,1],"--")
plt.title("ROC Curve - Best Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig(f"{MODELS_DIR}/ROC_BestModel.png")
plt.close()

cm = confusion_matrix(y, pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Best Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"{MODELS_DIR}/CM_BestModel.png")
plt.close()

print("\nROC Curve & Confusion Matrix saved")
print("\nGraph Generation Completed")
