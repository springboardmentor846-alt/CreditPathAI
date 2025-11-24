import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


# STEP 1 — Load saved model and cleaned dataset
print("[INFO] Loading model and dataset...")
model = joblib.load("models/logistic_baseline.pkl")
df = pd.read_csv("data/loan_data_clean.csv")
print("[INFO] Model and data loaded successfully!")
X = df.drop("Default", axis=1)
y = df["Default"]


# STEP 2 — Make predictions using saved model
print("[INFO] Generating predictions for graphs...")
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]


# GRAPH 1 — ROC CURVE
print("[INFO] Plotting ROC Curve...")
fpr, tpr, thresholds = roc_curve(y, y_prob)
auc = roc_auc_score(y, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Logistic Regression Baseline Model")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# GRAPH 2 — CONFUSION MATRIX HEATMAP
print("[INFO] Plotting Confusion Matrix...")
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# GRAPH 3 — FEATURE DISTRIBUTION PLOTS
print("[INFO] Plotting feature distributions...")
plt.figure(figsize=(10, 5))
sns.histplot(df["LoanAmount"], bins=40, kde=True)
plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 5))
sns.histplot(df["CreditScore"], bins=40, kde=True, color='orange')
plt.title("Credit Score Distribution")
plt.xlabel("Credit Score")
plt.tight_layout()
plt.show()
print("\n[INFO] All graphs generated successfully!")
