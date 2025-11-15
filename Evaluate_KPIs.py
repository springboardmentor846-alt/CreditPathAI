import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# DATA INGESTION TIME
start_time = time.time()
df = pd.read_csv("data/loan_data.csv")
ingestion_time = time.time() - start_time

print("\nBASIC KPIs:-")


# BASIC KPIs
total_borrowers = len(df)
print(f"Total Borrowers: {total_borrowers}")

total_loan_amount = df["LoanAmount"].sum()
print(f"Total Loan Amount: {total_loan_amount:,.2f}")

avg_loan_amount = df["LoanAmount"].mean()
print(f"Average Loan Amount: {avg_loan_amount:,.2f}")

default_rate = (df[df["Default"] == 1].shape[0] / total_borrowers) * 100
print(f"Default Rate (%): {default_rate:.2f}%")

avg_interest_rate = df["InterestRate"].mean()
print(f"Average Interest Rate: {avg_interest_rate:.2f}")

avg_dti = df["DTIRatio"].mean()
print(f"Average DTI Ratio: {avg_dti:.2f}")

avg_credit_score = df["CreditScore"].mean()
print(f"Average Credit Score: {avg_credit_score:.2f}")

avg_employment = df["MonthsEmployed"].mean()
print(f"Average Months Employed: {avg_employment:.2f}")

print(f"\nData Ingestion Time: {ingestion_time:.4f} seconds")


# MODEL KPIs (Classification Metrics)
print("\nMODEL PERFORMANCE KPIs:-")

features = ["LoanAmount", "InterestRate", "DTIRatio", "CreditScore", "MonthsEmployed"]
X = df[features]
y = df["Default"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# KPI Calculations
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC–ROC Score: {auc:.4f}")

print("\nConfusion Matrix:")
print(f"TP (True Positives): {tp}")
print(f"TN (True Negatives): {tn}")
print(f"FP (False Positives): {fp}")
print(f"FN (False Negatives): {fn}")
