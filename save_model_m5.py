import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv("data/loan_data.csv")
# 
features = [
    "LoanAmount",
    "InterestRate",
    "DTIRatio",
    "CreditScore",
    "MonthsEmployed"
]

X = df[features]
y = df["Default"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(
    max_iter=1000,
    class_weight={0: 1, 1: 2}
)

model.fit(X_scaled, y)

joblib.dump(model, "credit_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully")
