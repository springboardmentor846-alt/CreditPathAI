import os
import warnings
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

DATA_PATH = "Loan_default.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Default rate: {df['Default'].mean():.2%}")

def feature_engineering(df):
    print("=== FEATURE ENGINEERING ===")

    df_processed = df.copy()

    # Drop identifier
    if "LoanID" in df_processed.columns:
        df_processed.drop(columns=["LoanID"], inplace=True)

    # Derived features
    df_processed['LoanToIncome'] = df_processed['LoanAmount'] / df_processed['Income']

    df_processed['MonthlyPayment'] = (
        (df_processed['LoanAmount'] *
         (df_processed['InterestRate'] / 100 / 12) *
         (1 + df_processed['InterestRate'] / 100 / 12) ** df_processed['LoanTerm']) /
        ((1 + df_processed['InterestRate'] / 100 / 12) ** df_processed['LoanTerm'] - 1)
    )

    df_processed['PaymentToIncome'] = (
        df_processed['MonthlyPayment'] / (df_processed['Income'] / 12)
    )

    df_processed['CreditUtilization'] = (
        df_processed['LoanAmount'] / df_processed['CreditScore']
    )

    # Categorical columns
    categorical_cols = [
        'Education', 'EmploymentType', 'MaritalStatus',
        'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
    ]

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col + "_encoded"] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le

    # Final feature list
    feature_cols = [
        'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
        'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
        'LoanToIncome', 'MonthlyPayment', 'PaymentToIncome', 'CreditUtilization'
    ] + [col + "_encoded" for col in categorical_cols]

    X = df_processed[feature_cols]
    y = df_processed['Default']

    print(f"Features: {X.shape[1]} | Records: {X.shape[0]}")

    return X, y, label_encoders, feature_cols

X, y, label_encoders, feature_cols = feature_engineering(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== LOGISTIC REGRESSION ===")

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

lr_model.fit(X_train_scaled, y_train)
auc_lr = roc_auc_score(y_test, lr_model.predict_proba(X_test_scaled)[:, 1])
print(f"AUC-ROC: {auc_lr:.4f}")

print("\n=== XGBOOST ===")

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    eval_metric="auc"
)

xgb_model.fit(X_train, y_train)
auc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
print(f"AUC-ROC: {auc_xgb:.4f}")

print("\n=== LIGHTGBM ===")

lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    objective="binary",
    metric="auc",
    reg_alpha=0.1,
    reg_lambda=0.1,
    verbose=-1
)

lgb_model.fit(X_train, y_train)
auc_lgb = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])
print(f"AUC-ROC: {auc_lgb:.4f}")

print("\n=== MODEL COMPARISON ===")
print(f"Logistic Regression: {auc_lr:.4f}")
print(f"XGBoost:            {auc_xgb:.4f}")
print(f"LightGBM:           {auc_lgb:.4f}")

best_model = max(
    [("Logistic Regression", auc_lr), ("XG Boost", auc_xgb), ("LightGBM", auc_lgb)],
    key=lambda x: x[1]
)[0]

print(f"\n✅ Best Model: {best_model}")

joblib.dump(lr_model, f"{MODEL_DIR}/logistic_regression_model.pkl")
joblib.dump(xgb_model, f"{MODEL_DIR}/xgboost_model.pkl")
joblib.dump(lgb_model, f"{MODEL_DIR}/lightgbm_model.pkl")

joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(label_encoders, f"{MODEL_DIR}/label_encoders.pkl")
joblib.dump(feature_cols, f"{MODEL_DIR}/feature_columns.pkl")

joblib.dump(X_test, f"{MODEL_DIR}/X_test.pkl")
joblib.dump(y_test, f"{MODEL_DIR}/y_test.pkl")

print("\n✅ All models and preprocessing artifacts saved in /models/")
