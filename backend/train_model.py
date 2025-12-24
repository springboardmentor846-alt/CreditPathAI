import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import warnings
import joblib

warnings.filterwarnings('ignore')

DATA_PATH = "Loan_default.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDefault rate: {df['Default'].mean():.2%}")

def preprocess_data(df):
    df_clean = df.copy()

    # Drop identifier
    df_clean.drop(columns=["LoanID"], inplace=True)

    # Binary categorical variables
    binary_cols = ['HasMortgage', 'HasDependents', 'HasCoSigner']
    for col in binary_cols:
        df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})

    # Encode categorical variables
    cat_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']
    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le

    # Features & target
    X = df_clean.drop('Default', axis=1)
    y = df_clean['Default']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale numerical features
    num_cols = [
        'Age', 'Income', 'LoanAmount', 'CreditScore',
        'MonthsEmployed', 'NumCreditLines',
        'InterestRate', 'LoanTerm', 'DTIRatio'
    ]

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test, y_train, y_test, scaler, label_encoders, num_cols, cat_cols, binary_cols

X_train, X_test, y_train, y_test, scaler, label_encoders, num_cols, cat_cols, binary_cols = preprocess_data(df)

print(f"\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

auc_scores = {}

# 1. Logistic Regression
print("\n")
print("1. LOGISTIC REGRESSION")

lr_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)

lr_model.fit(X_train, y_train)
auc_lr = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])
auc_scores["LogisticRegression"] = auc_lr

joblib.dump(lr_model, f"{MODEL_DIR}/logistic_regression_model.pkl")

print(f"AUC-ROC: {auc_lr:.4f}")

# 2. XGBoost
print("\n")
print("2. XGBOOST")

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    eval_metric='auc',
    use_label_encoder=False
)

xgb_model.fit(X_train, y_train)
auc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
auc_scores["XGBoost"] = auc_xgb

joblib.dump(xgb_model, f"{MODEL_DIR}/xgboost_model.pkl")

print(f"AUC-ROC: {auc_xgb:.4f}")

# 3. LightGBM
print("\n")
print("3. LIGHTGBM")

lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    objective='binary',
    metric='auc',
    reg_alpha=0.1,
    reg_lambda=0.1,
    verbose=-1
)

lgb_model.fit(X_train, y_train)
auc_lgb = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])
auc_scores["LightGBM"] = auc_lgb

joblib.dump(lgb_model, f"{MODEL_DIR}/lightgbm_model.pkl")

print(f"AUC-ROC: {auc_lgb:.4f}")

joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(label_encoders, f"{MODEL_DIR}/label_encoders.pkl")
joblib.dump(X_train.columns.tolist(), f"{MODEL_DIR}/feature_columns.pkl")

print("\n")
print("MODEL TRAINING SUMMARY")

for model, auc in auc_scores.items():
    print(f"{model}: AUC-ROC = {auc:.4f}")

best_model = max(auc_scores, key=auc_scores.get)
print(f"\n✅ Best Model: {best_model}")

print("\n✅ All models and preprocessing artifacts saved in /models/")

joblib.dump(X_test, "models/X_test.pkl")
joblib.dump(y_test, "models/y_test.pkl")
