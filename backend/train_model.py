import os
import warnings
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

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

    # -------- Numerical Feature Engineering --------
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

    # -------- OneHot Encoding --------
    categorical_cols = [
        'Education', 'EmploymentType', 'MaritalStatus',
        'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner'
    ]

    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    encoded = ohe.fit_transform(df_processed[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(categorical_cols),
        index=df_processed.index
    )

    df_processed = pd.concat(
        [df_processed.drop(columns=categorical_cols), encoded_df],
        axis=1
    )

    df_processed.columns = df_processed.columns.str.replace(" ", "_")

    # -------- Final features --------
    feature_cols = df_processed.drop(columns=["Default"]).columns.tolist()

    X = df_processed[feature_cols]
    y = df_processed["Default"]

    print(f"Features: {X.shape[1]} | Records: {X.shape[0]}")

    return X, y, ohe, feature_cols

X, y, ohe, feature_cols = feature_engineering(df)

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
    n_estimators=900,
    max_depth=5,
    learning_rate=0.03,
    min_child_weight=40,
    gamma=0.25,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_alpha=0.6,
    reg_lambda=1.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    objective="binary:logistic",
    tree_method="hist",
    random_state=42
)

xgb_model.fit(X_train, y_train)

auc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
print(f"AUC-ROC: {auc_xgb:.4f}")

print("\n=== LIGHTGBM ===")

# ---- monotonic constraints (credit-risk best practice) ----
monotone_constraints = {
    "CreditScore": -1,
    "Income": -1,
    "LoanToIncome": 1,
    "DTIRatio": 1,
    "PaymentToIncome": 1
}

lgb_model = lgb.LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.75,
    reg_alpha=0.3,
    reg_lambda=1.0,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    objective="binary",
    metric="auc",
    random_state=42,
    monotone_constraints=[
        monotone_constraints.get(f, 0) for f in X_train.columns
    ]
)

lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(0)  # silences output
    ]
)

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
joblib.dump(ohe, f"{MODEL_DIR}/onehot_encoder.pkl")
joblib.dump(feature_cols, f"{MODEL_DIR}/feature_columns.pkl")

joblib.dump(X_test, f"{MODEL_DIR}/X_test.pkl")
joblib.dump(y_test, f"{MODEL_DIR}/y_test.pkl")

print("\n✅ All models and preprocessing artifacts saved in /models/")
