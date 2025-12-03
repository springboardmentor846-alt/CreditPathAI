import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA_PATH = "data/loan_data_clean.csv"
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

print("\nLoading cleaned dataset...")
df = pd.read_csv(DATA_PATH)

X = df.drop("Default", axis=1)
y = df["Default"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

cat_cols = []
for col in X.select_dtypes(include=["object"]).columns:
    if X[col].nunique() <= 100:
        cat_cols.append(col)
    else:
        print(f"Skipping high-cardinality column: {col} ({X[col].nunique()} unique values)")

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Data split completed")

print("\nTraining Logistic Regression...")
baseline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])
baseline.fit(X_train, y_train)
baseline_auc = roc_auc_score(y_test, baseline.predict_proba(X_test)[:, 1])
print(f"Baseline AUC: {baseline_auc:.4f}")

print("\nTraining XGBoost...")
xgb_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", XGBClassifier(eval_metric="auc", use_label_encoder=False, random_state=42))
])
xgb_model.fit(X_train, y_train)
xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
print(f"XGBoost AUC: {xgb_auc:.4f}")

print("\nTraining LightGBM...")
lgbm_model = Pipeline([
    ("preprocess", preprocessor),
    ("model", LGBMClassifier(random_state=42))
])
lgbm_model.fit(X_train, y_train)
lgbm_auc = roc_auc_score(y_test, lgbm_model.predict_proba(X_test)[:, 1])
print(f"LightGBM AUC: {lgbm_auc:.4f}")

print("\nHyperparameter tuning for XGBoost...")
param_grid = {
    "model__n_estimators": [200, 300, 500],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__subsample": [0.7, 0.8, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 1.0],
}

search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_grid,
    scoring="roc_auc",
    n_iter=10,
    cv=3,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

search.fit(X_train, y_train)

tuned_xgb = search.best_estimator_
tuned_auc = roc_auc_score(y_test, tuned_xgb.predict_proba(X_test)[:, 1])
print(f"Tuned XGBoost AUC: {tuned_auc:.4f}")

scores = {
    "Baseline": baseline_auc,
    "XGBoost": xgb_auc,
    "LightGBM": lgbm_auc,
    "XGBoost_Tuned": tuned_auc
}

print("\nFinal AUC Scores:")
for name, auc in scores.items():
    print(f"{name}: {auc:.4f}")

best_model_name = max(scores, key=scores.get)
print(f"\nBest Model: {best_model_name}")

best_model = {
    "Baseline": baseline,
    "XGBoost": xgb_model,
    "LightGBM": lgbm_model,
    "XGBoost_Tuned": tuned_xgb
}[best_model_name]

save_path = f"{MODELS_DIR}/best_model.pkl"
joblib.dump(best_model, save_path)

print(f"\nBest model saved at: {save_path}")
print("\nModel Training + Optimization Completed")
