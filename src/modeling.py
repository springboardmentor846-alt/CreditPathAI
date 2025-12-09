"""
Modeling pipeline for Prosper engineered dataset.

Usage:
    cd C:\\Users\\LENOVO T440\\Desktop\\CreditPathAI\\DayaSagarCreditPath\\src
    python modeling.py
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             precision_score, recall_score, f1_score, classification_report)
from sklearn.ensemble import RandomForestClassifier

# Optional: xgboost and lightgbm (install if you want them)
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

# ----------------- Paths -----------------
BASE = Path(__file__).resolve().parents[1]   # project root
DATA_IN = BASE / "data" / "prosper_engineered.csv"
MODELS_DIR = BASE / "models"
REPORTS_DIR = BASE / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Config -----------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ----------------- Helpers -----------------
def save_plot(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def evaluate_model(name, model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n===== {name} EVALUATION =====")
    print(f"AUC-ROC : {auc:.4f}")
    print(f"Precision: {prec:.4f}   Recall: {rec:.4f}   F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    # ROC curve
    fig, ax = plt.subplots(figsize=(6,5))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    ax.plot([0,1],[0,1],"--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve: {name}")
    ax.legend(loc="lower right")
    save_plot(fig, REPORTS_DIR / f"roc_{name}.png")

    # Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(4,3))
    cax = ax.matshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha="center", va="center", color="black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {name}")
    save_plot(fig, REPORTS_DIR / f"cm_{name}.png")

    return {"auc": auc, "precision": prec, "recall": rec, "f1": f1, "cm": cm}

# ----------------- Main -----------------
def main():
    print("Loading dataset:", DATA_IN)
    if not DATA_IN.exists():
        raise FileNotFoundError(f"Engineered data not found at: {DATA_IN}")

    df = pd.read_csv(DATA_IN, low_memory=False)
    print("Loaded shape:", df.shape)

    # Basic column selection: drop non-feature columns if present
    drop_cols = [
        "loanstatus", "loanstatus_clean", "target_default", "listingcreationdate",
        "closeddate", "loanoriginationdate", "datecreditpulled"
    ]
    # Keep target separate
    if "target_default" not in df.columns:
        raise KeyError("target_default column not found in engineered data. Create target first.")

    y = df["target_default"].astype(int)
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + ["target_default"], errors='ignore')

    # Keep only numeric and a few encoded columns. Convert non-numeric to numeric where possible
    X = X.select_dtypes(include=[np.number]).fillna(0)  # simple; pipelines can do better later
    print("Features shape after numeric selection:", X.shape)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    print("Train / Test sizes:", X_train.shape, X_test.shape)
    print("Class balance (train):", y_train.value_counts().to_dict())
    print("Class balance (test):", y_test.value_counts().to_dict())

    # ---------------- Baseline: Logistic Regression ----------------
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="liblinear", class_weight="balanced", random_state=RANDOM_STATE))
    ])

    print("\nTraining: Logistic Regression (baseline)...")
    pipe_lr.fit(X_train, y_train)
    joblib.dump(pipe_lr, MODELS_DIR / "logistic_baseline.pkl")
    print("Saved logistic_baseline.pkl")

    lr_metrics = evaluate_model("logistic_baseline", pipe_lr, X_test, y_test)

    # ---------------- Random Forest (fast tree baseline) ----------------
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced")
    print("\nTraining: Random Forest (baseline)...")
    rf.fit(X_train, y_train)
    joblib.dump(rf, MODELS_DIR / "rf_baseline.pkl")
    print("Saved rf_baseline.pkl")

    rf_metrics = evaluate_model("rf_baseline", rf, X_test, y_test)

    # ----------------- XGBoost (if available) -----------------
    xgb_metrics = None
    if xgb is not None:
        print("\nTraining: XGBoost (simple params)...")
        xgb_clf = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        xgb_clf.fit(X_train, y_train)
        joblib.dump(xgb_clf, MODELS_DIR / "xgb_baseline.pkl")
        xgb_metrics = evaluate_model("xgb_baseline", xgb_clf, X_test, y_test)
    else:
        print("XGBoost not installed; skipping XGBoost step.")

    # ----------------- LightGBM (if available) -----------------
    lgb_metrics = None
    if lgb is not None:
        print("\nTraining: LightGBM (simple params)...")
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=-1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        lgb_clf.fit(X_train, y_train)
        joblib.dump(lgb_clf, MODELS_DIR / "lgb_baseline.pkl")
        lgb_metrics = evaluate_model("lgb_baseline", lgb_clf, X_test, y_test)
    else:
        print("LightGBM not installed; skipping LightGBM step.")

    # ---------------- Feature importance from RF ----------------
    try:
        feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        feat_imp.head(30).to_csv(REPORTS_DIR / "feature_importance_rf_top30.csv")
        print("\nSaved RF feature importance to reports.")
    except Exception as e:
        print("Feature importance error:", e)

    # ---------------- Summary ----------------
    results = {
        "logistic": lr_metrics,
        "random_forest": rf_metrics,
        "xgboost": xgb_metrics,
        "lightgbm": lgb_metrics
    }
    pd.DataFrame.from_dict({k:v for k,v in results.items() if v is not None}, orient='index').to_csv(REPORTS_DIR / "model_performance_summary.csv")
    print("\nModeling completed. Reports saved to:", REPORTS_DIR)
    print("Models saved to:", MODELS_DIR)

if __name__ == "__main__":
    main()
