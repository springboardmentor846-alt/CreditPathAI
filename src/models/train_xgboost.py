import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from src.features.feature_engineering import engineer_features


def load_data():
    conn = sqlite3.connect("data/loans.db")
    df = pd.read_sql("SELECT * FROM kaggle_Loan_default", conn)
    conn.close()
    return df


def train():
    df = load_data()
    df = engineer_features(df)

    target = "Default"

    features = [
        "Age", "Income", "LoanAmount", "CreditScore",
        "MonthsEmployed", "NumCreditLines",
        "InterestRate", "LoanTerm",
        "DTIRatio", "LoanIncomeRatio"
    ]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nXGBoost ROC-AUC: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_pred_proba)
    plt.title("XGBoost ROC Curve")
    plt.savefig("reports/plots/xgboost_roc.png")
    plt.close()

    # Precision-Recall Curve
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba)
    plt.title("XGBoost Precision-Recall Curve")
    plt.savefig("reports/plots/xgboost_precision_recall.png")
    plt.close()

    print("XGBoost plots saved in reports/plots/")


if __name__ == "__main__":
    train()
