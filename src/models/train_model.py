import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay
)

from src.features.feature_engineering import engineer_features


def load_data():
    conn = sqlite3.connect("data/loans.db")
    df = pd.read_sql("SELECT * FROM kaggle_Loan_default", conn)
    conn.close()
    return df


def train():
    # Load & prepare data
    df = load_data()
    df = engineer_features(df)

    # 🔍 Class imbalance check
    print("\nClass distribution (counts):")
    print(df["Default"].value_counts())

    print("\nClass distribution (percentage):")
    print(df["Default"].value_counts(normalize=True))

    target = "Default"

    numeric_features = [
        "Age", "Income", "LoanAmount", "CreditScore",
        "MonthsEmployed", "NumCreditLines",
        "InterestRate", "LoanTerm",
        "DTIRatio", "LoanIncomeRatio"
    ]

    X = df[numeric_features]
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # ✅ Logistic Regression with class imbalance handling
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nLogistic Regression (Balanced) ROC-AUC: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ✅ SAVE ROC curve (NO display issues)
    RocCurveDisplay.from_predictions(y_test, y_pred_proba)
    plt.title("Logistic Regression ROC Curve")
    plt.savefig("reports/plots/logistic_regression_roc.png")
    plt.close()

    print("\nROC curve saved to: reports/plots/logistic_regression_roc.png")


if __name__ == "__main__":
    train()
