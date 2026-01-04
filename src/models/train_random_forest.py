from pyexpat import model
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_score,
    recall_score,
    f1_score
)

from src.features.feature_engineering import engineer_features


def load_data():
    conn = sqlite3.connect("data/loans.db")
    df = pd.read_sql("SELECT * FROM kaggle_Loan_default", conn)
    conn.close()
    return df


def train():
    # ======================
    # Load & prepare data
    # ======================
    df = load_data()
    df = engineer_features(df)
    target = "Default"
    print("\nTarget Distribution:")
    print(df[target].value_counts())
    print("\nTarget Distribution (Normalized):")
    print(df[target].value_counts(normalize=True))


   

    numeric_features = [
        "Age", "Income", "LoanAmount", "CreditScore",
        "MonthsEmployed", "NumCreditLines",
        "InterestRate", "LoanTerm",
        "DTIRatio", "LoanIncomeRatio"
    ]

    X = df[numeric_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # ======================
    # Train Random Forest
    # ======================
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    # ======================
    # Save trained model
    # ======================
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest.pkl")
    print("Model saved to: models/random_forest.pkl")


    # ======================
    # Predictions
    # ======================
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # ======================
    # Evaluation
    # ======================
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nRandom Forest ROC-AUC: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ======================
    # ROC Curve
    # ======================
    RocCurveDisplay.from_predictions(y_test, y_pred_proba)
    plt.title("Random Forest ROC Curve")
    plt.savefig("reports/plots/random_forest_roc.png")
    plt.close()
    print("ROC curve saved to: reports/plots/random_forest_roc.png")

    # ======================
    # Precision-Recall Curve
    # ======================
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba)
    plt.title("Random Forest Precision-Recall Curve")
    plt.savefig("reports/plots/random_forest_precision_recall.png")
    plt.close()
    print("Precision-Recall curve saved to: reports/plots/random_forest_precision_recall.png")

    # ======================
    # Threshold Tuning
    # ======================
    print("\nThreshold Tuning Results:")
    print("Threshold | Precision | Recall | F1-score")

    for threshold in [0.3, 0.4, 0.5]:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)

        precision = precision_score(y_test, y_pred_thresh)
        recall = recall_score(y_test, y_pred_thresh)
        f1 = f1_score(y_test, y_pred_thresh)

        print(f"{threshold:9} | {precision:.2f}      | {recall:.2f}  | {f1:.2f}")

    # ======================
    # Feature Importance
    # ======================
    importances = pd.Series(
        model.feature_importances_,
        index=numeric_features
    ).sort_values(ascending=False)

    print("\nTop Feature Importances:")
    print(importances)

    plt.figure(figsize=(8, 5))
    importances.plot(kind="bar")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig("reports/plots/random_forest_feature_importance.png")
    plt.close()

    print("Feature importance plot saved to: reports/plots/random_forest_feature_importance.png")

    # ======================
    # Interview Summary
    # ======================
    print("""
Model Summary:
- Random Forest captures non-linear risk patterns
- Recall improved for defaulters using threshold tuning
- Feature importance provides business insight
""")


if __name__ == "__main__":
    train()
