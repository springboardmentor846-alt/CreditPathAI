import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay

from src.features.feature_engineering import engineer_features


def load_data():
    conn = sqlite3.connect("data/loans.db")
    df = pd.read_sql("SELECT * FROM kaggle_Loan_default", conn)
    conn.close()
    return df


def plot_roc():
    df = load_data()
    df = engineer_features(df)

    target = "Default"

    numeric_features = [
        "Age", "Income", "LoanAmount", "CreditScore",
        "MonthsEmployed", "NumCreditLines",
        "InterestRate", "LoanTerm",
        "DTIRatio", "LoanIncomeRatio"
    ]

    X = df[numeric_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    RocCurveDisplay.from_estimator(model, X_test, y_test)

    plt.title("ROC Curve - Logistic Regression")
    plt.savefig("reports/plots/roc_curve.png")
    plt.show()   # 👈 THIS LINE IS REQUIRED


if __name__ == "__main__":
    plot_roc()
