import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from pathlib import Path

DATA_PATH = Path("data/processed/cleaned_loan_data.csv")
MODEL_PATH = Path("src/models/baseline_model.pkl")

def load_data(path):
    print(f"Loading cleaned dataset from: {path}")
    df = pd.read_csv(path)
    return df

def train_baseline(df):
    # Target variable
    if "status" not in df.columns:
        raise ValueError("Target column 'status' not found.")

    X = df.drop("status", axis=1)
    y = df["status"]

    # Separate categorical and numeric columns
    numeric_cols = X.select_dtypes(include=["int64","float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    # Create preprocessing transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Pipeline: preprocessing → logistic regression
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training baseline Logistic Regression model...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    print("\n📌 Classification Report:")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_prob)
    print(f"📌 AUC-ROC Score: {auc:.4f}")

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved at: {MODEL_PATH}")

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    train_baseline(df)
