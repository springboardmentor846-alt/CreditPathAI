import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from src.utils.risk_logic import risk_category
from src.utils.recommendation_engine import recommend_action
from src.utils.recovery_recommendations import recovery_action
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def load_data():
    path = Path("data/processed/cleaned_loan_data.csv")
    print(f"📌 Loading cleaned data from: {path}")
    df = pd.read_csv(path)
    return df


def preprocess(df):
    print("📌 Preprocessing data...")

    y = df["status"]
    X = df.drop("status", axis=1)

    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_xgboost(X_train, X_test, y_train, y_test):
    print("🚀 Training XGBoost model...")

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    # Example risk category for the first prediction
    example_prob = prob[0]
    example_category = risk_category(example_prob)

    print("\nExample predicted probability:", example_prob)
    print("Risk Category:", example_category)
    prob = model.predict_proba(X_test)[0][1]
    category = risk_category(prob)
    print("Example Risk Category:", category)
    
    # Example usage
    category = risk_category(prob)
    recommendation = recommend_action(category)
    recommendation = recovery_action(category)


    print("Example Probability:", prob)
    print("Risk Category:", category)
    print("Recommended Action:", recommendation)
    print("Recovery Recommendation:", recommendation)


    print("\n📌 XGBoost Classification Report:")
    print(classification_report(y_test, preds))

    print("📌 XGBoost AUC-ROC:", roc_auc_score(y_test, proba))

    joblib.dump(model, "src/models/xgboost_model.pkl")
    print("💾 XGBoost model saved at src/models/xgboost_model.pkl")


def train_lightgbm(X_train, X_test, y_train, y_test):
    print("🚀 Training LightGBM model...")

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("\n📌 LightGBM Classification Report:")
    print(classification_report(y_test, preds))

    print("📌 LightGBM AUC-ROC:", roc_auc_score(y_test, proba))

    joblib.dump(model, "src/models/lightgbm_model.pkl")
    print("💾 LightGBM model saved at src/models/lightgbm_model.pkl")


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)

    train_xgboost(X_train, X_test, y_train, y_test)
    train_lightgbm(X_train, X_test, y_train, y_test)
