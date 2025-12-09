import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def run_eda():

    # ---------- Load Ingested Dataset ----------
    data_path = Path("../data/prosper_ingested.csv")
    df = pd.read_csv(data_path, low_memory=False)

    print("\nBASIC INFO")
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())

    print("\nDATA TYPES")
    print(df.dtypes)

    # ---------- Missing Values ----------
    print("\nMISSING VALUES")
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing.head(20))

    # Save missing table
    missing.to_csv("../data/missing_values_report.csv")

    # ---------- Target Variable (LoanStatus) ----------
    if "loanstatus" in df.columns:
        print("\nLOAN STATUS DISTRIBUTION")
        print(df["loanstatus"].value_counts(dropna=False))

        df["loanstatus"].value_counts().plot(kind="bar", figsize=(10,5))
        plt.title("LoanStatus Distribution")
        plt.xlabel("Loan Status")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("../data/loanstatus_distribution.png")
        plt.close()

    # ---------- Numeric Columns Summary ----------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\nNUMERIC SUMMARY")
    print(df[numeric_cols].describe().T.head(15))

    df[numeric_cols].describe().T.to_csv("../data/numeric_summary.csv")

    # ---------- Correlation Heatmap ----------
    print("\nCORRELATION HEATMAP")

    corr = df[numeric_cols].corr()

    plt.figure(figsize=(14,12))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.savefig("../data/correlation_heatmap.png")
    plt.close()

    # ---------- Sample Save ----------
    df.head(50).to_csv("../data/sample_head.csv", index=False)

    print("\nEDA COMPLETED SUCCESSFULLY")
    print("Missing values saved at: ../data/missing_values_report.csv")
    print("LoanStatus plot saved at: ../data/loanstatus_distribution.png")
    print("Correlation heatmap saved at: ../data/correlation_heatmap.png")
    print("Numeric summary saved at: ../data/numeric_summary.csv")

if __name__ == "__main__":
    run_eda()
