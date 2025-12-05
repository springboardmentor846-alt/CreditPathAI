import pandas as pd
from pathlib import Path

CLEAN_DATA_PATH = Path("data/processed/cleaned_loan_data.csv")

def load_data(path):
    print(f"\n📌 Loading cleaned data from: {path}")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    return df

def basic_info(df):
    print("\n📌 Dataset Info:")
    print(df.info())

def missing_values(df):
    print("\n📌 Missing Values Count:")
    print(df.isnull().sum())

def summary_stats(df):
    print("\n📌 Summary Statistics:")
    print(df.describe())

def value_counts_example(df):
    print("\n📌 Value Counts (for default_flag if exists):")
    if "default_flag" in df.columns:
        print(df["default_flag"].value_counts())
    else:
        print("Column 'default_flag' not found.")

def correlations(df):
    print("\n📌 Correlation Matrix (numeric columns only):")

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    # Compute correlation
    corr = numeric_df.corr()

    print(corr)


if __name__ == "__main__":
    df = load_data(CLEAN_DATA_PATH)
    basic_info(df)
    missing_values(df)
    summary_stats(df)
    value_counts_example(df)
    correlations(df)
