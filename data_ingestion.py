import pandas as pd
import time
import numpy as np

def load_data(path):
    start = time.time()
    df = pd.read_csv(path)
    end = time.time()

    print(f"[INFO] Data loaded in {end - start:.3f} seconds")
    print(f"[INFO] Initial shape: {df.shape}")

    return df

def clean_data(df):
    print("[INFO] Cleaning data...")

    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    print(f"[INFO] Removed duplicates: {before - after}")

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])


    df = df[df['Age'] >= 18]
    df = df[df['Income'] > 0]
    df = df[df['LoanAmount'] > 0]

    print(f"[INFO] Shape after cleaning: {df.shape}")

    return df


def transform_data(df):
    print("[INFO] Transforming data...")


    cols_to_encode = [
        "Education",
        "EmploymentType",
        "MaritalStatus",
        "LoanPurpose"
    ]

    cols_to_encode = [col for col in cols_to_encode if col in df.columns]

    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    print(f"[INFO] Shape after encoding: {df.shape}")

    return df

def save_data(df, path):
    df.to_csv(path, index=False)
    print(f"[INFO] Cleaned data saved to {path}")

def data_quality_report(df):
    print("\nDATA QUALITY KPIs:-")
    print(f"Total Rows: {len(df)}")
    print(f"Missing Values After Cleaning:\n{df.isnull().sum()}")
    print(f"Memory Usage: {df.memory_usage().sum() / 1024**2:.2f} MB")

if __name__ == "__main__":
    df = load_data("data/loan_data.csv")
    df = clean_data(df)
    df = transform_data(df)
    data_quality_report(df)
    save_data(df, "data/loan_data_clean.csv")
