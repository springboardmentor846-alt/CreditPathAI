import pandas as pd
from pathlib import Path

# Paths
RAW_DATA_PATH = Path("data/raw/Loan_Default.csv")      # change filename if needed
CLEAN_DATA_PATH = Path("data/processed/cleaned_loan_data.csv")

def load_data(path):
    """Load CSV file into a DataFrame"""
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    return df

def clean_data(df):
    """Basic cleaning: handle missing values, drop duplicates, standardize columns"""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop duplicates
    df = df.drop_duplicates()

    # Fill numeric NaNs with median
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical NaNs with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    # Optional: remove unrealistic loan amounts
    if 'loan_amount' in df.columns:
        df = df[df['loan_amount'] > 0]

    print("✅ Data cleaned successfully!")
    print(f"New shape: {df.shape}")
    return df

def save_data(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Cleaned data saved to: {path}")

if __name__ == "__main__":
    df_raw = load_data(RAW_DATA_PATH)
    df_clean = clean_data(df_raw)
    save_data(df_clean, CLEAN_DATA_PATH)
