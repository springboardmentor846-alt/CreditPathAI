import pandas as pd
import os

def load_csv(path: str) -> pd.DataFrame:
    """Loads a CSV file and checks if it exists."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")

    df = pd.read_csv(path)
    print(f"[INFO] Loaded: {path} | Shape before cleaning: {df.shape}")
    return df


def clean_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Removes null values and returns cleaned dataset."""
    null_count = df.isnull().sum().sum()

    if null_count > 0:
        print(f"[WARNING] {name} contains {null_count} null values. Removing them...")
        df = df.dropna()
    else:
        print(f"[INFO] {name} has no null values.")

    print(f"[INFO] {name} Shape after cleaning: {df.shape}")
    return df


def load_datasets():
    """
    Loads BOTH datasets:
    - loan_data.csv
    - Bank_loan.csv
    Cleans them by removing null values.
    Returns cleaned DataFrames.
    """
    loan_data_path = r"C:\\Users\\DELL\\OneDrive\\Desktop\\CreditPathAI\\CreditPathAIDayasagar\\loan_data.csv"
    bank_loan_path = r"C:\\Users\\DELL\\OneDrive\\Desktop\\CreditPathAI\\CreditPathAIDayasagar\\Bank_loan.csv"

    # Load
    loan_df = load_csv(loan_data_path)
    bank_df = load_csv(bank_loan_path)

    # Clean
    loan_df = clean_dataset(loan_df, "Loan Data")
    bank_df = clean_dataset(bank_df, "Bank Loan Data")

    print("\n[INFO] Both datasets loaded & cleaned successfully.")
    return loan_df, bank_df


if __name__ == "__main__":
    loan_df, bank_df = load_datasets()

    print("\n[INFO] Loan Data Columns:")
    print(loan_df.columns.tolist())

    print("\n[INFO] Bank Loan Data Columns:")
    print(bank_df.columns.tolist())

