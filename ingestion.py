import pandas as pd
import os

def load_csv(path: str) -> pd.DataFrame:
    """Loads any CSV file and checks if it exists."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")

    df = pd.read_csv(path)
    print(f"[INFO] Loaded: {path} | Shape: {df.shape}")
    return df


def load_datasets():
    """
    Loads BOTH datasets:
    - loan_data.csv
    - bank_loan.csv
    Returns them as two separate DataFrames.
    """
    loan_data_path = "C:\\Users\\LENOVO T440\\Desktop\\CreditPathAI\\CreditPathAI\\CreditPathAIDayasagar\\Bank_Loan.csv"

    bank_loan_path = "C:\\Users\\LENOVO T440\\Desktop\\CreditPathAI\\CreditPathAI\\CreditPathAIDayasagar\\loan_data.csv"

    loan_df = load_csv(loan_data_path)
    bank_df = load_csv(bank_loan_path)

    print("\n[INFO] Both datasets loaded successfully.")
    return loan_df, bank_df


if __name__ == "__main__":
    loan_df, bank_df = load_datasets()

    print("\n[INFO] Loan Data Columns:")
    print(loan_df.columns.tolist())

    print("\n[INFO] Bank Loan Data Columns:")
    print(bank_df.columns.tolist())
