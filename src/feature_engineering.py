import pandas as pd
import numpy as np
from pathlib import Path

def feature_engineering():

    # Load ingested dataset
    df = pd.read_csv("../data/prosper_ingested.csv", low_memory=False)

    # ============== 1. TARGET ENGINEERING ==============
    # Map LoanStatus to binary default (1 = default, 0 = paid)
    default_statuses = [
        "Defaulted", "Chargedoff", "Charged Off", 
        "Past Due", "Delinquent"
    ]
    
    df["loanstatus_clean"] = df["loanstatus"].astype(str).str.replace(" ", "").str.lower()

    df["target_default"] = df["loanstatus_clean"].apply(
        lambda x: 1 if any(s.replace(" ", "").lower() in x for s in default_statuses) else 0
    )

    # ============== 2. CREDIT AGE FEATURES ==============
    if "listingcreationdate" in df.columns and "firstrecordedcreditline" in df.columns:
        df["firstrecordedcreditline"] = pd.to_datetime(df["firstrecordedcreditline"], errors='coerce')
        df["listingcreationdate"] = pd.to_datetime(df["listingcreationdate"], errors='coerce')

        df["credit_age_months"] = (
            (df["listingcreationdate"] - df["firstrecordedcreditline"]).dt.days / 30
        )

    # ============== 3. LOAN AMOUNT LOG SCALE ==============
    if "loanoriginalamount" in df.columns:
        df["log_loan_amount"] = np.log1p(df["loanoriginalamount"])

    # ============== 4. INCOME FEATURES ==============
    if "statedmonthlyincome" in df.columns:
        df["log_monthly_income"] = np.log1p(df["statedmonthlyincome"])

        if "loanoriginalamount" in df.columns:
            df["income_to_loan_ratio"] = df["statedmonthlyincome"] / df["loanoriginalamount"]

    # ============== 5. INSTALLMENT RATIOS ==============
    if "monthlyloanpayment" in df.columns and "statedmonthlyincome" in df.columns:
        df["installment_to_income_ratio"] = df["monthlyloanpayment"] / (df["statedmonthlyincome"] + 1)

    # ============== 6. CREDIT UTILIZATION (if available) ==============
    if "currentcreditlines" in df.columns and "opencreditlines" in df.columns:
        df["credit_utilization"] = df["opencreditlines"] / (df["currentcreditlines"] + 1)

    # ============== 7. DELINQUENCY FLAGS ==============
    if "delinquencieslast7years" in df.columns:
        df["has_delinquency"] = df["delinquencieslast7years"].apply(lambda x: 1 if x > 0 else 0)

    # ============== 8. PUBLIC RECORD FLAG ==============
    if "publicrecords" in df.columns:
        df["public_record_flag"] = df["publicrecords"].apply(lambda x: 1 if x > 0 else 0)

    # ============== 9. ENCODE RISK RATINGS ==============
    if "prosperrating_alpha" in df.columns:
        df = pd.get_dummies(df, columns=["prosperrating_alpha"], prefix="rating", drop_first=True)

    if "employmentstatus" in df.columns:
        df = pd.get_dummies(df, columns=["employmentstatus"], prefix="emp", drop_first=True)

    if "incomerange" in df.columns:
        df = pd.get_dummies(df, columns=["incomerange"], prefix="inc", drop_first=True)

    # ============== 10. SAVE FILE ==============
    output_path = "../data/prosper_engineered.csv"
    df.to_csv(output_path, index=False)

    print("\nFeature Engineering Completed Successfully!")
    print("Saved to:", output_path)
    print("Final Shape:", df.shape)


if __name__ == "__main__":
    feature_engineering()
