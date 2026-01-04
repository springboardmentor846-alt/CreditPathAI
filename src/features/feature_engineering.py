import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features for modeling
    """

    df = df.copy()

    # Debt-to-Income bins
    df["DTI_Band"] = pd.cut(
        df["DTIRatio"],
        bins=[0, 0.2, 0.4, 0.6, 1.0],
        labels=["Low", "Medium", "High", "Very High"]
    )

    # Credit score bins
    df["CreditScore_Band"] = pd.cut(
        df["CreditScore"],
        bins=[300, 580, 670, 740, 850],
        labels=["Poor", "Fair", "Good", "Excellent"]
    )

    # Loan to income ratio
    df["LoanIncomeRatio"] = df["LoanAmount"] / (df["Income"] + 1)

    return df
