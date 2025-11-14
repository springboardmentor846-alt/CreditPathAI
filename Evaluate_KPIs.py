import pandas as pd

df = pd.read_csv("data/loan_data.csv")

print("\nBASIC KPIs:-")

total_borrowers = len(df)
print(f"Total Borrowers: {total_borrowers}")

total_loan_amount = df["LoanAmount"].sum()
print(f"Total Loan Amount: {total_loan_amount:,.2f}")

avg_loan_amount = df["LoanAmount"].mean()
print(f"Average Loan Amount: {avg_loan_amount:,.2f}")

default_rate = (df[df["Default"] == 1].shape[0] / total_borrowers) * 100
print(f"Default Rate (%): {default_rate:.2f}%")

avg_interest_rate = df["InterestRate"].mean()
print(f"Average Interest Rate: {avg_interest_rate:.2f}")

avg_dti = df["DTIRatio"].mean()
print(f"Average DTI Ratio: {avg_dti:.2f}")

avg_credit_score = df["CreditScore"].mean()
print(f"Average Credit Score: {avg_credit_score:.2f}")

avg_employment = df["MonthsEmployed"].mean()
print(f"Average Months Employed: {avg_employment:.2f}")
