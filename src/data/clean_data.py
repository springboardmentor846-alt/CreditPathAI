import pandas as pd

df = pd.read_csv("data/Loan_Default.csv")

df = df.fillna(0)

df.to_csv("data/processed.csv", index=False)
print("Data cleaned and saved!")
