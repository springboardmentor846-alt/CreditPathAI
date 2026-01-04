# ================================
# EXPLORATORY DATA ANALYSIS (EDA)
# Loan Default Dataset
# ================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------
# 1. Load Dataset
# ----------------
df = pd.read_csv("Loan_Default.csv")

print("\nDataset Shape:", df.shape)
print("\nData Types:\n")
print(df.dtypes)

# -----------------------
# 2. Missing Value Check
# -----------------------
print("\nMissing Values:\n")
print(df.isnull().sum())

# -----------------------
# 3. Target Variable
# -----------------------
print("\nTarget Variable Distribution:\n")
print(df["Default"].value_counts())

plt.figure()
sns.countplot(x="Default", data=df)
plt.title("Loan Default Distribution")
plt.show()

# -----------------------
# 4. Numerical Summary
# -----------------------
print("\nNumerical Summary:\n")
print(df.describe())

num_cols = [
    "Age", "Income", "LoanAmount", "CreditScore",
    "MonthsEmployed", "NumCreditLines",
    "InterestRate", "LoanTerm", "DTIRatio"
]

# -----------------------
# 5. Histograms
# -----------------------
for col in num_cols:
    plt.figure()
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# -----------------------
# 6. Boxplots (Outliers)
# -----------------------
for col in num_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# -----------------------
# 7. Categorical Analysis
# -----------------------
cat_cols = [
    "Education", "EmploymentType", "MaritalStatus",
    "HasMortgage", "HasDependents",
    "LoanPurpose", "HasCoSigner"
]

for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=col, data=df, order=df[col].value_counts().index)
    plt.title(f"Count Plot of {col}")
    plt.show()

# ------------------------------------
# 8. Default vs Categorical Features
# ------------------------------------
for col in cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, hue="Default", data=df)
    plt.xticks(rotation=45)
    plt.title(f"{col} vs Loan Default")
    plt.show()

# ------------------------------------
# 9. Correlation Heatmap
# ------------------------------------
plt.figure(figsize=(12, 8))
corr = df[num_cols + ["Default"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------------
# 10. Feature Engineering (Optional)
# ------------------------------------
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[18, 25, 35, 45, 60, 100],
    labels=["18-25", "26-35", "36-45", "46-60", "60+"]
)

plt.figure()
sns.countplot(x="AgeGroup", hue="Default", data=df)
plt.title("Age Group vs Default")
plt.show()

print("\nEDA Completed Successfully ✅")
