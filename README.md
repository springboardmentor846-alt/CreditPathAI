CreditPathAI – Loan Default Prediction & Recovery Recommendation System

CreditPathAI is a lightweight machine learning project that predicts loan default risk and recommends actions to improve recovery.
It uses open-source tools, simple datasets, and a clean workflow so anyone can understand or extend it.

📌 What This Project Does

Predicts default risk for loan applicants

Categorizes borrowers into Low / Medium / High Risk

Suggests recovery actions based on predicted risk

Exposes predictions through a FastAPI endpoint

Includes a simple dashboard for visualizing risk distribution

📁 Datasets

You only need the dataset downloaded from Kaggle.

Dataset Description:
loan_data_final.csv

It contains 9,578 rows and 14 columns.

📑 Column-by-Column Description


| Column                | Type        | Description                                                                                                                        |
| --------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **credit.policy**     | int (0/1)   | Whether the applicant meets the lending company's credit standards. *(1 = approved under credit policy, 0 = lower credit quality)* |
| **purpose**           | categorical | Purpose of the loan such as: `credit_card`, `debt_consolidation`, `home_improvement`, `small_business`, etc.                       |
| **int.rate**          | float       | Interest rate on the loan. Higher interest rates often indicate higher borrower risk.                                              |
| **installment**       | float       | Fixed monthly payment amount the borrower must repay.                                                                              |
| **log.annual.inc**    | float       | Natural logarithm of the borrower’s annual income. Useful for income normalization.                                                |
| **dti**               | float       | Debt-to-income ratio — total debt divided by annual income. Higher DTI indicates higher risk.                                      |
| **fico**              | int         | Borrower’s FICO credit score. Strong predictor of repayment behavior.                                                              |
| **days.with.cr.line** | float       | Number of days the borrower has held credit lines. Indicates credit age.                                                           |
| **revol.bal**         | float       | Revolving balance — total outstanding credit card debt.                                                                            |
| **revol.util**        | float or %  | Revolving line utilization rate (percentage of credit limit used).                                                                 |
| **inq.last.6mths**    | int         | Number of credit inquiries made in the past 6 months. More inquiries = higher risk.                                                |
| **delinq.2yrs**       | int         | Number of times the borrower was delinquent in the past 2 years.                                                                   |
| **pub.rec**           | int         | Number of derogatory public records (bankruptcies, judgments, etc.).                                                               |
| **not.fully.paid**    | int (0/1)   | **Target variable** — whether the borrower failed to fully repay the loan. *(1 = default, 0 = repaid)*                             |

After Processing:

📑 Derived Feature Table

| Feature Name                    | Meaning                                                                          | Why It Matters                                                                                                                                | How It Was Derived                                                                                                                           |
| ------------------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **repayment_velocity_proxy**    | An approximate measure of how quickly and reliably a borrower repays their debt. | Borrowers with higher repayment velocity are less likely to default. Repayment history is one of the strongest predictors of credit behavior. | `repayment_velocity_proxy = (fico / 850) / (dti + ε)` Combines **high credit score** and **low debt burden** to estimate repayment tendency. |
| **approx_credit_limit**         | Estimated total credit limit of the borrower’s revolving accounts.               | Useful when actual credit limit is not present. Credit utilization accuracy improves default prediction.                                      | `approx_credit_limit = revol_bal / (revol_util + ε)` Only available if `revol_util` > 0.                                                     |
| **credit_utilization**          | Actual percentage of credit used by the borrower (0–1 scale).                    | Higher utilization strongly correlates with higher default risk.                                                                              | `credit_utilization = revol_bal / approx_credit_limit`                                                                                       |
| **annual_inc**                  | Actual annual income reconstructed from its logarithmic form.                    | Many credit risk models perform better when both raw and transformed versions are available.                                                  | `annual_inc = exp(log_annual_inc)`                                                                                                           |
| **debt_to_income_calc**         | Alternative DTI metric measured using actual income.                             | Provides more accurate and interpretable debt burden compared to raw DTI.                                                                     | `debt_to_income_calc = installment / (annual_inc / 12 + ε)`                                                                                  |
| **credit_age_years**            | Age of borrower’s credit history in years.                                       | Borrowers with longer credit histories tend to be more financially stable.                                                                    | `credit_age_years = days_with_cr_line / 365` (or `days_with_credit_line / 365`)                                                              |
| **has_recent_inq**              | Indicates whether the borrower had a credit inquiry in the last 6 months.        | Frequent inquiries often signal financial stress.                                                                                             | `has_recent_inq = 1 if inq_last_6mths > 0 else 0`                                                                                            |
| **has_past_delinquency**        | Indicates if borrower was delinquent in the past 2 years.                        | Strong early warning signal for future defaults.                                                                                              | `has_past_delinquency = 1 if delinq_2yrs > 0 else 0`                                                                                         |
| **target_default**              | Final binary label representing loan outcome.                                    | Required for supervised ML models. Simplifies inconsistent original labels.                                                                   | `target_default = 1 if not_fully_paid in [1, yes, y, true] else 0`                                                                           |
| **purpose_*** (one-hot encoded) | One-hot encoded categories for loan purpose.                                     | Helps model detect risk patterns specific to certain loan types.                                                                              | Created using `pd.get_dummies(purpose)`                                                                                                      |

🛠 Tech Used

Python, Pandas, scikit-learn

XGBoost / LightGBM

FastAPI for serving predictions

React + Plotly.js for dashboard

SQLite/PostgreSQL for data storage

Docker for containerization

🚀 Project Structure
creditpathai/
│
├── data/
│   ├── raw/          # Kaggle dataset here
│   └── processed/    # Cleaned data saved here
│
├── notebooks/        # EDA and experiments
│
├── src/
│   ├── ingestion/    # Load data
│   ├── features/     # Feature engineering
│   ├── models/       # ML models
│   ├── api/          # FastAPI app
│   └── dashboard/    # React UI
│
├── requirements.txt
├── README.md
└── .gitignore



