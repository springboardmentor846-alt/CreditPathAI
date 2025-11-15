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



