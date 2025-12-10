# Milestone 1 — Sample Datasets Collected

This project uses publicly available datasets from the Home Credit Default Risk competition on Kaggle.  
These datasets simulate real-world borrower profiles and repayment behavior, making them suitable for the CreditPathAI risk prediction and recovery model.

## 1. Source of Data
All files were downloaded from:
- Kaggle: Home Credit Default Risk Dataset

## 2. Files Collected (Stored Locally in data/raw/)
- application_train.csv
- bureau.csv
- bureau_balance.csv
- previous_application.csv
- POS_CASH_balance.csv
- credit_card_balance.csv
- installments_payments.csv

(Note: Raw data is NOT uploaded to GitHub as per best practices.  
A full preview is available in notebooks/00_data_overview.ipynb.)

## 3. Purpose of Each File
- application_train.csv → Main customer + target variable
- bureau.csv → Customer's past external loans
- bureau_balance.csv → Monthly bureau loan status
- previous_application.csv → Past loan applications
- POS_CASH_balance.csv → POS loan details
- credit_card_balance.csv → Credit card behavior
- installments_payments.csv → Payment patterns

## 4. Evaluation Criteria
- Data arranged in required fields (see docs/KPIs.md)
- Data preview notebook added (notebooks/00_data_overview.ipynb)

