📊 EDA Report
Dataset Overview

Total rows: 500

Total columns: 21

Data ingestion: Successful

Target variable: Recovery_Status

Data Types

Numerical columns: Age, Monthly_Income, Num_Dependents, Loan_Amount, Loan_Tenure, Interest_Rate, Collateral_Value, Outstanding_Loan_Amount, Monthly_EMI, Num_Missed_Payments, Days_Past_Due, Collection_Attempts

Categorical columns: Gender, Employment_Type, Loan_Type, Payment_History, Recovery_Status, Collection_Method, Legal_Action_Taken, Borrower_ID, Loan_ID

Missing Values

No missing values found in the dataset.

Duplicates

Checked for duplicates (drop if needed).

Numerical Summary

Values analyzed using mean, median, min, max & distribution

Loan & Income columns show variability across borrowers

Missed payments and days past due are key behavioral indicators

Categorical Summary

Gender, Employment Type, Loan Type distribution observed

Payment_History shows risk category behavior

Recovery_Status is used for default prediction

Correlation Insights

Num_Missed_Payments and Days_Past_Due positively linked with default risk

Higher outstanding loan amount increases recovery difficulty

Income vs loan balance influences repayment capability

Key Findings

Dataset is clean and ready for modeling

No null values → smooth preprocessing

Strong predictors of default: Payment_History, Num_Missed_Payments, Days_Past_Due, Outstanding_Loan_Amount

Suitable for ML classification model for default prediction
