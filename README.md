# CreditPath AI – Loan Repayment Prediction

CreditPath AI is a machine learning project that predicts whether a borrower is **likely to repay a loan or not**, based on key financial and credit-related features.

The project covers the **complete ML workflow** from data cleaning to final prediction.

---

## Features

- Data preprocessing and cleaning  
- Model training and comparison  
- Algorithms used:
  - Logistic Regression
  - XGBoost
  - LightGBM
- Model evaluation using AUC-ROC and Confusion Matrix  
- Best model selection (XGBoost)  
- Final prediction using user input  
- Input validation with minimal required features  

---

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost, LightGBM  
- Matplotlib  

---

## Model Performance

| Model | AUC-ROC |
|------|--------|
| Logistic Regression | 0.70 |
| LightGBM | 0.79 |
| **XGBoost** | **0.83** |

---

## How It Works

1. Clean and preprocess loan data  
2. Train multiple ML models  
3. Evaluate and compare performance  
4. Select the best model  
5. Predict loan repayment outcome  

---

## Sample Output

```text
Prediction: Borrower is likely to repay the loan
