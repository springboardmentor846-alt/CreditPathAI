Milestone 1 – Requirements & Data Collection

## 1. Project Requirements

### Functional Requirements
- Ingest datasets (Kaggle + Microsoft Loan Credit Risk).
- Perform cleaning, normalization, merging, and feature engineering.
- Train Logistic Regression, XGBoost, LightGBM models.
- Evaluate using AUC-ROC, F1-score, Precision, Recall.
- Generate risk categories for borrowers.
- Recommendation engine for recovery actions.
- Serve model through FastAPI.
- Visualize insights in React + Plotly dashboard.

### Non-Functional Requirements
- Response time < 500ms for API.
- Fully open-source and scalable.
- Dashboard must load under 2 seconds.

---

## 2. KPIs

### Model KPIs
- AUC-ROC
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix values

### Business KPIs
- Default detection rate
- False positive reduction
- Recovery success rate
- Agent efficiency improvement

---

## 3. Sample Dataset Fields
- loan_id  
- customer_id  
- age  
- employment_type  
- income  
- loan_amount  
- term  
- interest_rate  
- credit_score  
- credit_utilization  
- num_of_defaults_past  
- num_of_loans_active  
- num_of_payments_made  
- num_of_payments_remaining  
- last_payment_date  
- delinquency_status  
- repayment_velocity  
- default_status  

---

## 4. Milestone Evaluation
- Dataset fields must match requirements.
- Repository must be structured.
- All documentation must be completed.
