CreditPathAI - Milestone 1: Requirements & Data Collection
Project Name: CreditPathAI - AI-Powered Smart Loan Recovery System Objective: Predict borrower default risk and recommend personalized recovery actions using ML.

Functional Requirements
The system must load and store loan datasets (CSV/Excel -> SQLite/PostgreSQL).
The model must classify borrowers into risk categories (Low / Medium / High).
The ML workflow must support Logistic Regression, XGBoost, LightGBM.
The system must compute KPIs like:
Default Risk Probability
Recovery Priority Score
Repayment Velocity
Credit Utilization
It must generate recommendations such as:
Early reminder
Phone call
Structured repayment plan
Legal escalation (high risk only)
Non-Functional Requirements
Accuracy target: AUC-ROC >= 0.80 (initial target).
Prediction latency target: < 1 second via FastAPI.
Dashboard must be interactive (React + Plotly).
All services must be containerized with Docker.
KPIs for Business
Recovery Rate Improvement (%)
Reduction in Agent Manual Effort (%)
Default Probability Accuracy (AUC-ROC)
Time-to-Recovery Optimization
Deliverables for Milestone 1
Documented KPIs and requirements (this document).
GitHub repository structure and README.
Two sample datasets placed in /data/raw/.
Verification checklist: data fields present, README and docs uploaded.
Notes & Next Steps
Milestone 2: build ingestion pipeline and perform EDA.
Use MLflow (local) to track experiments.
Use SQLite for initial prototyping; move to PostgreSQL for scale.
6. Data Requirements
The following fields are required for the ML pipeline:

loan_id
borrower_id (optional)
loan_amount
term
interest_rate
installment
grade
sub_grade
employment_length
home_ownership
annual_income
verification_status
issue_date
dti
delinq_2yrs
total_acc
revol_util
default_flag
loan_status
These fields are sufficient for baseline modeling and EDA

Reference: User-provided project brief PDF located at /mnt/data/AI - CreditPathAI (1) (1).pdf
