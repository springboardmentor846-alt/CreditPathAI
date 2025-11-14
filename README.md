CreditPathAI – ML-Driven Smart Loan Recovery System


CreditPathAI is a machine-learning–powered platform designed to predict borrower default risk, classify delinquency behavior, and recommend personalized recovery actions.
It automates the loan-recovery lifecycle by analyzing repayment patterns, communication history, and borrower risk signals.
This project is built using 100% open-source technologies, scalable for both academic and real-world fintech use cases.


Project Objective
To build a complete ML-driven system that:
1.	Predicts the probability of loan default / delinquency
2.	Identifies risk buckets (Current, 15+, 30+, 60+ DPD)
3.	Generates actionable recovery recommendations for collection agents
4.	Provides an API service for real-time scoring
5.	Offers a dashboard for insights, charts, and portfolio health visualization


 Project Workflow (End-to-End)
The project follows a 6-stage pipeline:
1. Data Ingestion & Collection


Sources:
1.	Kaggle Loan Default Datasets
2.	Microsoft Loan Credit Risk sample data
3.	Transactional data (Payments, EMI, Spends)
4.	Collections & communication logs


Tools:
Pandas, SQLite/PostgreSQL, CSV/Excel
2. Data Cleaning & Feature Engineering
Operations:
1.	Missing value handling
2.	Normalization (MinMax/Standard Scaling)
3.	Derived features such as:
o	Repayment Velocity
o	On-time Payment Ratio
o	Days Past Due Trend
o	Credit Utilization
o	Communication Engagement Score
o	Promise-to-Pay frequency
o	Borrower segmentation
Tools:
Pandas, Dask


3. Model Training & Evaluation
Algorithms used:
1.	Logistic Regression (Baseline)
2.	Random Forest / XGBoost / LightGBM
3.	Optional: CatBoost
Metrics:
1.	AUC-ROC
2.	F1 Score
3.	Confusion Matrix
4.	KS-Statistic
Tools:
scikit-learn, XGBoost, LightGBM, MLflow


4. Recommendation Engine
Maps ML-predicted risk to personalized interventions, such as:
Risk Level	Recommended Action
Low Risk	Automated SMS/Push
Medium Risk	Call attempts + Payment plan
High Risk	Field visit, restructure, settlement offer
Critical Risk	Escalation to recovery manager




5. Serving Layer – FastAPI
A FastAPI microservice exposes:
1.	/predict → returns risk score + recommended action
2.	/features → returns model features
3.	/health → service health check
Tools:
FastAPI, Uvicorn, Docker




6. Visualization Dashboard (Frontend)
Frontend for:
1.	Portfolio risk heatmap
2.	Time-series repayment trends
3.	Top risky borrowers
4.	Agent task recommendations
Tools:
React.js, Plotly.js

