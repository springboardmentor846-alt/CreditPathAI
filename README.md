📊 CreditPathAI – Smart Loan Risk & Recovery System 📌 Project Overview

CreditPathAI is an end-to-end Machine Learning–based Loan Risk Assessment and Recommendation System built using the Home Credit dataset. The project aims to help financial institutions identify risky borrowers, reduce loan defaults, and provide actionable recommendations for loan recovery agents.

This project follows a milestone-driven approach covering:

Data ingestion & EDA

Feature engineering

Baseline & advanced ML models

Recommendation engine

API & frontend prototype planning

🎯 Problem Statement

Financial institutions face challenges in:

Identifying high-risk loan applicants

Reducing Non-Performing Assets (NPA)

Supporting loan recovery agents with insights

This system predicts probability of default and provides clear recommendations based on borrower risk.

🧠 Solution Approach

Clean and analyze borrower data

Engineer meaningful features

Train ML models to predict loan default risk

Generate risk-based recommendations

Design API & frontend architecture for real-world usage

🚀 Milestone-wise Implementation 🔹 Milestone 1: Data Overview & Ingestion

Loaded Home Credit datasets safely

Verified schema, missing values, and data consistency

Stored cleaned datasets for analysis

Deliverables:

Clean datasets

Data ingestion pipeline

🔹 Milestone 2: Exploratory Data Analysis (EDA)

Key insights from EDA:

Strong class imbalance (~8% defaulters)

Income, credit amount, annuity strongly affect default risk

Previous loan behavior is a strong indicator

Many missing values handled via imputation

Deliverables:

EDA notebooks

Summary logs for all tables

🔹 Milestone 3: Feature Engineering & Baseline Model

Created 127 final features

Encoded categorical variables

Scaled numerical features

Trained Logistic Regression baseline model

Results:

Train AUC-ROC: 0.75

Test AUC-ROC: 0.74

This model serves as a benchmark.

🔹 Milestone 4: Advanced Model Training

Due to system limitations, data was sampled efficiently.

Models trained:

XGBoost

Results:

Train AUC-ROC: 0.73

Classification Report:

              precision    recall  f1-score   support

           0       0.95      0.84      0.89      9189
           1       0.20      0.47      0.28       811

    accuracy                           0.81     10000
   macro avg       0.58      0.65      0.59     10000
weighted avg       0.89      0.81      0.84     10000

Confusion Matrix:
 [[7700 1489]
 [ 430  381]]

📌 Shows better learning capacity but highlights overfitting risk.

(LightGBM planned – limited by environment constraints)

🔹 Milestone 5: Recommendation Engine

Built a rule-based recommendation engine based on predicted risk score.

Risk Categories:

🟢 Low Risk → Normal follow-up

🟡 Medium Risk → Reminder calls

🔴 High Risk → Immediate recovery action

Deliverables:

Actionable borrower recommendations

Risk-based decision logic

🔹 Milestone 6: Frontend & API (Prototype Level)

Planned Architecture:

FastAPI for model inference

React.js frontend

Plotly.js for dashboards

Components Designed:

frontend/src/services/api.js → API communication

Agent dashboard concept

Documentation & rollout plan

(Full deployment can be extended further)

📊 Model Evaluation Metric – AUC-ROC

AUC-ROC measures how well the model distinguishes between:

Defaulters (1)

Non-defaulters (0)

Why AUC-ROC?

Handles imbalanced data

More reliable than accuracy for credit risk

🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

Jupyter Notebook

FastAPI (prototype)

React.js (planned)

Plotly.js

Git & GitHub

⚙️ How to Run the Project pip install -r requirements.txt jupyter notebook

Run notebooks in order from 00_ to 07_.

📌 Key Learnings

Handling large datasets efficiently

Feature engineering for financial data

Model evaluation on imbalanced datasets

End-to-end ML project structuring

Industry-level project documentation

👨‍💻 Author

Rachit Bhadade AI / ML Intern | Data Analytics Enthusiast
