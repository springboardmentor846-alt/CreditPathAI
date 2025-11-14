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
Place the CSV file inside:

data/raw/


The project will load and process it automatically.

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



