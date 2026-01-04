<<<<<<< HEAD
# CreditpathAI 🚀  
AI-powered Credit Risk Prediction System

CreditpathAI is a full-stack machine learning project that predicts **credit risk / loan default probability** using user-provided financial and demographic inputs.  
The system uses a **trained ML model served via FastAPI** and a **React-based frontend UI** for user interaction.

---

## Project Features

-  Credit risk prediction using Machine Learning
-  FastAPI backend for fast inference
-  React.js frontend with a clean, responsive UI
-  REST API integration between frontend & backend
-  Model trained on structured loan data
-  Supports real-time predictions

---

## Project Structure
CreditpathAI/
│
├── backend/ # FastAPI backend
│ ├── inference.py # Model inference API
│ ├── best_model.pkl # Trained ML model
│ ├── requirements.txt # Backend dependencies
│ └── venv/ # Virtual environment (not pushed)
│
├── frontend/ # React frontend
│ ├── public/
│ ├── src/
│ │ ├── App.js
│ │ ├── App.css
│ │ ├── index.js
│ │ └── index.css
│ ├── package.json
│ └── package-lock.json
│
├── .gitignore
└── README.md


---

##  Tech Stack

### Frontend
- React.js
- HTML5, CSS3
- JavaScript (ES6)
- Fetch API / Axios

### Backend
- Python
- FastAPI
- Pydantic
- Uvicorn

### Machine Learning
- Scikit-learn
- NumPy
- Pandas
- Joblib

---

##  Setup Instructions

### 1 Clone the Repository
```bash
git clone https://github.com/your-username/CreditpathAI.git
cd CreditpathAI

Backend Setup (FastAPI)
Create Virtual Environment
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate 

Install Dependencies
pip install -r requirements.txt

Run Backend Server
uvicorn inference:app --reload --port 5000

Backend will run at:

http://127.0.0.1:5000


Swagger Docs:

http://127.0.0.1:5000/docs

🔹 Frontend Setup (React)
cd frontend
npm install
npm start


Frontend will run at:

http://localhost:5000

API Endpoint
Predict Credit Risk

POST /predict

Request Body

{
  "features": [value1, value2, value3, ...]
}


Response

{
  "prediction": "Low Risk / High Risk"
}

UI Preview

The UI allows users to input loan details on the left and view prediction results instantly on the right.
=======
﻿# 📊 CreditPathAI

CreditPathAI is an **end-to-end AI-powered credit risk assessment system** that predicts loan default probability and provides **actionable recommendations** for lenders.  
The project covers the **complete machine learning lifecycle** — from data ingestion to model deployment and frontend dashboards.

---

## 🚀 Project Overview

Loan defaults cause significant financial losses for lending institutions.  
CreditPathAI helps mitigate this risk by:

- Predicting the probability of loan default using ML models
- Exposing predictions via a FastAPI backend
- Visualizing risk and recommendations through a React dashboard

This project is built as a **production-ready ML system**, not just a model.

---

## 🎯 Project KPIs

- AUC-ROC for loan default prediction
- Precision & Recall for high-risk borrowers
- Confusion Matrix and Threshold Analysis
- Model inference time via FastAPI
- Dashboard usability & recommendation clarity

---

## 🗂️ Project Structure

```text
CreditPathAI/
│
├── data/
│   └── loans.db
│
├── src/
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train_model.py
│   │   └── train_random_forest.py
│   ├── recommendation/
│   │   └── recommend.py
│   └── api/
│       └── app.py
│
├── reports/
│   └── plots/
│
├── frontend/
│   └── (React + Plotly dashboard)
│
├── models/
│   └── random_forest.pkl
│
├── Dockerfile
├── requirements.txt
└── README.md
=======
# \# CreditPathAI

# AI-based Smart Loan Recovery System

# 

# This project aims to predict borrower default risk and recommend personalized loan recovery actions using open-source machine learning tools.

# Workflow: Data ingestion → Feature engineering → Model training → Recommendation engine → API → Dashboard.


>>>>>>> ce824d8196dbaefd6a0bbbb27a520ae6cb565eb7
