# 💳 CreditPathAI  
**Automating and optimizing the loan recovery lifecycle using machine learning**

---

## 🧠 Project Overview
**CreditPathAI** is a machine learning–driven platform designed to **predict borrower default risk** and recommend **personalized recovery actions**.  
The goal is to make the loan recovery process **data-driven, efficient, and scalable**, helping financial institutions and collection agents improve delinquency recovery.

---

## 🎯 Objectives
- Predict borrower default risk using diverse financial and behavioral data.  
- Recommend personalized recovery strategies based on predicted risk.  
- Provide actionable insights to collection agents through interactive dashboards.  
- Build an end-to-end, open-source, cost-effective ML pipeline.

---

## ⚙️ Project Workflow
1. **Data Ingestion**
   - Load datasets from Kaggle Loan Default Dataset and Microsoft R Server Loan Credit Risk.
   - Store and manage data using SQLite or PostgreSQL.

2. **Feature Engineering**
   - Data cleaning, normalization, and feature extraction (e.g., repayment velocity, credit utilization).
   - Tools: Pandas, Dask (for large-scale data).

3. **Model Training & Evaluation**
   - Algorithms: Logistic Regression (baseline), XGBoost, LightGBM.
   - Metrics: AUC-ROC, Confusion Matrix.
   - Tools: Python, scikit-learn, MLflow (local tracking).

4. **Action Recommendation Engine**
   - Maps predicted risk categories to personalized recovery interventions.

5. **Serving Layer & API**
   - Model predictions exposed via **FastAPI**.
   - Containerized using **Docker** for portability.

6. **Visualization & Dashboard**
   - React.js + Plotly.js dashboard for interactive analytics and insights.

---

## 🧩 Tech Stack
| Layer | Technologies Used |
|-------|--------------------|
| **Data Storage & Processing** | CSV/Excel, SQLite, PostgreSQL, Pandas, Dask |
| **ML & MLOps** | Python, scikit-learn, XGBoost, LightGBM, MLflow |
| **API & Backend** | FastAPI, Docker |
| **Frontend** | React.js, Plotly.js |
| **Monitoring & Logging** | FastAPI built-in logging, SQLite logs |
| **CI/CD (optional)** | GitHub Actions |

---

## 🗂️ Project Structure
```
CreditPathAI/
├── data/                     # Sample or reference datasets
│   ├── loan_default_sample.csv
│   └── credit_risk_sample.csv
├── notebooks/                # Jupyter notebooks for EDA & experiments
├── src/                      # Source code (ML pipeline, API)
│   ├── models/
│   ├── api/
│   └── utils/
├── dashboard/                # React.js frontend
├── requirements.txt          # Python dependencies
├── Dockerfile
├── README.md
└── LICENSE
```

---

## 📊 Datasets Used
- **Kaggle Loan Default Dataset**  
  [Link](https://www.kaggle.com/datasets/) – Contains borrower-level credit and repayment information.  

- **Microsoft R Server Loan Credit Risk Dataset**  
  [Link](https://github.com/Microsoft/ML-Server) – Provides credit risk and repayment performance data.  

> ⚠️ Due to licensing restrictions, only **sample data** is included in this repository.  
> For full datasets, please refer to the official data sources linked above.

---

## 🚀 How to Run the Project
1. **Clone this repository**
   ```bash
   git clone https://github.com/<springboardmentor846-alt>/CreditPathAI.git
   cd CreditPathAI
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the API**
   ```bash
   uvicorn src.api.main:app --reload
   ```
4. **Access the API docs**
   - Open: `http://127.0.0.1:8000/docs`

5. **Run the Dashboard**
   ```bash
   cd dashboard
   npm install
   npm start
   ```

---

## 🧱 Milestones
| Milestone | Description | Status |
|------------|-------------|--------|
| 1. Data Collection & Requirements | Gather datasets, define KPIs | ✅ |
| 2. Data Ingestion & EDA | Build ingestion pipeline & analysis | 🟡 |
| 3. Baseline Model | Logistic Regression | 🟡 |
| 4. Advanced Models | XGBoost, LightGBM | ⬜ |
| 5. Recommendation API | FastAPI + Recommendation logic | ⬜ |
| 6. Dashboard & Final Delivery | React.js frontend & testing | ⬜ |

---

## 📈 Expected Outcomes
- Improved loan recovery efficiency through intelligent risk modeling.  
- Automated recommendation engine for collection strategies.  
- Real-time API and interactive visualization for agents.  

---

## 👩‍💻 Contributors
- Susmitha Nalla – Project Developer  
- Rohit - Mentor

---

## 📜 License
This project is open-sourced under the **MIT License**.  
Feel free to use, modify, and contribute with credit.

---
