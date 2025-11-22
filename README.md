# **CreditPathAI**

Automating and optimizing the loan recovery lifecycle by modeling borrower repayment behavior using machine learning and actionable risk insights.

## 📌 **Project Objective**

CreditPathAI is a machine learning–driven platform designed to:

- Predict borrower default risk
- Recommend personalized loan recovery interventions
- Improve delinquency recovery efficiency
- Equip collection agents with data-driven insights

Built using open-source tools to ensure cost-effectiveness, scalability, and reliability.

---

## 🚀 **Project Workflow**

### 1️⃣ **Data Ingestion**

- Load datasets from Kaggle Loan Default & Microsoft R Server Loan Credit Risk
- Tools: CSV/Excel ingestion, SQLite/PostgreSQL

### 2️⃣ **Feature Engineering**

- Cleaning, normalization, joining
- Derived metrics: repayment velocity, credit utilization
- Tools: Pandas, Dask

### 3️⃣ **Model Training & Evaluation**

- Models: Logistic Regression, XGBoost, LightGBM
- Metrics: AUC-ROC, Confusion Matrix
- Tools: Python, scikit-learn, MLflow

### 4️⃣ **Recommendation Engine**

- Maps risk scores to actionable interventions

### 5️⃣ **API Layer**

- FastAPI service to expose predictions
- Containerized using Docker

### 6️⃣ **Dashboard & Visualization**

- React.js for frontend
- Plotly.js charts for analytics visualization

---

## 🏗 **System Architecture**

---

## ⚙ **Tech Stack**

| Layer                     | Technologies                            |
| ------------------------- | --------------------------------------- |
| Data Storage & Processing | CSV, SQLite/PostgreSQL, Pandas, Dask    |
| ML & MLOps                | scikit-learn, XGBoost, LightGBM, MLflow |
| Backend/API               | FastAPI, Docker                         |
| Frontend                  | React.js, Plotly.js                     |

---

## 📅 **Project Milestones**

| Milestone                              | Deliverables                       | Evaluation                    |
| -------------------------------------- | ---------------------------------- | ----------------------------- |
| **M1: Requirements & Data Collection** | KPIs, sample data, repo setup      | Data prepared                 |
| **M2: Ingestion & EDA**                | CSV → SQL pipeline, EDA report     | Ingestion verified            |
| **M3: Baseline Model**                 | LR model, feature pipeline         | Baseline AUC-ROC              |
| **M4: Advanced Models**                | XGBoost, LightGBM + tuning         | Better AUC-ROC                |
| **M5: API + Recommendations**          | FastAPI scoring + logic validation | Local API tested              |
| **M6: Frontend UI**                    | Dashboard, rollout docs            | UAT validated recommendations |

---
