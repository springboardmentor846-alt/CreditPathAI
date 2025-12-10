# Milestone 1 — Requirements & Data Collection  
This document defines the KPIs, datasets, and requirements for the CreditPathAI project.

---

## 1. Project KPIs (Key Performance Indicators)

### Business KPIs
1. Improve loan recovery rate by **10–15%**.
2. Reduce average customer follow-up time by **20%** using prioritization.
3. Reduce agent manual effort by **25%** via automated recommendations.
4. Identify high-risk customers with at least **80% accuracy**.

### Technical / Model KPIs
1. AUC-ROC ≥ **0.80**.
2. Precision in high-risk segment ≥ **0.70**.
3. False Negative Rate ≤ **0.20**.
4. End-to-end pipeline reproducible with logs and scripts.

---

## 2. Project Requirements

### Functional Requirements
- Predict borrower’s default probability.
- Recommend recovery action based on risk level.
- Provide API endpoint for prediction.
- Create basic data visualization and EDA reports.
- Store processed datasets in a database.

### Non-Functional Requirements
- Code must be modular (src/ structure).
- Logs must be generated during pipeline execution.
- GitHub repo must contain clean folder structure.
- Notebook files must demonstrate workflow.

---

## 3. Sample Dataset Description (Collected from Kaggle)

The following CSVs have been downloaded and placed into:
`data/raw/`

- application_train.csv
- bureau.csv
- bureau_balance.csv
- previous_application.csv
- POS_CASH_balance.csv
- credit_card_balance.csv
- installments_payments.csv

These datasets will be used for:
- data ingestion
- EDA
- feature engineering
- model building
- recommendation logic

(These datasets are NOT pushed to GitHub.)

---

## 4. Data Arrangement (Required Fields Identified)

### Main Fields for Modeling (from application_train):
- SK_ID_CURR  
- TARGET  
- AMT_INCOME_TOTAL  
- AMT_CREDIT  
- AMT_ANNUITY  
- AMT_GOODS_PRICE  
- NAME_INCOME_TYPE  
- NAME_EDUCATION_TYPE  
- NAME_FAMILY_STATUS  
- DAYS_EMPLOYED  
- DAYS_BIRTH  
- CNT_CHILDREN  
- etc.

### Supporting Tables (for aggregated features):
- bureau → past external loans
- previous_application → past applications with Home Credit
- installments_payments → payment behavior
- credit_card_balance → monthly credit card history
- POS_CASH_balance → POS loan history

Aggregations will be performed in Milestone 3.

---

## Milestone 1 Deliverable Status
- [x] KPIs documented  
- [ ] GitHub repository setup  
- [ ] Sample datasets collected  
- [ ] Data arranged in required fields  

