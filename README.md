# CreditPathAI 🚀

📄 Overview  
**CreditPathAI** is an open-source project aiming to build a credit-risk / credit-scoring system using AI / data-driven methods.  
Currently this repository contains the *Milestone 1* deliverables: requirement definitions and sample data for prototyping.  

📁 Repository Structure  

CreditPathAI/
├── data/
│ └── raw/
│ ├── kaggle_loan_default.csv
│ └── ms_credit_risk.csv
├── docs/
│ └── requirements_milestone1.md
├── README.md
└── requirements_milestone1.md # (optional duplicate / stub)


- `data/raw/` — sample datasets (synthetic or anonymized) for initial prototyping and experimentation.  
- `docs/requirements_milestone1.md` — detailed functional and non-functional requirements and KPIs defined for Milestone 1.  
- (Optional) `requirements_milestone1.md` — alternate or supplementary requirements documentation.  

 
 ✅ Milestone 1 — Completed  

For Milestone 1 the following deliverables are included:

- A complete requirements specification document (`docs/requirements_milestone1.md`) covering functional & non-functional requirements and KPIs.  
- Sample dataset(s) under `data/raw/`, for use in prototyping experiments.  
- README and metadata to describe project status and repository structure.  

**Validation checklist for Milestone 1:**  
1. Ensure the requirements document is complete and clearly specifies project scope and objectives.  
2. Confirm that sample CSV datasets in `data/raw/` are present and structured correctly (i.e. columns/fields are properly defined).  
3. Repository is pushed to GitHub and accessible for review / collaboration.  

🔮 Next Steps (Milestone 2 & Beyond)  

Planned upcoming tasks:  

- Build ingestion scripts to load raw CSV data into a structured database (e.g. SQLite / PostgreSQL).  
- Perform Exploratory Data Analysis (EDA), data cleaning / preprocessing.  
- Generate visualizations to understand data distributions, correlations, missing values, etc.  
- Develop baseline models for credit-risk prediction: classification/regression using classical ML (e.g. Logistic Regression, Random Forest) or possibly neural networks.  
- Evaluate model performance, define metrics, document results.  
- Expand to data pipelines: preprocessing → training → evaluation → deployment (if applicable).  



