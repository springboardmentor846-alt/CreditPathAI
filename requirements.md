--> Project Requirements

Functional Requirements

Data Ingestion – Load and store loan-related datasets (CSV/Excel → SQLite/PostgreSQL).
1. Feature Engineering – Generate features such as repayment velocity, credit utilization, and income-to-loan ratio.
2. Model Training – Build baseline and advanced ML models (Logistic Regression, XGBoost, LightGBM).
3. Prediction Service – Expose model results via FastAPI REST API.
4. Recommendation Engine – Map predicted risk levels to personalized recovery strategies.
5. Visualization Dashboard – Provide interactive dashboards for collection agents using React.js + Plotly.js.

Non-Functional Requirements

1. Scalability: System should support larger datasets using Dask or PostgreSQL.
2. Cost Efficiency: Entire project must use open-source and lightweight tools.
3. Explainability: Include model interpretability (e.g., SHAP values or feature importance).
4. Reliability: Implement error handling, logging, and retry mechanisms for data and API failures.
5. Security: Basic API authentication for model endpoints.
6. Maintainability: Modular, documented codebase with CI/CD compatibility (GitHub Actions optional).
