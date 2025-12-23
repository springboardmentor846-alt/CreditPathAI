# CreditPathAI

Automating and optimizing  the loan recovery lifecycle by modeling  repayment behavior using diverse data.  

# Project Objective 

To design and develop a machine learning–driven platform that predicts borrower default risk and recommends personalized recovery actions. Using open-source technologies to ensure cost-effectiveness, scalability, and reliability. The system aims to improve delinquency recovery efficiency and equip collection agents with actionable insights.  

The XGBoost model predicts the probability of loan default based on borrower financial attributes such as income, credit score, LTV, and DTI. Based on probability thresholds, borrowers are classified into Low, Medium, or High risk categories.

# Tech Stack

# Frontend
> React.js
> Plotly.js (Charts & dashboards)

# Backend
> FastAPI
> Pydantic
> Uvicorn

# Machine Learning
> XGBoost
> Pandas, NumPy

# Development & Deployment
> GitHub Codespaces

# Features
- Loan risk prediction (Low / Medium / High)
- Probability-based risk scoring
- Visual dashboards
- Agent action recommendations

# How to Run

# Backend
  bash
- uvicorn src.api.main:app --reload
- Backend will be available at: http://localhost:8000

# Frontend
> Use Frontend directory
- cd frontend
- npm install
- npm start
- Frontend will be available at: http://localhost:3000

# Dashboards & Visualizations

- Risk Probability Bar Chart
- Risk Distribution Pie Chart
- Risk Summary Dashboard
- Agent Action Recommendations

# Future Enhancements-

# DevOps & Deployment Enhancements

Docker and Containerization
  - Containerize frontend, backend, and model services
  - Enable consistent deployment across environments

CI/CD Pipeline
  - GitHub Actions for:
    - Automated testing
    - Linting & formatting
    - Build & deployment on push

Cloud Deployment
  - Deploy backend on AWS / Azure / GCP
  - Host frontend using Netlify or Vercel
  - Use managed services for scalability

Monitoring & Logging
  - API health monitoring (Prometheus / Grafana)
  - Centralized logging (ELK stack)

Security Enhancements
  - Authentication & role-based access
  - API rate limiting
  - Secure secrets using environment variables



