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

http://localhost:3000

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
