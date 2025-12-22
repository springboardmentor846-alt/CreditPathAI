🚀 CreditPathAI – Credit Default Prediction System

CreditPathAI is a Machine Learning–based system that predicts the probability of credit default for a customer and recommends an **appropriate business action**.  
The trained model is deployed using **FastAPI** for real-time predictions.



🎯 Project Objective

- Predict whether a credit card customer is likely to default next month
- Assign a risk category based on probability
- Provide actionable recommendations for banks and financial institutions


 🧠 Model Used

- Algorithm: XGBoost Classifier  
- Why XGBoost?
  - Handles non-linear relationships well
  - Performs strongly on tabular financial data
  - Robust to class imbalance



 🗂️ Dataset Used

- UCI Credit Card Default Dataset
- Contains customer demographics, billing history, and repayment behavior
- Target variable: default payment next month
⚙️ Installation & Setup

1️⃣ Clone the repository
```bash
git clone https://github.com/springboardmentor846-alt/CreditPathAI.git
cd CreditPathAI or simply run locally by downloading it.

2️⃣ Create a virtual environment
python -m venv venv


Activate it:

Windows

venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

▶️ Running the Application

Start the FastAPI server:

uvicorn creditpath_api:app --reload


Open your browser:

http://127.0.0.1:8000/docs


You will see Swagger UI where you can test the API.

🔌 API Endpoint
POST /predict

Input:
Customer credit and repayment details

Output:

default_probability

risk_category

recommended_action

🧪 Sample Response
{
  "default_probability": 0.1254,
  "risk_category": "Low Risk",
  "recommended_action": "Send gentle SMS reminder"
}

📊 Risk Categories & Actions
Default Probability	Risk Level	
