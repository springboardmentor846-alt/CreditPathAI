#!/bin/bash

echo "🔹 Creating virtual environment..."
python -m venv venv

echo "🔹 Activating virtual environment..."
source venv/bin/activate

echo "🔹 Upgrading pip..."
pip install --upgrade pip

echo "🔹 Installing required packages..."
pip install fastapi uvicorn pydantic numpy pandas scikit-learn joblib

echo "✅ Setup completed successfully!"
echo "👉 To activate later: source venv/bin/activate"
echo "👉 To run server: uvicorn inference:app --reload"
