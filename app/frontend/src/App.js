import React, { useState } from "react";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    Age: 35,
    Income: 60000,
    LoanAmount: 250000,
    CreditScore: 720,
    MonthsEmployed: 60,
    NumCreditLines: 5,
    InterestRate: 11.5,
    LoanTerm: 36,
    DTIRatio: 0.35,

    HasMortgage: "Yes",
    HasDependents: "No",
    HasCoSigner: "No",

    Education: "Bachelor",
    EmploymentType: "Salaried",
    MaritalStatus: "Married",
    LoanPurpose: "Home"
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const predictRisk = async () => {
    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData)
    });
    const data = await res.json();
    setResult(data);
  };

  const getRiskClass = () => {
    if (!result) return "";
    if (result.risk_level === "Low Risk") return "low";
    if (result.risk_level === "Medium Risk") return "medium";
    return "high";
  };

  return (
    <div className="dashboard">
      <h2 className="title">CreditPathAI - Loan Risk Dashboard</h2>

      <div className="main-grid">

        {/* INPUT PANEL */}
        <div className="card input-card">
          <h3>Loan Inputs</h3>

          <div className="form-grid">

            <div className="field">
              <label>Age</label>
              <input
                name="Age"
                value={formData.Age}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Income</label>
              <input
                name="Income"
                value={formData.Income}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Loan Amount</label>
              <input
                name="LoanAmount"
                value={formData.LoanAmount}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>CreditScore</label>
              <input
                name="CreditScore"
                value={formData.CreditScore}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>MonthsEmployed</label>
              <input
                name="MonthsEmployed"
                value={formData.MonthsEmployed}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>NumCreditLines</label>
              <input
                name="NumCreditLines"
                value={formData.NumCreditLines}
                onChange={handleChange}
              />
            </div>

            <div className="field">
              <label>Interest Rate (%)</label>
              <input
                name="InterestRate"
                value={formData.InterestRate}
                onChange={handleChange}
              />
            </div>

            <div className="field">
              <label>Loan Term (Months)</label>
              <input
                name="LoanTerm"
                value={formData.LoanTerm}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>DTIRatio</label>
              <input
                name="DTIRatio"
                value={formData.DTIRatio}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>HasMortgage</label>
              <select
                name="HasMortgage"
                value={formData.HasMortgage}
                onChange={handleChange}
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
            <div className="field">
              <label>HasDependents</label>
              <select
                name="HasDependents"
                value={formData.HasDependents}
                onChange={handleChange}
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
            <div className="field">
              <label>HasCoSigner</label>
              <select
                name="HasCoSigner"
                value={formData.HasCoSigner}
                onChange={handleChange}
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
            <div className="field">
              <label>Education</label>
              <select
                name="Education"
                value={formData.Education}
                onChange={handleChange}
              >
                <option value="Bachelor's">Bachelor's</option>
                <option value="Master's">Master's</option>
                <option value="High School">High School</option>
                <option value="PhD">PhD</option>
              </select>
            </div>
            <div className="field">
              <label>EmploymentType</label>
              <select
                name="EmploymentType"
                value={formData.EmploymentType}
                onChange={handleChange}
              >
                <option value="Full-time">Full-time</option>
                <option value="Unemployed">Unemployed</option>
                <option value="Self-employed">Self-employed</option>
                <option value="Part-time">Part-time</option>
              </select>
            </div>
            <div className="field">
              <label>MaritalStatus</label>
              <select
                name="MaritalStatus"
                value={formData.MaritalStatus}
                onChange={handleChange}
              >
                <option value="Divorced">Divorced</option>
                <option value="Married">Married</option>
                <option value="Single">Single</option>
              </select>
            </div>
            <div className="field">
              <label>Loan Purpose</label>
              <select
                name="LoanPurpose"
                value={formData.LoanPurpose}
                onChange={handleChange}
              >
                <option value="Other">Other</option>
                <option value="Auto">Auto</option>
                <option value="Business">Business</option>
                <option value="Home">Home</option>
                <option value="Education">Education</option>
              </select>
            </div>

          </div>

          <button onClick={predictRisk}>Predict Risk</button>
        </div>


        {/* OUTPUT PANEL */}
        <div className="card output-card">
          <h3>Prediction</h3>

          {!result && <p className="hint">Enter details & click Predict</p>}

          {result && (
            <>
              <div className={`risk-badge ${getRiskClass()}`}>
                {result.risk_level}
              </div>

              <p className="probability">
                Default Probability: <b>{result.default_probability}%</b>
              </p>

              <div className="progress">
                <div
                  className={`progress-bar ${getRiskClass()}`}
                  style={{ width: `${result.default_probability}%` }}
                />
              </div>

              <div className="action-box">
                <b>Recommended Action</b>
                <p>{result.recommended_action}</p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
