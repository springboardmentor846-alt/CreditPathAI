import React, { useState } from "react";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    loan_amount: 3000000,
    rate_of_interest: 10.2,
    term: 240,
    LTV: 82,
    Upfront_charges: 1800,
    Credit_Worthiness: "l1",
    loan_type: "type1",
    Security_Type: "direct",
    loan_purpose: "p1",
    open_credit: "nopc",
    business_or_commercial: "nob/c",
    approv_in_adv: "pre",
    Neg_ammortization: "not_neg"
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const predictRisk = async () => {
    const res = await fetch("http://127.0.0.1:5000/predict", {
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
      <h2 className="title">CreditPathAI – Loan Risk Dashboard</h2>

      <div className="main-grid">

        {/* INPUT PANEL */}
        <div className="card input-card">
          <h3>Loan Inputs</h3>

          <div className="form-grid">

            <div className="field">
              <label>Loan Amount</label>
              <input
                name="loan_amount"
                value={formData.loan_amount}
                onChange={handleChange}
              />
            </div>

            <div className="field">
              <label>Rate of Interest (%)</label>
              <input
                name="rate_of_interest"
                value={formData.rate_of_interest}
                onChange={handleChange}
              />
            </div>

            <div className="field">
              <label>Loan Term (Months)</label>
              <input
                name="term"
                value={formData.term}
                onChange={handleChange}
              />
            </div>

            <div className="field">
              <label>Loan to Value (LTV)</label>
              <input
                name="LTV"
                value={formData.LTV}
                onChange={handleChange}
              />
            </div>

            <div className="field">
              <label>Upfront Charges</label>
              <input
                name="Upfront_charges"
                value={formData.Upfront_charges}
                onChange={handleChange}
              />
            </div>

            <div className="field">
              <label>Credit Worthiness</label>
              <select
                name="Credit_Worthiness"
                value={formData.Credit_Worthiness}
                onChange={handleChange}
              >
                <option value="l1">l1</option>
                <option value="l2">l2</option>
              </select>
            </div>

            <div className="field">
              <label>Loan Type</label>
              <select
                name="loan_type"
                value={formData.loan_type}
                onChange={handleChange}
              >
                <option value="type1">type1</option>
                <option value="type2">type2</option>
              </select>
            </div>

            <div className="field">
              <label>Security Type</label>
              <select
                name="Security_Type"
                value={formData.Security_Type}
                onChange={handleChange}
              >
                <option value="direct">Direct</option>
                <option value="Indirect">Indirect</option>
              </select>
            </div>

            <div className="field">
              <label>Loan Purpose</label>
              <select
                name="loan_purpose"
                value={formData.loan_purpose}
                onChange={handleChange}
              >
                <option value="p1">p1</option>
                <option value="p2">p2</option>
              </select>
            </div>

            <div className="field">
              <label>Open Credit</label>
              <select
                name="open_credit"
                value={formData.open_credit}
                onChange={handleChange}
              >
                <option value="opc">Yes</option>
                <option value="nopc">No</option>
              </select>
            </div>

            <div className="field">
              <label>Business / Commercial</label>
              <select
                name="business_or_commercial"
                value={formData.business_or_commercial}
                onChange={handleChange}
              >
                <option value="b/c">Yes</option>
                <option value="nob/c">No</option>
              </select>
            </div>

            <div className="field">
              <label>Approval in Advance</label>
              <select
                name="approv_in_adv"
                value={formData.approv_in_adv}
                onChange={handleChange}
              >
                <option value="pre">Yes</option>
                <option value="nopre">No</option>
              </select>
            </div>

            <div className="field">
              <label>Negative Amortization</label>
              <select
                name="Neg_ammortization"
                value={formData.Neg_ammortization}
                onChange={handleChange}
              >
                <option value="neg_amm">Yes</option>
                <option value="not_neg">No</option>
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
