import React, { useState } from "react";
import { predictRisk } from "./api";
import RiskCharts from "./components/RiskCharts";
import RiskSummaryDashboard from "./components/RiskSummaryDashboard";
import RiskSummaryChart from "./components/RiskSummaryChart";
import AgentDashboard from "./components/AgentDashboard";

function PredictionForm() {
  const [formData, setFormData] = useState({
    income: "",
    loan_amount: "",
    credit_score: "",
    ltv: "",
    dtir1: ""
  });

  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);

    try {
      const payload = {
        income: Number(formData.income),
        loan_amount: Number(formData.loan_amount),
        credit_score: Number(formData.credit_score),
        ltv: Number(formData.ltv),
        dtir1: Number(formData.dtir1)
      };

      const response = await predictRisk(payload);

      if (!response || response.error) {
        throw new Error("Prediction failed");
      }

      setResult(response);
    } catch (err) {
      console.error(err);
      setError("Prediction failed. Check backend.");
    }
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.heading}>Loan Risk Prediction</h2>

      <form onSubmit={handleSubmit} style={styles.form}>
        <input name="income" placeholder="Income" type="number" onChange={handleChange} required />
        <input name="loan_amount" placeholder="Loan Amount" type="number" onChange={handleChange} required />
        <input name="credit_score" placeholder="Credit Score" type="number" onChange={handleChange} required />
        <input name="ltv" placeholder="LTV (%)" type="number" onChange={handleChange} required />
        <input name="dtir1" placeholder="DTI (%)" type="number" onChange={handleChange} required />

        <button type="submit" style={styles.button}>Predict</button>
      </form>

      {error && <p style={styles.error}>{error}</p>}

      {result && (
        <div style={styles.result}>
          <h3>Prediction Result</h3>
          <p><b>Risk Category:</b> {result.risk_category}</p>
          <p><b>Probability:</b> {result.probability}</p>
        
          <div className="result-card">
          <RiskCharts probability={result.probability} />
          <RiskSummaryDashboard result={result} /> 
          <RiskSummaryChart probability={result.probability} />
          <AgentDashboard result={result} />
          </div>


          {result.recommendation && (
            <>
              <p><b>Action:</b> {result.recommendation.action}</p>
              <p><b>Message:</b> {result.recommendation.message}</p>
              <p><b>Priority:</b> {result.recommendation.priority}</p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    maxWidth: "400px",
    margin: "40px auto",
    padding: "20px",
    borderRadius: "10px",
    background: "linear-gradient(135deg, #667eea, #764ba2)",
    color: "white"
  },
  heading: { textAlign: "center" },
  form: { display: "flex", flexDirection: "column", gap: "10px" },
  button: {
    padding: "10px",
    background: "#fff",
    color: "#333",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer"
  },
  result: {
    marginTop: "20px",
    background: "rgba(255,255,255,0.2)",
    padding: "15px",
    borderRadius: "8px"
  },
  error: { color: "#ffdddd", marginTop: "10px" }
};

export default PredictionForm;
