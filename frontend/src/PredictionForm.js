import React, { useState } from "react";
import { predictRisk } from "./api";

function PredictionForm() {
  const [formData, setFormData] = useState({
    income: "",
    loan_amount: "",
    credit_score: "",
    ltv: "",
    dtir1: "",
  });

  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: Number(e.target.value),
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResult(null);

    try {
      const response = await predictRisk(formData);
      setResult(response);
    } catch (err) {
      setError("Prediction failed. Check backend.");
    }
  };

  return (
    <div style={styles.container}>
      <h2>Loan Risk Prediction</h2>

      <form onSubmit={handleSubmit} style={styles.form}>
        <input name="income" placeholder="Income" onChange={handleChange} />
        <input name="loan_amount" placeholder="Loan Amount" onChange={handleChange} />
        <input name="credit_score" placeholder="Credit Score" onChange={handleChange} />
        <input name="ltv" placeholder="LTV" onChange={handleChange} />
        <input name="dtir1" placeholder="DTI Ratio" onChange={handleChange} />

        <button type="submit">Predict</button>
      </form>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {result && (
        <div style={styles.result}>
          <h3>Result</h3>
          <p><b>Risk Category:</b> {result.risk_category}</p>
          <p><b>Probability:</b> {result.probability}</p>

          <h4>Recommendation</h4>
          <p><b>Action:</b> {result.recommendation.action}</p>
          <p><b>Message:</b> {result.recommendation.message}</p>
          <p><b>Priority:</b> {result.recommendation.priority}</p>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    padding: "30px",
    fontFamily: "Arial",
  },
  form: {
    display: "flex",
    gap: "10px",
    flexWrap: "wrap",
    marginBottom: "20px",
  },
  result: {
    marginTop: "20px",
    padding: "15px",
    borderRadius: "8px",
    background: "#f4f6ff",
    border: "1px solid #ccc",
  },
};

export default PredictionForm;
