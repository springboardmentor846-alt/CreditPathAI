import React, { useState } from "react";
import { predictRisk } from "./api";

function PredictionForm() {
  const [income, setIncome] = useState("");
  const [loanAmount, setLoanAmount] = useState("");
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const data = {
      income: Number(income),
      loan_amount: Number(loanAmount),
    };

    const response = await predictRisk(data);
    setResult(response);
  };

  return (
    <div>
      <h2>Loan Risk Prediction</h2>

      <form onSubmit={handleSubmit}>
        <input
          type="number"
          placeholder="Income"
          value={income}
          onChange={(e) => setIncome(e.target.value)}
        />

        <input
          type="number"
          placeholder="Loan Amount"
          value={loanAmount}
          onChange={(e) => setLoanAmount(e.target.value)}
        />

        <button type="submit">Predict</button>
      </form>

      {result && (
        <div>
          <h3>Risk Category: {result.risk_category}</h3>
          <p>Default Probability: {result.probability}</p>
        </div>
      )}
    </div>
  );
}

export default PredictionForm;
