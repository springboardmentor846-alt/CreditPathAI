import React, { useState } from "react";
import Plot from "react-plotly.js";
import { getPrediction } from "./api";


function Dashboard() {
  const [result, setResult] = useState(null);

  const sampleInput = {
    Age: 35,
    Income: 50000,
    LoanAmount: 200000,
    CreditScore: 650,
    MonthsEmployed: 36,
    NumCreditLines: 4,
    InterestRate: 13.5,
    LoanTerm: 36,
    DTIRatio: 0.32,
    LoanIncomeRatio: 4
  };

  const handlePredict = async () => {
    const data = await getPrediction(sampleInput);
    setResult(data);
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>📊 CreditPathAI Dashboard</h2>

      <button onClick={handlePredict}>
        Run Risk Prediction
      </button>

      {result && (
        <>
          <h3>Prediction Result</h3>
          <p><b>Default Probability:</b> {result.default_probability}</p>
          <p><b>Risk Level:</b> {result.risk_level}</p>
          <p><b>Recommendation:</b> {result.recommendation}</p>

          <Plot
            data={[
              {
                type: "bar",
                x: ["No Default", "Default"],
                y: [
                  1 - result.default_probability,
                  result.default_probability
                ],
                marker: { color: ["green", "red"] }
              }
            ]}
            layout={{
              title: "Loan Default Risk Probability"
            }}
          />
        </>
      )}
    </div>
  );
}

export default Dashboard;
