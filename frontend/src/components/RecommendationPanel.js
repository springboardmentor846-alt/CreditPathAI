import { useState } from "react";
import { getRiskRecommendation } from "../services/api";

export default function RecommendationPanel() {
  const [amount, setAmount] = useState("");
  const [result, setResult] = useState(null);

  const handleCheck = async () => {
    const res = await getRiskRecommendation({
      loan_amount: amount
    });
    setResult(res);
  };

  return (
    <div className="recommend-box">
      <h3>Loan Recovery Recommendation</h3>

      <input
        placeholder="Enter Loan Amount"
        onChange={(e) => setAmount(e.target.value)}
      />

      <button onClick={handleCheck}>Check Risk</button>

      {result && (
        <div className="result-box">
          <p><b>Risk Level:</b> {result.risk}</p>
          <p><b>Recovery Probability:</b> {result.recovery_probability}</p>
          <p><b>Recommended Action:</b> {result.action}</p>
        </div>
      )}
    </div>
  );
}
