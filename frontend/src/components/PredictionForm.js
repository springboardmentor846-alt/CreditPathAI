import React, { useState } from "react";
import { predictRisk } from "../api/api";

function PredictionForm() {
  const [form, setForm] = useState({
    income: "",
    loan_amount: "",
    credit_score: "",
    ltv: "",
    dtir1: ""
  });

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const result = await predictRisk(form);
    if (result) alert(JSON.stringify(result, null, 2));
  };

  return (
    <div>
      <h2>Loan Risk Prediction</h2>
      <form onSubmit={handleSubmit}>
        {Object.keys(form).map((key) => (
          <input
            key={key}
            name={key}
            placeholder={key}
            value={form[key]}
            onChange={handleChange}
          />
        ))}
        <button type="submit">Predict</button>
      </form>
    </div>
  );
}

export default PredictionForm;