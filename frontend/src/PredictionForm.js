import React, { useState } from "react";
import { predictRisk } from "./api";

function PredictionForm() {
  const [form, setForm] = useState({
    income: "",
    loan_amount: "",
    credit_score: "",
    ltv: "",
    dtir1: "",
  });

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const payload = {
      income: Number(form.income),
      loan_amount: Number(form.loan_amount),
      credit_score: Number(form.credit_score),
      ltv: Number(form.ltv),
      dtir1: Number(form.dtir1),
    };

    const response = await predictRisk(payload);
    console.log("API RESPONSE:", response);
    alert(JSON.stringify(response)); // temp debug
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>Loan Risk Prediction</h2>

      <form onSubmit={handleSubmit}>
        <input name="income" placeholder="Income"
          value={form.income} onChange={handleChange} />

        <input name="loan_amount" placeholder="Loan Amount"
          value={form.loan_amount} onChange={handleChange} />

        <input name="credit_score" placeholder="Credit Score"
          value={form.credit_score} onChange={handleChange} />

        <input name="ltv" placeholder="LTV"
          value={form.ltv} onChange={handleChange} />

        <input name="dtir1" placeholder="DTI Ratio"
          value={form.dtir1} onChange={handleChange} />

        <button type="submit">Predict</button>
      </form>
    </div>
  );
}

export default PredictionForm;
