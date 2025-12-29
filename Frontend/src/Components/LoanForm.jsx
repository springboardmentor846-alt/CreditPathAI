import React, { useState } from "react";

const LoanForm = ({ onSubmit }) => {
  const [formData, setFormData] = useState({
    LoanAmount: "",
    InterestRate: "",
    DTIRatio: "",
    CreditScore: "",
    MonthsEmployed: ""
  });

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    // Convert values to numbers
    const payload = {
      LoanAmount: Number(formData.LoanAmount),
      InterestRate: Number(formData.InterestRate),
      DTIRatio: Number(formData.DTIRatio),
      CreditScore: Number(formData.CreditScore),
      MonthsEmployed: Number(formData.MonthsEmployed)
    };

    onSubmit(payload);
  };

  return (
    <form onSubmit={handleSubmit}>
      <h2>Loan Application</h2>

      <input
        type="number"
        name="LoanAmount"
        placeholder="Loan Amount"
        value={formData.LoanAmount}
        onChange={handleChange}
        required
      />

      <input
        type="number"
        name="InterestRate"
        placeholder="Interest Rate (%)"
        step="0.1"
        value={formData.InterestRate}
        onChange={handleChange}
        required
      />

      <input
        type="number"
        name="DTIRatio"
        placeholder="DTI Ratio (0–1)"
        step="0.01"
        value={formData.DTIRatio}
        onChange={handleChange}
        required
      />

      <input
        type="number"
        name="CreditScore"
        placeholder="Credit Score"
        value={formData.CreditScore}
        onChange={handleChange}
        required
      />

      <input
        type="number"
        name="MonthsEmployed"
        placeholder="Months Employed"
        value={formData.MonthsEmployed}
        onChange={handleChange}
        required
      />

      <button type="submit">Predict Risk</button>
    </form>
  );
};

export default LoanForm;
