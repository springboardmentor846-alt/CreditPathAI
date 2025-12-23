import { useState } from "react";
import "./LoanForm.css";

function LoanForm({ onResult }) {
  const [form, setForm] = useState({
    Age: "",
    Income: "",
    LoanAmount: "",
    CreditScore: "",
    MonthsEmployed: "",
    NumCreditLines: "",
    InterestRate: "",
    LoanTerm: "",
    DTIRatio: "",
    Education: "",
    EmploymentType: "",
    MaritalStatus: "",
    HasMortgage: "",
    HasDependents: "",
    LoanPurpose: "",
    HasCoSigner: "",
  });

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const submitForm = async () => {
    const res = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...form,
        Age: Number(form.Age),
        Income: Number(form.Income),
        LoanAmount: Number(form.LoanAmount),
        CreditScore: Number(form.CreditScore),
        MonthsEmployed: Number(form.MonthsEmployed),
        NumCreditLines: Number(form.NumCreditLines),
        InterestRate: Number(form.InterestRate),
        LoanTerm: Number(form.LoanTerm),
        DTIRatio: Number(form.DTIRatio),
      }),
    });

    const data = await res.json();
    onResult(data);
  };

  return (
    <div className="loan-card">
      <h2>Loan Applicant Details</h2>

      <div className="form-grid">
        {/* Numeric Inputs */}
        <input
          name="Age"
          type="number"
          placeholder="Age"
          onChange={handleChange}
        />
        <input
          name="Income"
          type="number"
          placeholder="Annual Income"
          onChange={handleChange}
        />
        <input
          name="LoanAmount"
          type="number"
          placeholder="Loan Amount"
          onChange={handleChange}
        />
        <input
          name="CreditScore"
          type="number"
          placeholder="Credit Score"
          onChange={handleChange}
        />
        <input
          name="MonthsEmployed"
          type="number"
          placeholder="Months Employed"
          onChange={handleChange}
        />
        <input
          name="NumCreditLines"
          type="number"
          placeholder="Credit Lines"
          onChange={handleChange}
        />
        <input
          name="InterestRate"
          type="number"
          placeholder="Interest Rate (%)"
          onChange={handleChange}
        />
        <input
          name="LoanTerm"
          type="number"
          placeholder="Loan Term (months)"
          onChange={handleChange}
        />
        <input
          name="DTIRatio"
          type="number"
          placeholder="DTI Ratio"
          onChange={handleChange}
        />

        {/* Dropdowns */}
        <select name="Education" onChange={handleChange}>
          <option value="">Education</option>
          <option>High School</option>
          <option>Bachelor's</option>
          <option>Master's</option>
          <option>PhD</option>
        </select>

        <select name="EmploymentType" onChange={handleChange}>
          <option value="">Employment Type</option>
          <option>Unemployed</option>
          <option>Self-employed</option>
          <option>Part-time</option>
          <option>Full-time</option>
        </select>

        <select name="MaritalStatus" onChange={handleChange}>
          <option value="">Marital Status</option>
          <option>Single</option>
          <option>Married</option>
          <option>Divorced</option>
        </select>

        <select name="LoanPurpose" onChange={handleChange}>
          <option value="">Loan Purpose</option>
          <option>Home</option>
          <option>Auto</option>
          <option>Education</option>
          <option>Business</option>
          <option>Other</option>
        </select>

        <select name="HasMortgage" onChange={handleChange}>
          <option value="">Has Mortgage?</option>
          <option>Yes</option>
          <option>No</option>
        </select>

        <select name="HasDependents" onChange={handleChange}>
          <option value="">Has Dependents?</option>
          <option>Yes</option>
          <option>No</option>
        </select>

        <select name="HasCoSigner" onChange={handleChange}>
          <option value="">Has Co-Signer?</option>
          <option>Yes</option>
          <option>No</option>
        </select>
      </div>

      <button className="predict-btn" onClick={submitForm}>
        Predict Default Risk
      </button>
    </div>
  );
}

export default LoanForm;
