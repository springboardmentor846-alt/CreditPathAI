import { useState } from "react";

const BorrowerForm = ({ onSubmit, isLoading }) => {
  const [formData, setFormData] = useState({
    loanAmount: 10000,
    interestRate: 12.5,
    annualIncome: 65000,
    monthlyPayment: 350,
    dtiRatio: 25,
    creditHistoryYears: 8,
    loanTerm: 36,
    loanPurpose: "Wedding",
    loanGrade: "B",
    homeOwnership: "Rent",
    state: "CA",
  });

  // 🔴 ADD: error state
  const [errors, setErrors] = useState({});

  // 🔴 ADD: validation logic
  const validateField = (name, value) => {
    switch (name) {
      case "loanAmount":
        return value < 1000 ? "Loan amount must be ≥ $1,000" : "";
      case "interestRate":
        return value < 1 || value > 36
          ? "Interest rate must be between 1% and 36%"
          : "";
      case "annualIncome":
        return value < 5000 ? "Annual income must be ≥ $5,000" : "";
      case "monthlyPayment":
        return value <= 0 ? "Monthly payment must be > 0" : "";
      case "dtiRatio":
        return value < 0 || value > 60
          ? "DTI ratio must be between 0% and 60%"
          : "";
      case "creditHistoryYears":
        return value < 0 || value > 40
          ? "Credit history must be between 0 and 40 years"
          : "";
      default:
        return "";
    }
  };

  // 🔴 UPDATED (but logic preserved)
  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData({ ...formData, [name]: value });

    const errorMsg = validateField(name, value);
    setErrors({ ...errors, [name]: errorMsg });
  };

  // 🔴 UPDATED submit validation (alerts kept)
  const handleSubmit = () => {
    const newErrors = {};

    Object.keys(formData).forEach((key) => {
      const error = validateField(key, formData[key]);
      if (error) newErrors[key] = error;
    });

    setErrors(newErrors);

    if (Object.keys(newErrors).length > 0) return;

    onSubmit(formData);
  };

  return (
    <div className="card">
      <h3>📋 Borrower Information</h3>

      <div className="borrower-form-grid">

        {/* Loan Amount */}
        <div className="input-group">
          <label>$ Loan Amount</label>
          <input
            name="loanAmount"
            type="number"
            min={1000}
            max={100000}
            step={500}
            required
            value={formData.loanAmount}
            onChange={handleChange}
            className={errors.loanAmount ? "input-error" : ""}
          />
          {errors.loanAmount && <span className="error-text">{errors.loanAmount}</span>}
        </div>

        {/* Interest Rate */}
        <div className="input-group">
          <label>% Interest Rate</label>
          <input
            name="interestRate"
            type="number"
            min={1}
            max={36}
            step={0.1}
            required
            value={formData.interestRate}
            onChange={handleChange}
            className={errors.interestRate ? "input-error" : ""}
          />
          {errors.interestRate && (
            <span className="error-text">{errors.interestRate}</span>
          )}
        </div>

        {/* Annual Income */}
        <div className="input-group">
          <label>📈 Annual Income</label>
          <input
            name="annualIncome"
            type="number"
            min={5000}
            step={1000}
            required
            value={formData.annualIncome}
            onChange={handleChange}
            className={errors.annualIncome ? "input-error" : ""}
          />
          {errors.annualIncome && (
            <span className="error-text">{errors.annualIncome}</span>
          )}
        </div>

        {/* Monthly Payment */}
        <div className="input-group">
          <label>$ Monthly Payment</label>
          <input
            name="monthlyPayment"
            type="number"
            min={10}
            step={10}
            required
            value={formData.monthlyPayment}
            onChange={handleChange}
            className={errors.monthlyPayment ? "input-error" : ""}
          />
          {errors.monthlyPayment && (
            <span className="error-text">{errors.monthlyPayment}</span>
          )}
        </div>

        {/* DTI Ratio */}
        <div className="input-group">
          <label>% DTI Ratio</label>
          <input
            name="dtiRatio"
            type="number"
            min={0}
            max={60}
            step={0.1}
            required
            value={formData.dtiRatio}
            onChange={handleChange}
            className={errors.dtiRatio ? "input-error" : ""}
          />
          {errors.dtiRatio && <span className="error-text">{errors.dtiRatio}</span>}
        </div>

        {/* Credit History */}
        <div className="input-group">
          <label>⏳ Credit History (Years)</label>
          <input
            name="creditHistoryYears"
            type="number"
            min={0}
            max={40}
            required
            value={formData.creditHistoryYears}
            onChange={handleChange}
            className={errors.creditHistoryYears ? "input-error" : ""}
          />
          {errors.creditHistoryYears && (
            <span className="error-text">{errors.creditHistoryYears}</span>
          )}
        </div>

        {/* Loan Term */}
        <div className="input-group">
          <label>📆 Loan Term</label>
          <select
            name="loanTerm"
            required
            value={formData.loanTerm}
            onChange={handleChange}
          >
            <option value={36}>36 months</option>
            <option value={60}>60 months</option>
          </select>
        </div>

        {/* Loan Purpose */}
        <div className="input-group">
          <label>🎯 Loan Purpose</label>
          <select
            name="loanPurpose"
            required
            value={formData.loanPurpose}
            onChange={handleChange}
          >
            <option>Debt Consolidation</option>
            <option>Wedding</option>
            <option>Medical</option>
            <option>Education</option>
            <option>Home Improvement</option>
          </select>
        </div>

        {/* Loan Grade */}
        <div className="input-group">
          <label>🏷️ Loan Grade</label>
          <select
            name="loanGrade"
            required
            value={formData.loanGrade}
            onChange={handleChange}
          >
            <option value="A">Grade A</option>
            <option value="B">Grade B</option>
            <option value="C">Grade C</option>
            <option value="D">Grade D</option>
          </select>
        </div>

        {/* Home Ownership */}
        <div className="input-group">
          <label>🏠 Home Ownership</label>
          <select
            name="homeOwnership"
            required
            value={formData.homeOwnership}
            onChange={handleChange}
          >
            <option>Rent</option>
            <option>Own</option>
            <option>Mortgage</option>
          </select>
        </div>

        {/* State */}
        <div className="input-group">
          <label>📍 State</label>
          <select
            name="state"
            required
            value={formData.state}
            onChange={handleChange}
          >
            <option>CA</option>
            <option>NY</option>
            <option>TX</option>
            <option>FL</option>
            <option>IL</option>
          </select>
        </div>
      </div>

      <button
        className="btn-primary"
        onClick={handleSubmit}
        disabled={isLoading}
      >
        {isLoading ? "Analyzing..." : "📑 Predict Risk"}
      </button>
    </div>
  );
};

export default BorrowerForm;
