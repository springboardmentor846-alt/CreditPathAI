// dashboard/src/pages/Predict.jsx
import React, { useState } from "react";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export default function Predict() {
  const [inputs, setInputs] = useState({
    loanoriginalamount: 5000,
    monthlyloanpayment: 160,
    statedmonthlyincome: 2000,
    credit_age_months: 40,
    delinquencieslast7years: 0,
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setInputs(s => ({ ...s, [name]: value === "" ? "" : Number(value) }));
  };

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    try {
      const resp = await axios.post(`${API_BASE}/predict?shap_top_k=3`, { data: inputs });
      setResult(resp.data);
    } catch (err) {
      setResult({ error: err.response?.data?.detail || err.message || "Network error" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card card-pretty p-4">
      <h3 className="mb-3">Score a Borrower</h3>

      <form onSubmit={submit}>
        <div className="row">
          <div className="col-md-4 mb-3">
            <label className="form-label">Loan Amount</label>
            <input name="loanoriginalamount" type="number" className="form-control" value={inputs.loanoriginalamount} onChange={handleChange} />
          </div>

          <div className="col-md-4 mb-3">
            <label className="form-label">Monthly Payment</label>
            <input name="monthlyloanpayment" type="number" className="form-control" value={inputs.monthlyloanpayment} onChange={handleChange} />
          </div>

          <div className="col-md-4 mb-3">
            <label className="form-label">Stated Monthly Income</label>
            <input name="statedmonthlyincome" type="number" className="form-control" value={inputs.statedmonthlyincome} onChange={handleChange} />
          </div>

          <div className="col-md-4 mb-3">
            <label className="form-label">Credit Age (months)</label>
            <input name="credit_age_months" type="number" className="form-control" value={inputs.credit_age_months} onChange={handleChange} />
          </div>

          <div className="col-md-4 mb-3">
            <label className="form-label">Delinquencies (last 7 yrs)</label>
            <input name="delinquencieslast7years" type="number" className="form-control" value={inputs.delinquencieslast7years} onChange={handleChange} />
          </div>
        </div>

        <div className="mt-2">
          <button className="btn btn-primary" disabled={loading}>{loading ? "Scoring..." : "Get Score"}</button>
        </div>
      </form>

      {result && (
        <div className="mt-3">
          {result.error ? (
            <div className="alert alert-danger">{String(result.error)}</div>
          ) : (
            <div className="card mt-2 p-3">
              <div><strong>Probability:</strong> {(result.default_probability * 100).toFixed(2)}%</div>
              <div><strong>Bucket:</strong> {result.risk_bucket}</div>
              <div><strong>Action:</strong> {result.recommended_action}</div>
              <div><strong>Top SHAP:</strong> {result.top_shap ? result.top_shap.join(", ") : "N/A"}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}