import React, { useState } from "react";
import Plot from "react-plotly.js";
import { getPrediction } from "./api";
import "bootstrap/dist/css/bootstrap.min.css";

const Dashboard = () => {
  const [customerId, setCustomerId] = useState("");
  const [data, setData] = useState(null);
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!customerId) return;

    setLoading(true);
    setError(false);
    setData(null);

    const result = await getPrediction(customerId);

    if (result) {
      setData(result);
    } else {
      setError(true);
    }
    setLoading(false);
  };

  const getRiskBadge = (risk) => {
    if (risk === "High") return "badge bg-danger";
    if (risk === "Medium") return "badge bg-warning text-dark";
    return "badge bg-success";
  };

  return (
    <div className="container-fluid bg-light min-vh-100 py-5">
      <div className="container">
        <div className="text-center mb-5">
          <h1 className="fw-bold text-dark display-5">🏦 CreditPath AI</h1>
          <p className="lead text-muted">
            Intelligent Collection & Recovery Dashboard
          </p>
        </div>

        <div className="row justify-content-center mb-5">
          <div className="col-md-6">
            <div className="card shadow border-0 rounded-3">
              <div className="card-body p-4">
                <form onSubmit={handleSearch} className="d-flex gap-2">
                  <input
                    type="number"
                    className="form-control form-control-lg"
                    placeholder="Enter Customer ID (e.g., 100002)"
                    value={customerId}
                    onChange={(e) => setCustomerId(e.target.value)}
                    autoFocus
                  />
                  <button
                    className="btn btn-primary btn-lg px-4"
                    type="submit"
                    disabled={loading}
                  >
                    {loading ? (
                      <span
                        className="spinner-border spinner-border-sm"
                        role="status"
                        aria-hidden="true"
                      ></span>
                    ) : (
                      "Search"
                    )}
                  </button>
                </form>
                {error && (
                  <div className="alert alert-danger mt-3 mb-0">
                    <strong>Customer Not Found.</strong> Try ID: 100002.
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {data && (
          <div className="row g-4 animate__animated animate__fadeIn">
            <div className="col-lg-4">
              <div className="card shadow-sm border-0 mb-4 h-100">
                <div className="card-header bg-white border-bottom-0 pt-4 px-4">
                  <h6 className="text-uppercase text-muted fw-bold small">
                    Risk Assessment
                  </h6>
                </div>
                <div className="card-body px-4 pt-0">
                  <div className="d-flex align-items-center justify-content-between">
                    <span
                      className={`display-6 fw-bold ${
                        data.prediction.risk_category === "High"
                          ? "text-danger"
                          : "text-success"
                      }`}
                    >
                      {data.prediction.default_probability}%
                    </span>
                    <span
                      className={`${getRiskBadge(
                        data.prediction.risk_category
                      )} fs-6 px-3 py-2 rounded-pill`}
                    >
                      {data.prediction.risk_category} Risk
                    </span>
                  </div>
                  <p className="text-muted mt-2 small">
                    Probability of Default within 12 months.
                  </p>
                </div>
              </div>
            </div>

            <div className="col-lg-8">
              <div
                className={`card shadow-sm border-0 mb-4 ${
                  data.prediction.risk_category === "High"
                    ? "border-start border-5 border-danger"
                    : "border-start border-5 border-success"
                }`}
              >
                <div className="card-body p-4">
                  <h6 className="text-uppercase text-muted fw-bold small mb-2">
                    Recommended Strategy
                  </h6>
                  <h3 className="fw-bold text-dark mb-2">
                    {data.recommendation.action}
                  </h3>
                  <div className="d-flex align-items-center text-muted">
                    <span className="me-2">💡 Reason:</span>
                    <span className="fst-italic">
                      {data.recommendation.reason || "Logic Rule Applied"}
                    </span>
                  </div>
                </div>
              </div>

              <div className="card shadow-sm border-0">
                <div className="card-header bg-white border-bottom-0 pt-4 px-4">
                  <h6 className="text-uppercase text-muted fw-bold small">
                    Key Risk Drivers
                  </h6>
                </div>
                <div className="card-body">
                  {/* --- FIX IS HERE: Changed 'context_data' to 'context' --- */}
                  <Plot
                    data={[
                      {
                        x: ["Income", "Ext. Debt", "Late Days", "Cash W/D"],
                        y: [
                          data.context.income,
                          data.context.debt,
                          data.context.late_days,
                          data.context.cash_withdrawals || 0,
                        ],
                        type: "bar",
                        marker: {
                          color: ["#0d6efd", "#dc3545", "#ffc107", "#198754"],
                          opacity: 0.8,
                        },
                        text: [
                          `$${data.context.income}`,
                          `$${data.context.debt}`,
                          `${data.context.late_days} Days`,
                          `$${data.context.cash_withdrawals || 0}`,
                        ],
                        textposition: "auto",
                      },
                    ]}
                    layout={{
                      autosize: true,
                      height: 350,
                      margin: { l: 50, r: 20, t: 30, b: 50 },
                      font: { family: "Arial, sans-serif" },
                      yaxis: { title: "Value", gridcolor: "#eee" },
                      paper_bgcolor: "rgba(0,0,0,0)",
                      plot_bgcolor: "rgba(0,0,0,0)",
                    }}
                    style={{ width: "100%", height: "100%" }}
                    config={{ displayModeBar: false }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
