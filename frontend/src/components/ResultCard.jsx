import RiskGauge from "./RiskGauge";
import "./ResultCard.css";

function ResultCard({ result }) {
  if (!result) return null;

  const renderRiskClass = (risk) => {
    if (risk === "Low") return "risk-low";
    if (risk === "Medium") return "risk-medium";
    return "risk-high";
  };

  return (
    <div className="result-card">
      <h3>Model Prediction Comparison</h3>

      <div className="model-grid">
        {/* Baseline Model */}
        <div className="model-box">
          <h4>Baseline Model</h4>
          <p className="model-name">Logistic Regression</p>

          <p className="probability">
            {(result.baseline_model.probability * 100).toFixed(2)}%
          </p>

          <span
            className={`risk-badge ${renderRiskClass(
              result.baseline_model.risk_level
            )}`}
          >
            {result.baseline_model.risk_level}
          </span>
        </div>

        {/* Final Model */}
        <div className="model-box highlight">
          <h4>Final Model</h4>
          <p className="model-name">LightGBM</p>

          <p className="probability">
            {(result.final_model.probability * 100).toFixed(2)}%
          </p>

          <span
            className={`risk-badge ${renderRiskClass(
              result.final_model.risk_level
            )}`}
          >
            {result.final_model.risk_level}
          </span>
        </div>
      </div>

      {/* Charts Section */}
      <div className="charts-section">
        <RiskGauge result={result} />
      </div>
    </div>
  );
}

export default ResultCard;
