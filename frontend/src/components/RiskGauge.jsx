import { PieChart, Pie, Cell, ResponsiveContainer } from "recharts";

const RiskGauge = ({ probability }) => {
  const data = [
    { value: probability },
    { value: 100 - probability },
  ];

  const riskLabel =
    probability < 30
      ? "Low Risk"
      : probability < 60
      ? "Medium Risk"
      : "High Risk";

  const riskColor =
    probability < 30
      ? "#22c55e"
      : probability < 60
      ? "#f59e0b"
      : "#ef4444";

  return (
    <div className="card">
      <h3>📈 Default Probability</h3>
      <p
        style={{
          fontSize: "12px",
          color: "#64748b",
          marginTop: "-10px",
        }}
      >
        LightGBM Model Prediction
      </p>

      {/* ===== Gauge Container ===== */}
      <div className="risk-gauge-container">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              startAngle={180}
              endAngle={0}
              innerRadius={60}
              outerRadius={80}
              paddingAngle={0}
              dataKey="value"
            >
              <Cell fill={riskColor} />
              <Cell fill="#e2e8f0" />
            </Pie>
          </PieChart>
        </ResponsiveContainer>

        {/* ===== Center Value ===== */}
        <div className="risk-gauge-value">
          <div className="percent" style={{ color: riskColor }}>
            {probability.toFixed(1)}%
          </div>
          <div className="label">{riskLabel}</div>
        </div>
      </div>

      {/* ===== Legend ===== */}
      <div className="gauge-legend">
        <span className="legend-low">● Low (0–30%)</span>
        <span className="legend-medium">● Medium (30–60%)</span>
        <span className="legend-high">● High (60–100%)</span>
      </div>
    </div>
  );
};

export default RiskGauge;
