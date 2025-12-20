import React from "react";
import Plot from "react-plotly.js";

const RiskSummaryDashboard = ({ probability }) => {
  if (probability === undefined || probability === null) return null;

  let low = 0, medium = 0, high = 0;

  if (probability < 0.33) {
    low = 100;
  } else if (probability < 0.66) {
    medium = 100;
  } else {
    high = 100;
  }

  const data = [
    {
      labels: ["Low Risk", "Medium Risk", "High Risk"],
      values: [low, medium, high],
      type: "pie",
      marker: {
        colors: ["#4CAF50", "#FFC107", "#F44336"]
      }
    }
  ];

  return (
    <div style={{ width: "100%", maxWidth: "420px", margin: "0 auto" }}>
      <Plot
        data={data}
        layout={{
          title: "Risk Distribution",
          height: 300,
          margin: { t: 40, l: 20, r: 20, b: 20 },
          responsive: true
        }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
};

export default RiskSummaryDashboard;
