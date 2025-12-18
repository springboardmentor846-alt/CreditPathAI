import React from "react";
import Plot from "react-plotly.js";

const RiskSummaryDashboard = () => {
  const data = [
    {
      labels: ["Low Risk", "Medium Risk", "High Risk"],
      values: [40, 35, 25], // mock/demo data
      type: "pie",
    },
  ];

  const layout = {
    title: "Overall Loan Risk Distribution",
    height: 300,
  };

  return (
    <div style={{ marginTop: "20px" }}>
      <Plot data={data} layout={layout} />
    </div>
  );
};

export default RiskSummaryDashboard;
