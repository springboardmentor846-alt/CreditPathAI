import React from "react";
import Plot from "react-plotly.js";

function RiskSummaryChart({ probability }) {
  if (probability === undefined) return null;

  return (
    <Plot
      data={[
        {
          type: "bar",
          x: ["Low Risk", "High Risk"],
          y: [1 - probability, probability],
        },
      ]}
      layout={{
        title: "Risk Summary",
        yaxis: { range: [0, 1] },
        height: 300,
        margin: { t: 40 },
      }}
    />
  );
}

export default RiskSummaryChart;
