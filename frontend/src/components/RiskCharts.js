import React from "react";
import Plot from "react-plotly.js";

function RiskCharts({ probability }) {
  if (probability === undefined || probability === null) return null;

  return (
    <div style={{ marginTop: "30px" }}>
      <h3>Risk Probability Chart</h3>

      <Plot
        data={[
          {
            x: ["Default Risk"],
            y: [probability * 100],
            type: "bar",
          },
        ]}
        layout={{
          width: 450,
          height: 300,
          yaxis: { range: [0, 100], title: "Probability (%)" },
          title: "Loan Default Probability",
        }}
      />
    </div>
  );
}

export default RiskCharts;
