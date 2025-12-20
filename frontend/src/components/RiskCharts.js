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
        autosize: true,
        yaxis: { range: [0, 100], title: "Probability (%)" },
        margin: { t: 40, l: 40, r: 20, b: 40 },
      }}
      useResizeHandler={true}
      style={{ width: "100%", height: "100%" }}
    />
    </div>
  );
}

export default RiskCharts;
