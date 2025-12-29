import React from "react";
import Plot from "react-plotly.js";

const RiskGauge = ({ probability }) => {
  return (
    <Plot
      data={[
        {
          type: "indicator",
          mode: "gauge+number",
          value: probability ? probability * 100 : 0,
          title: { text: "Default Risk (%)" },
          gauge: {
            axis: { range: [0, 100] },
            bar: { color: "red" }
          }
        }
      ]}
      layout={{ width: 400, height: 300 }}
    />
  );
};

export default RiskGauge;
