import Plot from "react-plotly.js";

function RiskGauge({ result }) {
  if (!result) return null;

  const prob = result.final_model.probability * 100;

  return (
    <Plot
      data={[
        {
          type: "indicator",
          mode: "gauge+number",
          value: prob,
          title: { text: "Final Model Risk Score (%)" },
          gauge: {
            axis: { range: [0, 100] },
            steps: [
              { range: [0, 40], color: "lightgreen" },
              { range: [40, 70], color: "orange" },
              { range: [70, 100], color: "red" },
            ],
          },
        },
      ]}
      layout={{ height: 350 }}
    />
  );
}

export default RiskGauge;
