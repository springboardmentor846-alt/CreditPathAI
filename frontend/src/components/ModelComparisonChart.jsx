import Plot from "react-plotly.js";

function ModelComparisonChart({ result }) {
  if (!result) return null;

  const baselineProb = result.baseline_model.probability * 100;
  const finalProb = result.final_model.probability * 100;

  return (
    <Plot
      data={[
        {
          x: ["Baseline (Logistic Regression)", "Final (LightGBM)"],
          y: [baselineProb, finalProb],
          type: "bar",
        },
      ]}
      layout={{
        title: "Default Probability Comparison",
        yaxis: { title: "Probability (%)", range: [0, 100] },
        xaxis: { title: "Models" },
        height: 400,
      }}
    />
  );
}

export default ModelComparisonChart;
