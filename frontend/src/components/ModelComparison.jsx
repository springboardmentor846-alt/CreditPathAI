import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import "./ModelComparison.css";

function ModelComparison() {
  const [rocData, setRocData] = useState(null);

  useEffect(() => {
    fetch("http://localhost:8000/roc-metrics")
      .then((res) => res.json())
      .then((data) => setRocData(data))
      .catch((err) => console.error(err));
  }, []);

  if (!rocData) return <p className="roc-loading">Loading ROC curves...</p>;

  const traces = Object.keys(rocData).map((modelName) => ({
    x: rocData[modelName].fpr,
    y: rocData[modelName].tpr,
    type: "scatter",
    mode: "lines",
    name: `${modelName} (AUC = ${rocData[modelName].auc})`,
  }));

  // Random baseline
  traces.push({
    x: [0, 1],
    y: [0, 1],
    type: "scatter",
    mode: "lines",
    name: "Random Guess",
    line: { dash: "dash" },
  });

  // Find best model
  const bestModel = Object.entries(rocData).reduce((best, curr) =>
    curr[1].auc > best[1].auc ? curr : best
  );

  return (
    <div className="roc-card">
      <h3>ROC–AUC Model Comparison</h3>

      <Plot
        data={traces}
        layout={{
          title: "ROC Curves for Loan Default Prediction Models",
          xaxis: { title: "False Positive Rate" },
          yaxis: { title: "True Positive Rate" },
          height: 500,
        }}
      />

      <div className="verdict-box">
        <h4>📌 Final Verdict</h4>
        <p>
          <strong>{bestModel[0]}</strong> achieved the highest ROC–AUC score (
          <strong>{bestModel[1].auc}</strong>) and is selected as the{" "}
          <strong>final model</strong>.
        </p>
      </div>
    </div>
  );
}

export default ModelComparison;
