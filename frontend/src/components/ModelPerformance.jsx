import { useEffect, useState } from "react";
import "./ModelPerformance.css";

function ModelPerformance() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://localhost:8000/model-performance")
      .then((res) => res.json())
      .then((result) => {
        setData(result);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading) return <p className="loading">Loading model performance...</p>;
  if (!data) return null;

  const models = Object.keys(data);

  // Find best model by F1 score
  const bestModel = models.reduce((best, model) =>
    data[model].f1 > data[best].f1 ? model : best
  );

  return (
    <div className="performance-card">
      <h3>Model Performance Comparison</h3>

      {/* Metric Cards */}
      <div className="metric-grid">
        {models.map((model) => (
          <div
            key={model}
            className={`metric-box ${model === bestModel ? "best-model" : ""}`}
          >
            <h4>{model}</h4>
            <p className="metric">
              Accuracy: <span>{data[model].accuracy}</span>
            </p>
            <p className="metric">
              Precision: <span>{data[model].precision}</span>
            </p>
            <p className="metric">
              Recall: <span>{data[model].recall}</span>
            </p>
            <p className="metric">
              F1 Score: <span>{data[model].f1}</span>
            </p>
          </div>
        ))}
      </div>

      {/* Table View */}
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1 Score</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr key={model}>
                <td>{model}</td>
                <td>{data[model].accuracy}</td>
                <td>{data[model].precision}</td>
                <td>{data[model].recall}</td>
                <td>{data[model].f1}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default ModelPerformance;
