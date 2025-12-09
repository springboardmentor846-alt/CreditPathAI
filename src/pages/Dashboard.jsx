// dashboard/src/pages/Dashboard.jsx
import React, { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const THEME_COLORS = ["#3b82f6","#7c3aed","#fb7185","#14b8a6","#f59e0b"];

export default function Dashboard() {
  const [logs, setLogs] = useState([]);
  const [limit, setLimit] = useState(200);
  const [minProb, setMinProb] = useState(0);
  const [loading, setLoading] = useState(false);

  async function fetchLogs() {
    try {
      setLoading(true);
      const resp = await axios.get(`${API_BASE}/logs`, {
        params: { limit, offset: 0, min_prob: minProb || undefined },
      });
      setLogs(resp.data.results || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchLogs();
    const id = setInterval(fetchLogs, 8000);
    return () => clearInterval(id);
  }, [limit, minProb]);

  const probs = useMemo(() => logs.map((r) => Number(r.pred_prob ?? 0)), [logs]);

  const bucketCounts = useMemo(() => {
    const counts = { low: 0, medium: 0, high: 0, very_high: 0 };
    logs.forEach((r) => {
      counts[r.risk_bucket] = (counts[r.risk_bucket] || 0) + 1;
    });
    return counts;
  }, [logs]);

  const pieLabels = ["low", "medium", "high", "very_high"];
  const pieValues = pieLabels.map((l) => bucketCounts[l] || 0);

  const timeseries = useMemo(() => {
    const map = {};
    logs.forEach((r) => {
      try {
        const dt = new Date(r.ts);
        const key = dt.toISOString().slice(0, 16);
        if (r.risk_bucket === "high" || r.risk_bucket === "very_high") {
          map[key] = (map[key] || 0) + 1;
        }
      } catch (e) {}
    });
    const keys = Object.keys(map).sort();
    return { keys, counts: keys.map((k) => map[k]) };
  }, [logs]);

  return (
    <div className="max-w-6xl mx-auto">
      <div className="card card-pretty p-3 mb-3">
        <div className="d-flex justify-content-between align-items-center mb-2">
          <h4 className="mb-0">Analytics Dashboard</h4>
          <div>
            <input type="number" step="0.01" min="0" max="1" className="form-control d-inline-block me-2" style={{ width: 120 }} value={minProb} onChange={(e) => setMinProb(e.target.value === "" ? 0 : Number(e.target.value))} />
            <input type="number" className="form-control d-inline-block me-2" style={{ width: 90 }} value={limit} onChange={(e) => setLimit(Number(e.target.value))} />
            <button className="btn btn-primary" onClick={fetchLogs}>Refresh</button>
          </div>
        </div>

        <div className="row">
          <div className="col-lg-8 mb-3">
            <div className="card p-3 h-100">
              <h6>Probability distribution</h6>
              <Plot data={[{ x: probs, type: "histogram", nbinsx: 30, marker: { color: THEME_COLORS[0] } }]} layout={{ height: 300, margin: { t: 20 } }} style={{ width: "100%" }} />
            </div>
          </div>

          <div className="col-lg-4 mb-3">
            <div className="card p-3 h-100">
              <h6>Bucket breakdown</h6>
              <Plot data={[{ labels: pieLabels, values: pieValues, type: "pie", textinfo: "label+percent", marker: { colors: THEME_COLORS } }]} layout={{ height: 300, margin: { t: 10 } }} style={{ width: "100%" }} />
              <div className="mt-2 small text-muted">
                low: {bucketCounts.low} • medium: {bucketCounts.medium} • high: {bucketCounts.high} • very_high: {bucketCounts.very_high}
              </div>
            </div>
          </div>
        </div>

        <div className="card p-3 mt-3">
          <h6>High-risk time-series (per minute)</h6>
          <Plot data={[{ x: timeseries.keys, y: timeseries.counts, type: "scatter", mode: "lines+markers", line: { color: THEME_COLORS[1] } }]} layout={{ height: 240, margin: { t: 10 } }} style={{ width: "100%" }} />
        </div>

        <div className="card p-3 mt-3">
          <h6>Recent predictions (top {limit})</h6>
          <div className="table-responsive">
            <table className="table table-striped table-sm">
              <thead className="table-light">
                <tr>
                  <th>ID</th><th>TS</th><th>Prob</th><th>Bucket</th><th>Action</th><th>Top drivers</th>
                </tr>
              </thead>
              <tbody>
                {logs.map(r => (
                  <tr key={r.id}>
                    <td>{r.id}</td>
                    <td>{new Date(r.ts).toLocaleString()}</td>
                    <td>{Number(r.pred_prob).toFixed(3)}</td>
                    <td>{r.risk_bucket}</td>
                    <td>{r.recommended_action}</td>
                    <td>{r.top_shap ? (Array.isArray(r.top_shap) ? r.top_shap.join(", ") : r.top_shap) : "N/A"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>
  );
}