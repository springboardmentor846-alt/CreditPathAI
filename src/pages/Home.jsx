// src/pages/Home.jsx
import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import axios from "axios";
import Plot from "react-plotly.js";

const API_BASE = "http://127.0.0.1:8000";

export default function Home() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [limit] = useState(200);

  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        setLoading(true);
        const resp = await axios.get(`${API_BASE}/logs`, { params: { limit, offset: 0 } });
        if (!mounted) return;
        setLogs(resp.data.results || []);
      } catch (e) {
        console.error("Failed to load logs for home:", e);
        if (mounted) setLogs([]);
      } finally {
        if (mounted) setLoading(false);
      }
    }
    load();
    return () => { mounted = false; };
  }, [limit]);

  // histogram data
  const probs = useMemo(() => logs.map(r => Number(r.pred_prob ?? 0)), [logs]);

  // bucket counts
  const bucketCounts = useMemo(() => {
    const counts = { low: 0, medium: 0, high: 0, very_high: 0 };
    logs.forEach(r => {
      const k = r.risk_bucket || "low";
      counts[k] = (counts[k] || 0) + 1;
    });
    return counts;
  }, [logs]);

  // small summary stats
  const avgProb = useMemo(() => {
    if (!probs.length) return 0;
    return probs.reduce((a,b) => a + b, 0) / probs.length;
  }, [probs]);

  return (
    // Single top-level element — required in JSX
    <div className="home-fullpage">

      {/* TOP: compact analytics strip */}
      <section className="card card-pretty p-3 mb-4">
        <div className="d-flex align-items-center justify-content-between flex-column flex-md-row">
          <div style={{minWidth: 280, flex: 1}}>
            <h2 style={{ margin: 0, color: "#eaf2ff" }}>CreditPathAI — Summary</h2>
            <div className="text-muted" style={{ marginTop: 6 }}>
              Live sample from recent predictions (showing {logs.length} rows)
            </div>

            <div className="d-flex gap-3 mt-3" style={{ flexWrap: "wrap" }}>
              <div className="chip" style={{ background: "linear-gradient(90deg,#3b82f6,#7c3aed)" }}>
                Avg Prob: {(avgProb * 100).toFixed(2)}%
              </div>
              <div className="chip" style={{ background: "linear-gradient(90deg,#60a5fa,#34d399)" }}>
                High: {bucketCounts.high}
              </div>
              <div className="chip" style={{ background: "linear-gradient(90deg,#fb7185,#f59e0b)" }}>
                Very High: {bucketCounts.very_high}
              </div>
              <div className="chip" style={{ background: "linear-gradient(90deg,#94a3b8,#c7d2fe)" }}>
                Medium: {bucketCounts.medium}
              </div>
            </div>
          </div>

          {/* small histogram visual */}
          <div style={{ width: 420, maxWidth: "42%", minWidth: 240 }}>
            <Plot
              data={[{ x: probs, type: "histogram", nbinsx: 20, marker: { color: "#7c3aed" } }]}
              layout={{
                margin: { t: 6, l: 30, r: 6, b: 30 },
                height: 120,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                xaxis: { showgrid: false, zeroline: false, color: "#cfe7ff" },
                yaxis: { showgrid: false, zeroline: false, color: "#cfe7ff" }
              }}
              config={{ displayModeBar: false }}
              style={{ width: "100%" }}
            />
          </div>
        </div>
      </section>

      {/* MAIN: three simple feature cards (Predict/Dashboard/Logs) */}
      <section className="row g-4">
        <div className="col-md-4">
          <div className="card card-pretty p-3 h-100">
            <h4 style={{ color: "#60a5fa" }}>Predict</h4>
            <p className="text-muted">Score a borrower quickly using the production model.</p>
            <Link to="/predict" className="btn btn-primary w-100">Go to Predict</Link>
          </div>
        </div>

        <div className="col-md-4">
          <div className="card card-pretty p-3 h-100">
            <h4 style={{ color: "#fb7185" }}>Dashboard</h4>
            <p className="text-muted">Analytics, risk distribution & time-series insights.</p>
            <Link to="/dashboard" className="btn btn-primary w-100">Open Dashboard</Link>
          </div>
        </div>

        <div className="col-md-4">
          <div className="card card-pretty p-3 h-100">
            <h4 style={{ color: "#34d399" }}>Logs</h4>
            <p className="text-muted">Inspect past predictions, inputs and SHAP drivers.</p>
            <Link to="/logs" className="btn btn-primary w-100">View Logs</Link>
          </div>
        </div>
      </section>

      <footer className="text-center mt-4" style={{ color: "#9fb0cc" }}>
        CreditPathAI • DayaSagar Doppalapudi • 2025
      </footer>
    </div>
  );
}