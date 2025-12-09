// dashboard/src/pages/Logs.jsx
import React, { useEffect, useState } from "react";
import axios from "axios";
import Spinner from "../components/Spinner";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export default function Logs() {
  const [logs, setLogs] = useState([]);
  const [limit, setLimit] = useState(300);
  const [minProb, setMinProb] = useState(0);
  const [loading, setLoading] = useState(false);

  async function fetchLogs() {
    try {
      setLoading(true);
      const resp = await axios.get(`${API_BASE}/logs`, { params: { limit, offset: 0, min_prob: minProb || undefined } });
      setLogs(resp.data.results || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { fetchLogs(); }, [limit, minProb]);

  return (
    <div className="card card-pretty p-3">
      <div className="d-flex justify-content-between align-items-center mb-2">
        <h4 className="mb-0">Logs</h4>
        <div>
          <input type="number" step="0.01" min="0" max="1" className="form-control d-inline-block me-2" style={{ width: 120 }} value={minProb} onChange={(e) => setMinProb(e.target.value === "" ? 0 : Number(e.target.value))} />
          <input type="number" className="form-control d-inline-block me-2" style={{ width: 90 }} value={limit} onChange={(e) => setLimit(Number(e.target.value))} />
          <button className="btn btn-primary" onClick={fetchLogs}>Refresh</button>
        </div>
      </div>

      {loading ? (
        <div className="text-center p-4"><Spinner size={48} /></div>
      ) : logs.length === 0 ? (
        <div className="text-center p-4 text-muted">No logs found.</div>
      ) : (
        <div className="table-responsive">
          <table className="table table-striped table-sm">
            <thead className="table-light">
              <tr>
                <th>ID</th><th>Timestamp</th><th>Probability</th><th>Bucket</th><th>Action</th><th>Top drivers</th><th>Input</th>
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
                  <td><pre className="small-mono">{JSON.stringify(r.input, null, 2)}</pre></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}