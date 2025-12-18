import React from "react";

function AgentDashboard({ result }) {
  if (!result) return null;

  return (
    <div style={{
      marginTop: "20px",
      padding: "15px",
      borderRadius: "8px",
      background: "rgba(0,0,0,0.2)"
    }}>
      <h3>Agent Action Panel</h3>

      <p><b>Risk:</b> {result.risk_category}</p>

      {result.recommendation && (
        <>
          <p><b>Action:</b> {result.recommendation.action}</p>
          <p><b>Priority:</b> {result.recommendation.priority}</p>
        </>
      )}
    </div>
  );
}

export default AgentDashboard;
