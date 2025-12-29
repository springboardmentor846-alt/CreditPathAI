import React from "react";

const RecommendationCard = ({ riskScore, riskLevel, recommendation }) => {
  if (riskScore === null) return null;

  return (
    <div style={{ marginTop: "20px", padding: "15px", border: "1px solid #ccc" }}>
      <h3>Agent Recommendation</h3>
      <p><strong>Risk Score:</strong> {riskScore}</p>
      <p><strong>Risk Level:</strong> {riskLevel}</p>
      <p><strong>Recommendation:</strong> {recommendation}</p>
    </div>
  );
};

export default RecommendationCard;
