function RiskDashboard() {
  const riskScore = 0.72; // demo value

  let riskLevel = "Low Risk";
  let color = "green";

  if (riskScore > 0.7) {
    riskLevel = "High Risk";
    color = "red";
  } else if (riskScore > 0.4) {
    riskLevel = "Medium Risk";
    color = "orange";
  }

  return (
    <div style={{
      border: "1px solid #ddd",
      padding: "15px",
      marginTop: "20px",
      borderRadius: "8px"
    }}>
      <h2>Borrower Risk Assessment</h2>

      <p>
        <strong>Predicted Risk Score:</strong> {riskScore}
      </p>

      <p style={{ color }}>
        <strong>Risk Category:</strong> {riskLevel}
      </p>
    </div>
  );
}

export default RiskDashboard;
