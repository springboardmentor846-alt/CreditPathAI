const RecommendationPanel = ({ result }) => {
  if (!result) return null;
  const isApprove = result.recommended_action?.toLowerCase().includes("approve");

  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
        <h3 style={{ margin: 0 }}>🛡️ Risk Assessment</h3>
        <span className="risk-badge" style={{ backgroundColor: isApprove ? '#22c55e' : '#ef4444', color: 'white' }}>
          {isApprove ? '✓ Low Risk' : '✕ High Risk'}
        </span>
      </div>

      <div className={`recommend-box ${isApprove ? 'recommend-approve' : 'recommend-reject'}`}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 'bold' }}>
          {isApprove ? '✓ Approve' : '✕ Reject'}
        </div>
        <p style={{ fontSize: '13px', margin: '8px 0 0' }}>
          {isApprove ? "Borrower meets all criteria. Proceed with loan origination." : "High risk detected."}
        </p>
      </div>

      <div className="analysis-summary">
        <p style={{ fontSize: '12px', color: '#64748b', fontWeight: 'bold', margin: '0 0 8px' }}>Analysis Summary</p>
        <p style={{ fontSize: '13px', margin: 0, lineHeight: '1.5' }}>{result.recommendation}</p>
      </div>

      <p style={{ marginTop: '20px', fontSize: '11px', color: '#94a3b8', textAlign: 'center' }}>
        Powered by LightGBM machine learning model
      </p>
    </div>
  );
};
export default RecommendationPanel;
