import "./RollOutPlan.css";

function RollOutPlan({ result }) {
  if (!result || !result.final_model) return null;

  // Backend risk string (e.g. "MEDIUM DEFAULT RISK")
  const backendRisk = result.final_model.risk_level;

  // Normalize backend risk → low | medium | high
  let normalizedRisk = "unknown";

  if (typeof backendRisk === "string") {
    const risk = backendRisk.toLowerCase();

    if (risk.includes("low")) normalizedRisk = "low";
    else if (risk.includes("medium")) normalizedRisk = "medium";
    else if (risk.includes("high")) normalizedRisk = "high";
  }

  const rolloutStrategies = {
    low: {
      title: "Low Risk Customer ✅",
      actions: [
        "Approve loan instantly",
        "Offer lower interest rate",
        "Provide higher credit limit",
        "Offer premium banking products",
      ],
    },
    medium: {
      title: "Medium Risk Customer ⚠️",
      actions: [
        "Send application for manual credit review",
        "Request additional income or employment documents",
        "Limit approved loan amount",
        "Apply standard interest rate",
      ],
    },
    high: {
      title: "High Risk Customer 🚨",
      actions: [
        "Reject or defer loan approval",
        "Request collateral or co-signer",
        "Offer secured or high-interest loan",
        "Suggest credit score improvement program",
      ],
    },
  };

  const plan = rolloutStrategies[normalizedRisk];

  // Safety fallback (should NOT happen now)
  if (!plan) {
    return (
      <div className="rollout-card">
        <h3>Risk-Based Rollout Plan</h3>
        <p>⚠️ Unable to determine rollout strategy.</p>
      </div>
    );
  }

  return (
    <div className={`rollout-card ${normalizedRisk}`}>
      <h3>Risk-Based Rollout Plan</h3>
      <h4>{plan.title}</h4>

      <ul>
        {plan.actions.map((action, index) => (
          <li key={index}>{action}</li>
        ))}
      </ul>
    </div>
  );
}

export default RollOutPlan;
