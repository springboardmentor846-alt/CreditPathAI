import "./RollOutPlan.css";

function RollOutPlan({ result }) {
  if (!result || !result.final_model) return null;

  const backendRisk = result.final_model.risk_level;

  let normalizedRisk = "unknown";

  if (typeof backendRisk === "string") {
    const risk = backendRisk.toLowerCase();

    if (risk.includes("low")) normalizedRisk = "low";
    else if (risk.includes("medium")) normalizedRisk = "medium";
    else if (risk.includes("high")) normalizedRisk = "high";
  }

  const rolloutStrategies = {
    low: {
      title: "Low Default Risk Customer ✅",
      actions: [
        "Continue standard repayment monitoring",
        "Send regular payment reminders and statements",
        "Offer loyalty benefits or pre-approved top-up loans",
        "Enable flexible repayment options if needed",
      ],
    },

    medium: {
      title: "Medium Default Risk Customer ⚠️",
      actions: [
        "Increase repayment monitoring frequency",
        "Send proactive reminders before due dates",
        "Limit additional credit or top-up loans",
        "Offer repayment restructuring or tenure adjustment if stress is detected",
      ],
    },

    high: {
      title: "High Default Risk Customer 🚨",
      actions: [
        "Flag account for high-risk monitoring",
        "Initiate early warning and collections workflow",
        "Offer restructuring, moratorium, or settlement options",
        "Restrict further credit and escalate to risk management team",
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
