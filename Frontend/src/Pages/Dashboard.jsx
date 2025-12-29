import React, { useState } from "react";
import LoanForm from "../Components/LoanForm";
import RiskGauge from "../Components/RiskGauge";
import RecommendationCard from "../Components/RecommendationCard";
import { predictRisk } from "../Services/api";

const Dashboard = () => {
  const [probability, setProbability] = useState(null);
  const [riskLevel, setRiskLevel] = useState("");
  const [recommendation, setRecommendation] = useState("");

  const handlePredict = async (data) => {
    try {
      const response = await predictRisk(data);

      
      setProbability(response.data.default_probability);
      setRiskLevel(response.data.risk_level);
      setRecommendation(response.data.recommendation);

    } catch (error) {
      console.error("Prediction error:", error);
      alert("Backend connection failed. Please check FastAPI server.");
    }
  };

  return (
    <div style={{ padding: "30px" }}>
      <h1>CreditPathAI – Agent Dashboard</h1>

      <LoanForm onSubmit={handlePredict} />

      {/* ✅ Send probability to gauge */}
      <RiskGauge probability={probability} />

      <RecommendationCard
        riskLevel={riskLevel}
        recommendation={recommendation}
      />
    </div>
  );
};

export default Dashboard;
