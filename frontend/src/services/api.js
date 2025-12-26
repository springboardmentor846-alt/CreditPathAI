import axios from "axios";

const API_URL = "http://127.0.0.1:8000/predict";

export const getRiskRecommendation = async (payload) => {
  try {
    const response = await axios.post(API_URL, payload);
    return response.data;
  } catch (error) {
    console.error("API Error:", error);
    return {
      risk: "High",
      recovery_probability: 0.42,
      action: "Immediate legal follow-up"
    };
  }
};
