import axios from "axios";

// Ensure this matches your FastAPI URL
const API_URL = "http://127.0.0.1:8000";

export const getPrediction = async (customerId) => {
  try {
    const response = await axios.get(`${API_URL}/predict/${customerId}`);
    console.log("API Response:", response.data); // Debugging log
    return response.data;
  } catch (error) {
    if (error.response) {
      // The request was made and the server responded with a status code
      console.error(
        "Server Error:",
        error.response.status,
        error.response.data
      );
    } else if (error.request) {
      // The request was made but no response was received
      console.error("Network Error: Is the backend running?");
    } else {
      console.error("Error:", error.message);
    }
    return null;
  }
};
