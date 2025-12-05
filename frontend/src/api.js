// src/api.js
const API_BASE = "https://orange-fortnight-69ppvpqgpx66hrvpv-8000.app.github.dev";

export async function predictRisk(inputData) {
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(inputData),
  });

  return response.json();
}

