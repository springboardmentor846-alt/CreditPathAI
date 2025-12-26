const API_URL = import.meta.env.VITE_API_URL;

export async function predictRisk(payload) {
  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error("Prediction failed");
  }

  return await res.json();
}
