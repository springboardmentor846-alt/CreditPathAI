const API_URL = "https://legendary-space-sniffle-r7g5pvq6j552p56q-8000.app.github.dev";

export async function predictRisk(data) {
  try {
    const response = await fetch(`${API_URL}/predict/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    if (!response.ok) throw new Error("Backend error");
    return await response.json();

  } catch (err) {
    alert("API failed");
    console.error(err);
    return null;
  }
}