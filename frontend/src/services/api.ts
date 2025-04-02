// frontend/src/services/api.ts
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function getLiveData() {
  const response = await fetch(`${API_BASE_URL}/data/live`);
  if (!response.ok) {
    throw new Error("Failed to fetch live data");
  }
  return response.json();
}

export async function getHistoricalData(date: string) {
  const response = await fetch(`${API_BASE_URL}/data/historical?date=${date}`);
  if (!response.ok) {
    throw new Error("Failed to fetch historical data");
  }
  return response.json();
}

export async function getBettingOdds() {
  const response = await fetch(`${API_BASE_URL}/data/odds`);
  if (!response.ok) {
    throw new Error("Failed to fetch betting odds");
  }
  return response.json();
}
