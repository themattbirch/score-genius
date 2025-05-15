// frontend/src/services/api.ts
import { apiFetch } from "@/api/client";

/**
 * Live NBA / MLB data currently in-play.
 */
export async function getLiveData() {
  const res = await apiFetch("/api/v1/data/live");
  if (!res.ok) throw new Error("Failed to fetch live data");
  return res.json();
}

/**
 * Historical box-scores for one day (YYYY-MM-DD, UTC).
 */
export async function getHistoricalData(date: string) {
  // Use URLSearchParams to be safe
  const qs = new URLSearchParams({ date }).toString();
  const res = await apiFetch(`/api/v1/data/historical?${qs}`);
  if (!res.ok) throw new Error("Failed to fetch historical data");
  return res.json();
}

/**
 * Betting lines / odds for today.
 */
export async function getBettingOdds() {
  const res = await apiFetch("/api/v1/data/odds");
  if (!res.ok) throw new Error("Failed to fetch betting odds");
  return res.json();
}
