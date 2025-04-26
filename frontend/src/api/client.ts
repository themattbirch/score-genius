// frontend/src/api/client.ts

// frontend/src/api/client.ts

// Pull in the full URL from your env (set in .env/.env.local)
const API = import.meta.env.VITE_API_BASE_URL || "";

export async function apiFetch(
  path: string,
  init?: RequestInit
): Promise<Response> {
  // In dev: API === "http://localhost:3001"
  // In prod: API === "https://score-genius-backend.onrender.com"
  return fetch(`${API}${path}`, init);
}
