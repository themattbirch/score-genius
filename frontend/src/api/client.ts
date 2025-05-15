// frontend/src/api/client.ts
const API = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/+$/, "");
// e.g. "https://score-genius-backend.onrender.com"

export async function apiFetch(path: string, init: RequestInit = {}) {
  // Allow callers to pass an absolute URL (handy for Supabase signed URLs etc.)
  const url = /^https?:\/\//.test(path)
    ? path
    : `${API}/${path.replace(/^\/+/, "")}`; // ensure exactly one “/” between

  const res = await fetch(url, {
    cache: "no-store",
    ...init,
    headers: {
      accept: "application/json",
      ...(init.headers || {}),
    },
  });

  return res;
}
