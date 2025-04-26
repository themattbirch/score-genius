// frontend/src/api/client.ts

// We only need a relative path here; Render rewrites `/api/v1/*` to your backend in prod,
// and Vite’s dev server proxies `/api/v1/*` to localhost:3001 in dev.
export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  // If you _did_ want to hit a different host in prod, you could read
  // import.meta.env.VITE_API_BASE_URL here—otherwise leave it empty.
  const base = "";
  return fetch(`${base}${path}`, init);
}
