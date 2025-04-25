// src/api/client.ts
const API = import.meta.env.VITE_API_BASE_URL ?? "";
export const apiFetch = (path: string, init?: RequestInit) =>
  fetch(`${API}${path}`, init);
