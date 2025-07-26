/// <reference lib="webworker" />
import {
  precacheAndRoute,
  cleanupOutdatedCaches,
  matchPrecache,
} from "workbox-precaching";
import { registerRoute } from "workbox-routing";
import { NetworkOnly } from "workbox-strategies";
import {
  NetworkFirst,
  StaleWhileRevalidate,
  CacheFirst,
} from "workbox-strategies";
import { ExpirationPlugin } from "workbox-expiration";
import { clientsClaim } from "workbox-core";

declare let self: ServiceWorkerGlobalScope;

/* ---------- precache core ---------- */
clientsClaim();
self.skipWaiting();

// Skip SW entirely for GTag.js
self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin === "https://www.googletagmanager.com") {
    // Prevent any other fetch listener (Workbox) from handling it
    event.stopImmediatePropagation();
    // Let the browser handle it natively
    return;
  }
});

precacheAndRoute(self.__WB_MANIFEST);
cleanupOutdatedCaches();

/* ---------- HTML navigate (network ↔ cache fallback) ---------- */
registerRoute(
  ({ request }) => request.mode === "navigate",
  new NetworkFirst({
    cacheName: "pages-cache",
    networkTimeoutSeconds: 3,
    plugins: [
      {
        // On network failure, serve the precached offline.html
        handlerDidError: () => matchPrecache("app/offline.html"),
      },
    ],
  })
);

/* ---------- assets & images (unchanged) ---------- */
registerRoute(
  ({ request }) => ["style", "script", "worker"].includes(request.destination),
  new StaleWhileRevalidate({
    cacheName: "assets-cache-v1",
    plugins: [new ExpirationPlugin({ maxEntries: 50, maxAgeSeconds: 86400 })],
  })
);

registerRoute(
  ({ request }) => request.destination === "image",
  new CacheFirst({
    cacheName: "img-cache-v1",
    plugins: [
      new ExpirationPlugin({ maxEntries: 60, maxAgeSeconds: 30 * 86400 }),
    ],
  })
);

/* ---------- skip‑waiting hook (unchanged) ---------- */
self.addEventListener("message", (event) => {
  if (event.data?.type === "SKIP_WAITING") self.skipWaiting();
});
