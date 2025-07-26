// src/app-sw.ts
/// <reference lib="webworker" />
import {
  precacheAndRoute,
  cleanupOutdatedCaches,
  matchPrecache,
} from "workbox-precaching";
import { registerRoute } from "workbox-routing";
import {
  NetworkFirst,
  StaleWhileRevalidate,
  CacheFirst,
} from "workbox-strategies";
import { ExpirationPlugin } from "workbox-expiration";
import { clientsClaim } from "workbox-core";
import type { HandlerCallbackOptions } from "workbox-core/types.js";

declare let self: ServiceWorkerGlobalScope;

// ✅ CHANGED: Removed leading slash to match the precache manifest key
const OFFLINE_URL = "app/offline.html";

/* ---------- core ---------- */
clientsClaim();
// ℹ️ self.skipWaiting() at the top is redundant with the message listener below.

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.origin === "https://www.googletagmanager.com") {
    event.stopImmediatePropagation();
    return;
  }
});

precacheAndRoute(self.__WB_MANIFEST);
cleanupOutdatedCaches();

/* ---------- HTML navigations (network-first + robust fallback) ---------- */
const networkFirstPages = new NetworkFirst({
  cacheName: "pages-cache",
  networkTimeoutSeconds: 3,
});

registerRoute(
  ({ request }) => request.mode === "navigate",
  async (options: HandlerCallbackOptions): Promise<Response> => {
    try {
      return await networkFirstPages.handle(options);
    } catch {
      // This will now correctly find the offline page
      const offlineResponse = await matchPrecache(OFFLINE_URL);
      if (offlineResponse) {
        return offlineResponse;
      }
      // This is the final fallback if offline.html is somehow not precached
      return new Response("You are offline. Please connect to the internet.", {
        status: 503,
        statusText: "Service Unavailable",
        headers: { "Content-Type": "text/html" },
      });
    }
  }
);

/* ---------- assets (css/js/workers) ---------- */
registerRoute(
  ({ request }) => ["style", "script", "worker"].includes(request.destination),
  new StaleWhileRevalidate({
    cacheName: "assets-cache-v1",
    plugins: [new ExpirationPlugin({ maxEntries: 50, maxAgeSeconds: 86_400 })],
  })
);

/* ---------- images ---------- */
registerRoute(
  ({ request }) => request.destination === "image",
  new CacheFirst({
    cacheName: "img-cache-v1",
    plugins: [
      new ExpirationPlugin({ maxEntries: 60, maxAgeSeconds: 30 * 86_400 }),
    ],
  })
);

/* ---------- skip-waiting hook ---------- */
self.addEventListener("message", (event) => {
  if (event.data?.type === "SKIP_WAITING") {
    self.skipWaiting();
  }
});
