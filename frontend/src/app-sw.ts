/// <reference lib="webworker" />

import { precacheAndRoute, cleanupOutdatedCaches } from "workbox-precaching";
import { registerRoute, NavigationRoute } from "workbox-routing";
import {
  NetworkFirst,
  StaleWhileRevalidate,
  CacheFirst,
} from "workbox-strategies";
import { ExpirationPlugin } from "workbox-expiration";
import { clientsClaim } from "workbox-core";

declare let self: ServiceWorkerGlobalScope;

/* ---------------- core boilerplate ---------------- */
clientsClaim(); // take control ASAP
self.skipWaiting(); // activate new SW immediately

// self.__WB_MANIFEST is injected by VitePWA
precacheAndRoute(self.__WB_MANIFEST);
cleanupOutdatedCaches();

/* ---------- 1. fresh HTML for every navigate --------- */
registerRoute(
  ({ request }) =>
    request.mode === "navigate" && location.pathname.startsWith("/app"),
  new NetworkFirst({
    cacheName: "html-cache-v1",
    networkTimeoutSeconds: 3,
    plugins: [
      new ExpirationPlugin({ maxEntries: 20, maxAgeSeconds: 60 * 60 }), // 1 h
    ],
  })
);

/* ---------- 2. CSS & JS, fast but up‑to‑date --------- */
registerRoute(
  ({ request }) => ["style", "script", "worker"].includes(request.destination),
  new StaleWhileRevalidate({
    cacheName: "assets-cache-v1",
    plugins: [
      new ExpirationPlugin({ maxEntries: 50, maxAgeSeconds: 24 * 60 * 60 }), // 1 day
    ],
  })
);

/* ---------- 3. images / icons, long‑lived ------------- */
registerRoute(
  ({ request }) => request.destination === "image",
  new CacheFirst({
    cacheName: "img-cache-v1",
    plugins: [
      new ExpirationPlugin({
        maxEntries: 60,
        maxAgeSeconds: 30 * 24 * 60 * 60,
      }), // 30 days
    ],
  })
);
