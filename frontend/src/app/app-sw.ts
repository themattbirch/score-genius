// src/app/app-sw.ts
/// <reference lib="webworker" />
declare const self: ServiceWorkerGlobalScope;

import { clientsClaim } from "workbox-core";
import {
  precacheAndRoute,
  cleanupOutdatedCaches,
  matchPrecache,
} from "workbox-precaching";
import { registerRoute, setCatchHandler } from "workbox-routing";
import {
  NetworkOnly,
  StaleWhileRevalidate,
  CacheFirst,
  NetworkFirst,
} from "workbox-strategies";
import { ExpirationPlugin } from "workbox-expiration";

const OFFLINE_URL = "/app/offline.html";
const PAGE_CACHE = "pages-cache-v1";
const ASSET_CACHE = "assets-cache-v1";
const IMG_CACHE = "img-cache-v1";

// -----------------------------------------------------------------------------
// Pre-cache core assets (shell + offline fallback) and clean old caches
// -----------------------------------------------------------------------------
precacheAndRoute([...self.__WB_MANIFEST, { url: OFFLINE_URL, revision: null }]);
cleanupOutdatedCaches();

// -----------------------------------------------------------------------------
// Service-worker lifecycle
// -----------------------------------------------------------------------------
clientsClaim();
self.skipWaiting();

self.addEventListener("install", () => self.skipWaiting());

self.addEventListener("activate", (event) => {
  event.waitUntil(self.registration.navigationPreload.enable()); // speed up first load
  self.clients.claim();
});

// -----------------------------------------------------------------------------
// Navigation requests → NetworkFirst → cached shell → offline fallback
// -----------------------------------------------------------------------------
registerRoute(
  ({ request }) => request.mode === "navigate",
  new NetworkFirst({
    cacheName: PAGE_CACHE,
    networkTimeoutSeconds: 3, // quick network race
    plugins: [
      {
        // If both network & cache miss, still return something useful
        handlerDidError: async () => await matchPrecache(OFFLINE_URL),
      },
    ],
  })
);

// -----------------------------------------------------------------------------
// Static assets (JS, CSS, Web-workers) → Stale-While-Revalidate
// -----------------------------------------------------------------------------
registerRoute(
  ({ request }) => ["style", "script", "worker"].includes(request.destination),
  new StaleWhileRevalidate({
    cacheName: ASSET_CACHE,
    plugins: [new ExpirationPlugin({ maxEntries: 50, maxAgeSeconds: 86_400 })],
  })
);

// -----------------------------------------------------------------------------
// Images → CacheFirst
// -----------------------------------------------------------------------------
registerRoute(
  ({ request }) => request.destination === "image",
  new CacheFirst({
    cacheName: IMG_CACHE,
    plugins: [
      new ExpirationPlugin({ maxEntries: 60, maxAgeSeconds: 30 * 86_400 }),
    ],
  })
);

// -----------------------------------------------------------------------------
// API calls → NetworkFirst (short-lived cache)
// -----------------------------------------------------------------------------
registerRoute(
  ({ url }) => url.pathname.startsWith("/api/"),
  new NetworkFirst({
    cacheName: "api-data-cache-v1",
    plugins: [
      new ExpirationPlugin({
        maxEntries: 50,
        maxAgeSeconds: 5 * 60,
      }),
    ],
  })
);

// -----------------------------------------------------------------------------
// Handle “skipWaiting” messages from the app
// -----------------------------------------------------------------------------
self.addEventListener("message", (event) => {
  if (event.data?.type === "SKIP_WAITING") self.skipWaiting();
});

// -----------------------------------------------------------------------------
// Global catch-all: fall back to cached assets or images when possible
// -----------------------------------------------------------------------------
setCatchHandler(async ({ request }) => {
  // Try cached JS/CSS/workers
  if (["script", "style", "worker"].includes(request.destination)) {
    const cache = await caches.open(ASSET_CACHE);
    const res = await cache.match(request);
    if (res) {
      return res;
    }
  }

  // Try cached images
  if (request.destination === "image") {
    const cache = await caches.open(IMG_CACHE);
    const res = await cache.match(request);
    if (res) {
      return res;
    }
  }

  // Always return a Response here
  const fallback = await matchPrecache(OFFLINE_URL);
  return (
    fallback ||
    new Response("offline", {
      status: 503,
      headers: { "Content-Type": "text/html" },
    })
  );
});
