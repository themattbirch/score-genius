// src/app-sw.ts
/// <reference lib="webworker" />
/// <reference lib="webworker" />
declare const self: ServiceWorkerGlobalScope;

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

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
const OFFLINE_URL = "/app/offline.html";
const PAGE_CACHE = "pages-cache-v1";
const ASSET_CACHE = "assets-cache-v1";
const IMG_CACHE = "img-cache-v1";

// -----------------------------------------------------------------------------
// Core lifecycle
// -----------------------------------------------------------------------------
clientsClaim();
self.skipWaiting();

precacheAndRoute([...self.__WB_MANIFEST, { url: OFFLINE_URL, revision: null }]);
cleanupOutdatedCaches();

// -----------------------------------------------------------------------------
// Navigation requests → NetworkFirst with safer fallback
// -----------------------------------------------------------------------------
const networkFirstPages = new NetworkFirst({
  cacheName: PAGE_CACHE,
  networkTimeoutSeconds: 1.5,
  plugins: [
    {
      cacheWillUpdate: async ({ request, response }) => {
        if (!response || response.status !== 200) return null;
        if (new URL(request.url).pathname === OFFLINE_URL) return null;
        return response;
      },
    },
  ],
});

registerRoute(
  ({ request }) => request.mode === "navigate",
  async (options: HandlerCallbackOptions): Promise<Response> => {
    try {
      const resp = await networkFirstPages.handle(options);
      if (resp) return resp;
      throw new Error("No response from network");
    } catch {
      const offlineResp = await matchPrecache(OFFLINE_URL);
      if (offlineResp) return offlineResp;

      return new Response("You are offline.", {
        status: 503,
        statusText: "Service Unavailable",
        headers: { "Content-Type": "text/html" },
      });
    }
  }
);

// -----------------------------------------------------------------------------
// Assets → StaleWhileRevalidate
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
// Skip waiting on demand
// -----------------------------------------------------------------------------
self.addEventListener("message", (event) => {
  if (event.data?.type === "SKIP_WAITING") {
    self.skipWaiting();
  }
});
