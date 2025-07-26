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
  NetworkFirst,
  StaleWhileRevalidate,
  CacheFirst,
} from "workbox-strategies";
import { ExpirationPlugin } from "workbox-expiration";

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
const OFFLINE_URL = "/app/offline.html";
const PAGE_CACHE = "pages-cache-v1";
const ASSET_CACHE = "assets-cache-v1";
const IMG_CACHE = "img-cache-v1";

// -----------------------------------------------------------------------------
// Lifecycle
// -----------------------------------------------------------------------------
clientsClaim();
self.skipWaiting();

precacheAndRoute([...self.__WB_MANIFEST, { url: OFFLINE_URL, revision: null }]);

cleanupOutdatedCaches();

// -----------------------------------------------------------------------------
// Navigation: NetworkFirst with fallback
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
  async ({ request, event }) => {
    try {
      const resp = await networkFirstPages.handle({ request, event });
      if (resp) return resp;
      throw new Error("no-response");
    } catch {
      const offlineResp = await matchPrecache(OFFLINE_URL);
      return (
        offlineResp ??
        new Response("You are offline.", {
          status: 503,
          statusText: "Service Unavailable",
          headers: { "Content-Type": "text/html" },
        })
      );
    }
  }
);

// -----------------------------------------------------------------------------
// Static assets: Stale-While-Revalidate
// -----------------------------------------------------------------------------
registerRoute(
  ({ request }) => ["style", "script", "worker"].includes(request.destination),
  new StaleWhileRevalidate({
    cacheName: ASSET_CACHE,
    plugins: [new ExpirationPlugin({ maxEntries: 50, maxAgeSeconds: 86400 })],
  })
);

// -----------------------------------------------------------------------------
// Images: CacheFirst
// -----------------------------------------------------------------------------
registerRoute(
  ({ request }) => request.destination === "image",
  new CacheFirst({
    cacheName: IMG_CACHE,
    plugins: [
      new ExpirationPlugin({ maxEntries: 60, maxAgeSeconds: 30 * 86400 }),
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

// -----------------------------------------------------------------------------
// Global catch handler for any failures
// -----------------------------------------------------------------------------
setCatchHandler(async ({ request }) => {
  if (request.mode === "navigate") {
    return (await matchPrecache(OFFLINE_URL))!;
  }
  return Response.error();
});
