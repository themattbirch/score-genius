// src/app/app-sw.ts
/// <reference lib="webworker" />
declare const self: ServiceWorkerGlobalScope;

import {
  precacheAndRoute,
  cleanupOutdatedCaches,
  matchPrecache,
} from "workbox-precaching";
import { registerRoute, setCatchHandler } from "workbox-routing";
import {
  StaleWhileRevalidate,
  CacheFirst,
  NetworkFirst,
} from "workbox-strategies";
import { ExpirationPlugin } from "workbox-expiration";
import { enable as enableNavigationPreload } from "workbox-navigation-preload";
import { CacheableResponsePlugin } from "workbox-cacheable-response";

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
self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      // keep your existing skipWaiting if you like
      try {
        const cache = await caches.open(PAGE_CACHE);
        // ensure canonical entry matches your launch URL (/app/)
        await cache.add(new Request("/app/", { cache: "reload" }));
      } catch {}
      self.skipWaiting?.();
    })()
  );
});

// Enable nav preload once SW activates
self.addEventListener("activate", (event: ExtendableEvent) => {
  event.waitUntil(
    (async () => {
      try {
        enableNavigationPreload();
      } catch {
        // no-op if unsupported
      }
      await self.clients.claim();
    })()
  );
});

// Reuse existing strategy but prefer the preloaded response
const pageStrategy = new NetworkFirst({
  cacheName: PAGE_CACHE,
  networkTimeoutSeconds: 3,
  plugins: [
    new CacheableResponsePlugin({ statuses: [200] }),
    {
      handlerDidError: async () => await matchPrecache(OFFLINE_URL),
    },
  ],
});

// -----------------------------------------------------------------------------
// Navigation requests → NetworkFirst → cached shell → offline fallback
// -----------------------------------------------------------------------------
// Prefer nav-preload; otherwise use your NetworkFirst strategy; always return a Response
registerRoute(
  ({ request }) => request.mode === "navigate",
  async (ctx): Promise<Response> => {
    const fe = ctx.event as FetchEvent & {
      preloadResponse?: Promise<Response>;
    };

    // 1) Try navigation preload (when enabled)
    try {
      if (fe.preloadResponse) {
        const preload = await fe.preloadResponse;
        if (preload) return preload;
      }
    } catch {}

    // 2) Your existing strategy
    try {
      return (await pageStrategy.handle(ctx as any)) as Response;
    } catch {}

    // 3) Guaranteed fallback (precached offline), else a hard Response
    const offline = await matchPrecache(OFFLINE_URL);
    return (
      offline ||
      new Response("offline", {
        status: 503,
        headers: { "Content-Type": "text/html" },
      })
    );
  }
);

// -----------------------------------------------------------------------------
// Static assets (JS, CSS, Web-workers) → Stale-While-Revalidate
// -----------------------------------------------------------------------------
const assetStrategy = new StaleWhileRevalidate({
  cacheName: ASSET_CACHE,
  plugins: [
    new CacheableResponsePlugin({ statuses: [200] }),
    new ExpirationPlugin({ maxEntries: 50, maxAgeSeconds: 86_400 }),
  ],
});
registerRoute(
  ({ request }) => ["style", "script", "worker"].includes(request.destination),
  assetStrategy
);

// Images (add guard)
registerRoute(
  ({ request }) => request.destination === "image",
  new CacheFirst({
    cacheName: IMG_CACHE,
    plugins: [
      new CacheableResponsePlugin({ statuses: [200] }),
      new ExpirationPlugin({ maxEntries: 60, maxAgeSeconds: 30 * 86_400 }),
    ],
  })
);

// API (add guard too)
registerRoute(
  ({ url }) => url.pathname.startsWith("/api/"),
  new NetworkFirst({
    cacheName: "api-data-cache-v1",
    plugins: [
      new CacheableResponsePlugin({ statuses: [200] }),
      new ExpirationPlugin({ maxEntries: 50, maxAgeSeconds: 5 * 60 }),
      {
        handlerDidError: async ({ request }) => {
          const cache = await caches.open("api-data-cache-v1");
          const cached = await cache.match(request);
          return cached || Response.error();
        },
      },
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
