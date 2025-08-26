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
  NetworkOnly,
} from "workbox-strategies";
import { ExpirationPlugin } from "workbox-expiration";
import { enable as enableNavigationPreload } from "workbox-navigation-preload";
import { CacheableResponsePlugin } from "workbox-cacheable-response";
import type { HTTPMethod } from "workbox-routing/utils/constants";

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
      try {
        const cache = await caches.open(PAGE_CACHE);
        await cache.add(new Request("/app/", { cache: "reload" }));
      } catch {}
      self.skipWaiting?.();
    })()
  );
});

self.addEventListener("activate", (event: ExtendableEvent) => {
  event.waitUntil(
    (async () => {
      try {
        enableNavigationPreload();
      } catch {}
      await self.clients.claim();
    })()
  );
});

// Reuse existing strategy but prefer the preloaded response
const pageStrategy = new NetworkFirst({
  cacheName: PAGE_CACHE,
  networkTimeoutSeconds: 8, // was 3
  plugins: [
    new CacheableResponsePlugin({ statuses: [200] }),
    { handlerDidError: async () => await matchPrecache(OFFLINE_URL) },
  ],
});

// -----------------------------------------------------------------------------
// Explicitly bypass/cordon off Supabase (no caching, no fallbacks)
// Define BEFORE setCatchHandler since we reference it there.
// -----------------------------------------------------------------------------
const isSupabaseHost = (h: string) =>
  h.endsWith(".supabase.co") || h.endsWith(".supabase.in");

const matchSupabase = ({ url }: { url: URL }) => isSupabaseHost(url.hostname);

const supabaseMethods: HTTPMethod[] = ["GET", "POST", "PUT", "PATCH", "DELETE"];

supabaseMethods.forEach((method) => {
  registerRoute(matchSupabase, new NetworkOnly(), method);
});

// -----------------------------------------------------------------------------
// Navigation → NetworkFirst → cached shell (/app/) → offline fallback
// -----------------------------------------------------------------------------
registerRoute(
  ({ request }) => request.mode === "navigate",
  async (ctx): Promise<Response> => {
    const fe = ctx.event as FetchEvent & {
      preloadResponse?: Promise<Response>;
    };

    // Prefer navigation preload (reduces first-load races)
    try {
      if (fe.preloadResponse) {
        const preload = await fe.preloadResponse;
        if (preload) return preload;
      }
    } catch {}

    // Try normal NetworkFirst
    try {
      const res = (await pageStrategy.handle(ctx as any)) as
        | Response
        | undefined;
      if (res) return res;
    } catch {}

    // NEW: fallback to cached app shell so SPA can render deep links
    try {
      const cache = await caches.open(PAGE_CACHE);
      const shell = await cache.match("/app/");
      if (shell) return shell;
    } catch {}

    // Last resort: offline page
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

// Images
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

// Same-origin API guard
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
// Global catch-all with Supabase carve-out
// -----------------------------------------------------------------------------
setCatchHandler(async ({ request }) => {
  // Never return HTML for Supabase failures
  try {
    const url = new URL(request.url);
    if (isSupabaseHost(url.hostname)) return Response.error();
  } catch {}

  if (["script", "style", "worker"].includes(request.destination)) {
    const cache = await caches.open(ASSET_CACHE);
    const res = await cache.match(request);
    if (res) return res;
  }

  if (request.destination === "image") {
    const cache = await caches.open(IMG_CACHE);
    const res = await cache.match(request);
    if (res) return res;
  }

  const fallback = await matchPrecache(OFFLINE_URL);
  return (
    fallback ||
    new Response("offline", {
      status: 503,
      headers: { "Content-Type": "text/html" },
    })
  );
});
