/// <reference lib="webworker" />
declare const self: ServiceWorkerGlobalScope;

import { registerRoute, setCatchHandler } from "workbox-routing";
import {
  precacheAndRoute,
  cleanupOutdatedCaches,
  matchPrecache,
} from "workbox-precaching";
import {
  StaleWhileRevalidate,
  CacheFirst,
  NetworkFirst,
  NetworkOnly,
} from "workbox-strategies";
import { ExpirationPlugin } from "workbox-expiration";
import { CacheableResponsePlugin } from "workbox-cacheable-response";
import { enable as enableNavigationPreload } from "workbox-navigation-preload";

// --- Bypass the SW for third-party data hosts (no cache, no fallback) ---
const BYPASS_HOSTS = [
  "score-genius-backend.onrender.com",
  "supabase.co",
  "supabase.in",
];

registerRoute(
  // ensure we only bypass for *cross-origin* requests to those hosts
  ({ url }) =>
    url.origin !== self.location.origin &&
    BYPASS_HOSTS.some((h) => url.hostname.endsWith(h)),
  new NetworkOnly()
);

const OFFLINE_URL = "/app/offline.html";
const PAGE_CACHE = "pages-cache-v3";
const ASSET_CACHE = "assets-cache-v3";
const IMG_CACHE = "img-cache-v3";
const API_CACHE = "api-data-cache-v3";

// 1) never precache HTML (avoid stale shell)
const _WB = (self as any).__WB_MANIFEST as Array<{
  url: string;
  revision?: string;
}>;
const WB_NO_HTML = Array.isArray(_WB)
  ? _WB.filter((e) => !e.url.endsWith(".html"))
  : [];
precacheAndRoute([...WB_NO_HTML, { url: OFFLINE_URL, revision: null }]);

cleanupOutdatedCaches();

self.addEventListener("install", (event: ExtendableEvent) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event: ExtendableEvent) => {
  event.waitUntil(
    (async () => {
      try {
        enableNavigationPreload();
      } catch {}
      const keep = new Set([PAGE_CACHE, ASSET_CACHE, IMG_CACHE, API_CACHE]);
      const names = await caches.keys();
      await Promise.all(
        names.map((n) => {
          if (keep.has(n)) return; // keep our runtime caches
          if (n.startsWith("workbox-precache")) return; // **keep WB precache** (offline.html lives here)
          return caches.delete(n);
        })
      );
      await self.clients.claim();
    })()
  );
});

/* 1) Navigations: stock NetworkFirst with fresh shell; Workbox handles preload internally */
registerRoute(
  ({ request }) => request.mode === "navigate",
  new NetworkFirst({
    cacheName: PAGE_CACHE,
    networkTimeoutSeconds: 5,
    fetchOptions: { cache: "reload" }, // keeps HTML fresh
  })
);

/* 2) Assets (unchanged) */
registerRoute(
  ({ request, url }) =>
    url.origin === self.location.origin &&
    ["style", "script", "worker"].includes(request.destination),
  new StaleWhileRevalidate({ cacheName: ASSET_CACHE })
);

registerRoute(
  ({ request, url }) =>
    url.origin === self.location.origin && request.destination === "image",
  new CacheFirst({ cacheName: IMG_CACHE })
);

/* 3) APIs: NEVER cache or fallback — avoids “offline” being triggered by data routes */
registerRoute(
  ({ url }) =>
    url.origin === self.location.origin &&
    url.pathname.startsWith("/api/backend/"),
  new NetworkOnly()
);

registerRoute(
  ({ url }) =>
    url.origin === self.location.origin && url.pathname.startsWith("/api/"),
  new NetworkOnly()
);

/* 4) Single offline catch for failed navigations only */
setCatchHandler(async ({ request }) => {
  if (request.destination === "document") {
    const offline = await matchPrecache(OFFLINE_URL);
    if (offline) return offline;
    return new Response("<h1>Offline</h1>", {
      status: 503,
      headers: { "Content-Type": "text/html; charset=utf-8" },
    });
  }
  return Response.error();
});

self.addEventListener("message", (e: ExtendableMessageEvent) => {
  if ((e.data as any)?.type === "SKIP_WAITING") self.skipWaiting();
});
