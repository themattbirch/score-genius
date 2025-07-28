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
const ASSET_CACHE = "assets-cache-v1";
const IMG_CACHE = "img-cache-v1";

type PrecacheEntry = {
  url: string;
  revision?: string | null;
};

function isPrecacheEntry(
  entry: string | PrecacheEntry
): entry is PrecacheEntry {
  return typeof entry !== "string";
}

precacheAndRoute(self.__WB_MANIFEST);
cleanupOutdatedCaches();

clientsClaim();
self.skipWaiting();

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", () => self.clients.claim());

// --- Navigation: NetworkOnly + offline fallback ------------------------------
const offlineFallbackPlugin = {
  handlerDidError: async () => (await matchPrecache(OFFLINE_URL)) as Response,
};

registerRoute(
  ({ request }) => request.mode === "navigate",
  new NetworkOnly({ plugins: [offlineFallbackPlugin] })
);

// --- Static assets: SWR ------------------------------------------------------
registerRoute(
  ({ request }) => ["style", "script", "worker"].includes(request.destination),
  new StaleWhileRevalidate({
    cacheName: ASSET_CACHE,
    plugins: [new ExpirationPlugin({ maxEntries: 50, maxAgeSeconds: 86400 })],
  })
);

// --- Images: CacheFirst ------------------------------------------------------
registerRoute(
  ({ request }) => request.destination === "image",
  new CacheFirst({
    cacheName: IMG_CACHE,
    plugins: [
      new ExpirationPlugin({ maxEntries: 60, maxAgeSeconds: 30 * 86400 }),
    ],
  })
);

// --- Skip waiting on demand --------------------------------------------------
self.addEventListener("message", (event) => {
  if (event.data?.type === "SKIP_WAITING") self.skipWaiting();
});

// --- Global catch (extra safety) --------------------------------------------
setCatchHandler(async ({ request }) => {
  if (request.mode === "navigate") {
    return (await matchPrecache(OFFLINE_URL))!;
  }
  return Response.error();
});
