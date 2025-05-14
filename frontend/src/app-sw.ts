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
precacheAndRoute(self.__WB_MANIFEST);
cleanupOutdatedCaches();

/* ---------- HTML navigate ---------- */
registerRoute(
  ({ request, url }) =>
    request.mode === "navigate" && url.pathname.startsWith("/app"),
  new NetworkOnly({
    plugins: [
      {
        handlerDidError: async () => {
          // immediately reload all client windows into offline.html
          const clientsList = await self.clients.matchAll({ type: "window" });
          clientsList.forEach((win) => win.navigate("/app/offline.html"));
          // then return the offline page from cache
          return matchPrecache("/app/offline.html");
        },
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
