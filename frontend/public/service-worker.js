const CACHE_NAME = 'score-genius';
const PRECACHE_URLS = [
  '/app',
  '/',
  '/public/404.html',
  '/public/offline.html',
  '/public/privacy.html',
  '/public/terms.html',
  '/public/support.html',
  '/public/manifest.webmanifest',
  '/icons/football-icon-48.png',
];

const OFFLINE_FALLBACK_URL = '/offline.html';

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(PRECACHE_URLS).catch((err) => {
        console.error('Precache error:', err);
        console.error('Likely a missing resource in PRECACHE_URLS.');
        throw err;
      });
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      for (const key of keys) {
        if (key !== CACHE_NAME) {
          await caches.delete(key);
        }
      }
    })()
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  // If this is an HTML navigation request:
  if (event.request.mode === 'navigate') {
    event.respondWith(
      (async () => {
        try {
          // Try network first
          const networkResponse = await fetch(event.request);

          // Cache the fresh copy if successful
          const cache = await caches.open(CACHE_NAME);
          cache.put(event.request, networkResponse.clone());

          return networkResponse;
        } catch (err) {
          // If network fails, serve offline.html
          const fallback = await caches.match(OFFLINE_FALLBACK_URL);
          return fallback || new Response('Offline', { status: 503 });
        }
      })()
    );
  } else {
    // Everything else: CSS, JS, images, JSON, etc.
    event.respondWith(
      (async () => {
        // 1) Check the cache
        const cached = await caches.match(event.request);
        if (cached) {
          return cached; // quick return from cache
        }

        // 2) No cache hit, try network
        try {
          const networkResponse = await fetch(event.request);
          // If OK, cache it for future
          if (networkResponse.ok) {
            const cache = await caches.open(CACHE_NAME);
            cache.put(event.request, networkResponse.clone());
          }
          return networkResponse;
        } catch (err) {
          // 3) If network fails, return a fallback or minimal error response
          // If you want *every* resource to show offline.html, you *could* do:
          //
          // return caches.match(OFFLINE_FALLBACK_URL);
          //
          // But more common is to just return a small error or placeholder:
          if (event.request.destination === 'image') {
            // Optionally return a cached placeholder image
            // return caches.match("/icons/placeholder.png");
          }
          // Or a short message or 503
          return new Response('Offline resource not cached.', { status: 503 });
        }
      })()
    );
  }
});



