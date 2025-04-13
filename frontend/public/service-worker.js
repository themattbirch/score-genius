// service-worker.js

const CACHE_NAME = 'score-genius'; // Consider versioning this, e.g., 'score-genius-v1.1'

// IMPORTANT: Verify asset paths (manifest, icons, main JS/CSS) after deployment!
// Vite might hash them and put them in /assets/. Update paths below if needed.
const PRECACHE_URLS = [
  // Routes (will cache the index.html served for these)
  '/',
  '/app',
  '/help', // Changed from /public/support.html - caches the result of the /help route
  '/privacy', // Changed from /public/privacy.html - caches the result of the /privacy route
  '/terms',   // Changed from /public/terms.html - caches the result of the /terms route

  // Specific Files (relative to the 'frontend/dist' root)
  '/offline.html', // Changed from /public/offline.html - MUST match OFFLINE_FALLBACK_URL
  '/404.html',     // Changed from /public/404.html - caches the actual 404 file
  '/manifest.webmanifest', // Changed from /public/manifest.webmanifest - VERIFY if hashed to /assets/
  '/icons/football-icon-48.png', // Kept as is - VERIFY if hashed to /assets/

  // ===> RECOMMENDED ADDITIONS (find exact paths from Network tab after successful deploy) <===
  // '/assets/index.xxxxx.js',  // Replace with your actual main JS bundle path
  // '/assets/index.xxxxx.css', // Replace with your actual main CSS bundle path
  // '/icons/favicon-32x32.png', // Or hashed path from /assets/ if applicable
  // '/icons/favicon-192x192.png', // Or hashed path from /assets/ if applicable
  // etc. for other critical icons/assets
];

const OFFLINE_FALLBACK_URL = '/offline.html'; // This MUST exactly match an entry in PRECACHE_URLS

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[Service Worker] Precaching URLs:', PRECACHE_URLS);
      return cache.addAll(PRECACHE_URLS).catch((err) => {
        console.error('[Service Worker] Precache failed:', err);
        // Log which URL failed if possible (browser dependent)
        if (err instanceof Response && !err.ok) {
          console.error(`Failed to fetch: ${err.url} Status: ${err.status}`);
        } else {
          // Try fetching individually to identify the culprit
          PRECACHE_URLS.forEach(url => {
            fetch(url).catch(fetchErr => console.error(`Individual fetch failed for ${url}:`, fetchErr));
          });
        }
        console.error('Likely a missing resource or incorrect path in PRECACHE_URLS.');
        // Optional: Don't throw error to allow SW to install partially? Risky.
        // throw err; 
      });
    }).then(() => {
      console.log('[Service Worker] Precache completed.');
      // Force the waiting service worker to become the active service worker.
      return self.skipWaiting(); 
    })
  );
});

self.addEventListener('activate', (event) => {
  console.log('[Service Worker] Activating...');
  event.waitUntil(
    (async () => {
      // Delete old caches
      const keys = await caches.keys();
      for (const key of keys) {
        if (key !== CACHE_NAME) {
          console.log(`[Service Worker] Deleting old cache: ${key}`);
          await caches.delete(key);
        }
      }
      // Enable navigation preload if supported (helps with network-first)
      if ('navigationPreload' in self.registration) {
        await self.registration.navigationPreload.enable();
      }
      // Tell the active service worker to take control of the page immediately.
      await self.clients.claim(); 
      console.log('[Service Worker] Activated and claimed clients.');
    })()
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Ignore non-GET requests and requests to external domains for caching strategy
  if (event.request.method !== 'GET' || url.origin !== self.location.origin) {
    return;
  }

  // Strategy: Network first for navigation, Cache first (falling back to network) for assets
  if (event.request.mode === 'navigate') {
    event.respondWith(
      (async () => {
        // Try network first, potentially using navigation preload
        try {
          const preloadResponse = await event.preloadResponse;
          if (preloadResponse) {
            console.log('[Service Worker] Using preload response for navigation:', event.request.url);
            return preloadResponse;
          }

          const networkResponse = await fetch(event.request);
          // Cache the successful response
          const cache = await caches.open(CACHE_NAME);
          cache.put(event.request, networkResponse.clone());
          return networkResponse;
        } catch (error) {
          // Network failed, try cache
          console.log('[Service Worker] Network fetch failed for navigation, trying cache:', event.request.url, error);
          const cachedResponse = await caches.match(event.request);
          if (cachedResponse) {
            return cachedResponse;
          }
          // If navigation request is not cached, serve the offline fallback page
          console.log('[Service Worker] Serving offline fallback page.');
          const fallbackResponse = await caches.match(OFFLINE_FALLBACK_URL);
          // Return offline page or a generic error if offline page isn't cached
          return fallbackResponse || new Response("Network error occurred and offline page is not available.", {
            status: 404,
            headers: { 'Content-Type': 'text/plain' }
          });
        }
      })()
    );
  } else {
    // Cache-first strategy for assets
    event.respondWith(
      caches.match(event.request).then((cachedResponse) => {
        // Return cached response if found
        if (cachedResponse) {
          return cachedResponse;
        }

        // If not in cache, fetch from network
        return fetch(event.request).then((networkResponse) => {
          return (async () => {
            // Cache the new response if successful
            if (networkResponse.ok) {
              const cache = await caches.open(CACHE_NAME);
              event.waitUntil(cache.put(event.request, networkResponse.clone()));
            }
            return networkResponse;
          })();
        }).catch(error => {
          console.log('[Service Worker] Network fetch failed for asset:', event.request.url, error);
          return new Response("Network error trying to fetch resource.", {
            status: 503,
            headers: { 'Content-Type': 'text/plain' }
          });
        });
      })
    );
  }
});
