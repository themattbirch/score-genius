// frontend/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import { resolve } from "path";
import vitePluginImp from "vite-plugin-imp";

export default defineConfig(({ mode }) => {
  // In dev, use your env var (or fallback localhost).
  // In production, use relative URLs so fetch('/api/...') hits the same origin.
  console.log("VITE MODE:", mode, "→ proxy /api to http://localhost:10000");
  return {
    //
    plugins: [
      react(),
      vitePluginImp({
        libList: [
          {
            libName: "lodash",
            libDirectory: "",
            camel2DashComponentName: false,
          },
        ],
      }),

      // ---------- PWA (scoped to /app) ----------
      VitePWA({
        // Use the stable 'generateSW' strategy
        strategies: "generateSW",
        // registerType: 'autoUpdate' is fine
        registerType: "autoUpdate",
        workbox: {
          swDest: "dist/app/app-sw.js",
          globPatterns: ["**/*.{js,css,html,svg,json,woff2}"],

          // This remains our ultimate safety net for uncached pages.
          navigateFallback: "/app/offline.html",

          // This correctly ensures the offline page is available.
          additionalManifestEntries: [
            { url: "/app/offline.html", revision: null },
          ],

          runtimeCaching: [
            {
              urlPattern: ({ request }) =>
                ["style", "script", "worker"].includes(request.destination),
              handler: "StaleWhileRevalidate", // This is fine for assets
              options: {
                cacheName: "assets-cache",
                expiration: { maxEntries: 50, maxAgeSeconds: 86400 },
              },
            },
            {
              urlPattern: ({ request }) =>
                request.mode === "navigate" && request.url.includes("/app/"),
              handler: "NetworkFirst",
              options: {
                cacheName: "app-pages",
                networkTimeoutSeconds: 5,
                // ✅ ADDED: This plugin ensures only successful responses are cached.
                plugins: [
                  {
                    cacheWillUpdate: async ({ response }) => {
                      // If the response is valid (status 200), cache it.
                      if (response && response.status === 200) {
                        return response;
                      }
                      // Otherwise, return null to prevent it from being cached.
                      return null;
                    },
                  },
                ],
              },
            },
          ],
        },
        manifest: {
          name: "ScoreGenius",
          short_name: "ScoreGenius",
          description:
            "ScoreGenius: Powerful predictive stats for passionate fans",
          scope: "/app",
          start_url: "/app",
          theme_color: "#1F2937",
          background_color: "#ffffff",
          display: "standalone",
          display_override: ["fullscreen", "standalone", "minimal-ui"],
          orientation: "portrait",
          splash_pages: ["splash_screen.html"],
          icons: [
            {
              src: "/icons/football-icon-192.png",
              sizes: "192x192",
              type: "image/png",
            },
            {
              src: "/icons/football-icon-512.png",
              sizes: "512x512",
              type: "image/png",
            },
            {
              src: "/icons/football-icon-maskable-512.png",
              sizes: "512x512",
              type: "image/png",
              purpose: "maskable",
            },
          ],
        } as any,
      }),
    ],

    publicDir: "public",
    resolve: { alias: { "@": resolve(__dirname, "src"), lodash: "lodash-es" } },

    server: {
      open: "/app",
      port: 5173,
      strictPort: true,

      proxy: {
        // send every /api request in dev to localhost:10000
        "/api": {
          target: "http://localhost:10000",
          changeOrigin: true,
          secure: false,
        },
      },
    },

    build: {
      outDir: "dist",
      target: "es2022",
      rollupOptions: {
        input: {
          index: resolve(__dirname, "public/index.html"),
          app: resolve(__dirname, "app.html"),
        },

        output: {
          entryFileNames: "assets/[name].[hash].js",
          chunkFileNames: "assets/[name].[name].[hash].js", // Often useful to include [name] for better chunk naming
          assetFileNames: "assets/[name].[hash].[ext]",
          // ← manualChunks splits each npm package into its own chunk
          manualChunks(id: string) {
            if (id.includes("node_modules")) {
              // Ensure consistent chunk names, e.g., 'vendor-react', 'vendor-recharts'
              const parts = id.split("node_modules/")[1].split("/");
              return `vendor-${parts[0].replace("@", "")}`; // Handles scoped packages like @tanstack
            }
          },
        },
      },
    },

    preview: {
      port: 3000,
      // (optional) if you still want the history fallback, add the middleware here
    },
  };
});
