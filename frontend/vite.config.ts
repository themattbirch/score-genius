import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import vitePluginImp from "vite-plugin-imp";
import { resolve } from "path";

export default defineConfig({
  plugins: [
    react(),

    VitePWA({
      strategies: "injectManifest",
      srcDir: "src/app", // where app-sw.ts lives
      filename: "app-sw.ts", // input TS file -> outputs app-sw.js
      injectRegister: false, // you register manually
      registerType: "autoUpdate",

      // Workbox options for runtime caching
      workbox: {
        cleanupOutdatedCaches: true,
        clientsClaim: true,
        skipWaiting: true,

        runtimeCaching: [
          {
            // network‑first for support page
            urlPattern: /^\/support(?:\?.*)?$/,
            handler: "NetworkFirst",
            options: {
              cacheName: "support-page-cache",
              networkTimeoutSeconds: 5,
              expiration: { maxEntries: 1, maxAgeSeconds: 24 * 3600 },
            },
          },
        ],

        // only fallback under /app/*
        navigateFallback: "/app/app.html",
        navigateFallbackAllowlist: [/^\/app\//],
      },

      // include these static files in your precache manifest
      includeAssets: [
        "offline.html",
        "privacy.html",
        "support.html",
        "app.html",
        "icons/*",
      ],

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
      },
    }),

    vitePluginImp({
      libList: [
        { libName: "lodash", libDirectory: "", camel2DashComponentName: false },
      ],
    }),
  ],

  publicDir: "public",

  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
      lodash: "lodash-es",
    },
  },

  server: {
    open: "/app",
    port: 5173,
    strictPort: true,
    proxy: { "/api": { target: "http://localhost:10000", changeOrigin: true } },
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
        chunkFileNames: "assets/[name].[name].[hash].js",
        assetFileNames: "assets/[name].[hash].[ext]",
      },
    },
  },

  preview: { port: 3000 },
});
