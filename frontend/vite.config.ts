// frontend/vite.config.ts

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import { resolve } from "path";
import fs from "fs";

export default defineConfig({
  plugins: [
    react(),

    // ---------- PWA (scoped to /app) ----------
    VitePWA({
      devOptions: { enabled: true, type: "module" },
      strategies: "injectManifest",
      srcDir: "src",
      filename: "app-sw.ts",
      injectRegister: false,
      registerType: "autoUpdate",
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
      workbox: {
        navigateFallback: "/app/offline.html",
        navigateFallbackAllowlist: [/^\/app(?:\/.*)?$/],
        globPatterns: ["**/*.{js,css,html,ico,png,svg,json,woff2}"],
        cleanupOutdatedCaches: true,
      },
    }), // ← close your VitePWA call here

    // ★ copy app.html → index.html after build
    {
      name: "copy-app-to-index",
      apply: "build",
      writeBundle() {
        const src = resolve(__dirname, "dist/app.html");
        const dest = resolve(__dirname, "dist/index.html");
        fs.copyFileSync(src, dest);
      },
    },
  ],

  publicDir: "public",
  resolve: { alias: { "@": resolve(__dirname, "src") } },

  server: {
    open: "/app", // still auto‑opens the SPA
    proxy: {
      // existing API proxy
      "/api/v1": "http://localhost:3001",
      // new snapshots proxy
      "/snapshots": "http://localhost:3001",
    },
    strictPort: true,
    port: 5173,
  },

  build: {
    outDir: "dist",
    rollupOptions: {
      input: {
        app: resolve(__dirname, "app.html"),
      },
      output: {
        entryFileNames: "assets/[name].[hash].js",
        chunkFileNames: "assets/[name].[hash].js",
        assetFileNames: "assets/[name].[hash].[ext]",
      },
      plugins: [
        {
          name: "copy-app-html-to-index",
          writeBundle() {
            const src = resolve(__dirname, "dist/app.html");
            const dest = resolve(__dirname, "dist/index.html");
            fs.copyFileSync(src, dest);
          },
        },
      ],
    },
  },

  preview: {
    port: 3000,
    // (optional) if you still want the history fallback, add the middleware here
  },
});
