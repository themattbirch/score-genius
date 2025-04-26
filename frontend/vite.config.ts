// frontend/vite.config.ts

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import { resolve } from "path";

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      injectRegister: "auto",
      includeAssets: ["favicon.ico"],

      /* ---------- manifest ---------- */
      manifest: {
        name: "Score Genius",
        short_name: "ScoreGenius",
        description:
          "Score Genius: Powerful predictive stats for passionate fans",
        theme_color: "#1F2937",
        background_color: "#ffffff",
        display: "standalone",
        orientation: "portrait",
        scope: "/app",
        start_url: "/app",

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
        navigateFallback: "/app.html",
        navigateFallbackDenylist: [/\/api\/v1\/.*$/, /\/[^/?]+\.[^/]{2,}$/],
        globPatterns: ["**/*.{js,css,html,ico,png,svg,json,woff2}"],
      },
    }),
  ],

  publicDir: "public",

  resolve: { alias: { "@": resolve(__dirname, "src") } },

  server: {
    open: "/app", // auto-open SPA
    proxy: { "/api/v1": "http://localhost:3001" },
    strictPort: true,
    port: 5173,
  },

  build: {
    outDir: "dist",
    rollupOptions: {
      input: {
        index: resolve(__dirname, "home.html"),
        app: resolve(__dirname, "app.html"),
      },
      output: {
        entryFileNames: "assets/[name].[hash].js",
        chunkFileNames: "assets/[name].[hash].js",
        assetFileNames: "assets/[name].[hash].[ext]",
      },
    },
  },
});
