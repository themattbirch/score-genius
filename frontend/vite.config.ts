// frontend/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import { resolve } from "path";

export default defineConfig(({ command }) => ({
  /** -------------------------------------------------
   *  Base URL
   *  ------------------------------------------------*/
  // command === 'serve'  -> vite dev     ->  /
  // command === 'build'  -> vite build   ->  /app/
  base: "/app/",

  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      injectRegister: "auto",
      devOptions: { enabled: true },

      /**  PWA manifest  */
      manifest: {
        name: "Score Genius",
        short_name: "ScoreGenius",
        description:
          "Score Genius: Powerful predictive stats for passionate fans",
        theme_color: "#1F2937",
        background_color: "#ffffff",
        display: "standalone",
        orientation: "portrait",
        scope: "/app", // ← trailing slash matters
        start_url: "/app", // ← trailing slash matters
        icons: [
          {
            src: "/app/icons/football-icon-192.png",
            sizes: "192x192",
            type: "image/png",
          },
          {
            src: "/app/icons/football-icon-512.png",
            sizes: "512x512",
            type: "image/png",
          },
          {
            src: "/app/icons/football-icon-maskable-512.png",
            sizes: "512x512",
            type: "image/png",
            purpose: "maskable",
          },
        ],
      },

      /**  Workbox  */
      workbox: {
        // use the relative path *inside the scope* (no leading slash)
        navigateFallback: "app.html",

        // keep the deny-list exactly as before
        navigateFallbackDenylist: [
          /\/[^/?]+\.[^/]{2,}$/, // direct asset hits
        ],

        globPatterns: ["**/*.{js,css,html,ico,png,svg,json,woff2}"],
      },
    }),
  ],

  publicDir: "public",

  resolve: {
    alias: { "@": resolve(__dirname, "src") },
  },

  /** -------------------------------------------------
   *  Dev-server
   *  ------------------------------------------------*/
  server: {
    open: "/app/", // auto-open the SPA page
    port: 5173,
    strictPort: true,
    proxy: { "/api/v1": "http://localhost:3001" },
  },

  /** -------------------------------------------------
   *  Build / preview
   *  ------------------------------------------------*/
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        home: resolve(__dirname, "home.html"),
        app: resolve(__dirname, "app.html"),
      },
      output: {
        entryFileNames: "assets/[name].[hash].js",
        chunkFileNames: "assets/[name].[hash].js",
        assetFileNames: "assets/[name].[hash].[ext]",
      },
    },
  },
}));
