// /frontend/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig(({ command }) => ({
  plugins: [
    react(), // First plugin in the array
    VitePWA({
      // Second plugin in the array, separated by a comma
      registerType: "autoUpdate",
      injectRegister: "auto",
      devOptions: {
        enabled: true,
      },
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
        globPatterns: ["**/*.{js,css,html,ico,png,svg,json,woff2}"],
        // runtimeCaching: [ ... ] // Keep commented out for now unless needed
      },
    }), // End of VitePWA configuration object
  ], // End of the plugins array

  publicDir: "public", // static assets only

  resolve: {
    alias: { "@": resolve(__dirname, "src") },
  },

  server: {
    open: "/app",
    port: 5173,
    strictPort: true,
    proxy: { "/api/v1": "http://localhost:3001" },
    // dev uses index.html by default—no extra fallback needed
  },

  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      // MPA Setup: Building both home.html and app.html
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
