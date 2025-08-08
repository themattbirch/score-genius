// frontend/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import { resolve } from "path";

export default defineConfig({
  base: "/app/",

  plugins: [
    react(),
    VitePWA({
      strategies: "injectManifest",
      srcDir: "src/app",
      filename: "app-sw.ts",
      injectRegister: false,
      includeAssets: ["offline.html", "icons/*"],
      injectManifest: {
        globIgnores: [
          "**/*.appxbundle",
          "**/*.msixbundle",
          "**/*.appinstaller",
        ],
      },
      workbox: {
        navigateFallbackDenylist: [
          /\/app\/.*\.(appxbundle|msixbundle|appinstaller)$/i,
        ],
      },
      manifest: {
        name: "ScoreGenius",
        short_name: "ScoreGenius",
        description:
          "ScoreGenius: Powerful predictive stats for passionate fans",
        id: "/app",
        scope: "/app/",
        start_url: "/app/",
        theme_color: "#1F2937",
        background_color: "#ffffff",
        display: "standalone",
        orientation: "any",
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
            purpose: "any",
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
  ],

  publicDir: "public",

  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
      lodash: "lodash-es",
    },
  },

  server: {
    open: "/app/",
    port: 5173,
    strictPort: true,
    proxy: {
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
    assetsDir: "assets",
    // ❌ NO rollupOptions.input here — let Vite use root index.html
  },

  preview: { port: 3000 },
});
