// frontend/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import { resolve } from "path";

export default defineConfig({
  // Where your static assets that are not processed by Vite live
  publicDir: "public",

  // Vite plugins
  plugins: [
    react(),
    VitePWA({
      strategies: "injectManifest",
      srcDir: "src/app", // Directory where the service worker source is
      filename: "app-sw.ts", // The service worker source file
      injectRegister: false, // We register the service worker manually in app.ts

      // Static assets to be included in the service worker precache
      includeAssets: [
        "offline.html",
        "help.html", // Changed from support.html
        "privacy.html",
        "icons/*",
      ],

      // Web App Manifest configuration
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
  ],

  // Path aliases
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
      lodash: "lodash-es",
    },
  },

  // Development server configuration
  server: {
    open: "/app",
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

  // Build configuration
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
        manualChunks(id) {
          if (id.includes("node_modules")) {
            const pkg = id.split("node_modules/")[1].split("/")[0];
            return `vendor-${pkg.replace("@", "")}`;
          }
        },
      },
    },
  },

  // Production preview server configuration
  preview: {
    port: 3000,
  },
});
