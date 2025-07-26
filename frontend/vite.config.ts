// frontend/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import { resolve } from "path";
import vitePluginImp from "vite-plugin-imp";

export default defineConfig(({ mode }) => ({
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

    // ─── PWA (scoped to /app) ─────────────────────────────────
    VitePWA({
      strategies: "injectManifest",
      injectManifest: {
        // point at your custom TS service‑worker
        swSrc: resolve(__dirname, "src/app-sw.ts"),
        // where you want it in the build output:
        swDest: "app/app-sw.js",
      },
      workbox: {
        skipWaiting: true,
        clientsClaim: true,
      },
      // automatically update when a new SW is available
      registerType: "autoUpdate", // :contentReference[oaicite:0]{index=0}
      // control how the registration script is injected:
      // 'inline', 'script', 'script-defer', or null for manual
      injectRegister: false,

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
    rollupOptions: {
      input: {
        index: resolve(__dirname, "public/index.html"),
        app: resolve(__dirname, "app.html"),
      },
      output: {
        entryFileNames: "assets/[name].[hash].js",
        chunkFileNames: "assets/[name].[name].[hash].js",
        assetFileNames: "assets/[name].[hash].[ext]",
        manualChunks(id: string) {
          if (id.includes("node_modules")) {
            const parts = id.split("node_modules/")[1].split("/");
            return `vendor-${parts[0].replace("@", "")}`;
          }
        },
      },
    },
  },

  preview: {
    port: 3000,
  },
}));
