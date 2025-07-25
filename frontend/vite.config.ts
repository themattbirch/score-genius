// frontend/vite.config.ts
console.log("🐞 vite.config.ts loaded");
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import { resolve } from "path";
import vitePluginImp from "vite-plugin-imp";
console.log("🧪 Using vite.config.ts at", __filename);
const swSrc = resolve(__dirname, "src/app/app-sw.ts");

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      strategies: "injectManifest",
      srcDir: "src/app", // where app-sw.ts lives
      filename: "app-sw.ts", // input TS file -> outputs app-sw.js
      injectRegister: false, // you register manually
      includeAssets: ["offline.html", "icons/*"], // just static stuff to copy
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
        //splash_pages: ["splash_screen.html"],
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
        manualChunks(id) {
          if (id.includes("node_modules")) {
            const pkg = id.split("node_modules/")[1].split("/")[0];
            return `vendor-${pkg.replace("@", "")}`;
          }
        },
      },
    },
  },

  preview: {
    port: 3000,
  },
});
