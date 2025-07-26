import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      strategies: "injectManifest",
      filename: "app-sw.js", // final output name
      injectManifest: {
        swSrc: "src/app-sw.js", // <-- MUST match the JS file you just created
        swDest: "app-sw.js",
      },
      injectRegister: false,
      workbox: {
        skipWaiting: true,
        clientsClaim: true,
        navigateFallback: "/app/offline.html",
        navigateFallbackDenylist: [
          /^\/app\/app-sw\.js$/,
          /^\/app\/workbox-.*\.js$/,
        ],
      },
    }),
  ],
  build: {
    outDir: "dist",
  },
});
