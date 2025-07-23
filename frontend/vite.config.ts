// frontend/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import { nodePolyfills } from "vite-plugin-node-polyfills";
import { resolve } from "path";

export default defineConfig(({ mode }) => {
  console.log("VITE MODE:", mode, "→ proxy /api to http://localhost:10000");

  return {
    plugins: [
      // 1) Polyfill Node core modules in the browser
      nodePolyfills(),

      // 2) React support
      react(),

      // 3) PWA support
      VitePWA({
        devOptions: { enabled: true, type: "module" },
        strategies: "injectManifest",
        srcDir: "src",
        filename: "app-sw.ts",
        injectRegister: false,
        includeAssets: [
          "splash_screen.html",
          "manifest.webmanifest",
          "images/basketball.svg",
          "icons/*",
        ],
        injectManifest: {
          globPatterns: ["**/*.{js,css,html,svg,json,woff2}"],
          globIgnores: [
            "**/favicon.ico",
            "**/basketball_header_logo.png",
            "**/orange_football_header_logo.png",
            "**/data/**",
          ],
        },
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
        workbox: {
          navigateFallback: "/app/offline.html",
          navigateFallbackDenylist: [/^\/api\//],
          navigateFallbackAllowlist: [/^\/app(?:\/|$)/],
          globPatterns: ["**/*.{js,css,html,png,svg,json,woff2}"],
          globIgnores: ["**/favicon.ico"],
          cleanupOutdatedCaches: true,
        },
      }),
    ],

    // static assets
    publicDir: "public",

    resolve: {
      alias: {
        "@": resolve(__dirname, "src"),
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
  };
});
