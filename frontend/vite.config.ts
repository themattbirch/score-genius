// frontend/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import { resolve } from "path";

export default defineConfig(({ mode }) => {
  // In dev, use your env var (or fallback localhost).
  // In production, use relative URLs so fetch('/api/...') hits the same origin.
  const API_BASE_URL =
    mode === "development"
      ? process.env.VITE_API_BASE_URL || "http://localhost:10000"
      : "";
  return {
    //
    plugins: [
      react(),

      // ---------- PWA (scoped to /app) ----------
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
        // Control what Workbox scans & injects when using `injectManifest`
        injectManifest: {
          // drop the `.ico` extension so it's never injected
          globPatterns: ["**/*.{js,css,html,svg,json,woff2}"],
          globIgnores: [
            "**/favicon.ico",
            "**/basketball_header_logo.png",
            "**/orange_football_header_logo.png",
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

    publicDir: "public",
    resolve: { alias: { "@": resolve(__dirname, "src") } },

    server: {
      open: "/app", // still auto-opens the SPA
      // Use the API_BASE_URL defined above for Vite's local dev proxy
      proxy: {
        "/api/v1": API_BASE_URL, // <--- Use API_BASE_URL here for local proxy
      },
      strictPort: true,
      port: 5173,
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
          chunkFileNames: "assets/[name].[name].[hash].js", // Often useful to include [name] for better chunk naming
          assetFileNames: "assets/[name].[hash].[ext]",
          // ← manualChunks splits each npm package into its own chunk
          manualChunks(id: string) {
            if (id.includes("node_modules")) {
              // Ensure consistent chunk names, e.g., 'vendor-react', 'vendor-recharts'
              const parts = id.split("node_modules/")[1].split("/");
              return `vendor-${parts[0].replace("@", "")}`; // Handles scoped packages like @tanstack
            }
          },
        },
      },
    },

    // Move the 'define' block INSIDE the main configuration object,
    // before the 'preview' property. This is where it belongs.
    define: {
      // This is the crucial part: it replaces `import.meta.env.VITE_API_BASE_URL`
      // with the actual string value of `API_BASE_URL` at build time.
      // `JSON.stringify` ensures it's injected as a string literal.
      "import.meta.env.VITE_API_BASE_URL": JSON.stringify(API_BASE_URL),
    },

    preview: {
      port: 3000,
      // (optional) if you still want the history fallback, add the middleware here
    },
  };
});
