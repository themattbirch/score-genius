// frontend/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig(({ command }) => ({
  /* ------------------------------------------------------------ */
  /* Plugins                                                      */
  /* ------------------------------------------------------------ */
  plugins: [
    react(),

    // ⬇️ Copy /public/* to dist/* — but *only* when we run `vite build`
    command === "build" &&
      viteStaticCopy({
        targets: [
          { src: resolve(__dirname, "public/*"), dest: "." }, // dist/home.html, dist/support.html, …
        ],
        hook: "writeBundle",
      }),
  ].filter(Boolean),

  /* ------------------------------------------------------------ */
  /* Path alias (nice‑to‑have)                                    */
  /* ------------------------------------------------------------ */
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },

  /* ------------------------------------------------------------ */
  /* Dev‑server settings                                          */
  /* ------------------------------------------------------------ */
  server: {
    open: "/app.html", // opens the SPA for local dev
    port: 5173,
    strictPort: true,
      proxy: {
    '/api/v1': 'http://localhost:3001', // forwards to Express
  },
  },

  /* ------------------------------------------------------------ */
  /* Build settings                                               */
  /* ------------------------------------------------------------ */
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      /** Emit SPA + optional landing page.
       *  We do *not* include home.html here because the plugin will
       *  copy it from /public to the dist root. */
      input: {
        // marketing landing
        home: resolve(__dirname, "home.html"),
        // optional index if you want one
        // SPA
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
