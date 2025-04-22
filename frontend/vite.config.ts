// frontend/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig(({ command }) => ({
  // base: "/app/", // <-- REMOVE THIS LINE

  /* ------------------------------------------------------------ */
  /* Plugins                                                      */
  /* ------------------------------------------------------------ */
  plugins: [
    react(),
    command === "build" &&
      viteStaticCopy({
        targets: [{ src: resolve(__dirname, "public/*"), dest: "." }],
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
    // Let dev server open the root, React Router will handle routes
    open: true, // Changed from '/app' or '/app.html'
    port: 5173,
    strictPort: true,
    proxy: {
      "/api/v1": "http://localhost:3001",
    },
  },

  /* ------------------------------------------------------------ */
  /* Build settings                                               */
  /* ------------------------------------------------------------ */
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      // Keep both inputs as this was required for your Render setup
      input: {
        home: resolve(__dirname, "home.html"),
        app: resolve(__dirname, "app.html"), // Your SPA entry point
      },
      output: {
        entryFileNames: "assets/[name].[hash].js",
        chunkFileNames: "assets/[name].[hash].js",
        assetFileNames: "assets/[name].[hash].[ext]",
      },
    },
  },
}));
