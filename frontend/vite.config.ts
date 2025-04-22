import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig(({ command }) => ({
  /* ------------------------------------------------------------
   * Plugins
   * ------------------------------------------------------------ */
  plugins: [
    react(),
    command === "build" &&
      viteStaticCopy({
        targets: [
          {
            src: resolve(__dirname, "public/*"),
            dest: ".", // copies home.html, support.html, etc.
          },
        ],
        hook: "writeBundle",
      }),
  ].filter(Boolean),

  /* ------------------------------------------------------------
   * Path alias
   * ------------------------------------------------------------ */
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },

  /* ------------------------------------------------------------
   * Dev-server settings
   * ------------------------------------------------------------ */
  server: {
    open: "/app",      // I like the trailing slash here, too
    port: 5173,
    strictPort: true,
    proxy: { "/api/v1": "http://localhost:3001" },
    // catch ALL non‑asset GETs and return app.html
    historyApiFallback: {
      index: "/app.html",
    },
  },

  /* ------------------------------------------------------------
   * Build settings
   * ------------------------------------------------------------ */
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      /* Emit SPA + optional landing page.
         We do *not* include home.html here because the plugin
         will copy it from /public to the dist root. */
      input: {
        home: resolve(__dirname, "home.html"),
        app: resolve(__dirname, "app.html"), // SPA entry point
      },
      output: {
        entryFileNames: "assets/[name].[hash].js",
        chunkFileNames: "assets/[name].[hash].js",
        assetFileNames: "assets/[name].[hash].[ext]",
      },
    },
  },
}));
