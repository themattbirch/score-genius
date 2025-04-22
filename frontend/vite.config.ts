// /frontend/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig(({ command }) => ({
  plugins: [react()],

  publicDir: "public",   // static assets only

  resolve: {
    alias: { "@": resolve(__dirname, "src") },
  },

  server: {
    open: "/app",
    port: 5173,
    strictPort: true,
    proxy: { "/api/v1": "http://localhost:3001" },
    // dev uses index.html by default—no extra fallback needed
  },

  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      // THESE MUST POINT AT ROOT‐LEVEL HTML FILES:
      input: {
        home: resolve(__dirname, "home.html"),
        app:  resolve(__dirname, "app.html"),
      },
      output: {
        entryFileNames:   "assets/[name].[hash].js",
        chunkFileNames:   "assets/[name].[hash].js",
        assetFileNames:   "assets/[name].[hash].[ext]",
      },
    },
  },
}));
