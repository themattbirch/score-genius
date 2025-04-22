// /frontend/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig(({ command }) => ({
  plugins: [
    react(),
    // only needed if you build from non‐public folder; 
    // since we’ll keep home.html & app.html in /public, you can drop this.
    // command === "build" &&
    //   viteStaticCopy({ ... })
  ].filter(Boolean),

  // alias so imports like "@/..." still work
  resolve: {
    alias: { "@": resolve(__dirname, "src") },
  },

  // serve everything in /public at /
  publicDir: "public",

  server: {
    // open http://localhost:5173/app
    open: "/app",
    port: 5173,
    strictPort: true,
    proxy: { "/api/v1": "http://localhost:3001" },
    // ← no need for historyApiFallback or connect‐history plugin any more
  },

  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      // these two HTML shells (in /public) become your MPA entry points
      input: {
        home: resolve(__dirname, "public/home.html"),
        app:  resolve(__dirname, "public/app.html"),
      },
      output: {
        entryFileNames:   "assets/[name].[hash].js",
        chunkFileNames:   "assets/[name].[hash].js",
        assetFileNames:   "assets/[name].[hash].[ext]",
      },
    },
  },
}));
