/// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig({
  plugins: [
    react(),
    viteStaticCopy({
      // <-- Add the plugin configuration
      targets: [
        {
          // Copy everything from frontend/public/*
          src: resolve(__dirname, "public/*"),
          // To the root of frontend/dist/
          dest: ".",
        },
        // Note: Depending on how the glob (*) works, you might need separate entries
        // if it doesn't copy directories correctly, e.g.:
        // { src: resolve(__dirname, 'public/icons'), dest: '.' },
        // { src: resolve(__dirname, 'public/help.html'), dest: '.' }, etc.
        // But try the simple 'public/*' first.
      ],
    }),
  ],
  server: {
    open: "/app/",
    proxy: {
      "/app": {
        target: "http://localhost:5173/public",
        rewrite: (path) => path.replace(/^\/app/, ""),
        secure: false,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
      },
      output: {
        entryFileNames: "assets/[name].[hash].js",
        chunkFileNames: "assets/[name].[hash].js",
        assetFileNames: "assets/[name].[hash].[ext]",
      },
    },
  },
  publicDir: "public",
});
