import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig({
  plugins: [
    react(),
    viteStaticCopy({
      targets: [
        {
          // Assuming you want to copy everything from public/* to dist/*
          src: resolve(__dirname, "public/*"),
          dest: ".", // Copies to the root of the outDir (dist)
        },
      ],
      // Added hook to ensure copy happens after build cleans directory
      hook: "writeBundle",
    }),
  ],
  build: {
    outDir: "dist",
    emptyOutDir: true, // Good practice to keep this true
    rollupOptions: {
      // --- INPUT SECTION REMOVED ---
      // Vite will now default to using 'index.html' at the project root ('frontend/')
      // as the main entry point for the SPA build.

      // Keep the output options if you want specific asset naming
      output: {
        entryFileNames: "assets/[name].[hash].js",
        chunkFileNames: "assets/[name].[hash].js",
        assetFileNames: "assets/[name].[hash].[ext]",
      },
    },
  },
});
