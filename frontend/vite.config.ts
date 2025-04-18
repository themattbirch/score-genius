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
          input: {
           // Assuming 'home.html' is now handled by being in public/
           // or needs its own entry if complex
            app: resolve(__dirname, "app.html") // Point to app.html
        },
      output: {
        entryFileNames: "assets/[name].[hash].js",
        chunkFileNames: "assets/[name].[hash].js",
        assetFileNames: "assets/[name].[hash].[ext]",
      },
    },
  },
});
