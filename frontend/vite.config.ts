import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';
//import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
  /** ----------------------------------------------------------------
   *  Plugins
   *  ---------------------------------------------------------------- */
  plugins: [
    react(),

    /* Copy anything in /public → /dist at build time */
    //viteStaticCopy({
      //targets: [
       // {
         // src: resolve(__dirname, 'public/*'),
          //dest: '.', // dist/<file>
        //},
      //],
      //hook: 'writeBundle',
   // }),
  ],

  /** ----------------------------------------------------------------
   *  Aliases (optional but handy)
   *  ---------------------------------------------------------------- */
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },

  /** ----------------------------------------------------------------
   *  Dev‑server tweaks
   *  ---------------------------------------------------------------- */
  server: {
    open: '/app.html',          // 🚀 open the SPA template, not index.html
    port: 5173,                 // change if 5173 busy
    strictPort: true,
  },

  /** ----------------------------------------------------------------
   *  Build settings
   *  ---------------------------------------------------------------- */
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      /* Multi‑page build: app.html (SPA) + index.html (landing) */
      input: {
        index: resolve(__dirname, 'index.html'), // static homepage
        home:  resolve(__dirname, 'public/home.html'),
        app: resolve(__dirname, 'app.html'),     // React SPA
      },
      output: {
        entryFileNames: 'assets/[name].[hash].js',
        chunkFileNames: 'assets/[name].[hash].js',
        assetFileNames: 'assets/[name].[hash].[ext]',
      },
    },
  },
});
