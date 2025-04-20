import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig(({ command }) => ({
  plugins: [
    react(),

    // 👉 copy files from /public to dist root *only during build*
    command === 'build' &&
      viteStaticCopy({
        targets: [
          { src: resolve(__dirname, 'public/home.html'), dest: '.' },
          // copy other marketing pages if you ever add them
        ],
        hook: 'writeBundle',
      }),
  ].filter(Boolean),

  build: {
    outDir: 'dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        index: resolve(__dirname, 'index.html'), // keeps /index.html
        app:   resolve(__dirname, 'app.html'),   // SPA
        // ❌ remove the previous "home:" entry here
      },
    },
  },
}));
