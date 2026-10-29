import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  // 5180 is reserved for this app; 3000/5173/5174/8091 belong to other projects.
  server: {
    port: 5180,
    strictPort: true,
    host: '127.0.0.1',
    // Same-origin API in dev: the SPA calls /api/v1/... and Vite forwards to FastAPI, so CORS never applies
    // and the refresh cookie (Path=/api/v1/auth) is first-party. ws:true covers the training event stream.
    proxy: { '/api': { target: 'http://127.0.0.1:8010', changeOrigin: false, ws: true } },
  },
  preview: { port: 5180, strictPort: true },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
  },
});
