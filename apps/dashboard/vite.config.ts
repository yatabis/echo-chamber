import { resolve } from 'path';

import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  base: '/dashboard/',
  plugins: [react()],
  build: {
    outDir: resolve(__dirname, '../cloudflare-workers/public/dashboard'),
    emptyOutDir: true,
  },
});
