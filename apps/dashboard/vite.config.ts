import { resolve } from 'path';

import { defineConfig } from 'vite';

export default defineConfig({
  base: '/dashboard/',
  build: {
    outDir: resolve(__dirname, '../cloudflare-workers/public/dashboard'),
    emptyOutDir: true,
  },
});
