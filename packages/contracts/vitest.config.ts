import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineConfig } from 'vitest/config';

const packageDir = path.dirname(fileURLToPath(import.meta.url));
const coreSrcDir = path.resolve(packageDir, '../core/src');

export default defineConfig({
  resolve: {
    alias: [
      {
        find: /^@echo-chamber\/core\/(.*)$/,
        replacement: `${coreSrcDir}/$1`,
      },
    ],
  },
  test: {
    environment: 'node',
    globals: true,
    include: ['src/**/*.test.ts'],
  },
});
