import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { defineWorkersConfig } from '@cloudflare/vitest-pool-workers/config';

const appDir = path.dirname(fileURLToPath(import.meta.url));
const packagesDir = path.resolve(appDir, '../../packages');
const cloudflareRuntimeSrcDir = path.resolve(
  packagesDir,
  'cloudflare-runtime/src'
);
const contractsSrcDir = path.resolve(packagesDir, 'contracts/src');
const discordAdapterSrcDir = path.resolve(packagesDir, 'discord-adapter/src');
const openaiAdapterSrcDir = path.resolve(packagesDir, 'openai-adapter/src');

export default defineWorkersConfig({
  resolve: {
    alias: [
      {
        find: /^@echo-chamber\/cloudflare-runtime\/(.*)$/,
        replacement: `${cloudflareRuntimeSrcDir}/$1`,
      },
      {
        find: /^@echo-chamber\/contracts\/(.*)$/,
        replacement: `${contractsSrcDir}/$1`,
      },
      {
        find: /^@echo-chamber\/discord-adapter\/(.*)$/,
        replacement: `${discordAdapterSrcDir}/$1`,
      },
      {
        find: /^@echo-chamber\/openai-adapter\/(.*)$/,
        replacement: `${openaiAdapterSrcDir}/$1`,
      },
    ],
  },
  test: {
    poolOptions: {
      workers: {
        wrangler: {
          configPath: './wrangler.jsonc',
        },
        miniflare: {
          compatibilityFlags: ['nodejs_compat'],
        },
      },
    },
    globals: true,
    watch: true,
    reporters: ['verbose', 'html'],
    coverage: {
      provider: 'istanbul',
      reporter: ['text', 'html', 'lcov'],
      include: ['src/**/*.ts'],
      exclude: [
        'node_modules/**',
        'dist/**',
        '**/*.d.ts',
        'vitest.config.ts',
        '**/*.test.ts',
        '**/*.spec.ts',
        'src/utils/logger.ts',
      ],
    },
    testTimeout: 10000,
    hookTimeout: 10000,
    setupFiles: './test/setup.ts',
  },
  esbuild: {
    target: 'esnext',
  },
});
