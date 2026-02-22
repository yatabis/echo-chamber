import { defineWorkersConfig } from '@cloudflare/vitest-pool-workers/config';

export default defineWorkersConfig({
  test: {
    poolOptions: {
      workers: {
        wrangler: {
          configPath: '../../apps/cloudflare-workers/wrangler.jsonc',
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
