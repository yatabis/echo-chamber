import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

interface PackageManifest {
  exports: Record<string, string>;
}

describe('@echo-chamber/core package exports', () => {
  it('公開 subpath は存在する source file を参照する', () => {
    const manifestPath = fileURLToPath(
      new URL('../package.json', import.meta.url)
    );
    const manifest = JSON.parse(
      readFileSync(manifestPath, 'utf8')
    ) as PackageManifest;
    const packageDirectory = dirname(manifestPath);
    const missingExports = Object.entries(manifest.exports).filter(
      ([, target]) => !existsSync(resolve(packageDirectory, target))
    );

    expect(missingExports).toEqual([]);
  });
});
