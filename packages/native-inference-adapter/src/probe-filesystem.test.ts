import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { requireEmptyProbeDirectory } from './probe-filesystem';

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map(async (directory) => {
      await rm(directory, { force: true, recursive: true });
    })
  );
});

describe('requireEmptyProbeDirectory', () => {
  it('admits an existing empty directory', async () => {
    const directory = await createTemporaryDirectory();

    await expect(
      requireEmptyProbeDirectory(directory, 'recovery probe state root')
    ).resolves.toBe(directory);
  });

  it('rejects existing state without changing it', async () => {
    const directory = await createTemporaryDirectory();
    const currentPath = join(directory, 'current.safetensors');
    await writeFile(currentPath, 'authoritative state', 'utf8');

    await expect(
      requireEmptyProbeDirectory(directory, 'recovery probe state root')
    ).rejects.toThrow(
      'recovery probe state root must be an existing empty directory'
    );
    await expect(readFile(currentPath, 'utf8')).resolves.toBe(
      'authoritative state'
    );
  });
});

async function createTemporaryDirectory(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), 'echo-probe-filesystem-'));
  temporaryDirectories.push(directory);
  return directory;
}
