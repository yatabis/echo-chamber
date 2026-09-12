import { readdir } from 'node:fs/promises';

/**
 * Admits a user-supplied probe directory only when it already exists and is empty.
 *
 * Real probes intentionally write authoritative Native snapshots. Refusing a
 * non-empty directory keeps those destructive fixtures away from existing state.
 */
export async function requireEmptyProbeDirectory(
  directory: string,
  label: string
): Promise<string> {
  let entries: string[];
  try {
    entries = await readdir(directory);
  } catch {
    throw new Error(
      `${label} must be an existing empty directory: ${directory}`
    );
  }

  if (entries.length > 0) {
    throw new Error(
      `${label} must be an existing empty directory: ${directory}`
    );
  }

  return directory;
}
