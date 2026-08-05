import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { NativeInferenceClient } from '@echo-chamber/native-inference-adapter/native-inference-client';

/**
 * Owns empty per-instance state directories for one live Native benchmark.
 *
 * State opening is performed before a timed generation, so lifecycle
 * lifecycle setup does not contaminate request latency. The directories are
 * intentionally ephemeral because performance gates do not test restart.
 */
export class EphemeralNativeStateRoots {
  private readonly root = mkdtempSync(
    join(tmpdir(), 'echo-native-evaluation-')
  );
  private readonly openedInstances = new Set<string>();

  /** Opens one empty state root and retains its native owner lock. */
  async open(client: NativeInferenceClient, instanceId: string): Promise<void> {
    if (this.openedInstances.has(instanceId)) {
      throw new Error(
        `native evaluation instance is already open: ${instanceId}`
      );
    }
    const safeInstanceId = instanceId.replaceAll(/[^A-Za-z0-9_.-]/g, '_');
    const event = await client.openState({
      type: 'open_state',
      request_id: `${safeInstanceId}:open-state`,
      instance_id: instanceId,
      persistence: 'durable',
      snapshot_root: join(this.root, safeInstanceId),
    });
    if (event.restored || event.instance_id !== instanceId) {
      throw new Error(
        `native evaluation state root was not empty for ${instanceId}`
      );
    }
    this.openedInstances.add(instanceId);
  }

  /** Removes only the temporary root created by this owner. */
  dispose(): void {
    rmSync(this.root, { recursive: true, force: true });
  }
}

/** Number of committed state tokens intentionally hidden from visible output. */
export function hiddenClosingTokens(event: {
  response: { finish_reason: 'length' | 'stop_token' };
}): number {
  return event.response.finish_reason === 'length' ? 1 : 0;
}
