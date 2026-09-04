import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';

import { beforeEach, describe, expect, it, vi } from 'vitest';

import { NativeInferenceClient } from './native-inference-client';
import { NATIVE_INFERENCE_PROTOCOL_VERSION } from './protocol';

const spawnMock = vi.hoisted(() => vi.fn());

vi.mock('node:child_process', () => ({ spawn: spawnMock }));

class FakeChildProcess extends EventEmitter {
  readonly stdin = new PassThrough();
  readonly stdout = new PassThrough();
  readonly stderr = new PassThrough();
}

beforeEach(() => {
  spawnMock.mockReset();
});

describe('StdioNativeInferenceTransport shutdown', () => {
  it('accepts a protocol shutdown followed by a zero exit', async () => {
    const { child, client } = await spawnReadyClient();
    const shutdown = client.shutdown();
    await waitForCloseRequest(child);
    child.stdout.write('{"event":"shutdown"}\n');
    await flushEvents();
    child.emit('exit', 0, null);
    child.emit('close', 0, null);

    await expect(shutdown).resolves.toBeUndefined();
  });

  it.each([
    { code: 1, signal: null, label: 'code 1' },
    { code: null, signal: 'SIGTERM', label: 'signal SIGTERM' },
  ])(
    'rejects $label even after a protocol shutdown',
    async ({ code, signal }) => {
      const { child, client } = await spawnReadyClient();
      const shutdown = client.shutdown();
      await waitForCloseRequest(child);
      child.stdout.write('{"event":"shutdown"}\n');
      await flushEvents();
      child.emit('exit', code, signal);
      child.emit('close', code, signal);

      await expect(shutdown).rejects.toThrow('native inference exited');
    }
  );

  it('rejects a zero exit that omitted the protocol shutdown event', async () => {
    const { child, client } = await spawnReadyClient();
    const shutdown = client.shutdown();
    await waitForCloseRequest(child);
    child.emit('exit', 0, null);
    child.emit('close', 0, null);

    await expect(shutdown).rejects.toThrow('without a protocol shutdown event');
  });
});

async function spawnReadyClient(): Promise<{
  child: FakeChildProcess;
  client: NativeInferenceClient;
}> {
  const child = new FakeChildProcess();
  spawnMock.mockReturnValue(child);
  const client = NativeInferenceClient.spawn({
    binaryPath: '/opt/echo-inference',
    modelDirectory: '/models/qwen',
  });
  child.stdout.write(
    `${JSON.stringify({
      event: 'ready',
      protocol_version: NATIVE_INFERENCE_PROTOCOL_VERSION,
      engine: { engine_id: 1 },
      eos_token_id: 248_046,
      chat_template_sha256: 'template',
      max_new_tokens_per_request: 4_096,
      max_outstanding_requests: 8,
      max_active_batch_size: 6,
      max_late_join_batch_size: 4,
    })}\n`
  );
  await client.ready();
  return { child, client };
}

async function waitForCloseRequest(child: FakeChildProcess): Promise<void> {
  await vi.waitFor(() => {
    expect(child.stdin.writableEnded).toBe(true);
  });
}

async function flushEvents(): Promise<void> {
  await new Promise<void>((resolve) => {
    setImmediate(resolve);
  });
}
