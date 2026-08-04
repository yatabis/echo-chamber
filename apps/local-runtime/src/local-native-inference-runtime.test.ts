import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import type { ModelRequest } from '@echo-chamber/core/ports/model';
import {
  NATIVE_INFERENCE_PROTOCOL_VERSION,
  NativeInferenceClient,
} from '@echo-chamber/native-inference-adapter/native-inference-client';
import type {
  NativeInferenceTransport,
  SpawnNativeInferenceOptions,
} from '@echo-chamber/native-inference-adapter/native-inference-client';

import { LocalNativeInferenceRuntime } from './local-native-inference-runtime';

type NativeWireCommand = Parameters<NativeInferenceTransport['send']>[0];
type NativeWireEvent = Parameters<
  Parameters<NativeInferenceTransport['onEvent']>[0]
>[0];
type NativeGenerateCommand = Extract<NativeWireCommand, { type: 'generate' }>;
type NativeSnapshotCommand = Extract<NativeWireCommand, { type: 'snapshot' }>;
type NativeCompletedEvent = Extract<NativeWireEvent, { event: 'completed' }>;

const temporaryDirectories: string[] = [];

class FakeTransport implements NativeInferenceTransport {
  readonly commands: NativeWireCommand[] = [];
  closed = false;
  onSend:
    | ((command: NativeWireCommand, transport: FakeTransport) => void)
    | undefined;
  private readonly eventListeners = new Set<(event: NativeWireEvent) => void>();
  private readonly errorListeners = new Set<(error: Error) => void>();

  async send(command: NativeWireCommand): Promise<void> {
    this.commands.push(command);
    this.onSend?.(command, this);
    await Promise.resolve();
  }

  onEvent(listener: (event: NativeWireEvent) => void): () => void {
    this.eventListeners.add(listener);
    return (): void => {
      this.eventListeners.delete(listener);
    };
  }

  onError(listener: (error: Error) => void): () => void {
    this.errorListeners.add(listener);
    return (): void => {
      this.errorListeners.delete(listener);
    };
  }

  async close(): Promise<void> {
    this.closed = true;
    await Promise.resolve();
  }

  emit(event: NativeWireEvent): void {
    for (const listener of this.eventListeners) listener(event);
  }

  ready(): void {
    this.emit({
      event: 'ready',
      protocol_version: NATIVE_INFERENCE_PROTOCOL_VERSION,
      engine: { engine_id: 1 },
      eos_token_id: 248_046,
      chat_template_sha256: 'template',
      max_outstanding_requests: 8,
    });
  }
}

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map(async (directory) => {
      await rm(directory, { force: true, recursive: true });
    })
  );
});

describe('LocalNativeInferenceRuntime', () => {
  it('opens every owner and checkpoints each completed thinking session', async () => {
    const snapshotDirectory = await createSnapshotDirectory();
    const transport = new FakeTransport();
    const client = new NativeInferenceClient(transport);
    let spawnOptions: SpawnNativeInferenceOptions | undefined;
    const sequenceLengths = new Map([['rin', 419]]);
    transport.ready();
    installOwnerLifecycle(transport, snapshotDirectory, sequenceLengths, 'rin');

    const runtime = await LocalNativeInferenceRuntime.start(
      {
        binaryPath: '/opt/echo-inference',
        modelDirectory: '/models/qwen',
        snapshotDirectory,
      },
      {
        spawnClient: (options): NativeInferenceClient => {
          spawnOptions = options;
          return client;
        },
      }
    );

    expect(spawnOptions).toEqual({
      binaryPath: '/opt/echo-inference',
      modelDirectory: '/models/qwen',
    });
    expect(runtime.state('rin')).toMatchObject({
      instanceId: 'rin',
      stateOpened: true,
      hasState: true,
      snapshotDirty: false,
    });
    expect(runtime.state('rin')).not.toHaveProperty('responseToken');
    expect(runtime.state('marie')).toMatchObject({
      instanceId: 'marie',
      stateOpened: true,
      hasState: false,
    });

    let retainedModel: unknown;
    await runtime.runThinkingSession('rin', async (model) => {
      retainedModel = model;
      return await model.generate(modelRequest());
    });
    await runtime.runThinkingSession('rin', async (model) => {
      expect(model).toBe(retainedModel);
      return await model.generate(modelRequest());
    });

    const generateCommands = transport.commands.filter(
      (command): command is NativeGenerateCommand => command.type === 'generate'
    );
    expect(generateCommands.map((command) => command.state_transition)).toEqual(
      ['new_session', 'new_session']
    );
    expect(
      transport.commands
        .filter((command) => command.type === 'open_state')
        .map((command) => ({
          instanceId: command.instance_id,
          snapshotRoot: command.snapshot_root,
        }))
    ).toEqual([
      { instanceId: 'rin', snapshotRoot: join(snapshotDirectory, 'rin') },
      { instanceId: 'marie', snapshotRoot: join(snapshotDirectory, 'marie') },
    ]);
    expect(
      transport.commands
        .filter((command) => command.type === 'snapshot')
        .map((command) => command.instance_id)
    ).toEqual(['rin', 'rin']);
    expect(runtime.state('rin')).toMatchObject({
      hasState: true,
      snapshotDirty: false,
      stateSequenceLength: 10,
    });

    await runtime.shutdown();
    expect(transport.commands[transport.commands.length - 1]).toEqual({
      type: 'shutdown',
    });
    expect(transport.closed).toBe(true);
  });

  it('checkpoints committed state even when later session work fails', async () => {
    const snapshotDirectory = await createSnapshotDirectory();
    const { client, transport } = successfulClient(snapshotDirectory);
    const runtime = await startWithClient(snapshotDirectory, client);
    const operationError = new Error('tool execution failed');

    await expect(
      runtime.runThinkingSession('rin', async (model) => {
        await model.generate(modelRequest());
        throw operationError;
      })
    ).rejects.toBe(operationError);

    expect(commandTypesAfterOpen(transport)).toEqual(['generate', 'snapshot']);
    expect(runtime.state('rin')).toMatchObject({
      hasState: true,
      snapshotDirty: false,
    });
    await runtime.shutdown();
  });

  it('serializes sessions per instance and waits for one to finish before shutdown', async () => {
    const snapshotDirectory = await createSnapshotDirectory();
    const { client, transport } = successfulClient(snapshotDirectory);
    const gate = deferred<undefined>();
    const runtime = await startWithClient(snapshotDirectory, client);

    const activeSession = runtime.runThinkingSession('rin', async () => {
      await gate.promise;
      return 'finished';
    });
    await expect(
      runtime.runThinkingSession('rin', async () => Promise.resolve('overlap'))
    ).rejects.toThrow('already has an active thinking session');

    const firstShutdown = runtime.shutdown();
    const secondShutdown = runtime.shutdown();
    await Promise.resolve();
    expect(commandTypesAfterOpen(transport)).toEqual([]);
    gate.resolve(undefined);

    await expect(activeSession).resolves.toBe('finished');
    await Promise.all([firstShutdown, secondShutdown]);
    expect(commandTypesAfterOpen(transport)).toEqual(['shutdown']);
  });

  it('cancels an active generation, rolls it back, and exits without snapshotting it', async () => {
    const snapshotDirectory = await createSnapshotDirectory();
    const transport = new FakeTransport();
    const client = new NativeInferenceClient(transport);
    let pendingGenerate: NativeGenerateCommand | undefined;
    transport.ready();
    transport.onSend = (command, current): void => {
      if (command.type === 'open_state') {
        emitOpened(current, command, snapshotDirectory, false);
      } else if (command.type === 'generate') {
        pendingGenerate = command;
      } else if (command.type === 'cancel') {
        if (pendingGenerate === undefined)
          throw new Error('missing generation');
        current.emit({
          event: 'cancelled',
          request_id: pendingGenerate.request_id,
        });
      }
    };
    const runtime = await startWithClient(snapshotDirectory, client);

    const activeSession = runtime.runThinkingSession('rin', async (model) => {
      await model.generate(modelRequest());
    });
    await waitForCommand(transport, 'generate');
    const shutdown = runtime.shutdown();

    await expect(activeSession).rejects.toThrow('cancelled before commit');
    await shutdown;
    expect(commandTypesAfterOpen(transport)).toEqual([
      'generate',
      'cancel',
      'shutdown',
    ]);
    expect(runtime.state('rin').hasState).toBe(false);
  });

  it('waits for an in-flight snapshot without sending it a cancellation', async () => {
    const snapshotDirectory = await createSnapshotDirectory();
    const transport = new FakeTransport();
    const client = new NativeInferenceClient(transport);
    let pendingSnapshot: NativeSnapshotCommand | undefined;
    transport.ready();
    transport.onSend = (command, current): void => {
      if (command.type === 'open_state') {
        emitOpened(current, command, snapshotDirectory, false);
      } else if (command.type === 'generate') {
        current.emit(completed(command, 10));
      } else if (command.type === 'snapshot') {
        pendingSnapshot = command;
      }
    };
    const runtime = await startWithClient(snapshotDirectory, client);

    const activeSession = runtime.runThinkingSession('rin', async (model) => {
      await model.generate(modelRequest());
    });
    await waitForCommand(transport, 'snapshot');
    const shutdown = runtime.shutdown();
    await Promise.resolve();
    expect(commandTypesAfterOpen(transport)).toEqual(['generate', 'snapshot']);
    if (pendingSnapshot === undefined) throw new Error('missing snapshot');
    transport.emit({
      event: 'snapshot_published',
      request_id: pendingSnapshot.request_id,
      instance_id: pendingSnapshot.instance_id,
      path: join(
        snapshotDirectory,
        pendingSnapshot.instance_id,
        'current.safetensors'
      ),
      physical_nbytes: 71_000_000,
    });

    await activeSession;
    await shutdown;
    expect(commandTypesAfterOpen(transport)).toEqual([
      'generate',
      'snapshot',
      'shutdown',
    ]);
  });

  it('fails closed and shuts down when opening current state fails', async () => {
    const snapshotDirectory = await createSnapshotDirectory();
    const transport = new FakeTransport();
    const client = new NativeInferenceClient(transport);
    transport.ready();
    transport.onSend = (command, current): void => {
      if (command.type === 'open_state') {
        current.emit({
          event: 'failed',
          request_id: command.request_id,
          phase: 'open_state',
          error: 'safetensors metadata mismatch',
        });
      }
    };

    await expect(startWithClient(snapshotDirectory, client)).rejects.toThrow(
      'safetensors metadata mismatch'
    );
    expect(transport.commands.map((command) => command.type)).toEqual([
      'open_state',
      'shutdown',
    ]);
    expect(transport.closed).toBe(true);
  });
});

async function createSnapshotDirectory(): Promise<string> {
  const directory = await mkdtemp(join(tmpdir(), 'echo-local-runtime-'));
  temporaryDirectories.push(directory);
  return directory;
}

function successfulClient(snapshotDirectory: string): {
  client: NativeInferenceClient;
  transport: FakeTransport;
} {
  const transport = new FakeTransport();
  const client = new NativeInferenceClient(transport);
  transport.ready();
  installOwnerLifecycle(transport, snapshotDirectory, new Map());
  return { client, transport };
}

function installOwnerLifecycle(
  transport: FakeTransport,
  snapshotDirectory: string,
  sequenceLengths: Map<string, number>,
  restoredInstance?: string
): void {
  transport.onSend = (command, current): void => {
    if (command.type === 'open_state') {
      emitOpened(
        current,
        command,
        snapshotDirectory,
        command.instance_id === restoredInstance
      );
    } else if (command.type === 'generate') {
      const sequenceLength =
        command.state_transition === 'new_session'
          ? 10
          : (sequenceLengths.get(command.instance_id) ?? 0) + 10;
      sequenceLengths.set(command.instance_id, sequenceLength);
      current.emit(completed(command, sequenceLength));
    } else if (command.type === 'snapshot') {
      current.emit({
        event: 'snapshot_published',
        request_id: command.request_id,
        instance_id: command.instance_id,
        path: join(
          snapshotDirectory,
          command.instance_id,
          'current.safetensors'
        ),
        physical_nbytes: 71_000_000,
      });
    }
  };
}

function emitOpened(
  transport: FakeTransport,
  command: Extract<NativeWireCommand, { type: 'open_state' }>,
  snapshotDirectory: string,
  restored: boolean
): void {
  transport.emit({
    event: 'state_opened',
    request_id: command.request_id,
    instance_id: command.instance_id,
    restored,
    current_path: join(
      snapshotDirectory,
      command.instance_id,
      'current.safetensors'
    ),
  });
}

async function startWithClient(
  snapshotDirectory: string,
  client: NativeInferenceClient
): Promise<LocalNativeInferenceRuntime> {
  return await LocalNativeInferenceRuntime.start(
    {
      binaryPath: '/opt/echo-inference',
      modelDirectory: '/models/qwen',
      snapshotDirectory,
    },
    { spawnClient: (): NativeInferenceClient => client }
  );
}

function modelRequest(): ModelRequest {
  return {
    input: [{ role: 'developer', content: 'Think.' }],
    tools: [],
  };
}

function completed(
  command: NativeGenerateCommand,
  stateSequenceLength: number
): NativeCompletedEvent {
  return {
    event: 'completed',
    request_id: command.request_id,
    response: {
      engine_id: 1,
      instance_id: command.instance_id,
      model: {},
      state_sequence_length: stateSequenceLength,
      generated_tokens: [1],
      finish_reason: 'stop_token',
      metrics: {
        queue_wait_nanos: 0,
        cached_prefix_tokens: 0,
        input_tokens_processed: 9,
        generated_tokens: 1,
        model_step_count: 2,
        input_model_execution_count: 1,
        input_execution_nanos: 1,
        input_graph_construction_nanos: 2,
        input_materialization_nanos: 3,
        first_generated_token_nanos: 4,
        decode_execution_nanos: 5,
        decode_graph_construction_nanos: 6,
        decode_schedule_nanos: 7,
        decode_token_wait_nanos: 8,
        decode_finalization_nanos: 9,
        model_execution_nanos: 10,
        request_nanos: 11,
        committed_state_logical_nbytes: 12,
        metal_memory: {
          active_nbytes: 13,
          cache_nbytes: 14,
          peak_nbytes: 15,
        },
      },
    },
    text: 'ok',
    output: [{ type: 'message', role: 'assistant', content: 'ok' }],
  };
}

function commandTypesAfterOpen(transport: FakeTransport): string[] {
  return transport.commands
    .filter((command) => command.type !== 'open_state')
    .map((command) => command.type);
}

interface Deferred<T> {
  promise: Promise<T>;
  resolve(value: T): void;
}

function deferred<T>(): Deferred<T> {
  let resolvePromise: ((value: T) => void) | undefined;
  const promise = new Promise<T>((resolve) => {
    resolvePromise = resolve;
  });
  return {
    promise,
    resolve: (value): void => {
      resolvePromise?.(value);
    },
  };
}

async function waitForCommand(
  transport: FakeTransport,
  commandType: NativeWireCommand['type']
): Promise<void> {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    if (transport.commands.some((command) => command.type === commandType)) {
      return;
    }
    // The fake transport settles command delivery on the next microtask.
    // eslint-disable-next-line no-await-in-loop
    await Promise.resolve();
  }
  throw new Error(`timed out waiting for ${commandType}`);
}
