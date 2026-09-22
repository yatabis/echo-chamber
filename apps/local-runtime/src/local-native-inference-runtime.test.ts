import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it, vi } from 'vitest';

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
import {
  CognitiveFixtureDomain,
  createCognitiveFixtureTools,
  COGNITIVE_FIXTURE_PROMPT,
} from './testing/cognitive-fixture';

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
    if (command.type === 'clear_input_cache') {
      this.emit({
        event: 'input_cache_cleared',
        request_id: command.request_id,
        instance_id: command.instance_id,
      });
      return;
    }
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
      max_new_tokens_per_request: 4_096,
      max_outstanding_requests: 8,
      max_active_batch_size: 6,
      max_late_join_batch_size: 4,
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
      persistence: 'durable',
      hasState: true,
      snapshotDirty: false,
    });
    expect(runtime.state('rin')).not.toHaveProperty('responseToken');
    expect(runtime.state('marie')).toMatchObject({
      instanceId: 'marie',
      stateOpened: true,
      persistence: 'durable',
      hasState: false,
    });
    expect(runtime.state('rin', 'memory')).toMatchObject({
      instanceId: 'rin.memory',
      stateOpened: true,
      persistence: 'ephemeral',
      hasState: false,
      snapshotDirty: false,
    });
    expect(runtime.state('rin', 'emotion')).toMatchObject({
      instanceId: 'rin.emotion',
      stateOpened: true,
      persistence: 'ephemeral',
      hasState: false,
      snapshotDirty: false,
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
          persistence: command.persistence,
          ...('snapshot_root' in command
            ? { snapshotRoot: command.snapshot_root }
            : {}),
        }))
    ).toEqual([
      {
        instanceId: 'rin',
        persistence: 'durable',
        snapshotRoot: join(snapshotDirectory, 'rin'),
      },
      { instanceId: 'rin.memory', persistence: 'ephemeral' },
      { instanceId: 'rin.emotion', persistence: 'ephemeral' },
      {
        instanceId: 'marie',
        persistence: 'durable',
        snapshotRoot: join(snapshotDirectory, 'marie'),
      },
      { instanceId: 'marie.memory', persistence: 'ephemeral' },
      { instanceId: 'marie.emotion', persistence: 'ephemeral' },
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

  it('runs memory and emotion from complete inputs without carrying prior inference state', async () => {
    const snapshotDirectory = await createSnapshotDirectory();
    const transport = new FakeTransport();
    const client = new NativeInferenceClient(transport);
    const pendingPair: NativeGenerateCommand[] = [];
    transport.ready();
    transport.onSend = (command, current): void => {
      if (command.type === 'open_state') {
        emitOpened(current, command, snapshotDirectory, false);
        return;
      }
      if (command.type !== 'generate') return;
      pendingPair.push(command);
      if (pendingPair.length !== 2) return;
      for (const pending of pendingPair.splice(0)) {
        current.emit(
          completed(pending, pending.state_transition === 'initial' ? 10 : 20, [
            {
              type: 'tool_call',
              call_id:
                pending.state_transition === 'initial'
                  ? 'main-observation'
                  : 'next-main-observation',
              tool_name: MODULE_TOOL.name,
              input: '{"summary":"updated"}',
            },
          ])
        );
      }
    };
    const runtime = await startWithClient(snapshotDirectory, client);

    await runtime.runThinkingSession('rin', async (_main, modules) => {
      await Promise.all([
        modules.memory.generate(moduleRequest('memory system prompt')),
        modules.emotion.generate(moduleRequest('emotion system prompt')),
      ]);
      await Promise.all([
        modules.memory.generate(moduleRequest('complete memory context')),
        modules.emotion.generate(moduleRequest('complete emotion context')),
      ]);
    });

    const generations = transport.commands.filter(
      (command): command is NativeGenerateCommand => command.type === 'generate'
    );
    expect(
      generations.map((command) => ({
        instanceId: command.instance_id,
        transition: command.state_transition,
      }))
    ).toEqual([
      { instanceId: 'rin.memory', transition: 'initial' },
      { instanceId: 'rin.emotion', transition: 'initial' },
      { instanceId: 'rin.memory', transition: 'reset' },
      { instanceId: 'rin.emotion', transition: 'reset' },
    ]);
    expect(runtime.state('rin')).toMatchObject({
      persistence: 'durable',
      hasState: false,
    });
    expect(runtime.state('rin', 'memory')).toMatchObject({
      persistence: 'ephemeral',
      hasState: true,
      snapshotDirty: false,
      stateSequenceLength: 20,
    });
    expect(runtime.state('rin', 'emotion')).toMatchObject({
      persistence: 'ephemeral',
      hasState: true,
      snapshotDirty: false,
      stateSequenceLength: 20,
    });
    expect(commandTypesAfterOpen(transport)).toEqual([
      'generate',
      'generate',
      'generate',
      'generate',
      'clear_input_cache',
      'clear_input_cache',
    ]);
    await runtime.shutdown();
  });

  it('runs real Core prompts through Native lanes, retries only the failed module and carries domain results into the next session', async () => {
    const snapshotDirectory = await createSnapshotDirectory();
    const { client, transport } = successfulClient(snapshotDirectory);
    const lifecycle = transport.onSend;
    const domain = new CognitiveFixtureDomain();
    const { tools, calls } = createCognitiveFixtureTools();
    let mainTurns = 0;
    let memoryAttempts = 0;
    let failedRequest: NativeGenerateCommand | undefined;
    let controller: AbortController | undefined;
    transport.onSend = (command, current): void => {
      if (command.type === 'cancel') {
        current.emit({
          event: 'cancelled',
          request_id: command.request_id,
          usage: {
            cached_prefix_tokens: 0,
            input_tokens_processed: 11,
            generated_tokens: 3,
          },
        });
        return;
      }
      if (command.type !== 'generate') {
        lifecycle?.(command, current);
        return;
      }
      if (command.instance_id === 'rin') {
        mainTurns += 1;
        expect(domain.commits).toHaveLength(
          mainTurns + Math.floor((mainTurns - 1) / 2)
        );
        current.emit(
          completed(command, mainTurns * 10, mainFixtureOutput(mainTurns))
        );
        return;
      }
      const memory = command.instance_id === 'rin.memory';
      if (memory && ++memoryAttempts === 1) {
        failedRequest = command;
        controller?.abort('injected timeout');
        return;
      }
      current.emit(completed(command, 10, cognitiveOutput(command)));
    };
    const runtime = await startWithClient(snapshotDirectory, client);
    const options = {
      tools,
      systemPrompt: COGNITIVE_FIXTURE_PROMPT,
      cognitive: {
        domain,
        retryPolicy: {
          maxAttempts: 2,
          shouldRetry: (input: { error: unknown }): boolean =>
            input.error instanceof Error && input.error.name === 'AbortError',
        },
        createRequestSignal: (): AbortSignal => {
          const next = new AbortController();
          controller ??= next;
          return next.signal;
        },
      },
    };
    const first = await runtime.think('rin', options);
    expect(first.cognitiveModules.phases.map((phase) => phase.phase)).toEqual([
      'pre_main',
      'pre_main',
      'post_main',
    ]);
    expect(first.cognitiveModules.phases[0]?.memory.attempts).toBe(2);
    expect(first.cognitiveModules.phases[0]?.emotion.attempts).toBe(1);
    expect(first.cognitiveModules.usage.totalTokens).toBe(74);
    expect(domain.state.previousSessionMemory?.content).toBe(
      '図書館へ行く予定。'
    );
    await runtime.think('rin', options);
    expect(domain.commits).toHaveLength(6);
    expect(calls.filter((name) => name === 'finish_thinking')).toHaveLength(2);
    assertCognitiveRequests(transport, failedRequest);
    expect(
      transport.commands.filter((command) => command.type === 'snapshot')
    ).toHaveLength(2);
    await runtime.shutdown();
  });

  it.each(['module', 'commit'] as const)(
    'does not advance Main when %s fails in the Native Cognitive path',
    async (failure) => {
      const snapshotDirectory = await createSnapshotDirectory();
      const { client, transport } = successfulClient(snapshotDirectory);
      const lifecycle = transport.onSend;
      const domain = new CognitiveFixtureDomain();
      if (failure === 'commit')
        vi.spyOn(domain, 'commitPhase').mockRejectedValue(
          new Error('storage unavailable')
        );
      transport.onSend = (command, current): void => {
        if (command.type !== 'generate') {
          lifecycle?.(command, current);
          return;
        }
        if (failure === 'module' && command.instance_id === 'rin.memory') {
          current.emit({
            event: 'failed',
            request_id: command.request_id,
            phase: 'inference',
            error: 'injected engine failure',
          });
        } else {
          const value =
            command.instance_id === 'rin.memory'
              ? { query: '図書館' }
              : { valence: 0, arousal: 0, labels: [] };
          current.emit(
            completed(command, 10, [
              {
                type: 'message',
                role: 'assistant',
                content: JSON.stringify(value),
              },
            ])
          );
        }
      };
      const runtime = await startWithClient(snapshotDirectory, client);
      await expect(
        runtime.think('rin', {
          tools: createCognitiveFixtureTools().tools,
          systemPrompt: COGNITIVE_FIXTURE_PROMPT,
          cognitive: {
            domain,
            retryPolicy: { maxAttempts: 1, shouldRetry: (): boolean => false },
          },
        })
      ).rejects.toMatchObject({ name: 'ThinkingEngineExecutionError' });
      expect(domain.commits).toHaveLength(0);
      expect(domain.failures).toHaveLength(1);
      expect(
        transport.commands.some(
          (command) =>
            command.type === 'generate' && command.instance_id === 'rin'
        )
      ).toBe(false);
      expect(runtime.state('rin').hasState).toBe(false);
      expect(
        transport.commands
          .filter((command) => command.type === 'clear_input_cache')
          .map((command) => command.instance_id)
      ).toEqual(['rin.memory', 'rin.emotion']);
      await runtime.shutdown();
    }
  );

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
          usage: {
            cached_prefix_tokens: 0,
            input_tokens_processed: 0,
            generated_tokens: 0,
          },
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

  it('rejects a generation that follows tool work after shutdown starts', async () => {
    const snapshotDirectory = await createSnapshotDirectory();
    const transport = new FakeTransport();
    const client = new NativeInferenceClient(transport);
    const toolWorkStarted = deferred<undefined>();
    const finishToolWork = deferred<undefined>();
    let generationCount = 0;
    transport.ready();
    transport.onSend = (command, current): void => {
      if (command.type === 'open_state') {
        emitOpened(current, command, snapshotDirectory, false);
      } else if (command.type === 'generate') {
        generationCount += 1;
        current.emit(
          completed(
            command,
            generationCount * 10,
            generationCount === 1
              ? [
                  {
                    type: 'tool_call',
                    call_id: 'rin:1:tool:1',
                    tool_name: 'lookup',
                    input: '{}',
                  },
                ]
              : undefined
          )
        );
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
    const runtime = await startWithClient(snapshotDirectory, client);

    const activeSession = runtime.runThinkingSession('rin', async (model) => {
      const first = await model.generate(modelRequest());
      const responseToken = first.responseToken;
      if (responseToken === undefined) {
        throw new Error('missing response token before tool work');
      }
      toolWorkStarted.resolve(undefined);
      await finishToolWork.promise;
      return await model.generate({
        input: [
          {
            type: 'tool_result',
            callId: 'rin:1:tool:1',
            output: '{"found":true}',
          },
        ],
        tools: [],
        previousResponseToken: responseToken,
      });
    });
    await toolWorkStarted.promise;
    const shutdown = runtime.shutdown();
    finishToolWork.resolve(undefined);

    await expect(activeSession).rejects.toThrow(
      'native instance rin is stopping'
    );
    await shutdown;
    expect(generationCount).toBe(1);
    expect(commandTypesAfterOpen(transport)).toEqual([
      'generate',
      'snapshot',
      'shutdown',
    ]);
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

  it('retains state-open and cleanup failures for diagnostics', async () => {
    const snapshotDirectory = await createSnapshotDirectory();
    const transport = new FakeTransport();
    const client = new NativeInferenceClient(transport);
    const cleanupError = new Error('owner cleanup failed');
    vi.spyOn(client, 'shutdown').mockRejectedValue(cleanupError);
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

    await expect(
      startWithClient(snapshotDirectory, client)
    ).rejects.toMatchObject({
      cause: cleanupError,
      errors: [
        {
          message:
            'native inference open_state failed: safetensors metadata mismatch',
        },
        cleanupError,
      ],
    });
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
        command.persistence === 'durable' &&
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
    persistence: command.persistence,
    restored: command.persistence === 'durable' && restored,
    ...(command.persistence === 'durable'
      ? {
          current_path: join(
            snapshotDirectory,
            command.instance_id,
            'current.safetensors'
          ),
        }
      : {}),
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
    input: [{ role: 'system', content: 'Think.' }],
    tools: [],
  };
}

const MODULE_TOOL = {
  name: 'publish_module_update',
  description: 'Publish one module update.',
  inputSchema: {
    type: 'object',
    properties: { summary: { type: 'string' } },
  },
};

function moduleRequest(content: string): ModelRequest {
  return {
    input: [{ role: 'system', content }],
    tools: [MODULE_TOOL],
  };
}

function completed(
  command: NativeGenerateCommand,
  stateSequenceLength: number,
  output: NativeCompletedEvent['output'] = [
    { type: 'message', role: 'assistant', content: 'ok' },
  ]
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
        maximum_decode_batch_size: 1,
        decode_batch_membership_changes: 0,
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
    output,
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
  // Observe the protocol boundary without depending on the adapter's number
  // of promise continuations before it dispatches the command.
  await vi.waitFor(() => {
    expect(
      transport.commands.some((command) => command.type === commandType)
    ).toBe(true);
  });
}

/** Confirm lineage isolation and the canonical shared input across two activations. */
function assertCognitiveRequests(
  transport: FakeTransport,
  failedRequest: NativeGenerateCommand | undefined
): void {
  const commands = transport.commands.filter(
    (command) => command.type === 'generate'
  );
  assertSystemInstructions(commands);
  const memory = commands.filter(
    (command) => command.instance_id === 'rin.memory'
  );
  const emotion = commands.filter(
    (command) => command.instance_id === 'rin.emotion'
  );
  expect(memory.map((command) => command.state_transition)).toEqual([
    'initial',
    'initial',
    'reset',
    'reset',
    'reset',
    'reset',
    'reset',
  ]);
  expect(emotion.map((command) => command.state_transition)).toEqual([
    'initial',
    'reset',
    'reset',
    'reset',
    'reset',
    'reset',
  ]);
  expect(memory[1]?.input).toEqual(failedRequest?.input);
  assertInputCacheScopes(memory, emotion, transport);
  expect(JSON.stringify(memory[4]?.input)).toContain('図書館へ行く予定。');
  expect(memory[0]?.input[0]).not.toEqual(emotion[0]?.input[0]);
  expect(memory[0]?.input.slice(1)).toEqual(emotion[0]?.input.slice(1));
  expect(JSON.stringify(memory[2]?.input)).toContain('KITE-47');
  for (const command of [...memory, ...emotion]) {
    expect(command.tools).toEqual([]);
    expect(JSON.stringify(command.input)).not.toContain(
      COGNITIVE_FIXTURE_PROMPT
    );
  }
  expect(
    commands
      .filter((command) => command.instance_id === 'rin')
      .map((command) => command.state_transition)
  ).toEqual(['initial', 'continuation', 'new_session', 'continuation']);
}

/** Every full input has exactly one system instruction, separate from observations. */
function assertSystemInstructions(commands: NativeGenerateCommand[]): void {
  for (const command of commands) {
    if (command.state_transition !== 'continuation') {
      expect(command.input[0]).toMatchObject({ role: 'system' });
      expect(
        command.input.filter((item) => 'role' in item && item.role === 'system')
      ).toHaveLength(1);
    }
  }
}

/** Deterministic transport fixture; production schemas still validate every response. */
function cognitiveOutput(
  command: NativeGenerateCommand
): NativeCompletedEvent['output'] {
  let value: unknown = { valence: 0.2, arousal: 0.3, labels: ['平静'] };
  if (command.instance_id === 'rin.memory') {
    value =
      command.response_format?.name === 'cognitive_memory_store'
        ? { content: '図書館へ行く予定。', type: 'episode' }
        : { query: '図書館' };
  }
  return [
    { type: 'message', role: 'assistant', content: JSON.stringify(value) },
  ];
}

/** Exercise a tool result followed by a successful Main finish in each activation. */
function mainFixtureOutput(turn: number): NativeCompletedEvent['output'] {
  return [
    {
      type: 'tool_call',
      call_id: `main-${turn}`,
      tool_name: turn % 2 === 1 ? 'inspect_note' : 'finish_thinking',
      input:
        turn % 2 === 1
          ? '{}'
          : JSON.stringify({
              reason: 'KITE-47 checked',
            }),
    },
  ];
}

/** A retry shares its session cache, while a new session gets a fresh identity. */
function assertInputCacheScopes(
  memory: NativeGenerateCommand[],
  emotion: NativeGenerateCommand[],
  transport: FakeTransport
): void {
  expect(
    memory.slice(0, 4).map((command) => command.input_cache_scope)
  ).toEqual(Array<string | undefined>(4).fill(memory[0]?.input_cache_scope));
  expect(memory[0]?.input_cache_scope).toBeTypeOf('string');
  expect(memory[4]?.input_cache_scope).not.toBe(memory[0]?.input_cache_scope);
  expect(emotion[0]?.input_cache_scope).toBe(memory[0]?.input_cache_scope);
  expect(emotion[3]?.input_cache_scope).toBe(memory[4]?.input_cache_scope);
  expect(
    transport.commands.filter((command) => command.type === 'clear_input_cache')
  ).toHaveLength(4);
}
