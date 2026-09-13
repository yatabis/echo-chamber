import { describe, expect, it, vi } from 'vitest';

import {
  formatCognitiveModuleHandoff,
  formatInitialCognitiveModuleContext,
} from '@echo-chamber/core/agent/cognitive-module-handoff';
import {
  ParallelCognitiveModuleOrchestrator,
  type CognitiveModuleCommittedState,
  type CognitiveModuleDomainPort,
} from '@echo-chamber/core/agent/cognitive-module-orchestrator';
import {
  createEmotionCognitiveModuleOutputFormat,
  createMemoryRecallCognitiveModuleOutputFormat,
  createMemoryStoreCognitiveModuleOutputFormat,
  parseEmotionCognitiveModuleOutput,
  parseMemoryRecallCognitiveModuleOutput,
  parseMemoryStoreCognitiveModuleOutput,
  type MemoryCognitiveModuleOutput,
  type EmotionCognitiveModuleOutput,
} from '@echo-chamber/core/agent/cognitive-module-schema';
import {
  ModelCognitiveModuleRunner,
  type ModelCognitiveModuleOutputContract,
} from '@echo-chamber/core/agent/model-cognitive-module';
import type { AgentSessionTool } from '@echo-chamber/core/agent/session';
import { ThinkingEngine } from '@echo-chamber/core/agent/thinking-engine';
import type {
  ModelPort,
  ModelRequest,
  ModelResponse,
} from '@echo-chamber/core/ports/model';

import {
  NativeInferenceClient,
  type NativeInferenceTransport,
} from './native-inference-client';
import { NativeInferenceModel } from './native-inference-model';
import {
  NATIVE_INFERENCE_PROTOCOL_VERSION,
  type NativeCompletedEvent,
  type NativeGenerateCommand,
  type NativeWireCommand,
  type NativeWireEvent,
} from './protocol';

const EMOTION = { valence: 0.1, arousal: 0.2, labels: ['calm'] };
const MODULE_USAGE = {
  cachedInputTokens: 0,
  cacheWriteInputTokens: 0,
  uncachedInputTokens: 10,
  totalInputTokens: 10,
  outputTokens: 1,
  reasoningTokens: 0,
  totalTokens: 11,
};

class FakeTransport implements NativeInferenceTransport {
  readonly commands: NativeWireCommand[] = [];
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
    await Promise.resolve();
  }

  emit(event: NativeWireEvent): void {
    for (const listener of this.eventListeners) {
      listener(event);
    }
  }

  ready(
    protocolVersion: number = NATIVE_INFERENCE_PROTOCOL_VERSION,
    maxNewTokensPerRequest = 4_096
  ): void {
    this.emit({
      event: 'ready',
      protocol_version: protocolVersion,
      engine: { engine_id: 1 },
      eos_token_id: 248_046,
      chat_template_sha256: 'template',
      max_new_tokens_per_request: maxNewTokensPerRequest,
      max_outstanding_requests: 8,
      max_active_batch_size: 6,
      max_late_join_batch_size: 4,
    });
  }
}

describe('Native Main with the Cognitive Module workflow', () => {
  it.each(['tool', 'message'] as const)(
    'commits Cognitive phases and continues Main after a %s completion',
    async (firstOutput) => {
      const fixture = await createFixture({ firstOutput });
      const result = await fixture.engine.think();
      const commands = fixture.transport.commands.filter(
        (command) => command.type === 'generate'
      );

      expect(commands).toHaveLength(2);
      expect(commands.map((command) => command.state_transition)).toEqual([
        'initial',
        'continuation',
      ]);
      expect(commands[1]?.tools).toEqual([]);
      expect(
        commands[0]?.tools.some((tool) => tool.name === 'update_emotion')
      ).toBe(false);
      const continuation = commands[1]?.input ?? [];
      const exchanges =
        firstOutput === 'tool' ? continuation.slice(1) : continuation;
      if (firstOutput === 'tool') {
        expect(continuation[0]).toMatchObject({
          type: 'tool_result',
          call_id: 'main-inspect',
          output: '{"success":true,"content":"current note"}',
        });
      }
      expect(exchanges).toMatchObject([
        {
          type: 'tool_call',
          call_id: 'cognitive:2:search_memory',
          tool_name: 'search_memory',
        },
        { type: 'tool_result', call_id: 'cognitive:2:search_memory' },
        {
          type: 'tool_call',
          call_id: 'cognitive:2:update_emotion',
          tool_name: 'update_emotion',
        },
        { type: 'tool_result', call_id: 'cognitive:2:update_emotion' },
      ]);
      expect(JSON.stringify(exchanges[1])).toContain('committed recall 2');
      expect(
        fixture.order.filter((entry) => /^(commit|main):/.test(entry))
      ).toEqual(['commit:1', 'main:1', 'commit:2', 'main:2', 'commit:3']);
      expect(
        result.cognitiveModules.phases.map((phase) => phase.phase)
      ).toEqual(['pre_main', 'pre_main', 'post_main']);
      expect(result.cognitiveModules.usage.totalTokens).toBe(66);
      expect(fixture.domain.commitPhase).toHaveBeenCalledTimes(3);
      expect(
        fixture.domain.commitPhase.mock.calls[
          fixture.domain.commitPhase.mock.calls.length - 1
        ]?.[0].memory.value
      ).toEqual({ content: 'session memory', type: 'episode' });
      expect(fixture.moduleRequests).toHaveLength(6);
      for (const request of fixture.moduleRequests) {
        expect(JSON.stringify(request.input)).not.toContain('MAIN_ONLY');
        expect(
          JSON.stringify(request.input).match(/MODULE_ONLY/g)
        ).toHaveLength(1);
      }
      for (const request of fixture.moduleRequests.filter(
        (item) => item.turnIndex === 2
      )) {
        expect(
          request.input.some(
            (item) =>
              'type' in item &&
              item.type === 'tool_call' &&
              item.toolName ===
                (firstOutput === 'tool' ? 'inspect_note' : 'think')
          )
        ).toBe(true);
      }
    }
  );

  it.each(['module', 'commit'] as const)(
    'does not advance Main when the next Cognitive %s fails and preserves paid usage',
    async (failure) => {
      const fixture = await createFixture({ failure });
      await expect(fixture.engine.think()).rejects.toMatchObject({
        name: 'ThinkingEngineExecutionError',
        cognitiveUsage: { totalTokens: 44 },
        mainUsage: { totalTokens: 2 },
        usage: { totalTokens: 46 },
      });
      expect(
        fixture.transport.commands.filter(
          (command) => command.type === 'generate'
        )
      ).toHaveLength(1);
      expect(fixture.domain.failPhase).toHaveBeenCalledTimes(1);
      expect(fixture.order).not.toContain('commit:2');
      expect(fixture.order).not.toContain('main:2');
    }
  );
});

type MockCognitiveDomain = Omit<
  CognitiveModuleDomainPort,
  'commitPhase' | 'failPhase'
> & {
  commitPhase: ReturnType<
    typeof vi.fn<CognitiveModuleDomainPort['commitPhase']>
  >;
  failPhase: ReturnType<typeof vi.fn<CognitiveModuleDomainPort['failPhase']>>;
};

interface IntegrationFixture {
  engine: ThinkingEngine;
  transport: FakeTransport;
  domain: MockCognitiveDomain;
  moduleRequests: ModelRequest[];
  order: string[];
}

/** Runs real coordinator, runner, session, adapter and client code with deterministic I/O. */
async function createFixture(
  options: {
    firstOutput?: 'tool' | 'message';
    failure?: 'module' | 'commit';
  } = {}
): Promise<IntegrationFixture> {
  const order: string[] = [];
  const moduleRequests: ModelRequest[] = [];
  let committed: CognitiveModuleCommittedState = {
    version: 0,
    emotion: null,
    previousSessionMemory: null,
    recalledMemories: [],
  };
  const domain = {
    beginActivation: vi.fn<CognitiveModuleDomainPort['beginActivation']>(
      async () => await Promise.resolve(committed)
    ),
    startPhase: vi.fn<CognitiveModuleDomainPort['startPhase']>(async () => {
      await Promise.resolve();
    }),
    failPhase: vi.fn<CognitiveModuleDomainPort['failPhase']>(async () => {
      await Promise.resolve();
    }),
    commitPhase: vi.fn<CognitiveModuleDomainPort['commitPhase']>(
      async ({ phase, emotion, memory }) => {
        if (options.failure === 'commit' && phase.sequence === 2) {
          throw new Error('domain commit unavailable');
        }
        order.push(`commit:${phase.sequence}`);
        committed = {
          version: committed.version + 1,
          emotion: emotion.value,
          previousSessionMemory:
            'content' in memory.value
              ? {
                  ...memory.value,
                  emotion: emotion.value,
                  createdAt: '2026-09-12T00:00:00Z',
                }
              : committed.previousSessionMemory,
          recalledMemories:
            phase.phase === 'pre_main'
              ? [
                  {
                    content: `committed recall ${phase.sequence}`,
                    type: 'semantic',
                    emotion: emotion.value,
                    createdAt: '2026-09-12T00:00:00Z',
                  },
                ]
              : [],
        };
        return await Promise.resolve(committed);
      }
    ),
  } satisfies CognitiveModuleDomainPort;
  const createModuleModel = (module: 'memory' | 'emotion'): ModelPort => ({
    generate: async (request): Promise<ModelResponse> => {
      moduleRequests.push(request);
      let value: MemoryCognitiveModuleOutput | EmotionCognitiveModuleOutput =
        EMOTION;
      if (module === 'memory') {
        value =
          request.responseFormat?.name ===
          createMemoryStoreCognitiveModuleOutputFormat().name
            ? { content: 'session memory', type: 'episode' }
            : { query: `recall ${request.turnIndex}` };
      }
      const content =
        options.failure === 'module' &&
        module === 'memory' &&
        request.turnIndex === 2
          ? '{invalid JSON'
          : JSON.stringify(value);
      return await Promise.resolve({
        output: [{ type: 'message', role: 'assistant', content }],
        usage: MODULE_USAGE,
      });
    },
  });
  const cognitiveModules = new ParallelCognitiveModuleOrchestrator({
    createActivationId: (): string => 'native-integration',
    memory: new ModelCognitiveModuleRunner<MemoryCognitiveModuleOutput>({
      model: createModuleModel('memory'),
      resolveSystemPrompt: (): string => 'MODULE_ONLY_memory',
      resolveOutputContract: ({
        phase,
      }): ModelCognitiveModuleOutputContract<MemoryCognitiveModuleOutput> =>
        phase === 'pre_main'
          ? {
              format: createMemoryRecallCognitiveModuleOutputFormat(),
              parse: parseMemoryRecallCognitiveModuleOutput,
            }
          : {
              format: createMemoryStoreCognitiveModuleOutputFormat(),
              parse: parseMemoryStoreCognitiveModuleOutput,
            },
    }),
    emotion: new ModelCognitiveModuleRunner({
      model: createModuleModel('emotion'),
      resolveSystemPrompt: (): string => 'MODULE_ONLY_emotion',
      resolveOutputContract:
        (): ModelCognitiveModuleOutputContract<EmotionCognitiveModuleOutput> => ({
          format: createEmotionCognitiveModuleOutputFormat(),
          parse: parseEmotionCognitiveModuleOutput,
        }),
    }),
    domain,
    retryPolicy: { maxAttempts: 1, shouldRetry: (): boolean => false },
    formatInitialContext: formatInitialCognitiveModuleContext,
    formatHandoff: formatCognitiveModuleHandoff,
  });
  const transport = new FakeTransport();
  const client = new NativeInferenceClient(transport);
  const model = new NativeInferenceModel({
    client,
    instanceId: 'rin',
    maxTokens: 128,
    seedSource: (): number => 42,
  });
  transport.ready();
  let generation = 0;
  transport.onSend = (wire, current): void => {
    if (wire.type === 'open_state') {
      current.emit({
        event: 'state_opened',
        request_id: wire.request_id,
        instance_id: wire.instance_id,
        persistence: 'ephemeral',
        restored: false,
      });
    } else if (wire.type === 'generate') {
      generation += 1;
      order.push(`main:${generation}`);
      let output: NativeCompletedEvent['output'] = [
        {
          type: 'tool_call',
          call_id: 'main-finish',
          tool_name: 'finish_thinking',
          input: '{"reason":"done"}',
        },
      ];
      if (generation === 1) {
        output =
          options.firstOutput === 'message'
            ? [
                {
                  type: 'message',
                  role: 'assistant',
                  content: 'Checking the current context.',
                },
              ]
            : [
                {
                  type: 'tool_call',
                  call_id: 'main-inspect',
                  tool_name: 'inspect_note',
                  input: '{}',
                },
              ];
      }
      current.emit(
        completed(wire, {
          output,
          stateSequenceLength: generation * 2,
          cachedPrefixTokens: (generation - 1) * 2,
        })
      );
    }
  };
  await model.openState({ persistence: 'ephemeral' });
  const tools: AgentSessionTool[] = [
    'check_notifications',
    'inspect_note',
    'finish_thinking',
  ].map((name) => ({
    name,
    contract: {
      name,
      description: name,
      inputSchema: { type: 'object', properties: {} },
    },
    execute: async (): Promise<string> =>
      await Promise.resolve(
        name === 'inspect_note'
          ? '{"success":true,"content":"current note"}'
          : '{"success":true}'
      ),
  }));
  return {
    engine: new ThinkingEngine({
      model,
      tools,
      systemPrompt: 'MAIN_ONLY',
      cognitiveModules,
    }),
    transport,
    domain,
    moduleRequests,
    order,
  };
}

interface CompletedOptions {
  stateSequenceLength?: number;
  generated?: number[];
  output?: NativeCompletedEvent['output'];
  cachedPrefixTokens?: number;
  finishReason?: NativeCompletedEvent['response']['finish_reason'];
}

function completed(
  command: NativeGenerateCommand,
  options: CompletedOptions = {}
): NativeCompletedEvent {
  const generated = options.generated ?? [20];
  const stateSequenceLength = options.stateSequenceLength ?? 2;
  const cachedPrefixTokens = options.cachedPrefixTokens ?? 0;
  return {
    event: 'completed',
    request_id: command.request_id,
    response: {
      engine_id: 1,
      instance_id: command.instance_id,
      model: {},
      state_sequence_length: stateSequenceLength,
      generated_tokens: generated,
      finish_reason: options.finishReason ?? 'stop_token',
      metrics: {
        queue_wait_nanos: 0,
        cached_prefix_tokens: cachedPrefixTokens,
        input_tokens_processed: Math.max(
          0,
          stateSequenceLength - generated.length - cachedPrefixTokens
        ),
        generated_tokens: generated.length,
        maximum_decode_batch_size: 1,
        decode_batch_membership_changes: 0,
        model_step_count: generated.length + 1,
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
    text: '',
    output: options.output ?? [
      { type: 'message', role: 'assistant', content: 'ok' },
    ],
  };
}
