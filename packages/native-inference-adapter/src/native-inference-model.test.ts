import { describe, expect, it, vi } from 'vitest';

import { canonicalRuntimeTools } from '@echo-chamber/core/agent/runtime-tools/catalog';
import { runAgentSession } from '@echo-chamber/core/agent/session';
import type { AgentSessionTurnBoundaryHandler } from '@echo-chamber/core/agent/session';
import type {
  EchoEvent,
  EchoEventPort,
} from '@echo-chamber/core/ports/echo-event';
import type {
  ModelInputItem,
  ModelRequest,
} from '@echo-chamber/core/ports/model';

import { NativeInferenceClient } from './native-inference-client';
import {
  ECHO_NATIVE_PRODUCTION_SAMPLING,
  NativeInferenceIncompleteGenerationError,
  NativeInferenceModel,
} from './native-inference-model';
import { NATIVE_INFERENCE_PROTOCOL_VERSION } from './protocol';

import type { NativeInferenceTransport } from './native-inference-client';
import type {
  NativeCompletedEvent,
  NativeGenerateCommand,
  NativeWireCommand,
  NativeWireEvent,
} from './protocol';

const TOOL = {
  name: 'finish_thinking',
  description: 'Finish',
  inputSchema: {
    type: 'object',
    properties: { session_record: { type: 'object' } },
  },
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

describe('NativeInferenceModel', () => {
  it('applies a request-local output limit without changing the default', async () => {
    const { model, transport } = setupModel();
    transport.onSend = autoResponder();
    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });
    await model.generate({ ...request('bounded'), maxOutputTokens: 7 });
    await model.generate(request('default'));
    expect(
      transport.commands
        .filter((item) => item.type === 'generate')
        .map((item) => item.max_new_tokens)
    ).toEqual([7, 128]);
  });

  it.each([0, -1, 1.5, NaN, Infinity, 129])(
    'rejects an invalid or over-budget request limit %s before sending',
    async (maxOutputTokens) => {
      const { model, transport } = setupModel();
      transport.onSend = autoResponder();
      await model.openState({
        persistence: 'durable',
        snapshotRoot: '/state/rin',
      });
      const before = model.state();
      await expect(
        model.generate({ ...request('invalid'), maxOutputTokens })
      ).rejects.toThrow('maxOutputTokens');
      expect(transport.commands).toHaveLength(1);
      expect(model.state()).toEqual(before);
    }
  );

  it('does not send a generation whose signal was already aborted', async () => {
    const { model, transport } = setupModel();
    transport.onSend = autoResponder();
    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });
    const before = model.state();
    await expect(
      model.generate({ ...request('cancelled'), signal: AbortSignal.abort() })
    ).rejects.toMatchObject({ name: 'AbortError' });
    expect(transport.commands).toHaveLength(1);
    expect(model.state()).toEqual(before);
  });

  it('waits for cancellation rollback before allowing a retry', async () => {
    const { model, transport } = setupModel();
    transport.onSend = autoResponder();
    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });
    await model.generate(request('base'));
    const before = model.state();
    const controller = new AbortController();
    transport.onSend = (wire): void => {
      if (wire.type === 'generate') controller.abort('deadline');
    };
    const pending = model.generate({
      ...request('interrupted'),
      signal: controller.signal,
    });
    const rejected = expect(pending).rejects.toMatchObject({
      name: 'AbortError',
      cause: 'deadline',
      usage: {
        cachedInputTokens: 2,
        uncachedInputTokens: 10,
        outputTokens: 3,
        totalTokens: 15,
      },
    });
    await vi.waitFor(() => {
      expect(transport.commands[transport.commands.length - 1]?.type).toBe(
        'cancel'
      );
    });
    const cancel = transport.commands[transport.commands.length - 1];
    if (cancel?.type !== 'cancel') throw new Error('missing cancel');
    await expect(model.generate(request('too early'))).rejects.toThrow(
      'active generation'
    );
    transport.emit({
      event: 'cancelled',
      usage: {
        cached_prefix_tokens: 2,
        input_tokens_processed: 10,
        generated_tokens: 3,
      },
      request_id: cancel.request_id,
    });
    await rejected;
    expect(model.state()).toEqual(before);
    transport.onSend = autoResponder();
    await expect(model.generate(request('retry'))).resolves.toHaveProperty(
      'responseToken'
    );
  });

  it('accepts authoritative completion when it wins an abort race and removes the listener', async () => {
    const { model, transport } = setupModel();
    transport.onSend = autoResponder();
    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });
    const controller = new AbortController();
    const removeListener = vi.spyOn(
      AbortSignal.prototype,
      'removeEventListener'
    );
    transport.onSend = (wire, current): void => {
      if (wire.type === 'generate') {
        controller.abort();
        current.emit(completed(wire, { stateSequenceLength: 19 }));
      }
    };
    await expect(
      model.generate({ ...request('race'), signal: controller.signal })
    ).resolves.toHaveProperty('usage');
    expect(model.state().stateSequenceLength).toBe(19);
    expect(removeListener).toHaveBeenCalledWith('abort', expect.any(Function));
    removeListener.mockRestore();
    expect(transport.commands.filter((item) => item.type === 'cancel')).toEqual(
      []
    );
  });

  it('keeps a committed response when cancellation is sent but loses the race', async () => {
    const { model, transport } = setupModel();
    transport.onSend = autoResponder();
    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });
    let command: NativeGenerateCommand | undefined;
    transport.onSend = (wire, current): void => {
      if (wire.type === 'generate') command = wire;
      if (wire.type === 'cancel') {
        if (command === undefined) throw new Error('missing generation');
        current.emit({
          event: 'cancel_acknowledged',
          request_id: wire.request_id,
          accepted: false,
        });
        current.emit(completed(command, { stateSequenceLength: 23 }));
      }
    };
    const controller = new AbortController();
    const pending = model.generate({
      ...request('race'),
      signal: controller.signal,
    });
    await vi.waitFor(() => {
      expect(command).toBeDefined();
    });
    controller.abort();
    const response = await pending;
    expect(model.state()).toMatchObject({
      stateSequenceLength: 23,
      responseToken: response.responseToken,
    });
    expect(
      transport.commands.filter((item) => item.type === 'cancel')
    ).toHaveLength(1);
  });

  it('cancels before dispatch through the owner API without sending an orphan cancel', async () => {
    const { model, transport } = setupModel();
    transport.onSend = autoResponder();
    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });
    const before = model.state();
    const pending = model.generate(request('cancel immediately'));
    const rejected = expect(pending).rejects.toMatchObject({
      name: 'AbortError',
    });
    expect(await model.cancelActive()).toBe(true);
    await rejected;
    expect(transport.commands).toHaveLength(1);
    expect(model.state()).toEqual(before);
  });

  it('fails the shared client if a cancellation cannot be delivered', async () => {
    const { model, transport } = setupModel();
    transport.onSend = autoResponder();
    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });
    const controller = new AbortController();
    transport.onSend = (wire): void => {
      if (wire.type === 'cancel') throw new Error('cancel transport failed');
    };
    const pending = model.generate({
      ...request('running'),
      signal: controller.signal,
    });
    const rejected = expect(pending).rejects.toThrow('cancel transport failed');
    await vi.waitFor(() => {
      expect(transport.commands).toHaveLength(2);
    });
    controller.abort();
    await rejected;
    await expect(model.generate(request('uncertain owner'))).rejects.toThrow(
      'cancel transport failed'
    );
    expect(
      transport.commands.filter((item) => item.type === 'generate')
    ).toHaveLength(1);
  });

  it('passes the schema separately without rewriting any prompt and preserves it across continuation', async () => {
    const { model, transport } = setupModel();
    transport.onSend = autoResponder({
      output: [{ type: 'message', role: 'assistant', content: '{"answer":7}' }],
    });
    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });
    const input: ModelRequest['input'] = [
      { role: 'system', content: 'Be precise.' },
      { role: 'user', content: 'Seven' },
    ];
    const format = {
      type: 'json_schema' as const,
      name: 'answer',
      strict: true as const,
      schema: {
        type: 'object',
        properties: { answer: { type: 'integer' } },
        required: ['answer'],
        additionalProperties: false,
      },
    };
    const response = await model.generate({
      input,
      tools: [],
      responseFormat: format,
    });
    expect(input[0]).toEqual({ role: 'system', content: 'Be precise.' });
    const first = transport.commands.find((item) => item.type === 'generate');
    expect(first?.input).toEqual(input);
    expect(first).toHaveProperty('response_format', format);
    const continued = await model.generate({
      input: [],
      tools: [],
      previousResponseToken: response.responseToken,
      responseFormat: format,
    });
    const continuation = transport.commands[transport.commands.length - 1];
    expect(continuation).toMatchObject({
      type: 'generate',
      state_transition: 'continuation',
      input: [],
      response_format: format,
    });
    await model.generate({
      input: [],
      tools: [],
      previousResponseToken: continued.responseToken,
    });
    expect(
      transport.commands[transport.commands.length - 1]
    ).not.toHaveProperty('response_format');
  });

  it.each(['not JSON', '{"answer":"7"}', '{"answer":7,"extra":true}'])(
    'detects an engine output contract violation %s while retaining committed state and usage',
    async (content) => {
      const { model, transport } = setupModel();
      transport.onSend = autoResponder({
        stateSequenceLength: 12,
        output: [{ type: 'message', role: 'assistant', content }],
      });
      await model.openState({
        persistence: 'durable',
        snapshotRoot: '/state/rin',
      });
      await expect(
        model.generate({
          input: [{ role: 'user', content: 'Seven' }],
          tools: [],
          responseFormat: {
            type: 'json_schema',
            name: 'answer',
            strict: true,
            schema: {
              type: 'object',
              properties: { answer: { type: 'integer' } },
              required: ['answer'],
              additionalProperties: false,
            },
          },
        })
      ).rejects.toMatchObject({
        name: 'NativeStructuredOutputError',
        usage: { outputTokens: 1, totalInputTokens: 11 },
      });
      expect(model.state()).toMatchObject({
        hasState: true,
        stateSequenceLength: 12,
      });
    }
  );

  it('rejects an incompatible native protocol before opening state', async () => {
    const transport = new FakeTransport();
    const client = new NativeInferenceClient(transport);
    transport.ready(1);

    await expect(client.ready()).rejects.toThrow(
      `protocol 1 is incompatible with required ${NATIVE_INFERENCE_PROTOCOL_VERSION}`
    );
    expect(transport.commands).toEqual([]);
  });

  it('opens an empty owner, generates initial state, and publishes one fixed path', async () => {
    const { model, transport } = setupModel();
    transport.onSend = (wire, current): void => {
      if (wire.type === 'open_state') {
        current.emit({
          event: 'state_opened',
          request_id: wire.request_id,
          instance_id: wire.instance_id,
          persistence: 'durable',
          restored: false,
          current_path: '/state/rin/current.safetensors',
        });
      } else if (wire.type === 'generate') {
        expect(wire.state_transition).toBe('initial');
        expect(wire.stream_tokens).toBe(false);
        expect(wire.tools).toHaveLength(1);
        expect(wire.sampling).toEqual({
          ...ECHO_NATIVE_PRODUCTION_SAMPLING,
          seed: 42,
        });
        current.emit(completed(wire, { stateSequenceLength: 7 }));
      } else if (wire.type === 'snapshot') {
        current.emit({
          event: 'snapshot_published',
          request_id: wire.request_id,
          instance_id: wire.instance_id,
          path: '/state/rin/current.safetensors',
          physical_nbytes: 71_000_000,
        });
      }
    };

    expect(
      await model.openState({
        persistence: 'durable',
        snapshotRoot: '/state/rin',
      })
    ).toMatchObject({
      stateOpened: true,
      persistence: 'durable',
      hasState: false,
      snapshotDirty: false,
    });
    const response = await model.generate(request('hello'));
    expect(response.responseToken).toMatch(/^echo-native-v3:/);
    expect(model.state()).toMatchObject({
      hasState: true,
      snapshotDirty: true,
      stateSequenceLength: 7,
    });
    const published = await model.snapshot();
    expect(published.path).toBe('/state/rin/current.safetensors');
    expect(model.needsSnapshot()).toBe(false);
  });

  it('uses the owner-advertised generation limit when maxTokens is omitted', async () => {
    const transport = new FakeTransport();
    const client = new NativeInferenceClient(transport);
    const model = new NativeInferenceModel({
      client,
      instanceId: 'rin',
      seedSource: (): number => 42,
    });
    transport.ready(NATIVE_INFERENCE_PROTOCOL_VERSION, 4_096);
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
        expect(wire.max_new_tokens).toBe(4_096);
        current.emit(completed(wire));
      }
    };

    await model.openState({ persistence: 'ephemeral' });
    await model.generate(request('owner limit'));
  });

  it('rejects an explicit generation limit above the resident owner limit', async () => {
    const transport = new FakeTransport();
    const client = new NativeInferenceClient(transport);
    const model = new NativeInferenceModel({
      client,
      instanceId: 'rin',
      maxTokens: 4_097,
      seedSource: (): number => 42,
    });
    transport.ready(NATIVE_INFERENCE_PROTOCOL_VERSION, 4_096);
    transport.onSend = (wire, current): void => {
      if (wire.type === 'open_state') {
        current.emit({
          event: 'state_opened',
          request_id: wire.request_id,
          instance_id: wire.instance_id,
          persistence: 'ephemeral',
          restored: false,
        });
      }
    };

    await expect(model.openState({ persistence: 'ephemeral' })).rejects.toThrow(
      'exceeds resident owner limit 4096'
    );
    expect(
      transport.commands.filter(
        (command) =>
          command.type === 'open_state' || command.type === 'generate'
      )
    ).toEqual([]);
  });

  it('enables diagnostic token events only when a listener is configured', async () => {
    const { model, transport } = setupModel(() => undefined);
    transport.onSend = (wire, current): void => {
      if (wire.type === 'open_state') {
        current.emit({
          event: 'state_opened',
          request_id: wire.request_id,
          instance_id: wire.instance_id,
          persistence: 'durable',
          restored: false,
          current_path: '/state/rin/current.safetensors',
        });
      } else if (wire.type === 'generate') {
        expect(wire.stream_tokens).toBe(true);
        current.emit(completed(wire, { stateSequenceLength: 7 }));
      }
    };

    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });
    await model.generate(request('hello'));
  });

  it('starts a new session after restore and does not recreate a live continuation token', async () => {
    const { model, transport } = setupModel();
    transport.onSend = (wire, current): void => {
      if (wire.type === 'open_state') {
        current.emit({
          event: 'state_opened',
          request_id: wire.request_id,
          instance_id: wire.instance_id,
          persistence: 'durable',
          restored: true,
          current_path: '/state/rin/current.safetensors',
        });
      } else if (wire.type === 'generate') {
        expect(wire.state_transition).toBe('new_session');
        current.emit(completed(wire, { stateSequenceLength: 5 }));
      }
    };

    const restored = await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });
    expect(restored).toMatchObject({ hasState: true, snapshotDirty: false });
    expect(restored).not.toHaveProperty('responseToken');
    await model.generate(request('fresh'));
  });

  it('uses only previousResponseToken presence for a live continuation', async () => {
    const { model, transport } = setupModel();
    let generation = 0;
    transport.onSend = (wire, current): void => {
      if (wire.type === 'open_state') {
        current.emit({
          event: 'state_opened',
          request_id: wire.request_id,
          instance_id: wire.instance_id,
          persistence: 'durable',
          restored: false,
          current_path: '/state/rin/current.safetensors',
        });
        return;
      }
      if (wire.type !== 'generate') return;
      generation += 1;
      if (generation === 1) {
        expect(wire.state_transition).toBe('initial');
        current.emit(
          completed(wire, {
            stateSequenceLength: 10,
            output: [toolCall('call-1')],
          })
        );
        return;
      }
      expect(wire.state_transition).toBe('continuation');
      expect(wire.tools).toEqual([]);
      expect(wire.input).toEqual([
        { type: 'tool_result', call_id: 'call-1', output: 'done' },
      ]);
      expect(wire).not.toHaveProperty('lineage_tokens');
      current.emit(
        completed(wire, { stateSequenceLength: 13, cachedPrefixTokens: 10 })
      );
    };

    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });
    await model.generate(request('hello'));
    await model.generate({
      previousResponseToken: 'the-value-is-deliberately-not-compared',
      input: [{ type: 'tool_result', callId: 'call-1', output: 'done' }],
      tools: [TOOL],
    });
  });

  it('rejects non-empty continuation when the preceding completion has no pending tool call', async () => {
    const { model, transport } = setupModel();
    transport.onSend = autoResponder();
    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });
    const initial = await model.generate(request('plain completion'));

    await expect(
      model.generate({
        previousResponseToken: initial.responseToken,
        input: [{ type: 'tool_result', callId: 'call-1', output: 'done' }],
        tools: [TOOL],
      })
    ).rejects.toThrow('requires a pending tool call');
    expect(
      transport.commands.filter((command) => command.type === 'generate')
    ).toHaveLength(1);
  });

  it('connects the core no-tool retry to an empty Native continuation', async () => {
    const { model, transport } = setupModel();
    let generation = 0;
    const nextWakeAt = '2026-09-13T00:00:00Z';
    const finishTool = canonicalRuntimeTools.find(
      (tool) => tool.name === 'finish_thinking'
    );
    if (finishTool === undefined)
      throw new Error('finish_thinking is required');
    const onTurnBoundary = vi
      .fn<AgentSessionTurnBoundaryHandler>()
      .mockResolvedValue([]);
    transport.onSend = (wire, current): void => {
      if (wire.type === 'open_state') {
        current.emit({
          event: 'state_opened',
          request_id: wire.request_id,
          instance_id: wire.instance_id,
          persistence: 'ephemeral',
          restored: false,
        });
        return;
      }
      if (wire.type !== 'generate') return;
      generation += 1;
      if (generation === 1) {
        current.emit(completed(wire));
        return;
      }
      expect(wire.state_transition).toBe('continuation');
      expect(wire.input).toEqual([]);
      current.emit(
        completed(wire, {
          output: [
            {
              type: 'tool_call',
              call_id: 'call-finish',
              tool_name: 'finish_thinking',
              input: JSON.stringify({
                reason: 'done',
                next_wake_at: nextWakeAt,
              }),
            },
          ],
        })
      );
    };

    await model.openState({ persistence: 'ephemeral' });
    const result = await runAgentSession({
      model,
      initialInput: [{ role: 'developer', content: 'continue until finished' }],
      tools: [
        {
          name: 'finish_thinking',
          contract: finishTool.contract,
          execute: async (): Promise<string> =>
            Promise.resolve('{"success":true}'),
        },
      ],
      onTurnBoundary,
    });

    expect(result).toMatchObject({
      nextWakeAt,
      terminationReason: 'finish_thinking',
    });
    expect(generation).toBe(2);
    expect(onTurnBoundary).toHaveBeenCalledTimes(2);
    expect(onTurnBoundary).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        terminationReason: null,
        resolvedInput: [],
      })
    );
    expect(onTurnBoundary).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        terminationReason: 'finish_thinking',
        resolvedInput: [
          {
            type: 'tool_result',
            callId: 'call-finish',
            output: '{"success":true}',
          },
        ],
      })
    );
  });

  it.each([
    'missing pending result',
    'wrong pending result',
    'reordered pending results',
    'duplicate pending results',
    'unmarked call',
    'missing runtime result',
    'wrong runtime result',
    'reused pending ID',
    'duplicate runtime ID',
    'extra result',
    'new user message',
  ])('rejects %s without advancing Native state', async (invalidCase) => {
    const { model, transport } = setupModel();
    const pendingCalls = invalidCase.endsWith('pending results')
      ? [toolCall('main-call'), toolCall('second-main-call')]
      : [toolCall('main-call')];
    transport.onSend = autoResponder({ output: pendingCalls });
    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });
    const initial = await model.generate(request('main'));
    const pending: ModelInputItem = {
      type: 'tool_result',
      callId: 'main-call',
      output: 'done',
    };
    const runtimeCall: ModelInputItem = {
      type: 'tool_call',
      origin: 'runtime',
      callId: 'cognitive:2:update_emotion',
      toolName: 'update_emotion',
      input: '{}',
    };
    const runtimeResult: ModelInputItem = {
      type: 'tool_result',
      callId: runtimeCall.callId,
      output: '{"success":true}',
    };
    const cases: Record<string, ModelInputItem[]> = {
      'reordered pending results': [
        { ...pending, callId: 'second-main-call' },
        pending,
        runtimeCall,
        runtimeResult,
      ],
      'duplicate pending results': [
        pending,
        pending,
        runtimeCall,
        runtimeResult,
      ],
      'missing pending result': [runtimeCall, runtimeResult],
      'wrong pending result': [
        { ...pending, callId: 'other' },
        runtimeCall,
        runtimeResult,
      ],
      'unmarked call': [
        pending,
        {
          type: 'tool_call',
          callId: runtimeCall.callId,
          toolName: runtimeCall.toolName,
          input: '{}',
        },
        runtimeResult,
      ],
      'missing runtime result': [pending, runtimeCall],
      'wrong runtime result': [
        pending,
        runtimeCall,
        { ...runtimeResult, callId: 'other' },
      ],
      'reused pending ID': [
        pending,
        { ...runtimeCall, callId: 'main-call' },
        pending,
      ],
      'duplicate runtime ID': [
        pending,
        runtimeCall,
        runtimeResult,
        runtimeCall,
        runtimeResult,
      ],
      'extra result': [
        pending,
        runtimeCall,
        runtimeResult,
        { ...runtimeResult, callId: 'orphan' },
      ],
      'new user message': [pending, { role: 'user', content: 'new query' }],
    };
    const state = model.state();
    await expect(
      model.generate({
        input: cases[invalidCase] ?? [],
        tools: [TOOL],
        previousResponseToken: initial.responseToken,
      })
    ).rejects.toThrow('native continuation');
    expect(
      transport.commands.filter((command) => command.type === 'generate')
    ).toHaveLength(1);
    expect(model.state()).toEqual(state);
  });

  it('retains the tool schema serialization failure for diagnostics', async () => {
    const { model, transport } = setupModel();
    const serializationError = new Error('schema serialization failed');
    transport.onSend = autoResponder();
    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });

    await expect(
      model.generate({
        ...request('inspect schema'),
        tools: [
          {
            ...TOOL,
            inputSchema: {
              toJSON(): never {
                throw serializationError;
              },
            },
          },
        ],
      })
    ).rejects.toMatchObject({ cause: serializationError });
    expect(transport.commands.some((wire) => wire.type === 'generate')).toBe(
      false
    );
  });

  it('retains both completion and token listener failures for diagnostics', async () => {
    const listenerError = new Error('token listener failed');
    const { model, transport } = setupModel(() => {
      throw listenerError;
    });
    transport.onSend = (wire, current): void => {
      if (wire.type === 'open_state') {
        autoResponder()(wire, current);
      } else if (wire.type === 'generate') {
        current.emit({
          event: 'token',
          request_id: wire.request_id,
          index: 0,
          token_id: 248_046,
          terminal: true,
        });
        const event = completed(wire);
        event.response.instance_id = 'another-owner';
        current.emit(event);
      }
    };
    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });

    await expect(
      model.generate(request('inspect completion'))
    ).rejects.toMatchObject({
      cause: {
        message: 'native response instance another-owner does not match rin',
      },
      errors: [
        {
          message: 'native response instance another-owner does not match rin',
        },
        listenerError,
      ],
    });
  });

  it('accepts committed state before surfacing a terminal token listener error', async () => {
    const listenerError = new Error('synthetic token listener failure');
    const { model, transport } = setupModel(() => {
      throw listenerError;
    });
    transport.onSend = (wire, current): void => {
      if (wire.type === 'open_state') {
        current.emit({
          event: 'state_opened',
          request_id: wire.request_id,
          instance_id: wire.instance_id,
          persistence: 'durable',
          restored: false,
          current_path: '/state/rin/current.safetensors',
        });
      } else if (wire.type === 'generate') {
        current.emit({
          event: 'token',
          request_id: wire.request_id,
          index: 0,
          token_id: 248_046,
          terminal: true,
        });
        current.emit(completed(wire, { stateSequenceLength: 9 }));
      }
    };

    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });
    await expect(model.generate(request('stream failure'))).rejects.toThrow(
      listenerError.message
    );
    expect(model.state()).toMatchObject({
      hasState: true,
      snapshotDirty: true,
      stateSequenceLength: 9,
    });
    expect(
      transport.commands.some((command) => command.type === 'cancel')
    ).toBe(true);
  });

  it('rejects continuation results that do not match the pending call IDs', async () => {
    const { model, transport } = setupModel();
    let generation = 0;
    transport.onSend = (wire, current): void => {
      if (wire.type === 'open_state') {
        current.emit({
          event: 'state_opened',
          request_id: wire.request_id,
          instance_id: wire.instance_id,
          persistence: 'durable',
          restored: false,
          current_path: '/state/rin/current.safetensors',
        });
      } else if (wire.type === 'generate') {
        generation += 1;
        current.emit(
          completed(wire, {
            output: [toolCall('expected-call')],
          })
        );
      }
    };
    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });
    const initial = await model.generate(request('tool completion'));

    await expect(
      model.generate({
        previousResponseToken: initial.responseToken,
        input: [{ type: 'tool_result', callId: 'wrong-call', output: 'done' }],
        tools: [TOOL],
      })
    ).rejects.toThrow('do not match pending calls');
    await expect(
      model.generate({
        previousResponseToken: initial.responseToken,
        input: [{ role: 'user', content: 'not a tool result' }],
        tools: [TOOL],
      })
    ).rejects.toThrow('accepts only results for the pending tool calls');
    expect(generation).toBe(1);
  });

  it('keeps an ephemeral module state in memory and retries a tool-response delta from its prior commit', async () => {
    const { model, transport } = setupModel();
    let generation = 0;
    transport.onSend = (wire, current): void => {
      if (wire.type === 'open_state') {
        expect(wire).toEqual({
          type: 'open_state',
          request_id: 'rin:1',
          instance_id: 'rin',
          persistence: 'ephemeral',
        });
        current.emit({
          event: 'state_opened',
          request_id: wire.request_id,
          instance_id: wire.instance_id,
          persistence: 'ephemeral',
          restored: false,
        });
        return;
      }
      if (wire.type !== 'generate') return;
      generation += 1;
      if (generation === 1) {
        expect(wire.state_transition).toBe('initial');
        current.emit(
          completed(wire, {
            stateSequenceLength: 10,
            output: [toolCall('main-observation-1')],
          })
        );
        return;
      }
      expect(wire.state_transition).toBe('continuation');
      expect(wire.input).toEqual([
        {
          type: 'tool_result',
          call_id: 'main-observation-1',
          output: 'new main-thought delta',
        },
      ]);
      expect(wire.tools).toEqual([]);
      if (generation === 2) {
        current.emit({
          event: 'failed',
          request_id: wire.request_id,
          phase: 'inference',
          error: 'synthetic auxiliary failure',
        });
      } else {
        current.emit(
          completed(wire, {
            stateSequenceLength: 14,
            cachedPrefixTokens: 10,
          })
        );
      }
    };

    await model.openState({ persistence: 'ephemeral' });
    const initial = await model.generate({
      input: [{ role: 'developer', content: 'memory system prompt' }],
      tools: [TOOL],
    });
    if (initial.responseToken === undefined) {
      throw new Error('ephemeral initial response has no live token');
    }
    const committed = model.state();
    expect(committed).toMatchObject({
      persistence: 'ephemeral',
      hasState: true,
      snapshotDirty: false,
      stateSequenceLength: 10,
    });
    const deltaRequest: ModelRequest = {
      input: [
        {
          type: 'tool_result',
          callId: 'main-observation-1',
          output: 'new main-thought delta',
        },
      ],
      tools: [TOOL],
      previousResponseToken: initial.responseToken,
    };
    await expect(model.generate(deltaRequest)).rejects.toThrow(
      'synthetic auxiliary failure'
    );
    expect(model.state()).toEqual(committed);
    await expect(model.generate(deltaRequest)).resolves.toMatchObject({
      output: [{ type: 'message', role: 'assistant', content: 'ok' }],
    });
    await expect(model.snapshot()).rejects.toThrow(
      'ephemeral state cannot be snapshotted'
    );
  });

  it('records the actual state transition on every native exchange event', async () => {
    const events: EchoEvent[] = [];
    const eventPort: EchoEventPort = {
      async emit(event): Promise<void> {
        events.push(event);
        await Promise.resolve();
      },
    };
    const { model, transport } = setupModel(undefined, eventPort);
    let generation = 0;
    transport.onSend = (wire, current): void => {
      if (wire.type === 'open_state') {
        current.emit({
          event: 'state_opened',
          request_id: wire.request_id,
          instance_id: wire.instance_id,
          persistence: 'durable',
          restored: false,
          current_path: '/state/rin/current.safetensors',
        });
      } else if (wire.type === 'generate') {
        generation += 1;
        current.emit(
          completed(
            wire,
            generation === 1 ? { output: [toolCall('call-1')] } : {}
          )
        );
      }
    };

    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });
    const initial = await model.generate(request('initial'));
    await model.generate({
      input: [{ type: 'tool_result', callId: 'call-1', output: 'done' }],
      tools: [TOOL],
      previousResponseToken: initial.responseToken,
    });
    await model.generate(request('new session'));

    expect(
      events
        .filter((event) => event.type === 'model.exchange.recorded')
        .map((event) => event.payload?.stateTransition)
    ).toEqual(['initial', 'continuation', 'new_session']);
  });

  it('rejects continuation after restart because no live token was issued', async () => {
    const { model, transport } = setupModel();
    transport.onSend = (wire, current): void => {
      if (wire.type === 'open_state') {
        current.emit({
          event: 'state_opened',
          request_id: wire.request_id,
          instance_id: wire.instance_id,
          persistence: 'durable',
          restored: true,
          current_path: '/state/rin/current.safetensors',
        });
      }
    };
    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });

    await expect(
      model.generate({
        ...request('tool'),
        previousResponseToken: 'old-process-token',
      })
    ).rejects.toThrow('issued by this live process');
    expect(transport.commands).toHaveLength(1);
  });

  it('keeps the EOS-closed state after a length result while surfacing incompleteness', async () => {
    const { model, transport } = setupModel();
    transport.onSend = autoResponder({
      finishReason: 'length',
      stateSequenceLength: 129,
    });
    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });

    const generation = model.generate(request('long'));
    await expect(generation).rejects.toBeInstanceOf(
      NativeInferenceIncompleteGenerationError
    );
    await expect(generation).rejects.toMatchObject({
      usage: { outputTokens: 1, totalTokens: 129 },
    });
    expect(model.state()).toMatchObject({
      hasState: true,
      snapshotDirty: true,
      stateSequenceLength: 129,
    });
  });

  it('rolls back adapter state when cancellation follows partial streaming', async () => {
    const { model, transport } = setupModel();
    transport.onSend = (wire, current): void => {
      if (wire.type === 'open_state') {
        current.emit({
          event: 'state_opened',
          request_id: wire.request_id,
          instance_id: wire.instance_id,
          persistence: 'durable',
          restored: false,
          current_path: '/state/rin/current.safetensors',
        });
      }
    };
    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });
    const before = model.state();
    const generation = model.generate(request('long'));
    await expect
      .poll(() => transport.commands[transport.commands.length - 1]?.type)
      .toBe('generate');
    const command = transport.commands[transport.commands.length - 1];
    if (command?.type !== 'generate') throw new Error('missing generation');
    transport.emit({
      event: 'token',
      request_id: command.request_id,
      index: 0,
      token_id: 10,
      text: 'a',
      terminal: false,
    });
    expect(await model.cancelActive()).toBe(true);
    transport.emit({
      event: 'cancelled',
      usage: {
        cached_prefix_tokens: 0,
        input_tokens_processed: 0,
        generated_tokens: 0,
      },
      request_id: command.request_id,
    });

    await expect(generation).rejects.toThrow('cancelled before commit');
    expect(model.state()).toEqual(before);
  });

  it('preserves the prior committed state when a new-session request fails', async () => {
    const { model, transport } = setupModel();
    let generation = 0;
    transport.onSend = (wire, current): void => {
      if (wire.type === 'open_state') {
        current.emit({
          event: 'state_opened',
          request_id: wire.request_id,
          instance_id: wire.instance_id,
          persistence: 'durable',
          restored: false,
          current_path: '/state/rin/current.safetensors',
        });
      } else if (wire.type === 'generate') {
        generation += 1;
        if (generation === 1) {
          current.emit(completed(wire, { stateSequenceLength: 8 }));
        } else {
          expect(wire.state_transition).toBe('new_session');
          current.emit({
            event: 'failed',
            request_id: wire.request_id,
            phase: 'inference',
            error: 'synthetic failure',
          });
        }
      }
    };
    await model.openState({
      persistence: 'durable',
      snapshotRoot: '/state/rin',
    });
    await model.generate(request('first'));
    const committed = model.state();

    await expect(model.generate(request('fresh'))).rejects.toThrow(
      'synthetic failure'
    );
    expect(model.state()).toEqual(committed);
  });
});

function setupModel(
  onToken?: () => void,
  events?: EchoEventPort
): {
  model: NativeInferenceModel;
  transport: FakeTransport;
} {
  const transport = new FakeTransport();
  const client = new NativeInferenceClient(transport);
  const model = new NativeInferenceModel({
    client,
    instanceId: 'rin',
    maxTokens: 128,
    seedSource: (): number => 42,
    ...(onToken === undefined ? {} : { onToken }),
    ...(events === undefined ? {} : { events }),
  });
  transport.ready();
  return { model, transport };
}

function request(content: string): ModelRequest {
  return {
    input: [{ role: 'user' as const, content }],
    tools: [TOOL],
  };
}

function toolCall(callId: string): NativeCompletedEvent['output'][number] {
  return {
    type: 'tool_call',
    call_id: callId,
    tool_name: TOOL.name,
    input: '{"session_record":{}}',
  };
}

function autoResponder(options: CompletedOptions = {}) {
  return (wire: NativeWireCommand, transport: FakeTransport): void => {
    if (wire.type === 'open_state') {
      transport.emit({
        event: 'state_opened',
        request_id: wire.request_id,
        instance_id: wire.instance_id,
        persistence: 'durable',
        restored: false,
        current_path: '/state/rin/current.safetensors',
      });
    } else if (wire.type === 'generate') {
      transport.emit(completed(wire, options));
    }
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
