import { describe, expect, it } from 'vitest';

import type {
  EchoEvent,
  EchoEventPort,
} from '@echo-chamber/core/ports/echo-event';
import type { ModelRequest } from '@echo-chamber/core/ports/model';

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

  ready(protocolVersion: number = NATIVE_INFERENCE_PROTOCOL_VERSION): void {
    this.emit({
      event: 'ready',
      protocol_version: protocolVersion,
      engine: { engine_id: 1 },
      eos_token_id: 248_046,
      chat_template_sha256: 'template',
      max_outstanding_requests: 8,
    });
  }
}

describe('NativeInferenceModel', () => {
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

    expect(await model.openState('/state/rin')).toMatchObject({
      stateOpened: true,
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

  it('enables diagnostic token events only when a listener is configured', async () => {
    const { model, transport } = setupModel(() => undefined);
    transport.onSend = (wire, current): void => {
      if (wire.type === 'open_state') {
        current.emit({
          event: 'state_opened',
          request_id: wire.request_id,
          instance_id: wire.instance_id,
          restored: false,
          current_path: '/state/rin/current.safetensors',
        });
      } else if (wire.type === 'generate') {
        expect(wire.stream_tokens).toBe(true);
        current.emit(completed(wire, { stateSequenceLength: 7 }));
      }
    };

    await model.openState('/state/rin');
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
          restored: true,
          current_path: '/state/rin/current.safetensors',
        });
      } else if (wire.type === 'generate') {
        expect(wire.state_transition).toBe('new_session');
        current.emit(completed(wire, { stateSequenceLength: 5 }));
      }
    };

    const restored = await model.openState('/state/rin');
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
          restored: false,
          current_path: '/state/rin/current.safetensors',
        });
        return;
      }
      if (wire.type !== 'generate') return;
      generation += 1;
      if (generation === 1) {
        expect(wire.state_transition).toBe('initial');
        current.emit(completed(wire, { stateSequenceLength: 10 }));
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

    await model.openState('/state/rin');
    await model.generate(request('hello'));
    await model.generate({
      previousResponseToken: 'the-value-is-deliberately-not-compared',
      input: [{ type: 'tool_result', callId: 'call-1', output: 'done' }],
      tools: [TOOL],
    });
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
    transport.onSend = autoResponder();

    await model.openState('/state/rin');
    const initial = await model.generate(request('initial'));
    await model.generate({
      ...request('continuation'),
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
          restored: true,
          current_path: '/state/rin/current.safetensors',
        });
      }
    };
    await model.openState('/state/rin');

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
    await model.openState('/state/rin');

    const generation = model.generate(request('long'));
    await expect(generation).rejects.toBeInstanceOf(
      NativeInferenceIncompleteGenerationError
    );
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
          restored: false,
          current_path: '/state/rin/current.safetensors',
        });
      }
    };
    await model.openState('/state/rin');
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
    transport.emit({ event: 'cancelled', request_id: command.request_id });

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
    await model.openState('/state/rin');
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

function autoResponder(options: CompletedOptions = {}) {
  return (wire: NativeWireCommand, transport: FakeTransport): void => {
    if (wire.type === 'open_state') {
      transport.emit({
        event: 'state_opened',
        request_id: wire.request_id,
        instance_id: wire.instance_id,
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
