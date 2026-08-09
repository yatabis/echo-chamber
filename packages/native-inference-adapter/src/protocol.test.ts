import { describe, expect, it } from 'vitest';

import {
  type NativeCompletedEvent,
  type NativeMetalMemoryStats,
  NATIVE_INFERENCE_PROTOCOL_VERSION,
  parseNativeWireEvent,
  toModelOutputItem,
  toNativeWireInput,
  toNativeWireTool,
} from './protocol';

describe('native inference protocol mapping', () => {
  it('admits the bounded continuous-batch ready contract', () => {
    expect(parseNativeWireEvent(JSON.stringify(readyEvent()))).toMatchObject({
      event: 'ready',
      max_new_tokens_per_request: 4_096,
      max_active_batch_size: 6,
      max_late_join_batch_size: 4,
    });
  });

  it.each([1, 2])(
    'admits ready batch widths bounded by an outstanding limit of %i',
    (maxOutstandingRequests) => {
      expect(
        parseNativeWireEvent(
          JSON.stringify({
            ...readyEvent(),
            max_outstanding_requests: maxOutstandingRequests,
            max_active_batch_size: maxOutstandingRequests,
            max_late_join_batch_size: maxOutstandingRequests,
          })
        )
      ).toMatchObject({
        max_outstanding_requests: maxOutstandingRequests,
        max_active_batch_size: maxOutstandingRequests,
        max_late_join_batch_size: maxOutstandingRequests,
      });
    }
  );

  it('distinguishes durable and process-local state owners', () => {
    expect(
      parseNativeWireEvent(
        '{"event":"state_opened","request_id":"rin:open","instance_id":"rin","persistence":"durable","restored":true,"current_path":"/state/rin/current.safetensors"}'
      )
    ).toMatchObject({ persistence: 'durable', restored: true });
    expect(
      parseNativeWireEvent(
        '{"event":"state_opened","request_id":"rin:memory:open","instance_id":"rin.memory","persistence":"ephemeral","restored":false}'
      )
    ).toMatchObject({ persistence: 'ephemeral', restored: false });
  });

  it('maps core camel-case request types to the Rust wire', () => {
    expect(
      toNativeWireInput({
        type: 'tool_call',
        callId: 'call-1',
        toolName: 'read_notes',
        input: '{"limit":2}',
      })
    ).toEqual({
      type: 'tool_call',
      call_id: 'call-1',
      tool_name: 'read_notes',
      input: '{"limit":2}',
    });
    expect(
      toNativeWireTool({
        name: 'read_notes',
        description: 'Read notes',
        inputSchema: { type: 'object' },
      })
    ).toEqual({
      name: 'read_notes',
      description: 'Read notes',
      input_schema: { type: 'object' },
      strict: false,
    });
  });

  it('maps parsed native tool output back to ModelOutputItem', () => {
    expect(
      toModelOutputItem({
        type: 'tool_call',
        call_id: 'rin:1:tool:1',
        tool_name: 'finish_thinking',
        input: '{"session_record":null}',
      })
    ).toEqual({
      type: 'tool_call',
      callId: 'rin:1:tool:1',
      toolName: 'finish_thinking',
      input: '{"session_record":null}',
    });
  });

  it('admits explicitly unavailable nullable runtime observations', () => {
    const completed = completedEvent();
    expect(
      parseNativeWireEvent(
        JSON.stringify({
          ...completed,
          response: {
            ...completed.response,
            metrics: {
              ...completed.response.metrics,
              first_generated_token_nanos: null,
              metal_memory: null,
            },
          },
        })
      )
    ).toMatchObject({
      event: 'completed',
      response: {
        metrics: { first_generated_token_nanos: null, metal_memory: null },
      },
    });
  });

  it.each([
    {
      event: 'queued',
      request_id: 'rin:queued',
      outstanding_requests: 1,
    },
    { event: 'started', request_id: 'rin:started', prompt_tokens: 42 },
    {
      event: 'token',
      request_id: 'rin:token',
      index: 0,
      token_id: 42,
      text: 'x',
      terminal: false,
    },
    {
      event: 'token',
      request_id: 'rin:terminal-token',
      index: 1,
      token_id: 248_046,
      terminal: true,
    },
    {
      event: 'cancel_acknowledged',
      request_id: 'rin:cancel',
      accepted: true,
    },
    { event: 'cancelled', request_id: 'rin:cancelled' },
    {
      event: 'failed',
      request_id: 'rin:failed',
      phase: 'inference',
      error: 'failed safely',
    },
    { event: 'failed', phase: 'startup', error: 'failed safely' },
    { event: 'shutdown' },
  ])('admits a complete $event event', (event) => {
    expect(parseNativeWireEvent(JSON.stringify(event))).toEqual(event);
  });

  it('admits a complete parsed tool-call output and warning', () => {
    expect(
      parseNativeWireEvent(
        JSON.stringify({
          ...completedEvent(),
          output: [
            {
              type: 'tool_call',
              call_id: 'rin:1:tool:1',
              tool_name: 'finish_thinking',
              input: '{"session_record":null}',
            },
          ],
          tool_parse_warning: 'diagnostic warning',
        })
      )
    ).toMatchObject({
      output: [{ type: 'tool_call', tool_name: 'finish_thinking' }],
      tool_parse_warning: 'diagnostic warning',
    });
  });

  it.each([
    {
      label: 'queued outstanding_requests',
      event: { event: 'queued', request_id: 'rin:queued' },
      error: 'outstanding_requests',
    },
    {
      label: 'started prompt_tokens',
      event: { event: 'started', request_id: 'rin:started' },
      error: 'prompt_tokens',
    },
    {
      label: 'token index',
      event: {
        event: 'token',
        request_id: 'rin:token',
        token_id: 42,
        terminal: false,
      },
      error: 'index',
    },
    {
      label: 'token token_id',
      event: {
        event: 'token',
        request_id: 'rin:token',
        index: 0,
        terminal: false,
      },
      error: 'token_id',
    },
    {
      label: 'token optional text',
      event: {
        event: 'token',
        request_id: 'rin:token',
        index: 0,
        token_id: 42,
        text: 42,
        terminal: false,
      },
      error: 'text',
    },
    {
      label: 'token terminal',
      event: {
        event: 'token',
        request_id: 'rin:token',
        index: 0,
        token_id: 42,
      },
      error: 'terminal',
    },
    {
      label: 'cancel acknowledgement',
      event: { event: 'cancel_acknowledged', request_id: 'rin:cancel' },
      error: 'accepted',
    },
    {
      label: 'optional failed request ID',
      event: {
        event: 'failed',
        request_id: 42,
        phase: 'inference',
        error: 'failed safely',
      },
      error: 'request_id',
    },
  ])('rejects an invalid $label field', ({ event, error }) => {
    expect(() => parseNativeWireEvent(JSON.stringify(event))).toThrow(error);
  });

  it.each([
    ['engine_id', { engine_id: 'one' }],
    ['model', { model: null }],
    ['generated_tokens', { generated_tokens: [1, -1] }],
    ['finish_reason', { finish_reason: 'cancelled' }],
  ])('rejects an invalid completed response.%s field', (field, override) => {
    const completed = completedEvent();
    expect(() =>
      parseNativeWireEvent(
        JSON.stringify({
          ...completed,
          response: { ...completed.response, ...override },
        })
      )
    ).toThrow(field);
  });

  it.each([
    ['text', { text: 42 }],
    ['tool_parse_warning', { tool_parse_warning: false }],
    [
      'output',
      {
        output: [{ type: 'message', role: 'user', content: 'wrong role' }],
      },
    ],
    [
      'output',
      {
        output: [
          {
            type: 'tool_call',
            call_id: 'rin:1:tool:1',
            tool_name: 'finish_thinking',
          },
        ],
      },
    ],
  ])('rejects an invalid completed %s field', (field, override) => {
    expect(() =>
      parseNativeWireEvent(JSON.stringify({ ...completedEvent(), ...override }))
    ).toThrow(field);
  });

  it.each([
    'queue_wait_nanos',
    'cached_prefix_tokens',
    'input_tokens_processed',
    'generated_tokens',
    'model_step_count',
    'input_model_execution_count',
    'input_execution_nanos',
    'input_graph_construction_nanos',
    'input_materialization_nanos',
    'decode_execution_nanos',
    'decode_graph_construction_nanos',
    'decode_schedule_nanos',
    'decode_token_wait_nanos',
    'decode_finalization_nanos',
    'model_execution_nanos',
    'request_nanos',
    'committed_state_logical_nbytes',
  ])('rejects an invalid completed metrics.%s field', (field) => {
    const completed = completedEvent();
    expect(() =>
      parseNativeWireEvent(
        JSON.stringify({
          ...completed,
          response: {
            ...completed.response,
            metrics: { ...completed.response.metrics, [field]: -1 },
          },
        })
      )
    ).toThrow(field);
  });

  it('rejects invalid nullable first-token timing and Metal counters', () => {
    const completed = completedEvent();
    expect(() =>
      parseNativeWireEvent(
        JSON.stringify({
          ...completed,
          response: {
            ...completed.response,
            metrics: {
              ...completed.response.metrics,
              first_generated_token_nanos: 'soon',
            },
          },
        })
      )
    ).toThrow('first_generated_token_nanos');
    expect(() =>
      parseNativeWireEvent(
        JSON.stringify({
          ...completed,
          response: {
            ...completed.response,
            metrics: {
              ...completed.response.metrics,
              metal_memory: {
                ...metalMemory(),
                cache_nbytes: -1,
              },
            },
          },
        })
      )
    ).toThrow('cache_nbytes');
  });

  it('rejects malformed or unknown native event envelopes', () => {
    expect(() => parseNativeWireEvent('{"event":"unknown"}')).toThrow(
      'unsupported native inference event'
    );
    expect(() =>
      parseNativeWireEvent(
        '{"event":"completed","request_id":"rin:1","output":[]}'
      )
    ).toThrow('missing response/output');
    const completed = completedEvent();
    expect(() =>
      parseNativeWireEvent(
        JSON.stringify({
          ...completed,
          response: {
            ...completed.response,
            metrics: {
              ...completed.response.metrics,
              metal_memory: undefined,
            },
          },
        })
      )
    ).toThrow('metrics.metal_memory must be an object or null');
    expect(() =>
      parseNativeWireEvent(
        '{"event":"state_opened","request_id":"rin:open","instance_id":"rin","persistence":"durable","restored":"yes","current_path":"/state/rin/current.safetensors"}'
      )
    ).toThrow('restored must be boolean');
    expect(() =>
      parseNativeWireEvent(
        '{"event":"state_opened","request_id":"rin:memory:open","instance_id":"rin.memory","persistence":"ephemeral","restored":false,"current_path":"/state/rin.memory/current.safetensors"}'
      )
    ).toThrow('ephemeral state_opened event cannot restore');
    expect(() =>
      parseNativeWireEvent(
        '{"event":"snapshot_published","request_id":"rin:snapshot","instance_id":"rin","path":"/state/rin/current.safetensors","physical_nbytes":-1}'
      )
    ).toThrow('physical_nbytes must be a nonnegative safe integer');
    expect(() =>
      parseNativeWireEvent(
        `{"event":"ready","protocol_version":${NATIVE_INFERENCE_PROTOCOL_VERSION},"engine":{},"eos_token_id":248046,"chat_template_sha256":"template","max_new_tokens_per_request":4096,"max_outstanding_requests":8,"max_active_batch_size":6,"max_late_join_batch_size":7}`
      )
    ).toThrow('max_late_join_batch_size exceeds max_active_batch_size');
    expect(() =>
      parseNativeWireEvent(
        `{"event":"ready","protocol_version":${NATIVE_INFERENCE_PROTOCOL_VERSION},"engine":{},"eos_token_id":248046,"chat_template_sha256":"template","max_outstanding_requests":8,"max_active_batch_size":6,"max_late_join_batch_size":4}`
      )
    ).toThrow('field max_new_tokens_per_request');
  });
});

function readyEvent(): Record<string, unknown> {
  return {
    event: 'ready',
    protocol_version: NATIVE_INFERENCE_PROTOCOL_VERSION,
    engine: {
      engine_id: 1,
      model: modelIdentity(),
      weight_shard_count: 16,
      weight_tensor_count: 320,
      model_load_nanos: 1_000,
      new_session_gdn_policy: 'carry_all',
      metal_memory: metalMemory(),
    },
    eos_token_id: 248_046,
    chat_template_sha256: 'template',
    max_new_tokens_per_request: 4_096,
    max_outstanding_requests: 8,
    max_active_batch_size: 6,
    max_late_join_batch_size: 4,
  };
}

function completedEvent(): NativeCompletedEvent {
  return {
    event: 'completed',
    request_id: 'rin:1',
    response: {
      engine_id: 1,
      instance_id: 'rin',
      model: modelIdentity(),
      state_sequence_length: 11,
      generated_tokens: [42],
      finish_reason: 'stop_token',
      metrics: {
        queue_wait_nanos: 1,
        cached_prefix_tokens: 8,
        input_tokens_processed: 2,
        generated_tokens: 1,
        maximum_decode_batch_size: 1,
        decode_batch_membership_changes: 0,
        model_step_count: 3,
        input_model_execution_count: 1,
        input_execution_nanos: 10,
        input_graph_construction_nanos: 2,
        input_materialization_nanos: 8,
        first_generated_token_nanos: 12,
        decode_execution_nanos: 20,
        decode_graph_construction_nanos: 3,
        decode_schedule_nanos: 4,
        decode_token_wait_nanos: 5,
        decode_finalization_nanos: 6,
        model_execution_nanos: 30,
        request_nanos: 32,
        committed_state_logical_nbytes: 4_096,
        metal_memory: metalMemory(),
      },
    },
    text: 'done',
    output: [{ type: 'message', role: 'assistant', content: 'done' }],
  };
}

function modelIdentity(): Record<string, string> {
  return {
    architecture: 'qwen3_5_moe',
    config_digest: 'config',
    weights_digest: 'weights',
    tokenizer_digest: 'tokenizer',
    template_digest: 'template',
  };
}

function metalMemory(): NativeMetalMemoryStats {
  return { active_nbytes: 1_024, cache_nbytes: 512, peak_nbytes: 2_048 };
}
