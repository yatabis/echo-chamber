import { describe, expect, it } from 'vitest';

import {
  NATIVE_INFERENCE_PROTOCOL_VERSION,
  parseNativeWireEvent,
  toModelOutputItem,
  toNativeWireInput,
  toNativeWireTool,
} from './protocol';

describe('native inference protocol mapping', () => {
  it('admits the bounded continuous-batch ready contract', () => {
    expect(
      parseNativeWireEvent(
        JSON.stringify({
          event: 'ready',
          protocol_version: NATIVE_INFERENCE_PROTOCOL_VERSION,
          engine: { engine_id: 1 },
          eos_token_id: 248_046,
          chat_template_sha256: 'template',
          max_outstanding_requests: 8,
          max_active_batch_size: 6,
          max_late_join_batch_size: 4,
        })
      )
    ).toMatchObject({
      event: 'ready',
      max_active_batch_size: 6,
      max_late_join_batch_size: 4,
    });
  });

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

  it('rejects malformed or unknown native event envelopes', () => {
    expect(() => parseNativeWireEvent('{"event":"unknown"}')).toThrow(
      'unsupported native inference event'
    );
    expect(() =>
      parseNativeWireEvent(
        '{"event":"completed","request_id":"rin:1","output":[]}'
      )
    ).toThrow('missing response/output');
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
        '{"event":"ready","protocol_version":9,"engine":{},"eos_token_id":248046,"chat_template_sha256":"template","max_outstanding_requests":8,"max_active_batch_size":6,"max_late_join_batch_size":7}'
      )
    ).toThrow('max_late_join_batch_size exceeds max_active_batch_size');
  });
});
