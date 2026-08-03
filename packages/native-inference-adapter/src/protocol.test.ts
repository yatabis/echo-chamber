import { describe, expect, it } from 'vitest';

import {
  parseNativeWireEvent,
  toModelOutputItem,
  toNativeWireInput,
  toNativeWireTool,
} from './protocol';

describe('native inference protocol mapping', () => {
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
        '{"event":"state_opened","request_id":"rin:open","instance_id":"rin","restored":"yes","current_path":"/state/rin/current.safetensors"}'
      )
    ).toThrow('restored must be boolean');
    expect(() =>
      parseNativeWireEvent(
        '{"event":"snapshot_published","request_id":"rin:snapshot","instance_id":"rin","path":"/state/rin/current.safetensors","physical_nbytes":-1}'
      )
    ).toThrow('physical_nbytes must be a nonnegative safe integer');
  });
});
