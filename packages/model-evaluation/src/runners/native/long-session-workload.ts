import type {
  NativeCompletedEvent,
  NativeGenerateCommand,
} from '@echo-chamber/native-inference-adapter/protocol';

export const LONG_SESSION_MAX_NEW_TOKENS = 64;
export const LONG_SESSION_FINAL_TEXT = 'LONG_SESSION_COMPLETE';
export const LONG_SESSION_DEVELOPER_PROMPT = [
  'This is a deterministic long-session state-cache benchmark.',
  'Use only the advance_probe tool until the benchmark is complete.',
  'Begin by calling advance_probe with integer step 1.',
  'After each tool result, ignore its padding field and read its integer step field.',
  'If step is lower than the final_step supplied in that result, call advance_probe with step + 1.',
  `If step equals final_step, reply with exactly ${LONG_SESSION_FINAL_TEXT} and nothing else.`,
].join('\n');
export const LONG_SESSION_TOOL: NativeGenerateCommand['tools'][number] = {
  name: 'advance_probe',
  description: 'Advances one deterministic long-session benchmark step.',
  input_schema: {
    type: 'object',
    properties: {
      step: { type: 'integer' },
    },
    required: ['step'],
    additionalProperties: false,
  },
  strict: true,
};
export const LONG_SESSION_GREEDY_SAMPLING = {
  temperature: 0,
  top_p: 1,
  top_k: 0,
  min_p: 0,
  repetition_penalty: 1,
  presence_penalty: 0,
  seed: 42,
} as const;

export type LongSessionToolCall = Extract<
  NativeCompletedEvent['output'][number],
  { type: 'tool_call' }
>;

export function requireLongSessionStepToolCall(
  event: NativeCompletedEvent,
  step: number
): LongSessionToolCall {
  const call = event.output.find(
    (item): item is LongSessionToolCall => item.type === 'tool_call'
  );
  if (call === undefined || call.tool_name !== LONG_SESSION_TOOL.name) {
    throw new Error(
      `request ${event.request_id} omitted ${LONG_SESSION_TOOL.name}`
    );
  }
  const input: unknown = JSON.parse(call.input);
  if (
    typeof input !== 'object' ||
    input === null ||
    !('step' in input) ||
    input.step !== step
  ) {
    throw new Error(
      `expected ${LONG_SESSION_TOOL.name} step ${step}, observed ${JSON.stringify(call)}`
    );
  }
  return call;
}

export function serializeLongSessionToolResult(
  step: number,
  finalStep: number,
  padding: string
): string {
  return JSON.stringify({
    padding,
    step,
    final_step: finalStep,
  });
}

export function longSessionNativePrefixCommand(
  instanceId: string
): NativeGenerateCommand {
  return {
    type: 'generate',
    request_id: `${instanceId}-step-0`,
    instance_id: instanceId,
    state_transition: 'initial',
    stream_tokens: true,
    input: [{ role: 'developer', content: LONG_SESSION_DEVELOPER_PROMPT }],
    tools: [LONG_SESSION_TOOL],
    max_new_tokens: LONG_SESSION_MAX_NEW_TOKENS,
    sampling: LONG_SESSION_GREEDY_SAMPLING,
  };
}

export function longSessionNativeContinuationCommand(
  instanceId: string,
  previous: NativeCompletedEvent,
  step: number,
  result: string
): NativeGenerateCommand {
  const call = requireLongSessionStepToolCall(previous, step);
  return {
    type: 'generate',
    request_id: `${instanceId}-step-${step}`,
    instance_id: instanceId,
    state_transition: 'continuation',
    stream_tokens: true,
    input: [{ type: 'tool_result', call_id: call.call_id, output: result }],
    tools: [],
    max_new_tokens: LONG_SESSION_MAX_NEW_TOKENS,
    sampling: LONG_SESSION_GREEDY_SAMPLING,
  };
}

export function longSessionNativeReplayCommand(
  instanceId: string,
  chain: readonly NativeCompletedEvent[],
  results: readonly string[]
): NativeGenerateCommand {
  const input: NativeGenerateCommand['input'] = [
    { role: 'developer', content: LONG_SESSION_DEVELOPER_PROMPT },
  ];
  for (const [index, result] of results.entries()) {
    const step = index + 1;
    const event = chain[index];
    if (event === undefined) {
      throw new Error(`missing cached event for replay step ${step}`);
    }
    const call = requireLongSessionStepToolCall(event, step);
    input.push({
      type: 'tool_call',
      call_id: call.call_id,
      tool_name: call.tool_name,
      input: call.input,
    });
    input.push({ type: 'tool_result', call_id: call.call_id, output: result });
  }
  return {
    type: 'generate',
    request_id: `${instanceId}-full-replay`,
    instance_id: instanceId,
    state_transition: 'initial',
    stream_tokens: true,
    input,
    tools: [LONG_SESSION_TOOL],
    max_new_tokens: LONG_SESSION_MAX_NEW_TOKENS,
    sampling: LONG_SESSION_GREEDY_SAMPLING,
  };
}
