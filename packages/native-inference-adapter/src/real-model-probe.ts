import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';

import type { ModelOutputItem } from '@echo-chamber/core/ports/model';

import { NativeInferenceClient } from './native-inference-client';
import { NativeInferenceModel } from './native-inference-model';

const TOOL = {
  name: 'lookup_probe_code',
  description:
    'Returns the integration probe code for the requested lookup key.',
  inputSchema: {
    type: 'object',
    properties: {
      key: { type: 'string' },
    },
    required: ['key'],
    additionalProperties: false,
  },
  strict: true,
};

const NEXT_SESSION_TOOL = {
  ...TOOL,
  description:
    'Returns the second-session integration probe code for the requested lookup key.',
};

const GREEDY_SAMPLING = {
  temperature: 0,
  top_p: 0,
  top_k: 0,
  min_p: 0,
  repetition_penalty: 1,
  presence_penalty: 0,
} as const;

const [binaryPath, modelDirectory, firstSeedArgument, profileArgument] =
  process.argv.slice(2);
if (binaryPath === undefined || modelDirectory === undefined) {
  throw new Error(
    'usage: pnpm probe:real-model <echo-inference-binary> <model-directory> [first-seed] [production|greedy]'
  );
}
const firstSeed = parseSeed(firstSeedArgument);
const greedy = parseProfile(profileArgument);
const nativeLibraryPath = process.env.ECHO_NATIVE_LIBRARY_PATH;
const nativeEnvironment =
  nativeLibraryPath === undefined
    ? process.env
    : { ...process.env, DYLD_LIBRARY_PATH: nativeLibraryPath };

const client = NativeInferenceClient.spawn({
  binaryPath,
  modelDirectory,
  maxOutstandingRequests: 2,
  environment: nativeEnvironment,
});
const ready = await client.ready();
const streamedTokens = [0, 0, 0, 0];
let activeTurn = 0;
const seeds = [firstSeed, firstSeed + 1, firstSeed + 2, firstSeed + 3] as const;
let seedIndex = 0;
const model = new NativeInferenceModel({
  client,
  instanceId: 'echo-native-real-probe',
  maxTokens: 128,
  ...(greedy ? { sampling: GREEDY_SAMPLING } : {}),
  seedSource: nextSeed,
  onToken: (): void => {
    const current = streamedTokens[activeTurn];
    if (current === undefined) {
      throw new Error(`unexpected streamed-token request index ${activeTurn}`);
    }
    streamedTokens[activeTurn] = current + 1;
  },
});
const stateRoot = await mkdtemp(join(tmpdir(), 'echo-native-real-probe-'));

try {
  await model.openState({
    persistence: 'durable',
    snapshotRoot: stateRoot,
  });
  const firstStarted = performance.now();
  const first = await model.generate({
    input: [
      {
        role: 'developer',
        content:
          'For this transport probe, your entire first reply must be exactly the following function call, with no prefix or suffix:\n\n<tool_call>\n<function=lookup_probe_code>\n<parameter=key>\necho_probe\n</parameter>\n</function>\n</tool_call>\n\nAfter its result arrives, reply with only the returned code and do not call another tool.',
      },
    ],
    tools: [TOOL],
    turnIndex: 1,
  });
  const firstElapsedMilliseconds = performance.now() - firstStarted;
  const firstState = model.state();
  const toolCall = requireToolCall(first.output, 'first session');
  const toolInput = parseToolInput(toolCall.input);
  if (toolCall.toolName !== TOOL.name || toolInput.key !== 'echo_probe') {
    throw new Error(
      `unexpected real-model tool call: ${JSON.stringify(toolCall)}`
    );
  }
  if (first.responseToken === undefined) {
    throw new Error('initial native request did not return a response token');
  }

  activeTurn = 1;
  const secondStarted = performance.now();
  const second = await model.generate({
    input: [
      {
        type: 'tool_result',
        callId: toolCall.callId,
        output: '{"code":"7391"}',
      },
    ],
    tools: [TOOL],
    previousResponseToken: first.responseToken,
    turnIndex: 2,
  });
  const secondElapsedMilliseconds = performance.now() - secondStarted;
  const secondState = model.state();
  const secondText = second.output
    .filter(
      (item): item is Extract<ModelOutputItem, { type: 'message' }> =>
        item.type === 'message'
    )
    .map((item) => item.content)
    .join('\n');
  if (!secondText.includes('7391')) {
    throw new Error(
      `real-model continuation did not consume the tool result: ${JSON.stringify(
        second.output
      )}`
    );
  }
  if (
    second.usage.cachedInputTokens !==
    requireSequenceLength(firstState, 'first session')
  ) {
    throw new Error(
      `native continuation reused ${second.usage.cachedInputTokens} tokens; expected ${String(firstState.stateSequenceLength)} resident tokens`
    );
  }

  activeTurn = 2;
  const newSessionStarted = performance.now();
  const newSession = await model.generate({
    input: [
      {
        role: 'developer',
        content:
          'This is a new thinking session. Your entire first reply must be exactly the following function call, with no prefix or suffix:\n\n<tool_call>\n<function=lookup_probe_code>\n<parameter=key>\necho_new_session\n</parameter>\n</function>\n</tool_call>\n\nAfter its result arrives, reply with only the returned code and do not call another tool.',
      },
    ],
    tools: [NEXT_SESSION_TOOL],
    turnIndex: 1,
  });
  const newSessionElapsedMilliseconds = performance.now() - newSessionStarted;
  const newSessionState = model.state();
  const newSessionToolCall = requireToolCall(newSession.output, 'new session');
  const newSessionToolInput = parseToolInput(newSessionToolCall.input);
  if (
    newSessionToolCall.toolName !== NEXT_SESSION_TOOL.name ||
    newSessionToolInput.key !== 'echo_new_session'
  ) {
    throw new Error(
      `unexpected new-session tool call: ${JSON.stringify(newSessionToolCall)}`
    );
  }
  if (newSession.responseToken === undefined) {
    throw new Error('native new session did not return a response token');
  }
  if (newSession.usage.cachedInputTokens !== 0) {
    throw new Error(
      `native new session unexpectedly reused ${newSession.usage.cachedInputTokens} old-lineage tokens`
    );
  }

  activeTurn = 3;
  const newSessionContinuationStarted = performance.now();
  const newSessionContinuation = await model.generate({
    input: [
      {
        type: 'tool_result',
        callId: newSessionToolCall.callId,
        output: '{"code":"8642"}',
      },
    ],
    tools: [NEXT_SESSION_TOOL],
    previousResponseToken: newSession.responseToken,
    turnIndex: 2,
  });
  const newSessionContinuationElapsedMilliseconds =
    performance.now() - newSessionContinuationStarted;
  const newSessionContinuationState = model.state();
  const newSessionContinuationText = newSessionContinuation.output
    .filter(
      (item): item is Extract<ModelOutputItem, { type: 'message' }> =>
        item.type === 'message'
    )
    .map((item) => item.content)
    .join('\n');
  if (!newSessionContinuationText.includes('8642')) {
    throw new Error(
      `new-session continuation did not consume the tool result: ${JSON.stringify(
        newSessionContinuation.output
      )}`
    );
  }
  if (
    newSessionContinuation.usage.cachedInputTokens !==
    requireSequenceLength(newSessionState, 'new session')
  ) {
    throw new Error(
      `new-session continuation reused ${newSessionContinuation.usage.cachedInputTokens} tokens; expected ${String(newSessionState.stateSequenceLength)} resident tokens`
    );
  }

  console.log(
    JSON.stringify(
      {
        schemaVersion: 3,
        protocolVersion: ready.protocol_version,
        engineId: ready.engine.engine_id,
        modelDirectory,
        samplingProfile: greedy ? 'greedy' : 'production',
        samplingSeeds: seeds,
        first: {
          elapsedMilliseconds: firstElapsedMilliseconds,
          streamedTokens: streamedTokens[0],
          output: first.output,
          usage: first.usage,
          state: firstState,
        },
        second: {
          elapsedMilliseconds: secondElapsedMilliseconds,
          streamedTokens: streamedTokens[1],
          output: second.output,
          usage: second.usage,
          state: secondState,
        },
        newSession: {
          elapsedMilliseconds: newSessionElapsedMilliseconds,
          streamedTokens: streamedTokens[2],
          output: newSession.output,
          usage: newSession.usage,
          state: newSessionState,
        },
        newSessionContinuation: {
          elapsedMilliseconds: newSessionContinuationElapsedMilliseconds,
          streamedTokens: streamedTokens[3],
          output: newSessionContinuation.output,
          usage: newSessionContinuation.usage,
          state: newSessionContinuationState,
        },
        checks: {
          qwenToolCallParsed: true,
          toolResultConsumed: true,
          exactResidentPrefixReused:
            second.usage.cachedInputTokens === firstState.stateSequenceLength,
          firstSessionStateAdvanced:
            requireSequenceLength(secondState, 'second session') >
            requireSequenceLength(firstState, 'first session'),
          newSessionStartedWithoutOldPrefix: true,
          newSessionToolCatalogAccepted: true,
          newSessionToolResultConsumed: true,
          newSessionLineageReusedExactly:
            newSessionContinuation.usage.cachedInputTokens ===
            newSessionState.stateSequenceLength,
          newSessionStateAdvanced:
            requireSequenceLength(
              newSessionContinuationState,
              'new-session continuation'
            ) > requireSequenceLength(newSessionState, 'new session'),
        },
      },
      undefined,
      2
    )
  );
} finally {
  await client.shutdown();
  await rm(stateRoot, { recursive: true, force: true });
}

function requireSequenceLength(
  state: ReturnType<NativeInferenceModel['state']>,
  phase: string
): number {
  if (state.stateSequenceLength === undefined) {
    throw new Error(`${phase} has no committed state sequence length`);
  }
  return state.stateSequenceLength;
}

function nextSeed(): number {
  const seed = seeds[seedIndex];
  if (seed === undefined) {
    throw new Error('real-model probe requested more than four sampling seeds');
  }
  seedIndex += 1;
  return seed;
}

function parseSeed(value: string | undefined): number {
  const seed = value === undefined ? 42 : Number(value);
  if (
    !Number.isSafeInteger(seed) ||
    seed < 0 ||
    seed >= Number.MAX_SAFE_INTEGER
  ) {
    throw new Error(`first-seed must be a non-negative safe integer: ${value}`);
  }
  return seed;
}

function parseProfile(value: string | undefined): boolean {
  if (value === undefined || value === 'production') {
    return false;
  }
  if (value === 'greedy') {
    return true;
  }
  throw new Error(`sampling profile must be production or greedy: ${value}`);
}

function requireToolCall(
  output: readonly ModelOutputItem[],
  phase: string
): Extract<ModelOutputItem, { type: 'tool_call' }> {
  const toolCall = output.find(
    (item): item is Extract<ModelOutputItem, { type: 'tool_call' }> =>
      item.type === 'tool_call'
  );
  if (toolCall === undefined) {
    throw new Error(
      `real-model ${phase} did not produce a parsed tool call: ${JSON.stringify(
        output
      )}`
    );
  }
  return toolCall;
}

function parseToolInput(input: string): { key?: unknown } {
  const parsed: unknown = JSON.parse(input);
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error(`real-model tool input is not an object: ${input}`);
  }
  return parsed as { key?: unknown };
}
