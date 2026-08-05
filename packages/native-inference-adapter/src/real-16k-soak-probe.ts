import { performance } from 'node:perf_hooks';

import { NativeInferenceClient } from './native-inference-client';

import type { NativeCompletedEvent, NativeGenerateCommand } from './protocol';

const [binaryPath, modelDirectory] = process.argv.slice(2);
if (binaryPath === undefined || modelDirectory === undefined) {
  throw new Error(
    'usage: pnpm probe:16k-soak <echo-inference-binary> <model-directory>'
  );
}

const CONTEXT_WORDS = parsePositiveInteger(
  process.env.ECHO_NATIVE_SOAK_CONTEXT_WORDS,
  15_800
);
const MEASURED_CYCLES = parsePositiveInteger(
  process.env.ECHO_NATIVE_SOAK_CYCLES,
  3
);
const MAX_NEW_TOKENS = parsePositiveInteger(
  process.env.ECHO_NATIVE_SOAK_MAX_NEW_TOKENS,
  64
);
const WIDTH_PATTERN = [3, 4, 5, 6] as const;
const TOOL_NAME = 'publish_soak_update';
const INSTANCE_IDS = Array.from(
  { length: 6 },
  (_, index) => `soak-16k-${index + 1}`
);
const PRODUCTION_SAMPLING = {
  temperature: 0.7,
  top_p: 0.8,
  top_k: 20,
  min_p: 0,
  repetition_penalty: 1,
  presence_penalty: 1.5,
} as const;
type NativeToolCallOutput = Extract<
  NativeCompletedEvent['output'][number],
  { type: 'tool_call' }
>;
const TOOL_CONTRACT: NativeGenerateCommand['tools'] = [
  {
    name: TOOL_NAME,
    description:
      '指定されたsummaryを状態更新として保存します。保存を求められた場合は必ず一度だけ呼び出してください。',
    input_schema: {
      type: 'object',
      properties: { summary: { type: 'string' } },
      required: ['summary'],
      additionalProperties: false,
    },
    strict: true,
  },
];
const nativeLibraryPath = process.env.ECHO_NATIVE_LIBRARY_PATH;
const client = NativeInferenceClient.spawn({
  binaryPath,
  modelDirectory,
  maxOutstandingRequests: 8,
  environment:
    nativeLibraryPath === undefined
      ? process.env
      : { ...process.env, DYLD_LIBRARY_PATH: nativeLibraryPath },
});

try {
  const ready = await client.ready();
  const opened = await Promise.all(
    INSTANCE_IDS.map(
      async (instanceId, index) =>
        await client.openState({
          type: 'open_state',
          request_id: `soak:open:${index + 1}`,
          instance_id: instanceId,
          persistence: 'ephemeral',
        })
    )
  );
  const longContext = `${' context'.repeat(CONTEXT_WORDS)}\n\n要求された状態更新は利用可能なツールで実行し、summaryを改変しないでください。説明文は出力しないでください。ツールの実行結果には次の観測が含まれます。`;
  const prefill = await timedCohort(
    INSTANCE_IDS.map((instanceId, index) =>
      generation({
        requestId: `soak:prefill:${index + 1}`,
        instanceId,
        stateTransition: 'initial',
        input: [
          { role: 'system', content: longContext },
          {
            role: 'user',
            content: toolInstruction(initialSummary(instanceId)),
          },
        ],
        tools: TOOL_CONTRACT,
        maxNewTokens: MAX_NEW_TOKENS,
        seed: 201_000 + index,
      })
    )
  );
  const committedLengths = sequenceLengths(prefill.events);
  const pendingToolCalls = pendingCalls(prefill.events, (instanceId) =>
    initialSummary(instanceId)
  );
  const initialInputLengths = prefill.events.map(
    (event) => event.response.metrics.input_tokens_processed
  );

  const warmup = await runContinuationRound({
    round: -1,
    instanceIds: [requireArrayItem(INSTANCE_IDS, 0)],
    committedLengths,
    pendingToolCalls,
    maxNewTokens: MAX_NEW_TOKENS,
  });
  const widthOne = await runContinuationRound({
    round: 0,
    instanceIds: [requireArrayItem(INSTANCE_IDS, 1)],
    committedLengths,
    pendingToolCalls,
    maxNewTokens: MAX_NEW_TOKENS,
  });

  const rounds: RoundResult[] = [];
  let selectionOffset = 2;
  for (let cycle = 0; cycle < MEASURED_CYCLES; cycle += 1) {
    for (const width of WIDTH_PATTERN) {
      const selected = Array.from({ length: width }, (_, index) =>
        requireArrayItem(INSTANCE_IDS, (selectionOffset + index) % 6)
      );
      const round = rounds.length + 1;
      // Each round depends on the preceding committed state per lane.
      // eslint-disable-next-line no-await-in-loop
      const result = await runContinuationRound({
        round,
        instanceIds: selected,
        committedLengths,
        pendingToolCalls,
        maxNewTokens: MAX_NEW_TOKENS,
      });
      rounds.push(result);
      selectionOffset = (selectionOffset + width) % 6;
    }
  }

  const measuredEvents = rounds.flatMap((round) => round.events);
  const allEvents = [
    ...prefill.events,
    ...warmup.events,
    ...widthOne.events,
    ...measuredEvents,
  ];
  const finalLengths = [...committedLengths.values()];
  const byWidth = WIDTH_PATTERN.map((width) => {
    const matching = rounds.filter((round) => round.width === width);
    return {
      width,
      rounds: matching.length,
      medianAggregateTokensPerSecond: median(
        matching.map((round) => round.aggregateTokensPerSecond)
      ),
      medianCompletionMilliseconds: median(
        matching.map((round) => round.elapsedMilliseconds)
      ),
      medianGeneratedTokensPerRound: median(
        matching.map((round) => round.generatedTokens)
      ),
      medianFirstTokenMilliseconds: median(
        matching.flatMap((round) =>
          round.events.map((event) =>
            nanosToMilliseconds(
              event.response.metrics.first_generated_token_nanos ?? 0
            )
          )
        )
      ),
    };
  });
  const checks = {
    protocolVersion: ready.protocol_version === 9,
    allOwnersAreEphemeral: opened.every(
      (event) =>
        event.persistence === 'ephemeral' &&
        !event.restored &&
        event.current_path === undefined
    ),
    initialContextsAreNear16k: initialInputLengths.every(
      (tokens) => tokens >= 15_000 && tokens <= 17_000
    ),
    chunkedPrefillWasUsed: prefill.events.every(
      (event) => event.response.metrics.input_model_execution_count > 1
    ),
    prefillDecodeUsedWidthSix: prefill.events.every(
      (event) => event.response.metrics.maximum_decode_batch_size === 6
    ),
    prefillStateExact: prefill.events.every((event) =>
      stateAccountingIsExact(event, 0)
    ),
    prefillToolCallsAreExact: prefill.events.every((event) =>
      hasExpectedToolCall(event, initialSummary(event.response.instance_id))
    ),
    warmupStayedWidthOne: warmup.observedWidthIsExact,
    measuredWidthOneStayedWidthOne: widthOne.observedWidthIsExact,
    everyVariableRoundUsedRequestedWidth: rounds.every(
      (round) => round.observedWidthIsExact
    ),
    everyContinuationReusedItsOwnPrefix: [warmup, widthOne, ...rounds].every(
      (round) => round.cachedPrefixesExact
    ),
    everyCommittedLengthIsExact: [warmup, widthOne, ...rounds].every(
      (round) => round.stateLengthsExact
    ),
    allSixLanesAdvanced: INSTANCE_IDS.every(
      (instanceId) =>
        requireSequenceLength(committedLengths, instanceId) >
        requireSequenceLength(sequenceLengths(prefill.events), instanceId)
    ),
    residentLengthsRemainNear16k: finalLengths.every(
      (tokens) => tokens >= 15_000 && tokens <= 20_000
    ),
    everyContinuationProducedAnExactToolCall: [
      warmup,
      widthOne,
      ...rounds,
    ].every((round) => round.toolCallsExact),
  };

  const generatedTokens = measuredEvents.reduce(
    (total, event) => total + event.response.metrics.generated_tokens,
    0
  );
  const measuredElapsedMilliseconds = rounds.reduce(
    (total, round) => total + round.elapsedMilliseconds,
    0
  );
  const report = {
    schemaVersion: 1,
    protocolVersion: ready.protocol_version,
    configuration: {
      contextWords: CONTEXT_WORDS,
      measuredCycles: MEASURED_CYCLES,
      widthPattern: WIDTH_PATTERN,
      maxNewTokens: MAX_NEW_TOKENS,
      sampling: PRODUCTION_SAMPLING,
    },
    prefill: {
      elapsedMilliseconds: prefill.elapsedMilliseconds,
      inputTokensPerLane: initialInputLengths,
      modelExecutionsPerLane: prefill.events.map(
        (event) => event.response.metrics.input_model_execution_count
      ),
      stateSequenceLengths: prefill.events.map(
        (event) => event.response.state_sequence_length
      ),
    },
    widthOne: summarizeRound(widthOne),
    variableWidth: {
      roundCount: rounds.length,
      generatedTokens,
      elapsedMilliseconds: measuredElapsedMilliseconds,
      aggregateTokensPerSecond:
        generatedTokens / (measuredElapsedMilliseconds / 1_000),
      byWidth,
      rounds: rounds.map(summarizeRound),
    },
    residency: {
      finalStateSequenceLengths: finalLengths,
      minimumFinalStateSequenceLength: Math.min(...finalLengths),
      maximumFinalStateSequenceLength: Math.max(...finalLengths),
      maximumActiveMetalBytes: Math.max(
        ...allEvents.map(
          (event) => event.response.metrics.metal_memory.active_nbytes
        )
      ),
      maximumCachedMetalBytes: Math.max(
        ...allEvents.map(
          (event) => event.response.metrics.metal_memory.cache_nbytes
        )
      ),
      maximumObservedMetalBytes: Math.max(
        ...allEvents.map(
          (event) => event.response.metrics.metal_memory.peak_nbytes
        )
      ),
    },
    checks,
  };
  console.log(JSON.stringify(report, undefined, 2));
  requirePassingChecks(checks);
} finally {
  await client.shutdown();
}

interface RoundResult {
  round: number;
  width: number;
  instanceIds: string[];
  events: NativeCompletedEvent[];
  elapsedMilliseconds: number;
  generatedTokens: number;
  aggregateTokensPerSecond: number;
  observedWidthIsExact: boolean;
  cachedPrefixesExact: boolean;
  stateLengthsExact: boolean;
  toolCallsExact: boolean;
}

async function runContinuationRound(input: {
  round: number;
  instanceIds: string[];
  committedLengths: Map<string, number>;
  pendingToolCalls: Map<string, NativeToolCallOutput>;
  maxNewTokens: number;
}): Promise<RoundResult> {
  const expectedPrefixes = new Map(
    input.instanceIds.map((instanceId) => [
      instanceId,
      requireSequenceLength(input.committedLengths, instanceId),
    ])
  );
  const cohort = await timedCohort(
    input.instanceIds.map((instanceId, index) =>
      generation({
        requestId: `soak:round:${input.round}:${index + 1}`,
        instanceId,
        stateTransition: 'continuation',
        input: [
          {
            type: 'tool_result',
            call_id: requirePendingCall(input.pendingToolCalls, instanceId)
              .call_id,
            output: `${JSON.stringify({
              stored: true,
              next_observation: { round: input.round, instance: instanceId },
            })}\n保存に成功し、次の観測が届きました。${toolInstruction(
              continuationSummary(instanceId, input.round)
            )}`,
          },
        ],
        maxNewTokens: input.maxNewTokens,
        seed: 202_000 + (input.round + 1) * 100 + index,
      })
    )
  );
  const generatedTokens = cohort.events.reduce(
    (total, event) => total + event.response.metrics.generated_tokens,
    0
  );
  const observedWidthIsExact = cohort.events.every(
    (event) =>
      event.response.metrics.maximum_decode_batch_size ===
      input.instanceIds.length
  );
  const cachedPrefixesExact = cohort.events.every(
    (event) =>
      event.response.metrics.cached_prefix_tokens ===
      requireSequenceLength(expectedPrefixes, event.response.instance_id)
  );
  const stateLengthsExact = cohort.events.every((event) =>
    stateAccountingIsExact(
      event,
      requireSequenceLength(expectedPrefixes, event.response.instance_id)
    )
  );
  const toolCallsExact = cohort.events.every((event) =>
    hasExpectedToolCall(
      event,
      continuationSummary(event.response.instance_id, input.round)
    )
  );
  for (const event of cohort.events) {
    input.committedLengths.set(
      event.response.instance_id,
      event.response.state_sequence_length
    );
    input.pendingToolCalls.set(
      event.response.instance_id,
      requireToolCall(
        event,
        continuationSummary(event.response.instance_id, input.round)
      )
    );
  }
  return {
    round: input.round,
    width: input.instanceIds.length,
    instanceIds: input.instanceIds,
    events: cohort.events,
    elapsedMilliseconds: cohort.elapsedMilliseconds,
    generatedTokens,
    aggregateTokensPerSecond:
      generatedTokens / (cohort.elapsedMilliseconds / 1_000),
    observedWidthIsExact,
    cachedPrefixesExact,
    stateLengthsExact,
    toolCallsExact,
  };
}

function generation(input: {
  requestId: string;
  instanceId: string;
  stateTransition: NativeGenerateCommand['state_transition'];
  input: NativeGenerateCommand['input'];
  tools?: NativeGenerateCommand['tools'];
  maxNewTokens: number;
  seed: number;
}): NativeGenerateCommand {
  return {
    type: 'generate',
    request_id: input.requestId,
    instance_id: input.instanceId,
    state_transition: input.stateTransition,
    stream_tokens: false,
    input: input.input,
    tools: input.tools ?? [],
    max_new_tokens: input.maxNewTokens,
    sampling: { ...PRODUCTION_SAMPLING, seed: input.seed },
  };
}

async function timedCohort(
  commands: readonly NativeGenerateCommand[]
): Promise<{ events: NativeCompletedEvent[]; elapsedMilliseconds: number }> {
  const started = performance.now();
  const events = await Promise.all(
    commands.map(async (command) => await client.generate(command))
  );
  return {
    events,
    elapsedMilliseconds: performance.now() - started,
  };
}

function stateAccountingIsExact(
  event: NativeCompletedEvent,
  cachedPrefix: number
): boolean {
  const metrics = event.response.metrics;
  const hiddenClosingToken = event.response.finish_reason === 'length' ? 1 : 0;
  return (
    event.response.state_sequence_length ===
    cachedPrefix +
      metrics.input_tokens_processed +
      metrics.generated_tokens +
      hiddenClosingToken
  );
}

function sequenceLengths(
  events: readonly NativeCompletedEvent[]
): Map<string, number> {
  return new Map(
    events.map((event) => [
      event.response.instance_id,
      event.response.state_sequence_length,
    ])
  );
}

function initialSummary(instanceId: string): string {
  return `${instanceId}-initial`;
}

function continuationSummary(instanceId: string, round: number): string {
  return `${instanceId}-round-${round}`;
}

function toolInstruction(summary: string): string {
  return `${TOOL_NAME}を一度だけ呼び出し、summaryを${summary}として保存してください。説明文は不要です。`;
}

function pendingCalls(
  events: readonly NativeCompletedEvent[],
  expectedSummary: (instanceId: string) => string
): Map<string, NativeToolCallOutput> {
  return new Map(
    events.map((event) => [
      event.response.instance_id,
      requireToolCall(event, expectedSummary(event.response.instance_id)),
    ])
  );
}

function requirePendingCall(
  calls: ReadonlyMap<string, NativeToolCallOutput>,
  instanceId: string
): NativeToolCallOutput {
  const call = calls.get(instanceId);
  if (call === undefined) {
    throw new Error(`missing pending tool call for ${instanceId}`);
  }
  return call;
}

function requireToolCall(
  event: NativeCompletedEvent,
  expectedSummary: string
): NativeToolCallOutput {
  const item = event.output[0];
  if (
    !hasExpectedToolCall(event, expectedSummary) ||
    item?.type !== 'tool_call'
  ) {
    throw new Error(
      `generation for ${event.response.instance_id} did not produce the exact tool call: ${JSON.stringify({ finishReason: event.response.finish_reason, generatedTokens: event.response.metrics.generated_tokens, output: event.output, warning: event.tool_parse_warning })}`
    );
  }
  return item;
}

function hasExpectedToolCall(
  event: NativeCompletedEvent,
  expectedSummary: string
): boolean {
  if (
    event.response.finish_reason !== 'stop_token' ||
    event.tool_parse_warning !== undefined ||
    event.output.length !== 1
  ) {
    return false;
  }
  const [item] = event.output;
  if (item?.type !== 'tool_call' || item.tool_name !== TOOL_NAME) {
    return false;
  }
  return parseOnlySummary(item.input) === expectedSummary;
}

function parseOnlySummary(input: string): string | undefined {
  try {
    const parsed: unknown = JSON.parse(input);
    if (
      typeof parsed !== 'object' ||
      parsed === null ||
      Array.isArray(parsed) ||
      Object.keys(parsed).length !== 1
    ) {
      return undefined;
    }
    const summary = (parsed as Record<string, unknown>).summary;
    return typeof summary === 'string' ? summary : undefined;
  } catch {
    return undefined;
  }
}

function requireSequenceLength(
  lengths: ReadonlyMap<string, number>,
  instanceId: string
): number {
  const length = lengths.get(instanceId);
  if (length === undefined) {
    throw new Error(`missing sequence length for ${instanceId}`);
  }
  return length;
}

function summarizeRound(round: RoundResult): Record<string, unknown> {
  return {
    round: round.round,
    width: round.width,
    instanceIds: round.instanceIds,
    elapsedMilliseconds: round.elapsedMilliseconds,
    generatedTokens: round.generatedTokens,
    aggregateTokensPerSecond: round.aggregateTokensPerSecond,
    generatedTokensPerRow: round.events.map(
      (event) => event.response.metrics.generated_tokens
    ),
    finishReasons: round.events.map((event) => event.response.finish_reason),
    membershipChanges: round.events.map(
      (event) => event.response.metrics.decode_batch_membership_changes
    ),
    firstTokenMilliseconds: round.events.map((event) =>
      nanosToMilliseconds(
        event.response.metrics.first_generated_token_nanos ?? 0
      )
    ),
  };
}

function nanosToMilliseconds(nanos: number): number {
  return nanos / 1_000_000;
}

function median(values: readonly number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const midpoint = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return requireArrayItem(sorted, midpoint);
  return (
    (requireArrayItem(sorted, midpoint - 1) +
      requireArrayItem(sorted, midpoint)) /
    2
  );
}

function requireArrayItem<T>(values: readonly T[], index: number): T {
  const value = values[index];
  if (value === undefined) {
    throw new Error(`missing array item at index ${index}`);
  }
  return value;
}

function parsePositiveInteger(
  value: string | undefined,
  fallback: number
): number {
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) {
    throw new Error(`expected a positive safe integer, observed ${value}`);
  }
  return parsed;
}

function requirePassingChecks(checks: Record<string, boolean>): void {
  const failed = Object.entries(checks)
    .filter(([, passed]) => !passed)
    .map(([name]) => name);
  if (failed.length > 0) {
    throw new Error(`16K soak probe failed: ${failed.join(', ')}`);
  }
}
