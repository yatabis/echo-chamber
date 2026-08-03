/* eslint-disable no-await-in-loop -- One resident GPU owner must execute state transitions in order. */

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  createReadStream,
  existsSync,
  mkdirSync,
  writeFileSync,
} from 'node:fs';
import { arch, cpus, platform, release, totalmem } from 'node:os';
import { dirname, join } from 'node:path';

import { expect, test } from 'vitest';

import { NativeInferenceClient } from '@echo-chamber/native-inference-adapter/native-inference-client';
import type {
  NativeCompletedEvent,
  NativeGenerateCommand,
  NativeRuntimeMetrics,
} from '@echo-chamber/native-inference-adapter/protocol';

import {
  EphemeralNativeStateRoots,
  hiddenClosingTokens,
} from './ephemeral-state-roots';

const LIVE_GATE_ENABLED =
  process.env.ECHO_NATIVE_STATEFUL_PERFORMANCE_GATE === '1';
const liveTest = LIVE_GATE_ENABLED ? test : test.skip;
const PREFIX_MAX_NEW_TOKENS = 64;
const CONTINUATION_MAX_NEW_TOKENS = 64;
const SWITCH_TTFT_TOLERANCE = 0.05;
const STATE_MEMORY_OVERHEAD_TOLERANCE = 1.25;
const DEVELOPER_PROMPT = [
  'This is a deterministic state-cache benchmark.',
  'Your entire first reply must be exactly the following function call, with no prefix or suffix:',
  '',
  '<tool_call>',
  '<function=lookup_probe_code>',
  '<parameter=key>',
  'echo_probe',
  '</parameter>',
  '</function>',
  '</tool_call>',
  '',
  'After its result arrives, do not call a tool again.',
  'Write the integers from 1 through 400 on separate lines.',
  'Every line must use exactly this format: 0001: stateful-native-performance',
  'Use four zero-padded digits, no preface, no explanation, and no closing text.',
].join('\n');
const TOOL_RESULT = '{"code":"7391"}';
const TOOL: NativeGenerateCommand['tools'][number] = {
  name: 'lookup_probe_code',
  description:
    'Returns the integration probe code for the requested lookup key.',
  input_schema: {
    type: 'object',
    properties: {
      key: { type: 'string' },
    },
    required: ['key'],
    additionalProperties: false,
  },
  strict: true,
};
const GREEDY_SAMPLING = {
  temperature: 0,
  top_p: 1,
  top_k: 0,
  min_p: 0,
  repetition_penalty: 1,
  presence_penalty: 0,
  seed: 42,
} as const;

interface GateConfig {
  nativeBinaryPath: string;
  modelDirectory: string;
  outputPath: string;
  measuredRuns: number;
}

type AttemptKind = 'prefix' | 'cached-continuation' | 'stateless-replay';

interface AttemptRecord {
  kind: AttemptKind;
  requestId: string;
  instanceId: string;
  engineId: number;
  stateSequenceLength: number;
  finishReason: NativeCompletedEvent['response']['finish_reason'];
  firstTokenEventMs: number;
  visibleTtftMs: number;
  totalMs: number;
  externalDecodeTokensPerSecond: number;
  streamedTokenEvents: number;
  outputSha256: string;
  metrics: NativeRuntimeMetrics;
}

interface TimedAttempt {
  event: NativeCompletedEvent;
  record: AttemptRecord;
}

interface RoundEvidence {
  index: number;
  direct: {
    prefix: AttemptRecord;
    continuation: AttemptRecord;
  };
  switched: {
    firstPrefix: AttemptRecord;
    secondPrefix: AttemptRecord;
    firstContinuation: AttemptRecord;
    secondContinuation: AttemptRecord;
  };
  statelessReplay: AttemptRecord;
}

interface AttemptSummary {
  count: number;
  medianVisibleTtftMs: number;
  medianInternalTtftMs: number;
  medianTotalMs: number;
  medianExternalDecodeTokensPerSecond: number;
  medianInputTokensProcessed: number;
  medianCachedPrefixTokens: number;
  medianCommittedStateLogicalNbytes: number;
}

type ToolCall = Extract<
  NativeCompletedEvent['output'][number],
  { type: 'tool_call' }
>;

function requiredEnvironmentVariable(name: string): string {
  const value = process.env[name];
  if (value === undefined || value.trim() === '') {
    throw new Error(`Required environment variable is missing: ${name}`);
  }
  return value;
}

function parsePositiveInteger(name: string, fallback: number): number {
  const raw = process.env[name];
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${name} must be a positive safe integer`);
  }
  return value;
}

function loadConfig(): GateConfig {
  return {
    nativeBinaryPath: requiredEnvironmentVariable(
      'ECHO_STATEFUL_NATIVE_INFERENCE_BIN'
    ),
    modelDirectory: requiredEnvironmentVariable('ECHO_STATEFUL_MODEL'),
    outputPath: requiredEnvironmentVariable('ECHO_STATEFUL_OUTPUT'),
    measuredRuns: parsePositiveInteger('ECHO_STATEFUL_MEASURED_RUNS', 5),
  };
}

function assertInputPaths(config: GateConfig): void {
  for (const [label, path] of [
    ['native binary', config.nativeBinaryPath],
    ['model directory', config.modelDirectory],
  ] as const) {
    if (!existsSync(path)) {
      throw new Error(`${label} does not exist: ${path}`);
    }
  }
}

function sha256Text(text: string): string {
  return createHash('sha256').update(text).digest('hex');
}

async function sha256File(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) {
    hash.update(chunk as Buffer);
  }
  return hash.digest('hex');
}

function writeResult(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(value, undefined, 2)}\n`, 'utf8');
}

function gitState(workingDirectory: string): Record<string, unknown> {
  const command = (args: string[]): string =>
    execFileSync('git', args, {
      cwd: workingDirectory,
      encoding: 'utf8',
    }).trim();
  return {
    commit: command(['rev-parse', 'HEAD']),
    dirtyPaths: command(['status', '--short'])
      .split('\n')
      .filter((line) => line !== ''),
  };
}

function median(values: readonly number[]): number {
  if (values.length === 0) {
    throw new Error('median requires at least one value');
  }
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  const upper = sorted[middle];
  if (upper === undefined) {
    throw new Error('median index is missing');
  }
  if (sorted.length % 2 === 1) {
    return upper;
  }
  const lower = sorted[middle - 1];
  if (lower === undefined) {
    throw new Error('median lower index is missing');
  }
  return (lower + upper) / 2;
}

function decodeRate(completionTokens: number, decodeWindowMs: number): number {
  if (completionTokens < 2 || decodeWindowMs <= 0) {
    throw new Error(
      'decode rate requires at least two tokens and positive time'
    );
  }
  return (completionTokens - 1) / (decodeWindowMs / 1_000);
}

async function runTimedAttempt(
  client: NativeInferenceClient,
  command: NativeGenerateCommand,
  kind: AttemptKind
): Promise<TimedAttempt> {
  const startedAt = performance.now();
  let firstTokenAt: number | undefined;
  let firstVisibleAt: number | undefined;
  let streamedTokenEvents = 0;
  const event = await client.generate(command, (token) => {
    const observedAt = performance.now();
    firstTokenAt ??= observedAt;
    if (token.text !== undefined && token.text !== '') {
      firstVisibleAt ??= observedAt;
    }
    streamedTokenEvents += 1;
  });
  const completedAt = performance.now();
  if (firstTokenAt === undefined || firstVisibleAt === undefined) {
    throw new Error(`${command.request_id} emitted no visible token`);
  }
  const decodeWindowMs = completedAt - firstVisibleAt;
  return {
    event,
    record: {
      kind,
      requestId: command.request_id,
      instanceId: command.instance_id,
      engineId: event.response.engine_id,
      stateSequenceLength: event.response.state_sequence_length,
      finishReason: event.response.finish_reason,
      firstTokenEventMs: firstTokenAt - startedAt,
      visibleTtftMs: firstVisibleAt - startedAt,
      totalMs: completedAt - startedAt,
      externalDecodeTokensPerSecond: decodeRate(
        event.response.metrics.generated_tokens,
        decodeWindowMs
      ),
      streamedTokenEvents,
      outputSha256: sha256Text(event.text),
      metrics: event.response.metrics,
    },
  };
}

function prefixCommand(
  instanceId: string,
  requestId: string
): NativeGenerateCommand {
  return {
    type: 'generate',
    request_id: requestId,
    instance_id: instanceId,
    state_transition: 'initial',
    stream_tokens: true,
    input: [{ role: 'developer', content: DEVELOPER_PROMPT }],
    tools: [TOOL],
    max_new_tokens: PREFIX_MAX_NEW_TOKENS,
    sampling: GREEDY_SAMPLING,
  };
}

function requireToolCall(prefix: NativeCompletedEvent): ToolCall {
  const toolCall = prefix.output.find(
    (item): item is ToolCall => item.type === 'tool_call'
  );
  if (toolCall === undefined) {
    throw new Error(
      `prefix did not produce a parsed tool call: ${JSON.stringify(prefix.output)}`
    );
  }
  const input: unknown = JSON.parse(toolCall.input);
  if (
    toolCall.tool_name !== TOOL.name ||
    typeof input !== 'object' ||
    input === null ||
    !('key' in input) ||
    input.key !== 'echo_probe'
  ) {
    throw new Error(
      `prefix produced an unexpected tool call: ${JSON.stringify(toolCall)}`
    );
  }
  return toolCall;
}

function continuationCommand(
  instanceId: string,
  requestId: string,
  prefix: NativeCompletedEvent
): NativeGenerateCommand {
  const toolCall = requireToolCall(prefix);
  return {
    type: 'generate',
    request_id: requestId,
    instance_id: instanceId,
    state_transition: 'continuation',
    stream_tokens: true,
    input: [
      {
        type: 'tool_result',
        call_id: toolCall.call_id,
        output: TOOL_RESULT,
      },
    ],
    tools: [],
    max_new_tokens: CONTINUATION_MAX_NEW_TOKENS,
    sampling: GREEDY_SAMPLING,
  };
}

function statelessReplayCommand(
  instanceId: string,
  requestId: string,
  prefix: NativeCompletedEvent
): NativeGenerateCommand {
  const toolCall = requireToolCall(prefix);
  return {
    type: 'generate',
    request_id: requestId,
    instance_id: instanceId,
    state_transition: 'initial',
    stream_tokens: true,
    input: [
      { role: 'developer', content: DEVELOPER_PROMPT },
      {
        type: 'tool_call',
        call_id: toolCall.call_id,
        tool_name: toolCall.tool_name,
        input: toolCall.input,
      },
      {
        type: 'tool_result',
        call_id: toolCall.call_id,
        output: TOOL_RESULT,
      },
    ],
    tools: [TOOL],
    max_new_tokens: CONTINUATION_MAX_NEW_TOKENS,
    sampling: GREEDY_SAMPLING,
  };
}

async function runPrefix(
  client: NativeInferenceClient,
  stateRoots: EphemeralNativeStateRoots,
  instanceId: string,
  requestId: string
): Promise<TimedAttempt> {
  await stateRoots.open(client, instanceId);
  const attempt = await runTimedAttempt(
    client,
    prefixCommand(instanceId, requestId),
    'prefix'
  );
  requireToolCall(attempt.event);
  const expectedSequenceLength =
    attempt.record.metrics.input_tokens_processed +
    attempt.record.metrics.generated_tokens +
    hiddenClosingTokens(attempt.event);
  if (attempt.record.stateSequenceLength !== expectedSequenceLength) {
    throw new Error(`${requestId} committed an unexpected state length`);
  }
  return attempt;
}

async function runContinuation(
  client: NativeInferenceClient,
  instanceId: string,
  requestId: string,
  prefix: NativeCompletedEvent
): Promise<TimedAttempt> {
  return await runTimedAttempt(
    client,
    continuationCommand(instanceId, requestId, prefix),
    'cached-continuation'
  );
}

async function runStatelessReplay(
  client: NativeInferenceClient,
  stateRoots: EphemeralNativeStateRoots,
  input: {
    instanceId: string;
    requestId: string;
    prefix: NativeCompletedEvent;
  }
): Promise<TimedAttempt> {
  await stateRoots.open(client, input.instanceId);
  return await runTimedAttempt(
    client,
    statelessReplayCommand(input.instanceId, input.requestId, input.prefix),
    'stateless-replay'
  );
}

function stateAdvanceMatches(
  base: AttemptRecord,
  continuation: AttemptRecord
): boolean {
  return (
    continuation.stateSequenceLength ===
    base.stateSequenceLength +
      continuation.metrics.input_tokens_processed +
      continuation.metrics.generated_tokens +
      (continuation.finishReason === 'length' ? 1 : 0)
  );
}

function summarize(attempts: readonly AttemptRecord[]): AttemptSummary {
  return {
    count: attempts.length,
    medianVisibleTtftMs: median(
      attempts.map((attempt) => attempt.visibleTtftMs)
    ),
    medianInternalTtftMs: median(
      attempts.map((attempt) => {
        const nanos = attempt.metrics.first_generated_token_nanos;
        if (nanos === undefined) {
          throw new Error(`${attempt.requestId} omitted internal TTFT`);
        }
        return nanos / 1_000_000;
      })
    ),
    medianTotalMs: median(attempts.map((attempt) => attempt.totalMs)),
    medianExternalDecodeTokensPerSecond: median(
      attempts.map((attempt) => attempt.externalDecodeTokensPerSecond)
    ),
    medianInputTokensProcessed: median(
      attempts.map((attempt) => attempt.metrics.input_tokens_processed)
    ),
    medianCachedPrefixTokens: median(
      attempts.map((attempt) => attempt.metrics.cached_prefix_tokens)
    ),
    medianCommittedStateLogicalNbytes: median(
      attempts.map((attempt) => attempt.metrics.committed_state_logical_nbytes)
    ),
  };
}

function numberField(value: unknown, field: string): number {
  if (typeof value !== 'object' || value === null) {
    throw new Error(`native ready engine omitted numeric ${field}`);
  }
  const record = value as Record<string, unknown>;
  const observed = record[field];
  if (typeof observed !== 'number') {
    throw new Error(`native ready engine omitted numeric ${field}`);
  }
  return observed;
}

liveTest(
  'measures cached continuation, instance switching, and resident-state memory',
  async () => {
    const config = loadConfig();
    assertInputPaths(config);
    mkdirSync(dirname(config.outputPath), { recursive: true });
    const nativeLibraryPath = requiredEnvironmentVariable(
      'ECHO_NATIVE_LIBRARY_PATH'
    );
    const result: Record<string, unknown> = {
      schemaVersion: 1,
      generatedAt: new Date().toISOString(),
      conditions: {
        prompt: DEVELOPER_PROMPT,
        promptSha256: sha256Text(DEVELOPER_PROMPT),
        toolResult: TOOL_RESULT,
        prefixMaxNewTokens: PREFIX_MAX_NEW_TOKENS,
        continuationMaxNewTokens: CONTINUATION_MAX_NEW_TOKENS,
        warmupRounds: 1,
        measuredRuns: config.measuredRuns,
        sampling: GREEDY_SAMPLING,
        concurrency: 1,
        switchPattern: [
          'direct-prefix',
          'direct-continuation',
          'switch-a-prefix',
          'switch-b-prefix',
          'switch-a-continuation',
          'switch-b-continuation',
          'stateless-replay',
        ],
      },
      machine: {
        platform: platform(),
        release: release(),
        architecture: arch(),
        cpu: cpus()[0]?.model ?? null,
        logicalCpuCount: cpus().length,
        unifiedMemoryBytes: totalmem(),
      },
      subjects: {
        echoChamber: gitState(process.cwd()),
        nativeBinaryPath: config.nativeBinaryPath,
        nativeBinarySha256: await sha256File(config.nativeBinaryPath),
        modelDirectory: config.modelDirectory,
        modelConfigSha256: await sha256File(
          join(config.modelDirectory, 'config.json')
        ),
        tokenizerConfigSha256: await sha256File(
          join(config.modelDirectory, 'tokenizer_config.json')
        ),
      },
      rounds: [],
    };
    writeResult(config.outputPath, result);

    const client = NativeInferenceClient.spawn({
      binaryPath: config.nativeBinaryPath,
      modelDirectory: config.modelDirectory,
      maxOutstandingRequests: 1,
      environment: {
        ...process.env,
        DYLD_LIBRARY_PATH: nativeLibraryPath,
      },
    });
    const stateRoots = new EphemeralNativeStateRoots();
    try {
      const readyStartedAt = performance.now();
      const ready = await client.ready();
      const readyElapsedMs = performance.now() - readyStartedAt;
      const readyEngineId = numberField(ready.engine, 'engine_id');
      result.engine = {
        readyElapsedMs,
        ready,
      };
      writeResult(config.outputPath, result);

      const warmupPrefix = await runPrefix(
        client,
        stateRoots,
        'stateful-warmup',
        'stateful-warmup-prefix'
      );
      const warmupContinuation = await runContinuation(
        client,
        'stateful-warmup',
        'stateful-warmup-continuation',
        warmupPrefix.event
      );
      result.warmup = {
        prefix: warmupPrefix.record,
        continuation: warmupContinuation.record,
      };
      writeResult(config.outputPath, result);

      const rounds: RoundEvidence[] = [];
      for (let index = 1; index <= config.measuredRuns; index += 1) {
        const directId = `stateful-direct-${index}`;
        const directPrefix = await runPrefix(
          client,
          stateRoots,
          directId,
          `${directId}-prefix`
        );
        const directContinuation = await runContinuation(
          client,
          directId,
          `${directId}-continuation`,
          directPrefix.event
        );

        const firstSwitchedId = `stateful-switch-a-${index}`;
        const secondSwitchedId = `stateful-switch-b-${index}`;
        const firstSwitchedPrefix = await runPrefix(
          client,
          stateRoots,
          firstSwitchedId,
          `${firstSwitchedId}-prefix`
        );
        const secondSwitchedPrefix = await runPrefix(
          client,
          stateRoots,
          secondSwitchedId,
          `${secondSwitchedId}-prefix`
        );
        const firstSwitchedContinuation = await runContinuation(
          client,
          firstSwitchedId,
          `${firstSwitchedId}-continuation`,
          firstSwitchedPrefix.event
        );
        const secondSwitchedContinuation = await runContinuation(
          client,
          secondSwitchedId,
          `${secondSwitchedId}-continuation`,
          secondSwitchedPrefix.event
        );

        const replayId = `stateful-replay-${index}`;
        const statelessReplay = await runStatelessReplay(client, stateRoots, {
          instanceId: replayId,
          requestId: `${replayId}-full-history`,
          prefix: directPrefix.event,
        });
        rounds.push({
          index,
          direct: {
            prefix: directPrefix.record,
            continuation: directContinuation.record,
          },
          switched: {
            firstPrefix: firstSwitchedPrefix.record,
            secondPrefix: secondSwitchedPrefix.record,
            firstContinuation: firstSwitchedContinuation.record,
            secondContinuation: secondSwitchedContinuation.record,
          },
          statelessReplay: statelessReplay.record,
        });
        result.rounds = rounds;
        writeResult(config.outputPath, result);
      }

      const directContinuations = rounds.map(
        (round) => round.direct.continuation
      );
      const switchedContinuations = rounds.flatMap((round) => [
        round.switched.firstContinuation,
        round.switched.secondContinuation,
      ]);
      const cachedContinuations = [
        ...directContinuations,
        ...switchedContinuations,
      ];
      const statelessReplays = rounds.map((round) => round.statelessReplay);
      const directSummary = summarize(directContinuations);
      const switchedSummary = summarize(switchedContinuations);
      const cachedSummary = summarize(cachedContinuations);
      const statelessSummary = summarize(statelessReplays);
      const allAttempts = rounds.flatMap((round) => [
        round.direct.prefix,
        round.direct.continuation,
        round.switched.firstPrefix,
        round.switched.secondPrefix,
        round.switched.firstContinuation,
        round.switched.secondContinuation,
        round.statelessReplay,
      ]);
      const allEngineIds = [
        warmupPrefix.event.response.engine_id,
        warmupContinuation.event.response.engine_id,
        ...allAttempts.map((attempt) => attempt.engineId),
      ];
      const completionHashes = cachedContinuations.map(
        (attempt) => attempt.outputSha256
      );
      const replayHashes = statelessReplays.map(
        (attempt) => attempt.outputSha256
      );
      const expectedOutputHash = completionHashes[0];
      if (expectedOutputHash === undefined) {
        throw new Error('stateful benchmark produced no continuation');
      }
      const switchTtftRatio =
        switchedSummary.medianVisibleTtftMs / directSummary.medianVisibleTtftMs;
      const cachedVsReplayTtftRatio =
        cachedSummary.medianVisibleTtftMs /
        statelessSummary.medianVisibleTtftMs;
      const warmupMemory = warmupContinuation.record.metrics.metal_memory;
      const warmupActive = warmupMemory.active_nbytes;
      const finalAttempt = statelessReplays[statelessReplays.length - 1];
      if (finalAttempt === undefined) {
        throw new Error('stateful benchmark produced no final attempt');
      }
      const finalMemory = finalAttempt.metrics.metal_memory;
      const measuredInstanceCount = config.measuredRuns * 4;
      const activeGrowthNbytes = finalMemory.active_nbytes - warmupActive;
      const activeGrowthPerMeasuredInstance =
        activeGrowthNbytes / measuredInstanceCount;
      const cacheGrowthNbytes =
        finalMemory.cache_nbytes - warmupMemory.cache_nbytes;
      const allocatorFootprintGrowthNbytes =
        activeGrowthNbytes + cacheGrowthNbytes;
      const allocatorFootprintGrowthPerMeasuredInstance =
        allocatorFootprintGrowthNbytes / measuredInstanceCount;
      const medianCommittedStateLogicalNbytes =
        cachedSummary.medianCommittedStateLogicalNbytes;
      const checks = {
        oneResidentEngine:
          allEngineIds.length === allAttempts.length + 2 &&
          allEngineIds.every((engineId) => engineId === readyEngineId),
        everyCachedContinuationReusedCurrentState: rounds.every(
          (round) =>
            round.direct.continuation.metrics.cached_prefix_tokens ===
              round.direct.prefix.stateSequenceLength &&
            round.switched.firstContinuation.metrics.cached_prefix_tokens ===
              round.switched.firstPrefix.stateSequenceLength &&
            round.switched.secondContinuation.metrics.cached_prefix_tokens ===
              round.switched.secondPrefix.stateSequenceLength
        ),
        everyStatefulRequestAdvancedCurrentState: rounds.every(
          (round) =>
            stateAdvanceMatches(
              round.direct.prefix,
              round.direct.continuation
            ) &&
            stateAdvanceMatches(
              round.switched.firstPrefix,
              round.switched.firstContinuation
            ) &&
            stateAdvanceMatches(
              round.switched.secondPrefix,
              round.switched.secondContinuation
            )
        ),
        statelessReplayProcessedTheSameLogicalContext: rounds.every(
          (round) =>
            round.statelessReplay.metrics.input_tokens_processed ===
            round.direct.prefix.stateSequenceLength +
              round.direct.continuation.metrics.input_tokens_processed
        ),
        fixedContinuationLength: [
          ...cachedContinuations,
          ...statelessReplays,
        ].every(
          (attempt) =>
            attempt.metrics.generated_tokens === CONTINUATION_MAX_NEW_TOKENS
        ),
        greedyOutputsMatch:
          completionHashes.every((hash) => hash === expectedOutputHash) &&
          replayHashes.every((hash) => hash === expectedOutputHash),
        switchedTtftWithinFivePercent:
          switchTtftRatio <= 1 + SWITCH_TTFT_TOLERANCE,
        cachedTtftFasterThanStatelessReplay: cachedVsReplayTtftRatio < 1,
        residentStateMemoryWithinTwentyFivePercent:
          activeGrowthPerMeasuredInstance <=
          medianCommittedStateLogicalNbytes * STATE_MEMORY_OVERHEAD_TOLERANCE,
      };
      result.summary = {
        directContinuation: directSummary,
        switchedContinuation: switchedSummary,
        allCachedContinuations: cachedSummary,
        statelessReplay: statelessSummary,
      };
      result.comparison = {
        switchTtftRatio,
        cachedVsReplayTtftRatio,
        expectedOutputHash,
        checks,
      };
      result.memory = {
        warmup: warmupMemory,
        final: finalMemory,
        measuredInstanceCount,
        activeGrowthNbytes,
        activeGrowthPerMeasuredInstance,
        cacheGrowthNbytes,
        allocatorFootprintGrowthNbytes,
        allocatorFootprintGrowthPerMeasuredInstance,
        medianCommittedStateLogicalNbytes,
        activeGrowthOverLogicalStateRatio:
          activeGrowthPerMeasuredInstance / medianCommittedStateLogicalNbytes,
        allocatorFootprintGrowthOverLogicalStateRatio:
          allocatorFootprintGrowthPerMeasuredInstance /
          medianCommittedStateLogicalNbytes,
      };
      writeResult(config.outputPath, result);

      expect(checks).toEqual({
        oneResidentEngine: true,
        everyCachedContinuationReusedCurrentState: true,
        everyStatefulRequestAdvancedCurrentState: true,
        statelessReplayProcessedTheSameLogicalContext: true,
        fixedContinuationLength: true,
        greedyOutputsMatch: true,
        switchedTtftWithinFivePercent: true,
        cachedTtftFasterThanStatelessReplay: true,
        residentStateMemoryWithinTwentyFivePercent: true,
      });
    } finally {
      try {
        await client.shutdown();
      } finally {
        stateRoots.dispose();
      }
    }
  },
  30 * 60_000
);
