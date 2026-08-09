/* eslint-disable no-await-in-loop -- One resident GPU owner must run context shapes sequentially. */

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
import {
  requireNativeMetalMemory,
  type NativeGenerateCommand,
  type NativeRuntimeMetrics,
} from '@echo-chamber/native-inference-adapter/protocol';

import { EphemeralNativeStateRoots } from './ephemeral-state-roots';

const LIVE_GATE_ENABLED = process.env.ECHO_NATIVE_CONTEXT_CURVE_GATE === '1';
const liveTest = LIVE_GATE_ENABLED ? test : test.skip;
const DEFAULT_TARGET_CONTEXT_TOKENS = [2_048, 8_192, 16_384, 32_768];
const MAX_NEW_TOKENS = 128;
const PROMPT_BASE_TOKENS = 108;
const PADDING_PHRASE_TOKENS = 3;
const PADDING_PHRASE = ' state-cache-padding';
const REMAINDER_TOKEN = ' x';
const BENCHMARK_PREFIX = [
  'This is a deterministic context-length performance benchmark.',
  'Ignore every padding token between <padding> and </padding>.',
  'After </padding>, write the integers from 1 through 400 on separate lines.',
  'Every line must use exactly this format: 0001: context-curve-performance',
  'Use four zero-padded digits, no preface, no explanation, and no closing text.',
  '<padding>',
].join('\n');
const BENCHMARK_SUFFIX = ['</padding>', 'Begin the numbered output now.'].join(
  '\n'
);
const GREEDY_SAMPLING = {
  temperature: 0,
  top_p: 1,
  top_k: 0,
  min_p: 0,
  repetition_penalty: 1,
  presence_penalty: 0,
  seed: 42,
} as const;

type AttemptPhase = 'warmup' | 'measured';

interface GateConfig {
  nativeBinaryPath: string;
  modelDirectory: string;
  outputPath: string;
  targetContextTokens: number[];
  warmupRuns: number;
  measuredRuns: number;
}

interface PaddingPlan {
  repetitions: number;
  remainderTokens: number;
}

interface AttemptRecord {
  phase: AttemptPhase;
  round: number;
  attemptIndexWithinTarget: number;
  targetContextTokens: number;
  promptSha256: string;
  stateTransition: NativeGenerateCommand['state_transition'];
  totalMs: number;
  outputSha256: string;
  stateSequenceLength: number;
  finishReason: 'length' | 'stop_token';
  metrics: NativeRuntimeMetrics;
  derived: {
    internalFirstGeneratedTokenMs: number;
    inputExecutionMs: number;
    inputGraphConstructionMs: number;
    inputMaterializationMs: number;
    decodeExecutionMs: number;
    generatedTokensPerDecodeSecond: number;
    requestMs: number;
    metalActiveGrowthFromPreviousAttemptNbytes: number | null;
  };
}

interface TargetSummary {
  targetContextTokens: number;
  measuredRuns: number;
  medianTotalMs: number;
  medianInternalFirstGeneratedTokenMs: number;
  medianInputExecutionMs: number;
  medianDecodeExecutionMs: number;
  medianGeneratedTokensPerDecodeSecond: number;
  medianCommittedStateLogicalNbytes: number;
  medianMetalActiveNbytes: number;
  medianMetalActiveGrowthPerRetainedInstanceNbytes: number;
  medianMetalCacheNbytes: number;
  maximumMetalPeakNbytes: number;
  decodeRateOverShortestContext: number;
  firstGeneratedTokenTimeOverShortestContext: number;
}

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

function parseTargetContextTokens(value: string | undefined): number[] {
  const targets =
    value === undefined
      ? [...DEFAULT_TARGET_CONTEXT_TOKENS]
      : value.split(',').map((item) => Number(item.trim()));
  if (
    targets.length === 0 ||
    targets.some(
      (target) => !Number.isSafeInteger(target) || target <= PROMPT_BASE_TOKENS
    ) ||
    new Set(targets).size !== targets.length
  ) {
    throw new Error(
      'ECHO_CONTEXT_CURVE_TARGETS must contain unique comma-separated integers greater than 108'
    );
  }
  return targets.sort((left, right) => left - right);
}

function loadConfig(): GateConfig {
  return {
    nativeBinaryPath: requiredEnvironmentVariable(
      'ECHO_CONTEXT_CURVE_NATIVE_INFERENCE_BIN'
    ),
    modelDirectory: requiredEnvironmentVariable('ECHO_CONTEXT_CURVE_MODEL'),
    outputPath: requiredEnvironmentVariable('ECHO_CONTEXT_CURVE_OUTPUT'),
    targetContextTokens: parseTargetContextTokens(
      process.env.ECHO_CONTEXT_CURVE_TARGETS
    ),
    warmupRuns: parsePositiveInteger('ECHO_CONTEXT_CURVE_WARMUP_RUNS', 1),
    measuredRuns: parsePositiveInteger('ECHO_CONTEXT_CURVE_MEASURED_RUNS', 3),
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

function paddingPlan(targetContextTokens: number): PaddingPlan {
  const paddingTokens = targetContextTokens - PROMPT_BASE_TOKENS;
  return {
    repetitions: Math.floor(paddingTokens / PADDING_PHRASE_TOKENS),
    remainderTokens: paddingTokens % PADDING_PHRASE_TOKENS,
  };
}

function benchmarkPrompt(targetContextTokens: number): string {
  const plan = paddingPlan(targetContextTokens);
  return `${BENCHMARK_PREFIX}${PADDING_PHRASE.repeat(plan.repetitions)}${REMAINDER_TOKEN.repeat(plan.remainderTokens)}\n${BENCHMARK_SUFFIX}`;
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

function writeResult(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(value, undefined, 2)}\n`, 'utf8');
}

function commandForAttempt(
  instanceId: string,
  attemptIdentity: string,
  targetContextTokens: number
): NativeGenerateCommand {
  return {
    type: 'generate',
    request_id: `context-curve-${targetContextTokens}-${attemptIdentity}`,
    instance_id: instanceId,
    state_transition: 'initial',
    stream_tokens: false,
    input: [{ role: 'user', content: benchmarkPrompt(targetContextTokens) }],
    tools: [],
    max_new_tokens: MAX_NEW_TOKENS,
    sampling: GREEDY_SAMPLING,
  };
}

function nanosToMilliseconds(nanos: number): number {
  return nanos / 1_000_000;
}

async function runAttempt(
  client: NativeInferenceClient,
  command: NativeGenerateCommand,
  context: {
    phase: AttemptPhase;
    round: number;
    attemptIndexWithinTarget: number;
    targetContextTokens: number;
    previousMetalActiveNbytes: number | null;
  }
): Promise<AttemptRecord> {
  const startedAt = performance.now();
  const event = await client.generate(command);
  const totalMs = performance.now() - startedAt;
  const metrics = event.response.metrics;
  const metalMemory = requireNativeMetalMemory(metrics, command.request_id);
  const firstGeneratedTokenNanos = metrics.first_generated_token_nanos;
  if (firstGeneratedTokenNanos === null) {
    throw new Error(`${command.request_id} omitted first-token timing`);
  }
  return {
    phase: context.phase,
    round: context.round,
    attemptIndexWithinTarget: context.attemptIndexWithinTarget,
    targetContextTokens: context.targetContextTokens,
    promptSha256: sha256Text(benchmarkPrompt(context.targetContextTokens)),
    stateTransition: command.state_transition,
    totalMs,
    outputSha256: sha256Text(event.text),
    stateSequenceLength: event.response.state_sequence_length,
    finishReason: event.response.finish_reason,
    metrics,
    derived: {
      internalFirstGeneratedTokenMs: nanosToMilliseconds(
        firstGeneratedTokenNanos
      ),
      inputExecutionMs: nanosToMilliseconds(metrics.input_execution_nanos),
      inputGraphConstructionMs: nanosToMilliseconds(
        metrics.input_graph_construction_nanos
      ),
      inputMaterializationMs: nanosToMilliseconds(
        metrics.input_materialization_nanos
      ),
      decodeExecutionMs: nanosToMilliseconds(metrics.decode_execution_nanos),
      generatedTokensPerDecodeSecond:
        metrics.generated_tokens / (metrics.decode_execution_nanos / 1e9),
      requestMs: nanosToMilliseconds(metrics.request_nanos),
      metalActiveGrowthFromPreviousAttemptNbytes:
        context.previousMetalActiveNbytes === null
          ? null
          : metalMemory.active_nbytes - context.previousMetalActiveNbytes,
    },
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

function summarizeTarget(
  targetContextTokens: number,
  attempts: readonly AttemptRecord[],
  shortestContext: {
    decodeRate: number;
    firstGeneratedTokenMs: number;
  }
): TargetSummary {
  const selected = attempts.filter(
    (attempt) =>
      attempt.phase === 'measured' &&
      attempt.targetContextTokens === targetContextTokens
  );
  const decodeRate = median(
    selected.map((attempt) => attempt.derived.generatedTokensPerDecodeSecond)
  );
  const firstGeneratedTokenMs = median(
    selected.map((attempt) => attempt.derived.internalFirstGeneratedTokenMs)
  );
  return {
    targetContextTokens,
    measuredRuns: selected.length,
    medianTotalMs: median(selected.map((attempt) => attempt.totalMs)),
    medianInternalFirstGeneratedTokenMs: firstGeneratedTokenMs,
    medianInputExecutionMs: median(
      selected.map((attempt) => attempt.derived.inputExecutionMs)
    ),
    medianDecodeExecutionMs: median(
      selected.map((attempt) => attempt.derived.decodeExecutionMs)
    ),
    medianGeneratedTokensPerDecodeSecond: decodeRate,
    medianCommittedStateLogicalNbytes: median(
      selected.map((attempt) => attempt.metrics.committed_state_logical_nbytes)
    ),
    medianMetalActiveNbytes: median(
      selected.map(
        (attempt) =>
          requireNativeMetalMemory(
            attempt.metrics,
            `context curve ${attempt.targetContextTokens}`
          ).active_nbytes
      )
    ),
    medianMetalActiveGrowthPerRetainedInstanceNbytes: median(
      selected.map((attempt) => {
        const growth =
          attempt.derived.metalActiveGrowthFromPreviousAttemptNbytes;
        if (growth === null) {
          throw new Error('measured attempt omitted prior Metal active bytes');
        }
        return growth;
      })
    ),
    medianMetalCacheNbytes: median(
      selected.map(
        (attempt) =>
          requireNativeMetalMemory(
            attempt.metrics,
            `context curve ${attempt.targetContextTokens}`
          ).cache_nbytes
      )
    ),
    maximumMetalPeakNbytes: Math.max(
      ...selected.map(
        (attempt) =>
          requireNativeMetalMemory(
            attempt.metrics,
            `context curve ${attempt.targetContextTokens}`
          ).peak_nbytes
      )
    ),
    decodeRateOverShortestContext: decodeRate / shortestContext.decodeRate,
    firstGeneratedTokenTimeOverShortestContext:
      firstGeneratedTokenMs / shortestContext.firstGeneratedTokenMs,
  };
}

function summarize(
  targets: readonly number[],
  attempts: readonly AttemptRecord[]
): TargetSummary[] {
  const shortestTarget = targets[0];
  if (shortestTarget === undefined) {
    throw new Error('context curve requires at least one target');
  }
  const shortestAttempts = attempts.filter(
    (attempt) =>
      attempt.phase === 'measured' &&
      attempt.targetContextTokens === shortestTarget
  );
  const shortestContext = {
    decodeRate: median(
      shortestAttempts.map(
        (attempt) => attempt.derived.generatedTokensPerDecodeSecond
      )
    ),
    firstGeneratedTokenMs: median(
      shortestAttempts.map(
        (attempt) => attempt.derived.internalFirstGeneratedTokenMs
      )
    ),
  };
  return targets.map((target) =>
    summarizeTarget(target, attempts, shortestContext)
  );
}

function outputHashesAreStable(
  targets: readonly number[],
  attempts: readonly AttemptRecord[]
): boolean {
  return targets.every((target) => {
    const hashes = attempts
      .filter(
        (attempt) =>
          attempt.phase === 'measured' && attempt.targetContextTokens === target
      )
      .map((attempt) => attempt.outputSha256);
    return hashes.length > 0 && new Set(hashes).size === 1;
  });
}

async function runTargetBlock(input: {
  config: GateConfig;
  targetContextTokens: number;
  libraryPath: string;
  attempts: AttemptRecord[];
  result: Record<string, unknown>;
  targetBlocks: Record<string, unknown>[];
}): Promise<void> {
  const spawnedAt = performance.now();
  const client = NativeInferenceClient.spawn({
    binaryPath: input.config.nativeBinaryPath,
    modelDirectory: input.config.modelDirectory,
    maxOutstandingRequests: 1,
    environment: {
      ...process.env,
      DYLD_LIBRARY_PATH: input.libraryPath,
    },
  });
  const stateRoots = new EphemeralNativeStateRoots();
  const block: Record<string, unknown> = {
    targetContextTokens: input.targetContextTokens,
    completedAttempts: 0,
  };
  input.targetBlocks.push(block);
  try {
    block.engine = await client.ready();
    block.readyElapsedMs = performance.now() - spawnedAt;
    let attemptIndexWithinTarget = 0;
    let previousMetalActiveNbytes: number | null = null;
    const runPhase = async (
      phase: AttemptPhase,
      count: number
    ): Promise<void> => {
      for (let round = 1; round <= count; round += 1) {
        attemptIndexWithinTarget += 1;
        const attemptIdentity = `${phase}-${round}`;
        const instanceId = `context-curve-${input.targetContextTokens}-${attemptIdentity}`;
        await stateRoots.open(client, instanceId);
        const attempt = await runAttempt(
          client,
          commandForAttempt(
            instanceId,
            attemptIdentity,
            input.targetContextTokens
          ),
          {
            phase,
            round,
            attemptIndexWithinTarget,
            targetContextTokens: input.targetContextTokens,
            previousMetalActiveNbytes,
          }
        );
        input.attempts.push(attempt);
        previousMetalActiveNbytes = requireNativeMetalMemory(
          attempt.metrics,
          `context curve ${attempt.targetContextTokens}`
        ).active_nbytes;
        block.completedAttempts = attemptIndexWithinTarget;
        writeResult(input.config.outputPath, input.result);
      }
    };
    await runPhase('warmup', input.config.warmupRuns);
    await runPhase('measured', input.config.measuredRuns);
  } finally {
    try {
      await client.shutdown();
    } finally {
      stateRoots.dispose();
    }
  }
}

test('parses ordered unique targets and constructs the calibrated plans', () => {
  expect(parseTargetContextTokens(undefined)).toEqual(
    DEFAULT_TARGET_CONTEXT_TOKENS
  );
  expect(parseTargetContextTokens('32768, 2048,8192')).toEqual([
    2_048, 8_192, 32_768,
  ]);
  expect(() => parseTargetContextTokens('2048,2048')).toThrow(
    'ECHO_CONTEXT_CURVE_TARGETS must contain unique comma-separated integers greater than 108'
  );
  expect(paddingPlan(2_048)).toEqual({
    repetitions: 646,
    remainderTokens: 2,
  });
  expect(paddingPlan(32_768)).toEqual({
    repetitions: 10_886,
    remainderTokens: 2,
  });
});

liveTest(
  'measures production-buffered decode and cold-prefill across context lengths',
  async () => {
    const config = loadConfig();
    assertInputPaths(config);
    const libraryPath = requiredEnvironmentVariable('ECHO_NATIVE_LIBRARY_PATH');
    const attempts: AttemptRecord[] = [];
    const result: Record<string, unknown> = {
      schemaVersion: 2,
      generatedAt: new Date().toISOString(),
      conditions: {
        targetContextTokens: config.targetContextTokens,
        warmupRunsPerTarget: config.warmupRuns,
        measuredRunsPerTarget: config.measuredRuns,
        targetExecutionOrder: 'ascending; one resident process per target',
        attemptIsolation:
          'one fresh instance with initial transition per attempt; no GDN carry-over',
        retainedInstancesPerTargetProcess:
          config.warmupRuns + config.measuredRuns,
        deliveryMode: 'buffered',
        streamTokens: false,
        maxNewTokens: MAX_NEW_TOKENS,
        sampling: GREEDY_SAMPLING,
        paddingCalibration: {
          basePromptTokens: PROMPT_BASE_TOKENS,
          paddingPhrase: PADDING_PHRASE,
          paddingPhraseTokens: PADDING_PHRASE_TOKENS,
          remainderToken: REMAINDER_TOKEN,
          runtimeExactTokenCountRequired: true,
        },
        decodeRateDenominator: 'generated_tokens / decode_execution_nanos',
        decodeExecutionIncludesHiddenLengthEos: true,
        inputTimingScope:
          'cold full-context initial prefill; not cached-continuation suffix prefill',
        concurrency: 1,
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
      attempts,
      targetBlocks: [],
    };
    writeResult(config.outputPath, result);

    const targetBlocks = result.targetBlocks as Record<string, unknown>[];
    for (const targetContextTokens of config.targetContextTokens) {
      await runTargetBlock({
        config,
        targetContextTokens,
        libraryPath,
        attempts,
        result,
        targetBlocks,
      });
    }

    const measured = attempts.filter((attempt) => attempt.phase === 'measured');
    const checks = {
      expectedAttemptCount:
        attempts.length ===
        config.targetContextTokens.length *
          (config.warmupRuns + config.measuredRuns),
      everyAttemptUsedIndependentInitialState: measured.every(
        (attempt) => attempt.stateTransition === 'initial'
      ),
      exactTargetInputTokenCounts: measured.every(
        (attempt) =>
          attempt.metrics.input_tokens_processed === attempt.targetContextTokens
      ),
      noCachedPrefixOnInitialStates: measured.every(
        (attempt) => attempt.metrics.cached_prefix_tokens === 0
      ),
      fixedCompletionLength: measured.every(
        (attempt) => attempt.metrics.generated_tokens === MAX_NEW_TOKENS
      ),
      everyAttemptReachedLength: measured.every(
        (attempt) => attempt.finishReason === 'length'
      ),
      forcedEosAdvancedState: measured.every(
        (attempt) =>
          attempt.stateSequenceLength ===
          attempt.targetContextTokens + MAX_NEW_TOKENS + 1
      ),
      outputStableWithinEachTarget: outputHashesAreStable(
        config.targetContextTokens,
        attempts
      ),
    };
    result.summary = summarize(config.targetContextTokens, attempts);
    result.checks = checks;
    writeResult(config.outputPath, result);

    expect(checks).toEqual({
      expectedAttemptCount: true,
      everyAttemptUsedIndependentInitialState: true,
      exactTargetInputTokenCounts: true,
      noCachedPrefixOnInitialStates: true,
      fixedCompletionLength: true,
      everyAttemptReachedLength: true,
      forcedEosAdvancedState: true,
      outputStableWithinEachTarget: true,
    });
  },
  30 * 60_000
);
