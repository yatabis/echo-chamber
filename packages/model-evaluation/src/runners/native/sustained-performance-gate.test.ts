/* eslint-disable no-await-in-loop -- Sustained load intentionally uses one serial resident GPU owner. */

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

const LIVE_GATE_ENABLED = process.env.ECHO_NATIVE_SUSTAINED_GATE === '1';
const liveTest = LIVE_GATE_ENABLED ? test : test.skip;
const MAX_NEW_TOKENS = 128;
const EXPECTED_PROMPT_TOKENS = 112;
const CHECKPOINT_INTERVAL = 10;
const MAX_DECODE_RATE_DEGRADATION = 0.2;
const MINIMUM_GENERATED_TOKEN_DUTY_FRACTION = 0.8;
const MAX_ACTIVE_GROWTH_NBYTES = 256 * 1_024 * 1_024;
const MAX_CACHE_GROWTH_NBYTES = 512 * 1_024 * 1_024;
const BENCHMARK_PROMPT = [
  'これは推論性能の測定です。ツールは使わず、次の形式だけを出力してください。',
  '1から400までの整数を昇順に、それぞれ独立した行へ書いてください。',
  '各行は「0001: native-rapid-performance」のように、4桁ゼロ埋めの番号、コロン、半角空白、固定文字列 native-rapid-performance の順にしてください。',
  '前置き、説明、後書き、省略記号は加えないでください。',
].join('\n');
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
  durationSeconds: number;
  warmupRuns: number;
}

interface AttemptRecord {
  phase: AttemptPhase;
  index: number;
  startedElapsedSeconds: number;
  completedElapsedSeconds: number;
  stateTransition: NativeGenerateCommand['state_transition'];
  totalMs: number;
  stateSequenceLength: number;
  finishReason: 'length' | 'stop_token';
  outputSha256: string;
  generatedTokenSha256: string;
  outputPreview: string;
  metrics: NativeRuntimeMetrics;
  derived: {
    internalFirstGeneratedTokenMs: number;
    decodeExecutionMs: number;
    generatedTokensPerDecodeSecond: number;
    requestMs: number;
  };
}

interface SustainedSummary {
  actualMeasuredDurationSeconds: number;
  measuredAttempts: number;
  totalGeneratedTokens: number;
  fullLengthAttempts: number;
  fullLengthFraction: number;
  generatedTokenDutyFraction: number;
  uniqueOutputHashes: number;
  comparisonWindowSeconds: number;
  firstWindowAttempts: number;
  lastWindowAttempts: number;
  firstWindowMedianDecodeTokensPerSecond: number;
  lastWindowMedianDecodeTokensPerSecond: number;
  lastOverFirstDecodeRate: number;
  medianDecodeTokensPerSecond: number;
  p10DecodeTokensPerSecond: number;
  p90DecodeTokensPerSecond: number;
  medianTotalMsForFullLengthAttempts: number;
  firstMetalActiveNbytes: number;
  lastMetalActiveNbytes: number;
  activeGrowthNbytes: number;
  firstMetalCacheNbytes: number;
  lastMetalCacheNbytes: number;
  cacheGrowthNbytes: number;
  maximumMetalPeakNbytes: number;
}

function requiredEnvironmentVariable(name: string): string {
  const value = process.env[name];
  if (value === undefined || value.trim() === '') {
    throw new Error(`Required environment variable is missing: ${name}`);
  }
  return value;
}

function parsePositiveNumber(name: string, fallback: number): number {
  const raw = process.env[name];
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${name} must be a positive finite number`);
  }
  return value;
}

function parsePositiveInteger(name: string, fallback: number): number {
  const value = parsePositiveNumber(name, fallback);
  if (!Number.isSafeInteger(value)) {
    throw new Error(`${name} must be a positive safe integer`);
  }
  return value;
}

function loadConfig(): GateConfig {
  return {
    nativeBinaryPath: requiredEnvironmentVariable(
      'ECHO_SUSTAINED_NATIVE_INFERENCE_BIN'
    ),
    modelDirectory: requiredEnvironmentVariable('ECHO_SUSTAINED_MODEL'),
    outputPath: requiredEnvironmentVariable('ECHO_SUSTAINED_OUTPUT'),
    durationSeconds: parsePositiveNumber(
      'ECHO_SUSTAINED_DURATION_SECONDS',
      20 * 60
    ),
    warmupRuns: parsePositiveInteger('ECHO_SUSTAINED_WARMUP_RUNS', 5),
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

function sha256Bytes(parts: readonly Uint8Array[]): string {
  const hash = createHash('sha256');
  for (const part of parts) {
    hash.update(part);
  }
  return hash.digest('hex');
}

function sha256Text(text: string): string {
  return sha256Bytes([Buffer.from(text)]);
}

function sha256Tokens(tokens: readonly number[]): string {
  const bytes = Buffer.allocUnsafe(tokens.length * 4);
  tokens.forEach((token, index) => {
    bytes.writeUInt32LE(token, index * 4);
  });
  return sha256Bytes([bytes]);
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

function generationCommand(input: {
  phase: AttemptPhase;
  index: number;
  instanceId: string;
  hasCommittedState: boolean;
}): NativeGenerateCommand {
  return {
    type: 'generate',
    request_id: `sustained-${input.phase}-${input.index}`,
    instance_id: input.instanceId,
    state_transition: input.hasCommittedState ? 'new_session' : 'initial',
    stream_tokens: false,
    input: [{ role: 'user', content: BENCHMARK_PROMPT }],
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
    index: number;
    measuredStartedAt: number;
  }
): Promise<AttemptRecord> {
  const startedAt = performance.now();
  const event = await client.generate(command);
  const completedAt = performance.now();
  const metrics = event.response.metrics;
  const firstGeneratedTokenNanos = metrics.first_generated_token_nanos;
  if (
    firstGeneratedTokenNanos === null ||
    metrics.decode_execution_nanos <= 0 ||
    metrics.generated_tokens <= 0
  ) {
    throw new Error(`${command.request_id} omitted usable timing metrics`);
  }
  return {
    phase: context.phase,
    index: context.index,
    startedElapsedSeconds: (startedAt - context.measuredStartedAt) / 1_000,
    completedElapsedSeconds: (completedAt - context.measuredStartedAt) / 1_000,
    stateTransition: command.state_transition,
    totalMs: completedAt - startedAt,
    stateSequenceLength: event.response.state_sequence_length,
    finishReason: event.response.finish_reason,
    outputSha256: sha256Text(event.text),
    generatedTokenSha256: sha256Tokens(event.response.generated_tokens),
    outputPreview: event.text.slice(0, 160),
    metrics,
    derived: {
      internalFirstGeneratedTokenMs: nanosToMilliseconds(
        firstGeneratedTokenNanos
      ),
      decodeExecutionMs: nanosToMilliseconds(metrics.decode_execution_nanos),
      generatedTokensPerDecodeSecond:
        metrics.generated_tokens / (metrics.decode_execution_nanos / 1e9),
      requestMs: nanosToMilliseconds(metrics.request_nanos),
    },
  };
}

function percentile(values: readonly number[], fraction: number): number {
  if (values.length === 0 || fraction < 0 || fraction > 1) {
    throw new Error(
      'percentile requires values and a fraction from zero to one'
    );
  }
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.floor((sorted.length - 1) * fraction);
  const value = sorted[index];
  if (value === undefined) {
    throw new Error('percentile index is missing');
  }
  return value;
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

function summarize(
  attempts: readonly AttemptRecord[],
  actualMeasuredDurationSeconds: number,
  configuredDurationSeconds: number
): SustainedSummary {
  const fullLength = attempts.filter(
    (attempt) => attempt.metrics.generated_tokens === MAX_NEW_TOKENS
  );
  const comparisonWindowSeconds = Math.min(
    5 * 60,
    configuredDurationSeconds / 4
  );
  const firstWindow = fullLength.filter(
    (attempt) => attempt.startedElapsedSeconds < comparisonWindowSeconds
  );
  const lastWindow = fullLength.filter(
    (attempt) =>
      attempt.startedElapsedSeconds >=
      actualMeasuredDurationSeconds - comparisonWindowSeconds
  );
  const decodeRates = fullLength.map(
    (attempt) => attempt.derived.generatedTokensPerDecodeSecond
  );
  const firstDecodeRate = median(
    firstWindow.map((attempt) => attempt.derived.generatedTokensPerDecodeSecond)
  );
  const lastDecodeRate = median(
    lastWindow.map((attempt) => attempt.derived.generatedTokensPerDecodeSecond)
  );
  const firstAttempt = attempts[0];
  const lastAttempt = attempts[attempts.length - 1];
  if (firstAttempt === undefined || lastAttempt === undefined) {
    throw new Error('sustained summary requires measured attempts');
  }
  const metalMemories = attempts.map((attempt) =>
    requireNativeMetalMemory(
      attempt.metrics,
      `sustained attempt ${attempt.index}`
    )
  );
  const firstMetalMemory = requireNativeMetalMemory(
    firstAttempt.metrics,
    'first sustained attempt'
  );
  const lastMetalMemory = requireNativeMetalMemory(
    lastAttempt.metrics,
    'last sustained attempt'
  );
  return {
    actualMeasuredDurationSeconds,
    measuredAttempts: attempts.length,
    totalGeneratedTokens: attempts.reduce(
      (sum, attempt) => sum + attempt.metrics.generated_tokens,
      0
    ),
    fullLengthAttempts: fullLength.length,
    fullLengthFraction: fullLength.length / attempts.length,
    generatedTokenDutyFraction:
      attempts.reduce(
        (sum, attempt) => sum + attempt.metrics.generated_tokens,
        0
      ) /
      (attempts.length * MAX_NEW_TOKENS),
    uniqueOutputHashes: new Set(attempts.map((attempt) => attempt.outputSha256))
      .size,
    comparisonWindowSeconds,
    firstWindowAttempts: firstWindow.length,
    lastWindowAttempts: lastWindow.length,
    firstWindowMedianDecodeTokensPerSecond: firstDecodeRate,
    lastWindowMedianDecodeTokensPerSecond: lastDecodeRate,
    lastOverFirstDecodeRate: lastDecodeRate / firstDecodeRate,
    medianDecodeTokensPerSecond: median(decodeRates),
    p10DecodeTokensPerSecond: percentile(decodeRates, 0.1),
    p90DecodeTokensPerSecond: percentile(decodeRates, 0.9),
    medianTotalMsForFullLengthAttempts: median(
      fullLength.map((attempt) => attempt.totalMs)
    ),
    firstMetalActiveNbytes: firstMetalMemory.active_nbytes,
    lastMetalActiveNbytes: lastMetalMemory.active_nbytes,
    activeGrowthNbytes:
      lastMetalMemory.active_nbytes - firstMetalMemory.active_nbytes,
    firstMetalCacheNbytes: firstMetalMemory.cache_nbytes,
    lastMetalCacheNbytes: lastMetalMemory.cache_nbytes,
    cacheGrowthNbytes:
      lastMetalMemory.cache_nbytes - firstMetalMemory.cache_nbytes,
    maximumMetalPeakNbytes: Math.max(
      ...metalMemories.map((memory) => memory.peak_nbytes)
    ),
  };
}

test('validates sustained gate numeric configuration', () => {
  expect(parsePositiveNumber('ECHO_UNUSED_TEST_VALUE', 60)).toBe(60);
  expect(parsePositiveInteger('ECHO_UNUSED_TEST_VALUE', 5)).toBe(5);
});

liveTest(
  'holds production-buffered generation under sustained resident load',
  async () => {
    const config = loadConfig();
    assertInputPaths(config);
    const libraryPath = requiredEnvironmentVariable('ECHO_NATIVE_LIBRARY_PATH');
    const warmupAttempts: AttemptRecord[] = [];
    const measuredAttempts: AttemptRecord[] = [];
    const result: Record<string, unknown> = {
      schemaVersion: 1,
      generatedAt: new Date().toISOString(),
      conditions: {
        configuredDurationSeconds: config.durationSeconds,
        warmupRuns: config.warmupRuns,
        workload:
          'one existence; initial once, then new_session with GDN retained and KV reset',
        deliveryMode: 'buffered',
        streamTokens: false,
        expectedPromptTokens: EXPECTED_PROMPT_TOKENS,
        maxNewTokens: MAX_NEW_TOKENS,
        sampling: GREEDY_SAMPLING,
        checkpointIntervalAttempts: CHECKPOINT_INTERVAL,
        thermalObservation:
          'throughput stability proxy; no privileged hardware temperature sensor',
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
      warmupAttempts,
      measuredAttempts,
    };
    writeResult(config.outputPath, result);

    const spawnedAt = performance.now();
    const client = NativeInferenceClient.spawn({
      binaryPath: config.nativeBinaryPath,
      modelDirectory: config.modelDirectory,
      maxOutstandingRequests: 1,
      environment: {
        ...process.env,
        DYLD_LIBRARY_PATH: libraryPath,
      },
    });
    const stateRoots = new EphemeralNativeStateRoots();
    const instanceId = 'sustained-production';
    let hasCommittedState = false;
    try {
      result.engine = await client.ready();
      result.readyElapsedMs = performance.now() - spawnedAt;
      await stateRoots.open(client, instanceId);
      const placeholderMeasuredStart = performance.now();
      for (let index = 1; index <= config.warmupRuns; index += 1) {
        const command = generationCommand({
          phase: 'warmup',
          index,
          instanceId,
          hasCommittedState,
        });
        warmupAttempts.push(
          await runAttempt(client, command, {
            phase: 'warmup',
            index,
            measuredStartedAt: placeholderMeasuredStart,
          })
        );
        hasCommittedState = true;
      }
      writeResult(config.outputPath, result);

      const measuredStartedAt = performance.now();
      const deadline = measuredStartedAt + config.durationSeconds * 1_000;
      let index = 0;
      while (performance.now() < deadline) {
        index += 1;
        const command = generationCommand({
          phase: 'measured',
          index,
          instanceId,
          hasCommittedState,
        });
        measuredAttempts.push(
          await runAttempt(client, command, {
            phase: 'measured',
            index,
            measuredStartedAt,
          })
        );
        hasCommittedState = true;
        if (index % CHECKPOINT_INTERVAL === 0) {
          writeResult(config.outputPath, result);
        }
      }
      const actualMeasuredDurationSeconds =
        (performance.now() - measuredStartedAt) / 1_000;
      const summary = summarize(
        measuredAttempts,
        actualMeasuredDurationSeconds,
        config.durationSeconds
      );
      const checks = {
        reachedConfiguredDuration:
          actualMeasuredDurationSeconds >= config.durationSeconds,
        firstAttemptUsedInitialState:
          warmupAttempts[0]?.stateTransition === 'initial',
        everyLaterAttemptUsedNewSession: [
          ...warmupAttempts.slice(1),
          ...measuredAttempts,
        ].every((attempt) => attempt.stateTransition === 'new_session'),
        everyAttemptReprocessedExactPrompt: measuredAttempts.every(
          (attempt) =>
            attempt.metrics.cached_prefix_tokens === 0 &&
            attempt.metrics.input_tokens_processed === EXPECTED_PROMPT_TOKENS
        ),
        everyStateLengthMatchesCommit: measuredAttempts.every(
          (attempt) =>
            attempt.stateSequenceLength ===
            EXPECTED_PROMPT_TOKENS +
              attempt.metrics.generated_tokens +
              (attempt.finishReason === 'length' ? 1 : 0)
        ),
        generatedTokenDutyCycle:
          summary.generatedTokenDutyFraction >=
          MINIMUM_GENERATED_TOKEN_DUTY_FRACTION,
        decodeRateRetainedAtLeastEightyPercent:
          summary.lastOverFirstDecodeRate >= 1 - MAX_DECODE_RATE_DEGRADATION,
        activeMemoryGrowthWithin256MiB:
          summary.activeGrowthNbytes <= MAX_ACTIVE_GROWTH_NBYTES,
        allocatorCacheGrowthWithin512MiB:
          summary.cacheGrowthNbytes <= MAX_CACHE_GROWTH_NBYTES,
      };
      result.summary = summary;
      result.checks = checks;
      writeResult(config.outputPath, result);

      expect(checks).toEqual({
        reachedConfiguredDuration: true,
        firstAttemptUsedInitialState: true,
        everyLaterAttemptUsedNewSession: true,
        everyAttemptReprocessedExactPrompt: true,
        everyStateLengthMatchesCommit: true,
        generatedTokenDutyCycle: true,
        decodeRateRetainedAtLeastEightyPercent: true,
        activeMemoryGrowthWithin256MiB: true,
        allocatorCacheGrowthWithin512MiB: true,
      });
    } finally {
      try {
        await client.shutdown();
      } finally {
        stateRoots.dispose();
      }
    }
  },
  35 * 60_000
);
