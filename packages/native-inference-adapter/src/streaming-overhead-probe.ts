/* eslint-disable no-await-in-loop -- One local GPU must run matched measurements sequentially. */

import { createHash } from 'node:crypto';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';

import { NativeInferenceClient } from './native-inference-client';

import type { NativeGenerateCommand, NativeRuntimeMetrics } from './protocol';

const MAX_NEW_TOKENS = 128;
const DEFAULT_MEASURED_ROUNDS = 7;
const BENCHMARK_PROMPT = [
  'これは推論性能の測定です。ツールは使わず、次の形式だけを出力してください。',
  '1から400までの整数を昇順に、それぞれ独立した行へ書いてください。',
  '各行は「0001: native-stream-overhead」のように、4桁ゼロ埋めの番号、コロン、半角空白、固定文字列 native-stream-overhead の順にしてください。',
  '前置き、説明、後書き、省略記号は加えないでください。',
].join('\n');

interface Attempt {
  phase: 'warmup' | 'measured';
  round: number;
  streamTokens: boolean;
  orderInRound: number;
  totalMs: number;
  requestMs: number;
  modelMs: number;
  decodeMs: number;
  generatedTokens: number;
  streamedTokenEvents: number;
  stateSequenceLength: number;
  finishReason: string;
  outputSha256: string;
  metrics: NativeRuntimeMetrics;
}

const [binaryPath, modelDirectory] = process.argv.slice(2);
if (binaryPath === undefined || modelDirectory === undefined) {
  throw new Error(
    'usage: pnpm probe:stream-overhead <echo-inference-binary> <model-directory>'
  );
}

const measuredRounds = parsePositiveInteger(
  process.env.ECHO_STREAM_OVERHEAD_ROUNDS,
  DEFAULT_MEASURED_ROUNDS
);
const stateRoot = await mkdtemp(join(tmpdir(), 'echo-native-stream-overhead-'));
const libraryPath = process.env.ECHO_NATIVE_LIBRARY_PATH;
const client = NativeInferenceClient.spawn({
  binaryPath,
  modelDirectory,
  maxOutstandingRequests: 1,
  environment:
    libraryPath === undefined
      ? process.env
      : { ...process.env, DYLD_LIBRARY_PATH: libraryPath },
});

const attempts: Attempt[] = [];
let attemptSequence = 0;

try {
  await client.ready();
  for (const streamTokens of [true, false]) {
    attempts.push(await runAttempt('warmup', 0, streamTokens, 0));
  }
  for (let round = 1; round <= measuredRounds; round += 1) {
    const modes = round % 2 === 1 ? [true, false] : [false, true];
    for (const [orderInRound, streamTokens] of modes.entries()) {
      attempts.push(
        await runAttempt('measured', round, streamTokens, orderInRound)
      );
    }
  }
} finally {
  await client.shutdown();
  await rm(stateRoot, { recursive: true, force: true });
}

const measured = attempts.filter((attempt) => attempt.phase === 'measured');
const streamed = measured.filter((attempt) => attempt.streamTokens);
const unstreamed = measured.filter((attempt) => !attempt.streamTokens);
const paired = Array.from({ length: measuredRounds }, (_, index) => {
  const round = index + 1;
  const streamedAttempt = streamed.find((attempt) => attempt.round === round);
  const unstreamedAttempt = unstreamed.find(
    (attempt) => attempt.round === round
  );
  if (streamedAttempt === undefined || unstreamedAttempt === undefined) {
    throw new Error(`missing paired attempts for round ${round}`);
  }
  return {
    round,
    streamedFirst: streamedAttempt.orderInRound === 0,
    totalMsSaved: streamedAttempt.totalMs - unstreamedAttempt.totalMs,
    requestMsSaved: streamedAttempt.requestMs - unstreamedAttempt.requestMs,
    modelMsSaved: streamedAttempt.modelMs - unstreamedAttempt.modelMs,
    decodeMsSaved: streamedAttempt.decodeMs - unstreamedAttempt.decodeMs,
  };
});
const outputHashes = new Set(measured.map((attempt) => attempt.outputSha256));

const result = {
  schemaVersion: 1,
  measuredRounds,
  benchmark: {
    maxNewTokens: MAX_NEW_TOKENS,
    prompt: BENCHMARK_PROMPT,
    order: 'odd rounds stream-first; even rounds no-stream-first',
  },
  checks: {
    outputHashesMatch: outputHashes.size === 1,
    generatedTokenCountsMatch:
      new Set(measured.map((attempt) => attempt.generatedTokens)).size === 1,
    stateSequenceLengthsMatch:
      new Set(measured.map((attempt) => attempt.stateSequenceLength)).size ===
      1,
    finishReasonsMatch:
      new Set(measured.map((attempt) => attempt.finishReason)).size === 1,
    streamedModeEmittedEveryVisibleToken: streamed.every(
      (attempt) => attempt.streamedTokenEvents === attempt.generatedTokens
    ),
    disabledModeEmittedNoTokenEvents: unstreamed.every(
      (attempt) => attempt.streamedTokenEvents === 0
    ),
  },
  summary: {
    streamEnabled: summarize(streamed),
    streamDisabled: summarize(unstreamed),
    disabledGainPercent: {
      totalMs: gainPercent(
        median(streamed.map((attempt) => attempt.totalMs)),
        median(unstreamed.map((attempt) => attempt.totalMs))
      ),
      requestMs: gainPercent(
        median(streamed.map((attempt) => attempt.requestMs)),
        median(unstreamed.map((attempt) => attempt.requestMs))
      ),
      modelMs: gainPercent(
        median(streamed.map((attempt) => attempt.modelMs)),
        median(unstreamed.map((attempt) => attempt.modelMs))
      ),
      decodeMs: gainPercent(
        median(streamed.map((attempt) => attempt.decodeMs)),
        median(unstreamed.map((attempt) => attempt.decodeMs))
      ),
    },
    paired: {
      disabledWins: paired.filter((pair) => pair.totalMsSaved > 0).length,
      enabledWins: paired.filter((pair) => pair.totalMsSaved < 0).length,
      ties: paired.filter((pair) => pair.totalMsSaved === 0).length,
      medianTotalMsSaved: median(paired.map((pair) => pair.totalMsSaved)),
      meanTotalMsSaved: mean(paired.map((pair) => pair.totalMsSaved)),
      medianDecodeMsSaved: median(paired.map((pair) => pair.decodeMsSaved)),
      meanDecodeMsSaved: mean(paired.map((pair) => pair.decodeMsSaved)),
      streamedFirst: summarizePairs(
        paired.filter((pair) => pair.streamedFirst)
      ),
      disabledFirst: summarizePairs(
        paired.filter((pair) => !pair.streamedFirst)
      ),
    },
  },
  ...(process.env.ECHO_STREAM_OVERHEAD_INCLUDE_ATTEMPTS === '1'
    ? { attempts, paired }
    : {}),
};

if (Object.values(result.checks).some((passed) => !passed)) {
  throw new Error(
    `stream overhead probe failed: ${JSON.stringify(result.checks)}`
  );
}
console.log(JSON.stringify(result, undefined, 2));

async function runAttempt(
  phase: Attempt['phase'],
  round: number,
  streamTokens: boolean,
  orderInRound: number
): Promise<Attempt> {
  attemptSequence += 1;
  const identity = `stream-overhead-${attemptSequence}`;
  await client.openState({
    type: 'open_state',
    request_id: `${identity}:open`,
    instance_id: identity,
    persistence: 'durable',
    snapshot_root: join(stateRoot, identity),
  });
  const command: NativeGenerateCommand = {
    type: 'generate',
    request_id: `${identity}:generate`,
    instance_id: identity,
    state_transition: 'initial',
    stream_tokens: streamTokens,
    input: [{ role: 'user', content: BENCHMARK_PROMPT }],
    tools: [],
    max_new_tokens: MAX_NEW_TOKENS,
    sampling: {
      temperature: 0,
      top_p: 1,
      top_k: 0,
      min_p: 0,
      repetition_penalty: 1,
      presence_penalty: 0,
      seed: 42,
    },
  };
  let streamedTokenEvents = 0;
  const startedAt = performance.now();
  const event = await client.generate(command, () => {
    streamedTokenEvents += 1;
  });
  const totalMs = performance.now() - startedAt;
  const metrics = event.response.metrics;
  return {
    phase,
    round,
    streamTokens,
    orderInRound,
    totalMs,
    requestMs: nanosToMilliseconds(metrics.request_nanos),
    modelMs: nanosToMilliseconds(metrics.model_execution_nanos),
    decodeMs: nanosToMilliseconds(metrics.decode_execution_nanos),
    generatedTokens: metrics.generated_tokens,
    streamedTokenEvents,
    stateSequenceLength: event.response.state_sequence_length,
    finishReason: event.response.finish_reason,
    outputSha256: createHash('sha256').update(event.text).digest('hex'),
    metrics,
  };
}

function summarize(
  attemptsForMode: readonly Attempt[]
): Record<string, number> {
  return {
    count: attemptsForMode.length,
    medianTotalMs: median(attemptsForMode.map((attempt) => attempt.totalMs)),
    medianRequestMs: median(
      attemptsForMode.map((attempt) => attempt.requestMs)
    ),
    medianModelMs: median(attemptsForMode.map((attempt) => attempt.modelMs)),
    medianDecodeMs: median(attemptsForMode.map((attempt) => attempt.decodeMs)),
  };
}

function summarizePairs(
  pairs: readonly (typeof paired)[number][]
): Record<string, number> {
  return {
    count: pairs.length,
    disabledWins: pairs.filter((pair) => pair.totalMsSaved > 0).length,
    medianTotalMsSaved: median(pairs.map((pair) => pair.totalMsSaved)),
    meanTotalMsSaved: mean(pairs.map((pair) => pair.totalMsSaved)),
    medianDecodeMsSaved: median(pairs.map((pair) => pair.decodeMsSaved)),
    meanDecodeMsSaved: mean(pairs.map((pair) => pair.decodeMsSaved)),
  };
}

function median(values: readonly number[]): number {
  if (values.length === 0) {
    throw new Error('median requires at least one value');
  }
  const sorted = [...values].sort((left, right) => left - right);
  const midpoint = Math.floor(sorted.length / 2);
  const upper = sorted[midpoint];
  if (upper === undefined) {
    throw new Error('median index is missing');
  }
  if (sorted.length % 2 === 1) {
    return upper;
  }
  const lower = sorted[midpoint - 1];
  if (lower === undefined) {
    throw new Error('median lower index is missing');
  }
  return (lower + upper) / 2;
}

function mean(values: readonly number[]): number {
  if (values.length === 0) {
    throw new Error('mean requires at least one value');
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function gainPercent(enabled: number, disabled: number): number {
  return ((enabled - disabled) / enabled) * 100;
}

function nanosToMilliseconds(nanos: number): number {
  return nanos / 1_000_000;
}

function parsePositiveInteger(
  raw: string | undefined,
  fallback: number
): number {
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error('ECHO_STREAM_OVERHEAD_ROUNDS must be a positive integer');
  }
  return value;
}
