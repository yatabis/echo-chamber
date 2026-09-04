/* eslint-disable no-await-in-loop -- One local GPU must run matched measurements sequentially. */

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

import OpenAI from 'openai';
import { expect, test } from 'vitest';

import { NativeInferenceClient } from '@echo-chamber/native-inference-adapter/native-inference-client';
import type {
  NativeGenerateCommand,
  NativeRuntimeMetrics,
} from '@echo-chamber/native-inference-adapter/protocol';

import { startLocalModelServer } from '../rapid-mlx/server-controller';

import { EphemeralNativeStateRoots } from './ephemeral-state-roots';

import type {
  ChatCompletionChunk,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
} from 'openai/resources/chat/completions';

const LIVE_GATE_ENABLED =
  process.env.ECHO_NATIVE_RAPID_PERFORMANCE_GATE === '1';
const liveTest = LIVE_GATE_ENABLED ? test : test.skip;
const MAX_NEW_TOKENS = 128;
const PERFORMANCE_TOLERANCE = 0.05;
const BENCHMARK_PROMPT = [
  'これは推論性能の測定です。ツールは使わず、次の形式だけを出力してください。',
  '1から400までの整数を昇順に、それぞれ独立した行へ書いてください。',
  '各行は「0001: native-rapid-performance」のように、4桁ゼロ埋めの番号、コロン、半角空白、固定文字列 native-rapid-performance の順にしてください。',
  '前置き、説明、後書き、省略記号は加えないでください。',
].join('\n');

type AttemptPhase = 'warmup' | 'measured';
type EngineName = 'native' | 'rapid-mlx';
type DeliveryMode = 'streaming' | 'buffered';
type FinishReason =
  | 'length'
  | 'stop'
  | 'stop_token'
  | 'cancelled'
  | 'content_filter'
  | 'tool_calls'
  | 'function_call'
  | null;

interface GateConfig {
  nativeBinaryPath: string;
  modelDirectory: string;
  rapidMlxBinaryPath: string;
  rapidMlxWorkingDirectory: string;
  outputPath: string;
  port: number;
  warmupRuns: number;
  measuredRuns: number;
  engineOrder: readonly [EngineName, EngineName];
  deliveryMode: DeliveryMode;
}

interface AttemptContext {
  phase: AttemptPhase;
  index: number;
  deliveryMode: DeliveryMode;
}

interface BaseAttempt {
  phase: AttemptPhase;
  index: number;
  promptTokens: number;
  cachedPromptTokens: number;
  completionTokens: number;
  finishReason: FinishReason;
  visibleTtftMs: number | null;
  totalMs: number;
  decodeWindowMs: number | null;
  externalDecodeTokensPerSecond: number | null;
  outputSha256: string;
}

interface NativeAttempt extends BaseAttempt {
  firstTokenEventMs: number | null;
  streamedTokenEvents: number;
  internal: NativeRuntimeMetrics;
}

interface RapidAttempt extends BaseAttempt {
  contentChunks: number | null;
}

interface RapidMlxStreamingParams extends ChatCompletionCreateParamsStreaming {
  chat_template_kwargs: { enable_thinking: false };
  top_k: number;
  min_p: number;
  repetition_penalty: number;
}

interface RapidMlxBufferedParams
  extends ChatCompletionCreateParamsNonStreaming {
  chat_template_kwargs: { enable_thinking: false };
  top_k: number;
  min_p: number;
  repetition_penalty: number;
}

interface RapidStreamState {
  firstVisibleAt?: number;
  finishReason: FinishReason;
  usage: ChatCompletionChunk['usage'];
  contentChunks: number;
  output: string;
}

interface AttemptSummary {
  count: number;
  promptTokens: number[];
  completionTokens: number[];
  medianVisibleTtftMs: number | null;
  medianTotalMs: number;
  medianExternalDecodeTokensPerSecond: number | null;
}

function requiredEnvironmentVariable(name: string): string {
  const value = process.env[name];
  if (value === undefined || value.trim() === '') {
    throw new Error(`Required environment variable is missing: ${name}`);
  }
  return value;
}

function parsePositiveInteger(
  name: string,
  fallback: number,
  minimum = 1
): number {
  const raw = process.env[name];
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new Error(
      `${name} must be an integer greater than or equal to ${minimum}`
    );
  }
  return value;
}

function parseEngineOrder(
  value = process.env.ECHO_PERF_ENGINE_ORDER
): readonly [EngineName, EngineName] {
  switch (value ?? 'native-first') {
    case 'native-first':
      return ['native', 'rapid-mlx'];
    case 'rapid-first':
      return ['rapid-mlx', 'native'];
    default:
      throw new Error(
        'ECHO_PERF_ENGINE_ORDER must be native-first or rapid-first'
      );
  }
}

function parseDeliveryMode(
  value = process.env.ECHO_PERF_DELIVERY_MODE
): DeliveryMode {
  const mode = value ?? 'streaming';
  switch (mode) {
    case 'streaming':
    case 'buffered':
      return mode;
    default:
      throw new Error('ECHO_PERF_DELIVERY_MODE must be streaming or buffered');
  }
}

function loadConfig(): GateConfig {
  return {
    nativeBinaryPath: requiredEnvironmentVariable(
      'ECHO_PERF_NATIVE_INFERENCE_BIN'
    ),
    modelDirectory: requiredEnvironmentVariable('ECHO_PERF_MODEL'),
    rapidMlxBinaryPath: requiredEnvironmentVariable('ECHO_PERF_RAPID_MLX_BIN'),
    rapidMlxWorkingDirectory: requiredEnvironmentVariable(
      'ECHO_PERF_RAPID_MLX_CWD'
    ),
    outputPath: requiredEnvironmentVariable('ECHO_PERF_OUTPUT'),
    port: parsePositiveInteger('ECHO_PERF_PORT', 87_66),
    warmupRuns: parsePositiveInteger('ECHO_PERF_WARMUP_RUNS', 1),
    measuredRuns: parsePositiveInteger('ECHO_PERF_MEASURED_RUNS', 7),
    engineOrder: parseEngineOrder(),
    deliveryMode: parseDeliveryMode(),
  };
}

function assertInputPaths(config: GateConfig): void {
  for (const [label, path] of [
    ['native binary', config.nativeBinaryPath],
    ['model directory', config.modelDirectory],
    ['Rapid-MLX binary', config.rapidMlxBinaryPath],
    ['Rapid-MLX working directory', config.rapidMlxWorkingDirectory],
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

function attemptPhases(config: GateConfig): AttemptPhase[] {
  return [
    ...Array.from({ length: config.warmupRuns }, () => 'warmup' as const),
    ...Array.from({ length: config.measuredRuns }, () => 'measured' as const),
  ];
}

function decodeRate(completionTokens: number, decodeWindowMs: number): number {
  if (completionTokens < 2 || decodeWindowMs <= 0) {
    throw new Error(
      'decode rate requires at least two tokens and positive time'
    );
  }
  return (completionTokens - 1) / (decodeWindowMs / 1_000);
}

function nativeCommand(
  phase: AttemptPhase,
  index: number,
  deliveryMode: DeliveryMode
): NativeGenerateCommand {
  const identity = `${phase}-${index}`;
  return {
    type: 'generate',
    request_id: `native-performance-${identity}`,
    instance_id: `native-performance-${identity}`,
    state_transition: 'initial',
    stream_tokens: deliveryMode === 'streaming',
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
}

async function runNativeAttempt(
  client: NativeInferenceClient,
  stateRoots: EphemeralNativeStateRoots,
  context: AttemptContext
): Promise<NativeAttempt> {
  const { phase, index, deliveryMode } = context;
  const command = nativeCommand(phase, index, deliveryMode);
  await stateRoots.open(client, command.instance_id);
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
  if (
    deliveryMode === 'streaming' &&
    (firstTokenAt === undefined || firstVisibleAt === undefined)
  ) {
    throw new Error(`native ${phase} ${index} emitted no visible token`);
  }
  if (deliveryMode === 'buffered' && streamedTokenEvents !== 0) {
    throw new Error(
      `native ${phase} ${index} emitted token events in buffered mode`
    );
  }
  const completionTokens = event.response.metrics.generated_tokens;
  const visibleTtftMs =
    firstVisibleAt === undefined ? null : firstVisibleAt - startedAt;
  const decodeWindowMs =
    firstVisibleAt === undefined ? null : completedAt - firstVisibleAt;
  return {
    phase,
    index,
    promptTokens: event.response.metrics.input_tokens_processed,
    cachedPromptTokens: event.response.metrics.cached_prefix_tokens,
    completionTokens,
    finishReason: event.response.finish_reason,
    firstTokenEventMs:
      firstTokenAt === undefined ? null : firstTokenAt - startedAt,
    visibleTtftMs,
    totalMs: completedAt - startedAt,
    decodeWindowMs,
    externalDecodeTokensPerSecond:
      decodeWindowMs === null
        ? null
        : decodeRate(completionTokens, decodeWindowMs),
    streamedTokenEvents,
    outputSha256: sha256Text(event.text),
    internal: event.response.metrics,
  };
}

async function runNativeBlock(
  config: GateConfig
): Promise<Record<string, unknown>> {
  const libraryPath = requiredEnvironmentVariable('ECHO_NATIVE_LIBRARY_PATH');
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
  const attempts: NativeAttempt[] = [];
  try {
    const ready = await client.ready();
    const readyElapsedMs = performance.now() - spawnedAt;
    let warmupIndex = 0;
    let measuredIndex = 0;
    for (const phase of attemptPhases(config)) {
      const index =
        phase === 'warmup' ? (warmupIndex += 1) : (measuredIndex += 1);
      attempts.push(
        await runNativeAttempt(client, stateRoots, {
          phase,
          index,
          deliveryMode: config.deliveryMode,
        })
      );
    }
    return {
      readyElapsedMs,
      ready,
      attempts,
      summary: summarizeAttempts(attempts),
    };
  } finally {
    try {
      await client.shutdown();
    } finally {
      stateRoots.dispose();
    }
  }
}

function rapidStreamingParams(model: string): RapidMlxStreamingParams {
  return {
    model,
    messages: [{ role: 'user', content: BENCHMARK_PROMPT }],
    max_tokens: MAX_NEW_TOKENS,
    temperature: 0,
    top_p: 1,
    top_k: 0,
    min_p: 0,
    repetition_penalty: 1,
    presence_penalty: 0,
    frequency_penalty: 0,
    seed: 42,
    stream: true,
    stream_options: { include_usage: true },
    chat_template_kwargs: { enable_thinking: false },
  };
}

function rapidBufferedParams(model: string): RapidMlxBufferedParams {
  return {
    model,
    messages: [{ role: 'user', content: BENCHMARK_PROMPT }],
    max_tokens: MAX_NEW_TOKENS,
    temperature: 0,
    top_p: 1,
    top_k: 0,
    min_p: 0,
    repetition_penalty: 1,
    presence_penalty: 0,
    frequency_penalty: 0,
    seed: 42,
    stream: false,
    chat_template_kwargs: { enable_thinking: false },
  };
}

function consumeRapidChunk(
  state: RapidStreamState,
  chunk: ChatCompletionChunk,
  observedAt: number
): void {
  state.usage = chunk.usage ?? state.usage;
  for (const choice of chunk.choices) {
    state.finishReason = choice.finish_reason ?? state.finishReason;
    const content = choice.delta.content;
    if (typeof content === 'string' && content !== '') {
      state.firstVisibleAt ??= observedAt;
      state.contentChunks += 1;
      state.output += content;
    }
  }
}

async function runRapidStreamingAttempt(
  client: OpenAI,
  model: string,
  phase: AttemptPhase,
  index: number
): Promise<RapidAttempt> {
  const startedAt = performance.now();
  const stream = await client.chat.completions.create(
    rapidStreamingParams(model)
  );
  const state: RapidStreamState = {
    finishReason: null,
    usage: undefined,
    contentChunks: 0,
    output: '',
  };
  for await (const chunk of stream) {
    consumeRapidChunk(state, chunk, performance.now());
  }
  const completedAt = performance.now();
  if (
    state.firstVisibleAt === undefined ||
    state.usage === undefined ||
    state.usage === null
  ) {
    throw new Error(
      `Rapid-MLX ${phase} ${index} omitted visible output or usage`
    );
  }
  const decodeWindowMs = completedAt - state.firstVisibleAt;
  return {
    phase,
    index,
    promptTokens: state.usage.prompt_tokens,
    cachedPromptTokens: state.usage.prompt_tokens_details?.cached_tokens ?? 0,
    completionTokens: state.usage.completion_tokens,
    finishReason: state.finishReason,
    visibleTtftMs: state.firstVisibleAt - startedAt,
    totalMs: completedAt - startedAt,
    decodeWindowMs,
    externalDecodeTokensPerSecond: decodeRate(
      state.usage.completion_tokens,
      decodeWindowMs
    ),
    contentChunks: state.contentChunks,
    outputSha256: sha256Text(state.output),
  };
}

async function runRapidBufferedAttempt(
  client: OpenAI,
  model: string,
  phase: AttemptPhase,
  index: number
): Promise<RapidAttempt> {
  const startedAt = performance.now();
  const response = await client.chat.completions.create(
    rapidBufferedParams(model)
  );
  const completedAt = performance.now();
  const choice = response.choices[0];
  const output = choice?.message.content;
  if (
    choice === undefined ||
    typeof output !== 'string' ||
    output === '' ||
    response.usage === undefined
  ) {
    throw new Error(`Rapid-MLX ${phase} ${index} omitted output or usage`);
  }
  return {
    phase,
    index,
    promptTokens: response.usage.prompt_tokens,
    cachedPromptTokens:
      response.usage.prompt_tokens_details?.cached_tokens ?? 0,
    completionTokens: response.usage.completion_tokens,
    finishReason: choice.finish_reason,
    visibleTtftMs: null,
    totalMs: completedAt - startedAt,
    decodeWindowMs: null,
    externalDecodeTokensPerSecond: null,
    contentChunks: null,
    outputSha256: sha256Text(output),
  };
}

async function runRapidAttempt(
  client: OpenAI,
  model: string,
  context: AttemptContext
): Promise<RapidAttempt> {
  return context.deliveryMode === 'streaming'
    ? await runRapidStreamingAttempt(
        client,
        model,
        context.phase,
        context.index
      )
    : await runRapidBufferedAttempt(
        client,
        model,
        context.phase,
        context.index
      );
}

async function readRapidStatus(baseURL: string): Promise<unknown> {
  const response = await fetch(`${baseURL}/status`);
  if (!response.ok) {
    throw new Error(`Rapid-MLX status returned HTTP ${response.status}`);
  }
  return await response.json();
}

async function runRapidBlock(
  config: GateConfig
): Promise<Record<string, unknown>> {
  const logPath = join(dirname(config.outputPath), 'rapid-mlx-server.log');
  const servedModelName = 'qwen36-native-rapid-performance';
  const server = await startLocalModelServer({
    target: {
      id: 'qwen36-native-rapid-performance',
      displayName: 'Qwen3.6 native versus Rapid performance',
      modelPath: config.modelDirectory,
      servedModelName,
    },
    rapidMlxBin: config.rapidMlxBinaryPath,
    rapidMlxWorkingDirectory: config.rapidMlxWorkingDirectory,
    port: config.port,
    logPath,
    kvCacheDtype: 'bf16',
    prefixCacheMode: 'disabled',
  });
  const client = new OpenAI({
    apiKey: 'local-native-rapid-performance',
    baseURL: server.baseURL,
    timeout: 180_000,
  });
  const attempts: RapidAttempt[] = [];
  const result: Record<string, unknown> = {
    readyElapsedMs: server.startupElapsedMs,
    logPath,
    readyStatus: await readRapidStatus(server.baseURL),
    attempts,
  };
  try {
    let warmupIndex = 0;
    let measuredIndex = 0;
    for (const phase of attemptPhases(config)) {
      const index =
        phase === 'warmup' ? (warmupIndex += 1) : (measuredIndex += 1);
      attempts.push(
        await runRapidAttempt(client, servedModelName, {
          phase,
          index,
          deliveryMode: config.deliveryMode,
        })
      );
      if (phase === 'warmup' && warmupIndex === config.warmupRuns) {
        result.statusAfterWarmup = await readRapidStatus(server.baseURL);
      }
    }
    result.finalStatus = await readRapidStatus(server.baseURL);
    result.summary = summarizeAttempts(attempts);
  } finally {
    result.serverExit = await server.stop();
    server.cleanup();
  }
  return result;
}

function measuredAttempts<T extends BaseAttempt>(attempts: readonly T[]): T[] {
  return attempts.filter((attempt) => attempt.phase === 'measured');
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

function nullableMedian(values: readonly (number | null)[]): number | null {
  const available = values.filter((value): value is number => value !== null);
  if (available.length === 0) {
    return null;
  }
  if (available.length !== values.length) {
    throw new Error('metric availability differs between measured attempts');
  }
  return median(available);
}

function summarizeAttempts(attempts: readonly BaseAttempt[]): AttemptSummary {
  const measured = measuredAttempts(attempts);
  return {
    count: measured.length,
    promptTokens: measured.map((attempt) => attempt.promptTokens),
    completionTokens: measured.map((attempt) => attempt.completionTokens),
    medianVisibleTtftMs: nullableMedian(
      measured.map((attempt) => attempt.visibleTtftMs)
    ),
    medianTotalMs: median(measured.map((attempt) => attempt.totalMs)),
    medianExternalDecodeTokensPerSecond: nullableMedian(
      measured.map((attempt) => attempt.externalDecodeTokensPerSecond)
    ),
  };
}

function allEqual(values: readonly number[], expected: number): boolean {
  return values.every((value) => value === expected);
}

function nullableRatio(
  numerator: number | null,
  denominator: number | null
): number | null {
  if (numerator === null || denominator === null) {
    return null;
  }
  return numerator / denominator;
}

function isAtMostWithTolerance(ratio: number | null): boolean | null {
  return ratio === null ? null : ratio <= 1 + PERFORMANCE_TOLERANCE;
}

function isAtLeastWithTolerance(ratio: number | null): boolean | null {
  return ratio === null ? null : ratio >= 1 - PERFORMANCE_TOLERANCE;
}

function compareBlocks(
  nativeAttempts: readonly NativeAttempt[],
  rapidAttempts: readonly RapidAttempt[]
): Record<string, unknown> {
  const nativeMeasured = measuredAttempts(nativeAttempts);
  const rapidMeasured = measuredAttempts(rapidAttempts);
  const nativeSummary = summarizeAttempts(nativeAttempts);
  const rapidSummary = summarizeAttempts(rapidAttempts);
  const comparableConditions = {
    measuredRunCountsMatch:
      nativeMeasured.length === rapidMeasured.length &&
      nativeMeasured.length > 0,
    promptTokenCountsMatch:
      nativeSummary.promptTokens.length === rapidSummary.promptTokens.length &&
      nativeSummary.promptTokens.every(
        (value, index) => value === rapidSummary.promptTokens[index]
      ),
    fixedNativeCompletionLength: allEqual(
      nativeSummary.completionTokens,
      MAX_NEW_TOKENS
    ),
    fixedRapidCompletionLength: allEqual(
      rapidSummary.completionTokens,
      MAX_NEW_TOKENS
    ),
    nativePrefixCacheDisabled: nativeMeasured.every(
      (attempt) => attempt.cachedPromptTokens === 0
    ),
    rapidPrefixCacheDisabled: rapidMeasured.every(
      (attempt) => attempt.cachedPromptTokens === 0
    ),
  };
  const greedyOutputHashesMatch = nativeMeasured.map(
    (attempt, index) =>
      attempt.outputSha256 === rapidMeasured[index]?.outputSha256
  );
  const visibleTtftNativeOverRapid = nullableRatio(
    nativeSummary.medianVisibleTtftMs,
    rapidSummary.medianVisibleTtftMs
  );
  const decodeRateNativeOverRapid = nullableRatio(
    nativeSummary.medianExternalDecodeTokensPerSecond,
    rapidSummary.medianExternalDecodeTokensPerSecond
  );
  const ratios = {
    visibleTtftNativeOverRapid,
    totalTimeNativeOverRapid:
      nativeSummary.medianTotalMs / rapidSummary.medianTotalMs,
    decodeRateNativeOverRapid,
  };
  const performance = {
    visibleTtftWithinFivePercent: isAtMostWithTolerance(
      ratios.visibleTtftNativeOverRapid
    ),
    totalTimeWithinFivePercent:
      ratios.totalTimeNativeOverRapid <= 1 + PERFORMANCE_TOLERANCE,
    decodeRateWithinFivePercent: isAtLeastWithTolerance(
      ratios.decodeRateNativeOverRapid
    ),
  };
  return {
    comparableConditions,
    greedyOutputHashesMatch,
    nativeSummary,
    rapidSummary,
    ratios,
    performance,
    admitted:
      Object.values(comparableConditions).every(Boolean) &&
      greedyOutputHashesMatch.every(Boolean) &&
      performance.totalTimeWithinFivePercent &&
      performance.visibleTtftWithinFivePercent !== false &&
      performance.decodeRateWithinFivePercent !== false,
  };
}

function asNativeAttempts(block: Record<string, unknown>): NativeAttempt[] {
  return block.attempts as NativeAttempt[];
}

function asRapidAttempts(block: Record<string, unknown>): RapidAttempt[] {
  return block.attempts as RapidAttempt[];
}

test('admits both sequential engine orders and rejects ambiguous input', () => {
  expect(parseEngineOrder('native-first')).toEqual(['native', 'rapid-mlx']);
  expect(parseEngineOrder('rapid-first')).toEqual(['rapid-mlx', 'native']);
  expect(() => parseEngineOrder('alternating')).toThrow(
    'ECHO_PERF_ENGINE_ORDER must be native-first or rapid-first'
  );
  expect(parseDeliveryMode('streaming')).toBe('streaming');
  expect(parseDeliveryMode('buffered')).toBe('buffered');
  expect(() => parseDeliveryMode('token-batches')).toThrow(
    'ECHO_PERF_DELIVERY_MODE must be streaming or buffered'
  );
});

// The model and two runtimes must be loaded sequentially on one unified-memory GPU.
liveTest(
  'measures matched native and Rapid-MLX warm inference',
  async () => {
    const config = loadConfig();
    assertInputPaths(config);
    mkdirSync(dirname(config.outputPath), { recursive: true });
    const result: Record<string, unknown> = {
      schemaVersion: 2,
      generatedAt: new Date().toISOString(),
      conditions: {
        order: config.engineOrder,
        prompt: BENCHMARK_PROMPT,
        promptSha256: sha256Text(BENCHMARK_PROMPT),
        maxNewTokens: MAX_NEW_TOKENS,
        warmupRuns: config.warmupRuns,
        measuredRuns: config.measuredRuns,
        deliveryMode: config.deliveryMode,
        externalTtftAndDecodeRateAvailable: config.deliveryMode === 'streaming',
        sampling: {
          temperature: 0,
          topP: 1,
          topK: 0,
          minP: 0,
          repetitionPenalty: 1,
          presencePenalty: 0,
          seed: 42,
        },
        kvCacheDtype: 'bf16',
        prefixCache: 'disabled',
        speculativeDecode: 'disabled',
        pflash: 'off',
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
        rapidMlx: gitState(config.rapidMlxWorkingDirectory),
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
      engines: {},
    };
    writeResult(config.outputPath, result);

    const engines = result.engines as Record<string, unknown>;
    for (const engine of config.engineOrder) {
      if (engine === 'native') {
        engines.native = await runNativeBlock(config);
      } else {
        engines.rapidMlx = await runRapidBlock(config);
      }
      writeResult(config.outputPath, result);
    }
    const native = engines.native as Record<string, unknown>;
    const rapid = engines.rapidMlx as Record<string, unknown>;
    const comparison = compareBlocks(
      asNativeAttempts(native),
      asRapidAttempts(rapid)
    );
    result.comparison = comparison;
    writeResult(config.outputPath, result);

    expect(comparison.comparableConditions).toEqual({
      measuredRunCountsMatch: true,
      promptTokenCountsMatch: true,
      fixedNativeCompletionLength: true,
      fixedRapidCompletionLength: true,
      nativePrefixCacheDisabled: true,
      rapidPrefixCacheDisabled: true,
    });
    expect(comparison.greedyOutputHashesMatch).toEqual(
      Array.from({ length: config.measuredRuns }, () => true)
    );
    expect(comparison.performance).toEqual(
      config.deliveryMode === 'streaming'
        ? {
            visibleTtftWithinFivePercent: true,
            totalTimeWithinFivePercent: true,
            decodeRateWithinFivePercent: true,
          }
        : {
            visibleTtftWithinFivePercent: null,
            totalTimeWithinFivePercent: true,
            decodeRateWithinFivePercent: null,
          }
    );
    expect(comparison.admitted).toBe(true);
  },
  30 * 60_000
);
