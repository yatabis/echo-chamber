/* eslint-disable no-await-in-loop -- Each engine must advance one linear session at a time. */

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
  NativeCompletedEvent,
  NativeGenerateCommand,
} from '@echo-chamber/native-inference-adapter/protocol';

import { startLocalModelServer } from '../rapid-mlx/server-controller';

import { EphemeralNativeStateRoots } from './ephemeral-state-roots';
import {
  LONG_SESSION_DEVELOPER_PROMPT,
  LONG_SESSION_FINAL_TEXT,
  LONG_SESSION_GREEDY_SAMPLING,
  LONG_SESSION_MAX_NEW_TOKENS,
  LONG_SESSION_TOOL,
  longSessionNativeContinuationCommand,
  longSessionNativePrefixCommand,
  requireLongSessionStepToolCall,
  serializeLongSessionToolResult,
} from './long-session-workload';

import type {
  ChatCompletion,
  ChatCompletionAssistantMessageParam,
  ChatCompletionChunk,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
  ChatCompletionMessageFunctionToolCall,
  ChatCompletionMessageParam,
  ChatCompletionTool,
} from 'openai/resources/chat/completions';

const LIVE_GATE_ENABLED =
  process.env.ECHO_NATIVE_RAPID_LONG_SESSION_PERFORMANCE_GATE === '1';
const liveTest = LIVE_GATE_ENABLED ? test : test.skip;
const PERFORMANCE_TOLERANCE = 0.05;
const RAPID_PREFILL_STEP_SIZE = 2_048;
const RETAINED_NATIVE_STRICT_FIELD_TOKENS = 5;
const RETAINED_RAPID_TOOL_SUFFIX_TOKENS = 144;
const RETAINED_RAPID_PROMPT_OVERHEAD_TOKENS =
  RETAINED_RAPID_TOOL_SUFFIX_TOKENS - RETAINED_NATIVE_STRICT_FIELD_TOKENS;

type AttemptPhase = 'warmup' | 'measured';
type SessionCacheSlot = 'rolling';

interface GateConfig {
  nativeBinaryPath: string;
  modelDirectory: string;
  rapidMlxBinaryPath: string;
  rapidMlxWorkingDirectory: string;
  outputPath: string;
  port: number;
  measuredRuns: number;
  continuationSteps: number;
  paddingRepetitions: number;
}

interface CacheStep {
  step: number;
  promptTokens: number;
  cachedPromptTokens: number;
  processedPromptTokens: number;
  completionTokens: number;
}

interface FinalAttempt {
  phase: AttemptPhase;
  index: number;
  promptTokens: number;
  cachedPromptTokens: number;
  processedPromptTokens: number;
  completionTokens: number;
  visibleTtftMs: number;
  totalMs: number;
  outputSha256: string;
}

interface SessionRound {
  phase: AttemptPhase;
  index: number;
  cacheSteps: CacheStep[];
  final: FinalAttempt;
}

interface AttemptSummary {
  count: number;
  promptTokens: number[];
  cachedPromptTokens: number[];
  processedPromptTokens: number[];
  completionTokens: number[];
  medianVisibleTtftMs: number;
  medianTotalMs: number;
}

interface RapidCacheControl {
  mode: 'auto';
  session_id: string;
  session_slot: SessionCacheSlot;
}

interface RapidLongSessionNonStreamingParams
  extends ChatCompletionCreateParamsNonStreaming {
  cache: RapidCacheControl;
  chat_template_kwargs: { enable_thinking: false };
  top_k: number;
  min_p: number;
  repetition_penalty: number;
}

interface RapidLongSessionStreamingParams
  extends ChatCompletionCreateParamsStreaming {
  cache: RapidCacheControl;
  chat_template_kwargs: { enable_thinking: false };
  top_k: number;
  min_p: number;
  repetition_penalty: number;
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
    throw new Error(`${name} must be an integer >= ${minimum}`);
  }
  return value;
}

function loadConfig(): GateConfig {
  return {
    nativeBinaryPath: requiredEnvironmentVariable(
      'ECHO_LONG_COMPARISON_NATIVE_INFERENCE_BIN'
    ),
    modelDirectory: requiredEnvironmentVariable('ECHO_LONG_COMPARISON_MODEL'),
    rapidMlxBinaryPath: requiredEnvironmentVariable(
      'ECHO_LONG_COMPARISON_RAPID_MLX_BIN'
    ),
    rapidMlxWorkingDirectory: requiredEnvironmentVariable(
      'ECHO_LONG_COMPARISON_RAPID_MLX_CWD'
    ),
    outputPath: requiredEnvironmentVariable('ECHO_LONG_COMPARISON_OUTPUT'),
    port: parsePositiveInteger('ECHO_LONG_COMPARISON_PORT', 8_785),
    measuredRuns: parsePositiveInteger('ECHO_LONG_COMPARISON_MEASURED_RUNS', 3),
    continuationSteps: parsePositiveInteger('ECHO_LONG_COMPARISON_STEPS', 8, 2),
    paddingRepetitions: parsePositiveInteger(
      'ECHO_LONG_COMPARISON_PADDING_REPETITIONS',
      256
    ),
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

function phases(config: GateConfig): AttemptPhase[] {
  return [
    'warmup',
    ...Array.from<AttemptPhase>({
      length: config.measuredRuns,
    }).fill('measured'),
  ];
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
    throw new Error('median upper value is missing');
  }
  if (sorted.length % 2 === 1) {
    return upper;
  }
  const lower = sorted[middle - 1];
  if (lower === undefined) {
    throw new Error('median lower value is missing');
  }
  return (lower + upper) / 2;
}

function measuredFinals(rounds: readonly SessionRound[]): FinalAttempt[] {
  return rounds
    .filter((round) => round.phase === 'measured')
    .map((round) => round.final);
}

function measuredRounds(rounds: readonly SessionRound[]): SessionRound[] {
  return rounds.filter((round) => round.phase === 'measured');
}

function arraysEqual(
  left: readonly number[],
  right: readonly number[]
): boolean {
  return (
    left.length === right.length &&
    left.every((value, index) => value === right[index])
  );
}

function promptGrowth(steps: readonly CacheStep[]): number[] {
  return steps.slice(1).map((step, index) => {
    const previous = steps[index];
    if (previous === undefined) {
      throw new Error('prompt-growth predecessor is missing');
    }
    return step.promptTokens - previous.promptTokens;
  });
}

function expectedNativeCachedTokens(steps: readonly CacheStep[]): number[] {
  return steps.map((_, index) => {
    if (index === 0) {
      return 0;
    }
    const previous = steps[index - 1];
    if (previous === undefined) {
      throw new Error('native cache predecessor is missing');
    }
    return previous.promptTokens + previous.completionTokens;
  });
}

function expectedRapidCachedTokens(steps: readonly CacheStep[]): number[] {
  let publishedCheckpoint = 0;
  return steps.map((_, index) => {
    const reusableForThisRequest = publishedCheckpoint;
    const previous = steps[index - 1];
    if (previous !== undefined) {
      const previousCommittedTokens =
        previous.promptTokens + previous.completionTokens;
      const checkpointPublishedByThisRequest =
        Math.floor(previousCommittedTokens / RAPID_PREFILL_STEP_SIZE) *
        RAPID_PREFILL_STEP_SIZE;
      publishedCheckpoint = Math.max(
        publishedCheckpoint,
        checkpointPublishedByThisRequest
      );
    }
    return reusableForThisRequest;
  });
}

function summarize(rounds: readonly SessionRound[]): AttemptSummary {
  const attempts = measuredFinals(rounds);
  return {
    count: attempts.length,
    promptTokens: attempts.map((attempt) => attempt.promptTokens),
    cachedPromptTokens: attempts.map((attempt) => attempt.cachedPromptTokens),
    processedPromptTokens: attempts.map(
      (attempt) => attempt.processedPromptTokens
    ),
    completionTokens: attempts.map((attempt) => attempt.completionTokens),
    medianVisibleTtftMs: median(
      attempts.map((attempt) => attempt.visibleTtftMs)
    ),
    medianTotalMs: median(attempts.map((attempt) => attempt.totalMs)),
  };
}

async function runNativeGeneration(
  client: NativeInferenceClient,
  command: NativeGenerateCommand
): Promise<{
  event: NativeCompletedEvent;
  visibleTtftMs: number;
  totalMs: number;
}> {
  const startedAt = performance.now();
  let firstVisibleAt: number | undefined;
  const event = await client.generate(command, (token) => {
    if (token.text !== undefined && token.text !== '') {
      firstVisibleAt ??= performance.now();
    }
  });
  const completedAt = performance.now();
  if (firstVisibleAt === undefined) {
    throw new Error(`${command.request_id} emitted no visible token`);
  }
  return {
    event,
    visibleTtftMs: firstVisibleAt - startedAt,
    totalMs: completedAt - startedAt,
  };
}

async function runNativeRound(
  client: NativeInferenceClient,
  stateRoots: EphemeralNativeStateRoots,
  input: {
    phase: AttemptPhase;
    index: number;
    continuationSteps: number;
    padding: string;
  }
): Promise<SessionRound> {
  const instanceId = `long-comparison-native-${input.phase}-${input.index}`;
  await stateRoots.open(client, instanceId);
  const prefix = await runNativeGeneration(
    client,
    longSessionNativePrefixCommand(instanceId)
  );
  requireLongSessionStepToolCall(prefix.event, 1);
  const cacheSteps: CacheStep[] = [
    {
      step: 0,
      promptTokens: prefix.event.response.metrics.input_tokens_processed,
      cachedPromptTokens: 0,
      processedPromptTokens:
        prefix.event.response.metrics.input_tokens_processed,
      completionTokens: prefix.event.response.metrics.generated_tokens,
    },
  ];
  let previous = prefix.event;
  let finalTiming:
    | { event: NativeCompletedEvent; visibleTtftMs: number; totalMs: number }
    | undefined;

  for (let step = 1; step <= input.continuationSteps; step += 1) {
    const result = serializeLongSessionToolResult(
      step,
      input.continuationSteps,
      input.padding
    );
    const attempt = await runNativeGeneration(
      client,
      longSessionNativeContinuationCommand(instanceId, previous, step, result)
    );
    const metrics = attempt.event.response.metrics;
    cacheSteps.push({
      step,
      promptTokens:
        metrics.cached_prefix_tokens + metrics.input_tokens_processed,
      cachedPromptTokens: metrics.cached_prefix_tokens,
      processedPromptTokens: metrics.input_tokens_processed,
      completionTokens: metrics.generated_tokens,
    });
    if (step < input.continuationSteps) {
      requireLongSessionStepToolCall(attempt.event, step + 1);
    } else {
      finalTiming = attempt;
    }
    previous = attempt.event;
  }

  if (
    finalTiming === undefined ||
    finalTiming.event.text.trim() !== LONG_SESSION_FINAL_TEXT
  ) {
    throw new Error('native final long-session output did not match fixture');
  }
  const finalStep = cacheSteps[cacheSteps.length - 1];
  if (finalStep === undefined) {
    throw new Error('native final cache step is missing');
  }
  return {
    phase: input.phase,
    index: input.index,
    cacheSteps,
    final: {
      phase: input.phase,
      index: input.index,
      ...finalStep,
      visibleTtftMs: finalTiming.visibleTtftMs,
      totalMs: finalTiming.totalMs,
      outputSha256: sha256Text(finalTiming.event.text),
    },
  };
}

async function runNativeBlock(
  config: GateConfig,
  padding: string
): Promise<Record<string, unknown>> {
  const libraryPath = requiredEnvironmentVariable('ECHO_NATIVE_LIBRARY_PATH');
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
  const rounds: SessionRound[] = [];
  try {
    const ready = await client.ready();
    let warmupIndex = 0;
    let measuredIndex = 0;
    for (const phase of phases(config)) {
      const index =
        phase === 'warmup' ? (warmupIndex += 1) : (measuredIndex += 1);
      rounds.push(
        await runNativeRound(client, stateRoots, {
          phase,
          index,
          continuationSteps: config.continuationSteps,
          padding,
        })
      );
    }
    return { ready, rounds, summary: summarize(rounds) };
  } finally {
    try {
      await client.shutdown();
    } finally {
      stateRoots.dispose();
    }
  }
}

function rapidTool(): ChatCompletionTool {
  return {
    type: 'function',
    function: {
      name: LONG_SESSION_TOOL.name,
      description: LONG_SESSION_TOOL.description,
      parameters: LONG_SESSION_TOOL.input_schema as Record<string, unknown>,
      strict: LONG_SESSION_TOOL.strict,
    },
  };
}

function rapidCache(sessionId: string): RapidCacheControl {
  return {
    mode: 'auto',
    session_id: sessionId,
    session_slot: 'rolling',
  };
}

function rapidBaseParams(input: {
  model: string;
  messages: ChatCompletionMessageParam[];
  sessionId: string;
}): Omit<RapidLongSessionNonStreamingParams, 'stream'> {
  return {
    model: input.model,
    messages: input.messages,
    tools: [rapidTool()],
    max_tokens: LONG_SESSION_MAX_NEW_TOKENS,
    temperature: LONG_SESSION_GREEDY_SAMPLING.temperature,
    top_p: LONG_SESSION_GREEDY_SAMPLING.top_p,
    top_k: LONG_SESSION_GREEDY_SAMPLING.top_k,
    min_p: LONG_SESSION_GREEDY_SAMPLING.min_p,
    repetition_penalty: LONG_SESSION_GREEDY_SAMPLING.repetition_penalty,
    presence_penalty: LONG_SESSION_GREEDY_SAMPLING.presence_penalty,
    frequency_penalty: 0,
    seed: LONG_SESSION_GREEDY_SAMPLING.seed,
    cache: rapidCache(input.sessionId),
    chat_template_kwargs: { enable_thinking: false },
  };
}

function requireRapidStepToolCall(
  response: ChatCompletion,
  step: number
): ChatCompletionMessageFunctionToolCall {
  const call = response.choices[0]?.message.tool_calls?.find(
    (candidate): candidate is ChatCompletionMessageFunctionToolCall =>
      candidate.type === 'function'
  );
  if (call === undefined || call.function.name !== LONG_SESSION_TOOL.name) {
    throw new Error(`Rapid-MLX step ${step} omitted ${LONG_SESSION_TOOL.name}`);
  }
  const input: unknown = JSON.parse(call.function.arguments);
  if (
    typeof input !== 'object' ||
    input === null ||
    !('step' in input) ||
    input.step !== step
  ) {
    throw new Error(
      `Rapid-MLX expected step ${step}, observed ${call.function.arguments}`
    );
  }
  return call;
}

function usageStep(step: number, response: ChatCompletion): CacheStep {
  const usage = (response as Partial<ChatCompletion>).usage;
  if (usage === undefined) {
    throw new Error(`Rapid-MLX step ${step} omitted usage`);
  }
  const cached = usage.prompt_tokens_details?.cached_tokens ?? 0;
  return {
    step,
    promptTokens: usage.prompt_tokens,
    cachedPromptTokens: cached,
    processedPromptTokens: usage.prompt_tokens - cached,
    completionTokens: usage.completion_tokens,
  };
}

function appendRapidToolExchange(
  messages: ChatCompletionMessageParam[],
  call: ChatCompletionMessageFunctionToolCall,
  output: string
): void {
  const assistant: ChatCompletionAssistantMessageParam = {
    role: 'assistant',
    content: null,
    tool_calls: [
      {
        id: call.id,
        type: 'function',
        function: {
          name: call.function.name,
          arguments: call.function.arguments,
        },
      },
    ],
  };
  messages.push(assistant, {
    role: 'tool',
    tool_call_id: call.id,
    content: output,
  });
}

function consumeRapidFinalChunk(
  state: {
    output: string;
    firstVisibleAt?: number;
    usage?: ChatCompletionChunk['usage'];
  },
  chunk: ChatCompletionChunk
): void {
  state.usage = chunk.usage ?? state.usage;
  for (const choice of chunk.choices) {
    const content = choice.delta.content;
    if (typeof content === 'string' && content !== '') {
      state.firstVisibleAt ??= performance.now();
      state.output += content;
    }
  }
}

async function runRapidFinal(
  client: OpenAI,
  params: Omit<RapidLongSessionNonStreamingParams, 'stream'>,
  phase: AttemptPhase,
  index: number
): Promise<FinalAttempt> {
  const startedAt = performance.now();
  const streamingParams: RapidLongSessionStreamingParams = {
    ...params,
    stream: true,
    stream_options: { include_usage: true },
  };
  const stream = await client.chat.completions.create(streamingParams);
  const state: {
    output: string;
    firstVisibleAt?: number;
    usage?: ChatCompletionChunk['usage'];
  } = { output: '' };
  for await (const chunk of stream) {
    consumeRapidFinalChunk(state, chunk);
  }
  const completedAt = performance.now();
  if (
    state.firstVisibleAt === undefined ||
    state.usage === undefined ||
    state.usage === null
  ) {
    throw new Error('Rapid-MLX final response omitted output or usage');
  }
  if (state.output.trim() !== LONG_SESSION_FINAL_TEXT) {
    throw new Error(
      `Rapid-MLX final output was ${JSON.stringify(state.output)}`
    );
  }
  const cached = state.usage.prompt_tokens_details?.cached_tokens ?? 0;
  return {
    phase,
    index,
    promptTokens: state.usage.prompt_tokens,
    cachedPromptTokens: cached,
    processedPromptTokens: state.usage.prompt_tokens - cached,
    completionTokens: state.usage.completion_tokens,
    visibleTtftMs: state.firstVisibleAt - startedAt,
    totalMs: completedAt - startedAt,
    outputSha256: sha256Text(state.output),
  };
}

async function runRapidRound(
  client: OpenAI,
  input: {
    model: string;
    phase: AttemptPhase;
    index: number;
    continuationSteps: number;
    padding: string;
  }
): Promise<SessionRound> {
  const sessionId = `long-comparison-rapid-${input.phase}-${input.index}`;
  const messages: ChatCompletionMessageParam[] = [
    { role: 'user', content: LONG_SESSION_DEVELOPER_PROMPT },
  ];
  const cacheSteps: CacheStep[] = [];
  const initial = await client.chat.completions.create({
    ...rapidBaseParams({ model: input.model, messages, sessionId }),
    stream: false,
  });
  cacheSteps.push(usageStep(0, initial));
  let call = requireRapidStepToolCall(initial, 1);

  for (let step = 1; step <= input.continuationSteps; step += 1) {
    appendRapidToolExchange(
      messages,
      call,
      serializeLongSessionToolResult(
        step,
        input.continuationSteps,
        input.padding
      )
    );
    const params = rapidBaseParams({
      model: input.model,
      messages,
      sessionId,
    });
    if (step === input.continuationSteps) {
      const final = await runRapidFinal(
        client,
        params,
        input.phase,
        input.index
      );
      cacheSteps.push({
        step,
        promptTokens: final.promptTokens,
        cachedPromptTokens: final.cachedPromptTokens,
        processedPromptTokens: final.processedPromptTokens,
        completionTokens: final.completionTokens,
      });
      return { phase: input.phase, index: input.index, cacheSteps, final };
    }
    const response = await client.chat.completions.create({
      ...params,
      stream: false,
    });
    cacheSteps.push(usageStep(step, response));
    call = requireRapidStepToolCall(response, step + 1);
  }
  throw new Error('Rapid-MLX long-session loop did not produce final output');
}

async function readRapidStatus(baseURL: string): Promise<unknown> {
  const response = await fetch(`${baseURL}/status`);
  if (!response.ok) {
    throw new Error(`Rapid-MLX status returned HTTP ${response.status}`);
  }
  return await response.json();
}

async function runRapidBlock(
  config: GateConfig,
  padding: string
): Promise<Record<string, unknown>> {
  const logPath = join(
    dirname(config.outputPath),
    'rapid-mlx-long-session-server.log'
  );
  const servedModelName = 'qwen36-native-rapid-long-session';
  const server = await startLocalModelServer({
    target: {
      id: 'qwen36-native-rapid-long-session',
      displayName: 'Qwen3.6 native versus Rapid long session',
      modelPath: config.modelDirectory,
      servedModelName,
    },
    rapidMlxBin: config.rapidMlxBinaryPath,
    rapidMlxWorkingDirectory: config.rapidMlxWorkingDirectory,
    port: config.port,
    logPath,
    kvCacheDtype: 'bf16',
    prefixCacheMode: 'enabled',
  });
  const client = new OpenAI({
    apiKey: 'local-native-rapid-long-session',
    baseURL: server.baseURL,
    timeout: 180_000,
  });
  const rounds: SessionRound[] = [];
  const result: Record<string, unknown> = {
    readyElapsedMs: server.startupElapsedMs,
    readyStatus: await readRapidStatus(server.baseURL),
    logPath,
    rounds,
  };
  try {
    let warmupIndex = 0;
    let measuredIndex = 0;
    for (const phase of phases(config)) {
      const index =
        phase === 'warmup' ? (warmupIndex += 1) : (measuredIndex += 1);
      rounds.push(
        await runRapidRound(client, {
          model: servedModelName,
          phase,
          index,
          continuationSteps: config.continuationSteps,
          padding,
        })
      );
    }
    result.finalStatus = await readRapidStatus(server.baseURL);
    result.summary = summarize(rounds);
  } finally {
    result.serverExit = await server.stop();
    server.cleanup();
  }
  return result;
}

function compareBlocks(
  nativeRounds: readonly SessionRound[],
  rapidRounds: readonly SessionRound[]
): Record<string, unknown> {
  const nativeMeasuredRounds = measuredRounds(nativeRounds);
  const rapidMeasuredRounds = measuredRounds(rapidRounds);
  const native = nativeMeasuredRounds.map((round) => round.final);
  const rapid = rapidMeasuredRounds.map((round) => round.final);
  const nativeSummary = summarize(nativeRounds);
  const rapidSummary = summarize(rapidRounds);
  const paired = (
    predicate: (nativeRound: SessionRound, rapidRound: SessionRound) => boolean
  ): boolean[] =>
    nativeMeasuredRounds.map((nativeRound, index) => {
      const rapidRound = rapidMeasuredRounds[index];
      return rapidRound !== undefined && predicate(nativeRound, rapidRound);
    });
  const nativeExpectedCached = nativeMeasuredRounds.map((round) =>
    expectedNativeCachedTokens(round.cacheSteps)
  );
  const rapidExpectedCached = rapidMeasuredRounds.map((round) =>
    expectedRapidCachedTokens(round.cacheSteps)
  );
  const rapidMinusNativePromptTokens = nativeMeasuredRounds.map(
    (nativeRound, index) => {
      const rapidRound = rapidMeasuredRounds[index];
      if (rapidRound === undefined) {
        return [];
      }
      return nativeRound.cacheSteps.map(
        (step, stepIndex) =>
          (rapidRound.cacheSteps[stepIndex]?.promptTokens ?? Number.NaN) -
          step.promptTokens
      );
    }
  );
  const comparableConditions = {
    measuredRunCountsMatch: native.length === rapid.length && native.length > 0,
    logicalStepCountsMatch: paired(
      (nativeRound, rapidRound) =>
        nativeRound.cacheSteps.length === rapidRound.cacheSteps.length
    ),
    logicalStepNumbersMatch: paired((nativeRound, rapidRound) =>
      arraysEqual(
        nativeRound.cacheSteps.map((step) => step.step),
        rapidRound.cacheSteps.map((step) => step.step)
      )
    ),
    perStepPromptGrowthMatches: paired((nativeRound, rapidRound) =>
      arraysEqual(
        promptGrowth(nativeRound.cacheSteps),
        promptGrowth(rapidRound.cacheSteps)
      )
    ),
    rapidPromptOverheadMatchesAuditedAdapters: rapidMinusNativePromptTokens.map(
      (offsets) =>
        offsets.length > 0 &&
        offsets.every(
          (offset) => offset === RETAINED_RAPID_PROMPT_OVERHEAD_TOKENS
        )
    ),
    perStepCompletionTokenCountsMatch: paired((nativeRound, rapidRound) =>
      arraysEqual(
        nativeRound.cacheSteps.map((step) => step.completionTokens),
        rapidRound.cacheSteps.map((step) => step.completionTokens)
      )
    ),
    finalOutputHashesMatch: native.map(
      (attempt, index) => attempt.outputSha256 === rapid[index]?.outputSha256
    ),
    nativeReusedExactPreviousState: nativeMeasuredRounds.map((round, index) =>
      arraysEqual(
        round.cacheSteps.map((step) => step.cachedPromptTokens),
        nativeExpectedCached[index] ?? []
      )
    ),
    rapidReusedPreviouslyPublishedAlignedCheckpoint: rapidMeasuredRounds.map(
      (round, index) =>
        arraysEqual(
          round.cacheSteps.map((step) => step.cachedPromptTokens),
          rapidExpectedCached[index] ?? []
        )
    ),
    rapidSessionCacheWasUsed: rapid.map(
      (attempt) => attempt.cachedPromptTokens > 0
    ),
  };
  const ratios = {
    visibleTtftNativeOverRapid:
      nativeSummary.medianVisibleTtftMs / rapidSummary.medianVisibleTtftMs,
    totalTimeNativeOverRapid:
      nativeSummary.medianTotalMs / rapidSummary.medianTotalMs,
  };
  const performance = {
    visibleTtftWithinFivePercent:
      ratios.visibleTtftNativeOverRapid <= 1 + PERFORMANCE_TOLERANCE,
    totalTimeWithinFivePercent:
      ratios.totalTimeNativeOverRapid <= 1 + PERFORMANCE_TOLERANCE,
  };
  const allComparable = Object.values(comparableConditions).every((value) =>
    Array.isArray(value) ? value.every(Boolean) : value
  );
  return {
    comparisonClass:
      'production-contract session comparison; logical history is matched while adapter-added prompt tokens and cache publication policies remain load-bearing',
    comparableConditions,
    promptAccounting: {
      exactPromptTokenCountsMatch: native.map(
        (attempt, index) => attempt.promptTokens === rapid[index]?.promptTokens
      ),
      rapidMinusNativePromptTokens,
      auditedRetainedInitialPromptTokens: {
        nativeWithStrictToolField: 374,
        rapidWithoutInjectedSuffixOrStrictField: 369,
        rapidWithInjectedToolSuffix: 513,
      },
      auditedAdapterTokenEffects: {
        nativeStrictToolField: RETAINED_NATIVE_STRICT_FIELD_TOKENS,
        rapidInjectedToolSuffix: RETAINED_RAPID_TOOL_SUFFIX_TOKENS,
        rapidNetOverNative: RETAINED_RAPID_PROMPT_OVERHEAD_TOKENS,
      },
    },
    checkpointAlignment: {
      rapidPrefillStepSize: RAPID_PREFILL_STEP_SIZE,
      nativeExpectedCachedTokens: nativeExpectedCached,
      nativeObservedCachedTokens: nativeMeasuredRounds.map((round) =>
        round.cacheSteps.map((step) => step.cachedPromptTokens)
      ),
      rapidExpectedCachedTokens: rapidExpectedCached,
      rapidObservedCachedTokens: rapidMeasuredRounds.map((round) =>
        round.cacheSteps.map((step) => step.cachedPromptTokens)
      ),
      nativeCachedMinusRapidCached: native.map(
        (attempt, index) =>
          attempt.cachedPromptTokens -
          (rapid[index]?.cachedPromptTokens ?? Number.NaN)
      ),
    },
    nativeSummary,
    rapidSummary,
    ratios,
    performance,
    admitted:
      allComparable &&
      performance.visibleTtftWithinFivePercent &&
      performance.totalTimeWithinFivePercent,
  };
}

test('models exact native commits and next-request Rapid checkpoints', () => {
  const buildSteps = (
    prompts: readonly number[],
    cached: readonly number[]
  ): CacheStep[] =>
    prompts.map((promptTokens, index) => ({
      step: index,
      promptTokens,
      cachedPromptTokens: cached[index] ?? Number.NaN,
      processedPromptTokens: promptTokens - (cached[index] ?? Number.NaN),
      completionTokens: index === prompts.length - 1 ? 4 : 27,
    }));
  const nativeSteps = buildSteps(
    [374, 1_199, 2_024, 2_849, 3_674, 4_499, 5_324, 6_149, 6_974],
    [0, 401, 1_226, 2_051, 2_876, 3_701, 4_526, 5_351, 6_176]
  );
  const rapidSteps = buildSteps(
    [513, 1_338, 2_163, 2_988, 3_813, 4_638, 5_463, 6_288, 7_113],
    [0, 0, 0, 0, 2_048, 2_048, 2_048, 4_096, 4_096]
  );
  const round = (
    cacheSteps: CacheStep[],
    visibleTtftMs: number,
    totalMs: number
  ): SessionRound => {
    const finalStep = cacheSteps[cacheSteps.length - 1];
    if (finalStep === undefined) {
      throw new Error('synthetic final cache step is missing');
    }
    return {
      phase: 'measured',
      index: 1,
      cacheSteps,
      final: {
        phase: 'measured',
        index: 1,
        ...finalStep,
        visibleTtftMs,
        totalMs,
        outputSha256: 'matching-output',
      },
    };
  };

  expect(
    compareBlocks([round(nativeSteps, 100, 110)], [round(rapidSteps, 300, 310)])
  ).toMatchObject({
    comparableConditions: {
      measuredRunCountsMatch: true,
      logicalStepCountsMatch: [true],
      logicalStepNumbersMatch: [true],
      perStepPromptGrowthMatches: [true],
      rapidPromptOverheadMatchesAuditedAdapters: [true],
      perStepCompletionTokenCountsMatch: [true],
      finalOutputHashesMatch: [true],
      nativeReusedExactPreviousState: [true],
      rapidReusedPreviouslyPublishedAlignedCheckpoint: [true],
      rapidSessionCacheWasUsed: [true],
    },
    promptAccounting: {
      exactPromptTokenCountsMatch: [false],
      rapidMinusNativePromptTokens: [
        [139, 139, 139, 139, 139, 139, 139, 139, 139],
      ],
    },
    admitted: true,
  });
});

liveTest(
  'compares the retained repeated-continuation workload with Rapid-MLX',
  async () => {
    const config = loadConfig();
    assertInputPaths(config);
    const padding = 'state-cache-padding '.repeat(config.paddingRepetitions);
    const result: Record<string, unknown> = {
      schemaVersion: 2,
      generatedAt: new Date().toISOString(),
      conditions: {
        prompt: LONG_SESSION_DEVELOPER_PROMPT,
        promptSha256: sha256Text(LONG_SESSION_DEVELOPER_PROMPT),
        continuationSteps: config.continuationSteps,
        paddingRepetitionsPerStep: config.paddingRepetitions,
        paddingSha256: sha256Text(padding),
        maxNewTokens: LONG_SESSION_MAX_NEW_TOKENS,
        sampling: LONG_SESSION_GREEDY_SAMPLING,
        warmupRuns: 1,
        measuredRuns: config.measuredRuns,
        concurrency: 1,
        kvCacheDtype: 'bf16',
        nativeStateBoundary: 'exact committed output boundary',
        rapidStateBoundary: `${RAPID_PREFILL_STEP_SIZE}-token aligned message boundary published during the following request`,
        comparisonClass:
          'same logical tool-result history through each production adapter; prompt token counts intentionally retain adapter policy differences',
        promptTokenAudit:
          'official retained Qwen tokenizer: native strict field +5 tokens; Rapid-MLX auto-injected tool suffix +144 tokens; Rapid net +139 tokens',
        performanceTolerance: PERFORMANCE_TOLERANCE,
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
    };
    writeResult(config.outputPath, result);

    result.native = await runNativeBlock(config, padding);
    writeResult(config.outputPath, result);
    result.rapidMlx = await runRapidBlock(config, padding);
    const native = result.native as { rounds: SessionRound[] };
    const rapid = result.rapidMlx as { rounds: SessionRound[] };
    result.comparison = compareBlocks(native.rounds, rapid.rounds);
    writeResult(config.outputPath, result);

    expect(result.comparison).toMatchObject({
      comparableConditions: {
        measuredRunCountsMatch: true,
        logicalStepCountsMatch: Array.from({
          length: config.measuredRuns,
        }).fill(true),
        logicalStepNumbersMatch: Array.from({
          length: config.measuredRuns,
        }).fill(true),
        perStepPromptGrowthMatches: Array.from({
          length: config.measuredRuns,
        }).fill(true),
        rapidPromptOverheadMatchesAuditedAdapters: Array.from({
          length: config.measuredRuns,
        }).fill(true),
        perStepCompletionTokenCountsMatch: Array.from({
          length: config.measuredRuns,
        }).fill(true),
        finalOutputHashesMatch: Array.from({
          length: config.measuredRuns,
        }).fill(true),
        nativeReusedExactPreviousState: Array.from({
          length: config.measuredRuns,
        }).fill(true),
        rapidReusedPreviouslyPublishedAlignedCheckpoint: Array.from({
          length: config.measuredRuns,
        }).fill(true),
        rapidSessionCacheWasUsed: Array.from({
          length: config.measuredRuns,
        }).fill(true),
      },
      performance: {
        visibleTtftWithinFivePercent: true,
        totalTimeWithinFivePercent: true,
      },
      admitted: true,
    });
  },
  15 * 60 * 1_000
);
