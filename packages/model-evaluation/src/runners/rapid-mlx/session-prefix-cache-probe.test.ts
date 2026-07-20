/* eslint-disable no-await-in-loop -- One local GPU must process probe calls sequentially. */

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

import OpenAI from 'openai';
import { expect, test } from 'vitest';

import { buildAgentPromptMessages } from '@echo-chamber/core/agent/prompt-builder';
import { canonicalRuntimeTools } from '@echo-chamber/core/agent/runtime-tools/catalog';
import systemPromptRin from '@echo-chamber/core/llm/prompts/rin';
import { toChatCompletionTool } from '@echo-chamber/openai-adapter/openai-chat-completions-model';

import { loadEvaluationTargets } from '../../qwen36-eat-readiness/evaluation-targets';

import { startLocalModelServer } from './server-controller';

import type { LocalEvaluationTarget } from '../../qwen36-eat-readiness/types';
import type {
  ChatCompletion,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionMessageFunctionToolCall,
  ChatCompletionMessageParam,
} from 'openai/resources/chat/completions';

const LIVE_PROBE_ENABLED =
  process.env.ECHO_LOCAL_SESSION_PREFIX_CACHE_PROBE === '1';
const liveTest = LIVE_PROBE_ENABLED ? test : test.skip;
const PROBE_SESSION_ID = 'echo:rin';
const PROBE_PROFILE =
  process.env.ECHO_SESSION_PREFIX_CACHE_PROBE_PROFILE === 'compact'
    ? 'compact'
    : 'full';
const LONG_OUTPUT_INSTRUCTION = [
  'これはキャッシュ一致検証です。ツールは使わず、次の形式だけを出力してください。',
  '1から400までの整数を昇順に、それぞれ独立した行へ書いてください。',
  '各行は「0001: cache-validation-sequence」のように、4桁ゼロ埋めの番号、コロン、半角空白、固定文字列 cache-validation-sequence の順にしてください。',
  '前置き、説明、後書き、省略記号は加えないでください。',
].join('\n');

type SessionCacheSlot = 'pinned' | 'rolling';

interface RapidMlxChatParams extends ChatCompletionCreateParamsNonStreaming {
  cache?: {
    mode: 'auto';
    session_id: string;
    session_slot: SessionCacheSlot;
  };
  chat_template_kwargs: { enable_thinking: false };
  top_k: number;
}

interface ProbeScenario {
  label: string;
  messages: ChatCompletionMessageParam[];
  sessionSlot: SessionCacheSlot;
}

interface ProbeAttempt {
  requestedMaxTokens: number;
  elapsedMs: number;
  cachedInputTokens: number;
  promptTokens: number;
  completionTokens: number;
  comparableOutput: unknown;
}

function requiredEnvironmentVariable(name: string): string {
  const value = process.env[name];
  if (value === undefined || value.trim() === '') {
    throw new Error(`Required environment variable is missing: ${name}`);
  }
  return value;
}

function commandOutput(command: string, args: string[], cwd: string): string {
  return execFileSync(command, args, { cwd, encoding: 'utf8' }).trim();
}

function writeResult(path: string, result: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(result, null, 2)}\n`, 'utf8');
}

function createBaseMessages(): ChatCompletionMessageParam[] {
  const promptMessages = buildAgentPromptMessages({
    systemPrompt: systemPromptRin,
    currentDatetime: new Date('2026-07-19T06:00:00.000Z'),
    latestContext: null,
    relatedMemories: [],
  });
  const convertedPrompt = promptMessages.map((message) => ({
    role: 'user' as const,
    content: message.content,
  }));

  return [
    ...convertedPrompt,
    {
      role: 'assistant',
      content: null,
      tool_calls: [
        {
          id: 'check_notifications',
          type: 'function',
          function: { name: 'check_notifications', arguments: '{}' },
        },
      ],
    },
    {
      role: 'tool',
      tool_call_id: 'check_notifications',
      content: JSON.stringify({ success: true, notifications: [] }),
    },
  ];
}

function appendToolExchange(
  messages: readonly ChatCompletionMessageParam[],
  input: {
    callId: string;
    toolName: string;
    arguments: string;
    result: unknown;
  }
): ChatCompletionMessageParam[] {
  return [
    ...messages,
    {
      role: 'assistant',
      content: null,
      tool_calls: [
        {
          id: input.callId,
          type: 'function',
          function: { name: input.toolName, arguments: input.arguments },
        },
      ],
    },
    {
      role: 'tool',
      tool_call_id: input.callId,
      content: JSON.stringify(input.result),
    },
  ];
}

function createProbeScenarios(): ProbeScenario[] {
  const startup = createBaseMessages();
  const afterListNotes = appendToolExchange(startup, {
    callId: 'list_notes_1',
    toolName: 'list_notes',
    arguments: '{}',
    result: { success: true, notes: [] },
  });
  const longCheckpointPayload = 'checkpoint datum '.repeat(1200).trim();
  const afterFirstLongResult = appendToolExchange(afterListNotes, {
    callId: 'search_memory_long_1',
    toolName: 'search_memory',
    arguments: '{"query":"checkpoint fixture one"}',
    result: {
      success: true,
      memories: [{ id: 'fixture-1', content: longCheckpointPayload }],
    },
  });
  const afterFirstCheckpoint = appendToolExchange(afterFirstLongResult, {
    callId: 'list_notes_checkpoint_1',
    toolName: 'list_notes',
    arguments: '{}',
    result: { success: true, notes: [] },
  });
  const afterSecondLongResult = appendToolExchange(afterFirstCheckpoint, {
    callId: 'search_memory_long_2',
    toolName: 'search_memory',
    arguments: '{"query":"checkpoint fixture two"}',
    result: {
      success: true,
      memories: [
        {
          id: 'fixture-2',
          content: `second ${longCheckpointPayload}`,
        },
      ],
    },
  });
  const afterSecondCheckpoint = appendToolExchange(afterSecondLongResult, {
    callId: 'list_notes_checkpoint_2',
    toolName: 'list_notes',
    arguments: '{}',
    result: { success: true, notes: [] },
  });
  const afterReplacementObserved = appendToolExchange(afterSecondCheckpoint, {
    callId: 'list_notes_observe_replacement',
    toolName: 'list_notes',
    arguments: '{}',
    result: { success: true, notes: [] },
  });

  return [
    { label: 'startup', messages: startup, sessionSlot: 'pinned' },
    {
      label: 'after-list-notes',
      messages: afterListNotes,
      sessionSlot: 'rolling',
    },
    {
      label: 'after-first-long-result',
      messages: afterFirstLongResult,
      sessionSlot: 'rolling',
    },
    {
      label: 'after-first-checkpoint',
      messages: afterFirstCheckpoint,
      sessionSlot: 'rolling',
    },
    {
      label: 'after-second-long-result',
      messages: afterSecondLongResult,
      sessionSlot: 'rolling',
    },
    {
      label: 'after-second-checkpoint',
      messages: afterSecondCheckpoint,
      sessionSlot: 'rolling',
    },
    {
      label: 'after-replacement-observed',
      messages: afterReplacementObserved,
      sessionSlot: 'rolling',
    },
  ];
}

function selectProbeScenarios(
  scenarios: readonly ProbeScenario[]
): ProbeScenario[] {
  if (PROBE_PROFILE === 'full') {
    return [...scenarios];
  }
  return [0, 3, 5, 6].map((index) => {
    const scenario = scenarios[index];
    if (scenario === undefined) {
      throw new Error(`Compact probe scenario is missing at index ${index}`);
    }
    return scenario;
  });
}

function comparableOutput(response: ChatCompletion): unknown {
  const choice = response.choices[0];
  const functionCalls = (choice?.message.tool_calls ?? []).filter(
    (call): call is ChatCompletionMessageFunctionToolCall =>
      call.type === 'function'
  );
  return {
    finishReason: choice?.finish_reason ?? null,
    content: choice?.message.content ?? null,
    toolCalls: functionCalls.map((call) => ({
      name: call.function.name,
      arguments: call.function.arguments,
    })),
  };
}

function sessionCacheParams(
  useSessionCache: boolean,
  slot: SessionCacheSlot
): Pick<RapidMlxChatParams, 'cache'> | Record<string, never> {
  if (!useSessionCache) {
    return {};
  }
  return {
    cache: {
      mode: 'auto',
      session_id: PROBE_SESSION_ID,
      session_slot: slot,
    },
  };
}

function forcedTextOutputParams(
  forceTextOutput: boolean | undefined
): { tool_choice: 'none' } | Record<string, never> {
  return forceTextOutput === true ? { tool_choice: 'none' } : {};
}

async function runAttempt(
  client: OpenAI,
  input: {
    model: string;
    scenario: ProbeScenario;
    useSessionCache: boolean;
    maxTokens?: number;
    forceTextOutput?: boolean;
  }
): Promise<ProbeAttempt> {
  const requestedMaxTokens = input.maxTokens ?? 16;
  const messages =
    input.forceTextOutput === true
      ? [
          ...input.scenario.messages,
          { role: 'user' as const, content: LONG_OUTPUT_INSTRUCTION },
        ]
      : input.scenario.messages;
  const params: RapidMlxChatParams = {
    model: input.model,
    messages,
    tools: canonicalRuntimeTools.map((tool) =>
      toChatCompletionTool(tool.contract)
    ),
    max_tokens: requestedMaxTokens,
    temperature: 0,
    top_p: 1,
    top_k: 1,
    presence_penalty: 0,
    frequency_penalty: 0,
    stream: false,
    ...forcedTextOutputParams(input.forceTextOutput),
    chat_template_kwargs: { enable_thinking: false },
    ...sessionCacheParams(input.useSessionCache, input.scenario.sessionSlot),
  };
  const startedAt = performance.now();
  const response = await client.chat.completions.create(params);
  return {
    requestedMaxTokens,
    elapsedMs: Math.round(performance.now() - startedAt),
    cachedInputTokens:
      response.usage?.prompt_tokens_details?.cached_tokens ?? 0,
    promptTokens: response.usage?.prompt_tokens ?? 0,
    completionTokens: response.usage?.completion_tokens ?? 0,
    comparableOutput: comparableOutput(response),
  };
}

async function clearPrefixCache(baseURL: string): Promise<void> {
  const response = await fetch(`${baseURL}/cache/clear`, { method: 'POST' });
  if (!response.ok) {
    throw new Error(`Rapid-MLX cache clear returned HTTP ${response.status}`);
  }
}

async function readStatus(baseURL: string): Promise<unknown> {
  const response = await fetch(`${baseURL}/status`);
  if (!response.ok) {
    throw new Error(`Rapid-MLX status returned HTTP ${response.status}`);
  }
  return await response.json();
}

function readCacheNumber(status: unknown, key: string): number | null {
  if (typeof status !== 'object' || status === null) {
    return null;
  }
  const cache = (status as Record<string, unknown>).cache;
  if (typeof cache !== 'object' || cache === null) {
    return null;
  }
  const value = (cache as Record<string, unknown>)[key];
  return typeof value === 'number' ? value : null;
}

async function runDisabledBaseline(input: {
  target: LocalEvaluationTarget;
  rapidMlxBin: string;
  rapidMlxWorkingDirectory: string;
  port: number;
  outputDirectory: string;
  scenarios: ProbeScenario[];
  longOutputScenarios: ProbeScenario[];
}): Promise<Record<string, unknown>> {
  const logPath = join(
    input.outputDirectory,
    `${input.target.id}.prefix-cache-disabled.server.log`
  );
  const server = await startLocalModelServer({
    target: input.target,
    rapidMlxBin: input.rapidMlxBin,
    rapidMlxWorkingDirectory: input.rapidMlxWorkingDirectory,
    port: input.port,
    logPath,
    kvCacheDtype: 'int4',
    prefixCacheMode: 'disabled',
  });
  const client = new OpenAI({
    apiKey: 'local-session-prefix-cache-probe',
    baseURL: server.baseURL,
  });
  const attempts: ProbeAttempt[] = [];
  const longOutputAttempts: ProbeAttempt[] = [];
  const result: Record<string, unknown> = {
    server: {
      startupElapsedMs: server.startupElapsedMs,
      temporaryModelPath: server.temporaryModelPath,
      logPath,
    },
    attempts,
    longOutputAttempts,
  };

  try {
    for (const scenario of input.scenarios) {
      attempts.push(
        await runAttempt(client, {
          model: input.target.servedModelName,
          scenario,
          useSessionCache: false,
        })
      );
    }
    for (const scenario of input.longOutputScenarios) {
      longOutputAttempts.push(
        await runAttempt(client, {
          model: input.target.servedModelName,
          scenario,
          useSessionCache: false,
          maxTokens: 512,
          forceTextOutput: true,
        })
      );
    }
  } finally {
    const serverExit = await server.stop();
    server.cleanup();
    Object.assign(result, { serverExit });
  }

  return result;
}

async function probeTarget(input: {
  target: LocalEvaluationTarget;
  rapidMlxBin: string;
  rapidMlxWorkingDirectory: string;
  port: number;
  outputDirectory: string;
}): Promise<Record<string, unknown>> {
  const allScenarios = createProbeScenarios();
  const scenarios = selectProbeScenarios(allScenarios);
  const firstLongOutputScenario = allScenarios[1];
  const lastLongOutputScenario = allScenarios[allScenarios.length - 1];
  if (
    firstLongOutputScenario === undefined ||
    lastLongOutputScenario === undefined
  ) {
    throw new Error('Long-output scenario is missing');
  }
  const longOutputScenarios: ProbeScenario[] = [
    firstLongOutputScenario,
    lastLongOutputScenario,
  ];
  const disabledBaseline = await runDisabledBaseline({
    ...input,
    scenarios,
    longOutputScenarios,
  });
  const logPath = join(
    input.outputDirectory,
    `${input.target.id}.session-prefix-cache.server.log`
  );
  const server = await startLocalModelServer({
    target: input.target,
    rapidMlxBin: input.rapidMlxBin,
    rapidMlxWorkingDirectory: input.rapidMlxWorkingDirectory,
    port: input.port,
    logPath,
    kvCacheDtype: 'int4',
    prefixCacheMode: 'enabled',
  });
  const client = new OpenAI({
    apiKey: 'local-session-prefix-cache-probe',
    baseURL: server.baseURL,
  });
  const cold: ProbeAttempt[] = [];
  const session: ProbeAttempt[] = [];
  const longOutputSession: ProbeAttempt[] = [];
  const sessionStatuses: unknown[] = [];
  const result: Record<string, unknown> = {
    candidate: input.target,
    disabledBaseline,
    server: {
      startupElapsedMs: server.startupElapsedMs,
      temporaryModelPath: server.temporaryModelPath,
      temporaryPrefixCachePath: server.temporaryPrefixCachePath,
      logPath,
    },
  };

  try {
    if (PROBE_PROFILE === 'full') {
      for (const scenario of scenarios) {
        await clearPrefixCache(server.baseURL);
        cold.push(
          await runAttempt(client, {
            model: input.target.servedModelName,
            scenario,
            useSessionCache: true,
          })
        );
      }
    }

    await clearPrefixCache(server.baseURL);
    for (const scenario of scenarios) {
      session.push(
        await runAttempt(client, {
          model: input.target.servedModelName,
          scenario,
          useSessionCache: true,
        })
      );
      sessionStatuses.push(await readStatus(server.baseURL));
    }
    for (const scenario of longOutputScenarios) {
      longOutputSession.push(
        await runAttempt(client, {
          model: input.target.servedModelName,
          scenario,
          useSessionCache: true,
          maxTokens: 512,
          forceTextOutput: true,
        })
      );
    }
    const finalStatus = sessionStatuses[sessionStatuses.length - 1];
    Object.assign(result, {
      scenarios: scenarios.map((scenario) => ({
        label: scenario.label,
        messageCount: scenario.messages.length,
        sessionSlot: scenario.sessionSlot,
      })),
      cold,
      session,
      longOutputSession,
      sessionStatuses,
      coldMatchesDisabled: cold.map(
        (attempt, index) =>
          JSON.stringify(attempt.comparableOutput) ===
          JSON.stringify(
            (disabledBaseline.attempts as ProbeAttempt[])[index]
              ?.comparableOutput
          )
      ),
      sessionMatchesDisabled: session.map(
        (attempt, index) =>
          JSON.stringify(attempt.comparableOutput) ===
          JSON.stringify(
            (disabledBaseline.attempts as ProbeAttempt[])[index]
              ?.comparableOutput
          )
      ),
      longOutputSessionMatchesDisabled: longOutputSession.map(
        (attempt, index) =>
          JSON.stringify(attempt.comparableOutput) ===
          JSON.stringify(
            (disabledBaseline.longOutputAttempts as ProbeAttempt[])[index]
              ?.comparableOutput
          )
      ),
      observations: {
        cachedInputTokens: session.map((attempt) => attempt.cachedInputTokens),
        sessionPinnedEntryCount: readCacheNumber(
          finalStatus,
          'session_pinned_entry_count'
        ),
        sessionRollingEntryCount: readCacheNumber(
          finalStatus,
          'session_rolling_entry_count'
        ),
        sessionReplacements: readCacheNumber(
          finalStatus,
          'session_replacements'
        ),
        boundaryEntryCount: readCacheNumber(
          finalStatus,
          'boundary_entry_count'
        ),
        sessionMemoryBytes: readCacheNumber(
          finalStatus,
          'session_memory_bytes'
        ),
      },
    });
  } finally {
    const serverExit = await server.stop();
    server.cleanup();
    Object.assign(result, { serverExit });
  }

  return result;
}

function assertCandidateProbeResult(candidate: Record<string, unknown>): void {
  const scenarios = candidate.scenarios as unknown[];
  expect(candidate.coldMatchesDisabled).toEqual(
    PROBE_PROFILE === 'full' ? scenarios.map(() => true) : []
  );
  expect(candidate.sessionMatchesDisabled).toEqual(scenarios.map(() => true));
  expect(candidate.longOutputSessionMatchesDisabled).toEqual([true, true]);
  expect(candidate.observations).toMatchObject({
    sessionPinnedEntryCount: 1,
    sessionRollingEntryCount: 1,
    boundaryEntryCount: 2,
  });
  const observations = candidate.observations as Record<string, unknown>;
  const sessionReplacements = observations.sessionReplacements;
  expect(typeof sessionReplacements).toBe('number');
  if (typeof sessionReplacements !== 'number') {
    throw new TypeError('sessionReplacements must be a number');
  }
  expect(sessionReplacements).toBeGreaterThanOrEqual(1);
  const attempts = candidate.session as ProbeAttempt[];
  expect(attempts[0]?.cachedInputTokens).toBe(0);
  expect(attempts[1]?.cachedInputTokens).toBeGreaterThan(0);
  expect(attempts[attempts.length - 1]?.cachedInputTokens).toBeGreaterThan(
    attempts[1]?.cachedInputTokens ?? 0
  );
  const longOutputAttempts = candidate.longOutputSession as ProbeAttempt[];
  expect(longOutputAttempts).toHaveLength(2);
  for (const attempt of longOutputAttempts) {
    expect(attempt.completionTokens).toBe(512);
    expect(
      (attempt.comparableOutput as { finishReason: unknown }).finishReason
    ).toBe('length');
  }
}

liveTest(
  'reuses one pinned root and one rolling state per Rapid-MLX session ID',
  // The live-only orchestration validates environment, records partial results,
  // runs evaluation targets sequentially, and asserts every resulting cache contract.
  async () => {
    const rapidMlxBin = requiredEnvironmentVariable('ECHO_EVAL_RAPID_MLX_BIN');
    const rapidMlxWorkingDirectory = requiredEnvironmentVariable(
      'ECHO_EVAL_RAPID_MLX_CWD'
    );
    const targetsPath = requiredEnvironmentVariable('ECHO_EVAL_TARGETS_FILE');
    const outputPath = requiredEnvironmentVariable(
      'ECHO_SESSION_PREFIX_CACHE_PROBE_OUTPUT'
    );
    const outputDirectory = dirname(outputPath);
    const targets = loadEvaluationTargets(targetsPath);
    const result: {
      schemaVersion: number;
      startedAt: string;
      completedAt?: string;
      protocol: Record<string, unknown>;
      candidates: Record<string, unknown>[];
    } = {
      schemaVersion: 1,
      startedAt: new Date().toISOString(),
      protocol: {
        purpose:
          'Validate isolated pinned/rolling prefix-cache reuse on growing E.C.H.O. histories.',
        probeProfile: PROBE_PROFILE,
        coldBaseline:
          PROBE_PROFILE === 'full'
            ? 'Clear the full prefix cache before each request while keeping the same pinned/rolling boundary path.'
            : 'Omitted in compact mode; each session output is compared directly with the prefix-cache-disabled reference.',
        disabledBaseline:
          'Run the same histories with Rapid-MLX prefix caching disabled; this is the ordinary prefill reference.',
        sessionRun:
          'Run the same histories in order with session_id=echo:rin; first pinned, later rolling.',
        checkpointPolicy:
          'Floor semantic message boundaries to the normal 2,048-token prefill step; session requests do not add artificial prompt segments.',
        histories:
          'Production Rin system prompt, generated 16-tool catalog, runtime context, and scripted assistant/tool exchanges including two long tool results that advance natural checkpoints.',
        generation:
          'Broad transition probe: temperature=0, top-p=1, top-k=1, presence/frequency penalty=0, thinking disabled, max output 16 tokens. Long-output diagnostic: append an explicit 400-line text request, use the same sampling with tool_choice=none, and require the 512-token output limit to be reached.',
        rapidMlxVersion: commandOutput(
          rapidMlxBin,
          ['--version'],
          rapidMlxWorkingDirectory
        ),
        rapidMlxHead: commandOutput(
          'git',
          ['rev-parse', 'HEAD'],
          rapidMlxWorkingDirectory
        ),
        rapidMlxWorkingTreeStatus: commandOutput(
          'git',
          ['status', '--short'],
          rapidMlxWorkingDirectory
        ),
        rapidMlxTrackedDiffSha256: createHash('sha256')
          .update(
            commandOutput(
              'git',
              ['diff', '--no-ext-diff', '--binary'],
              rapidMlxWorkingDirectory
            )
          )
          .digest('hex'),
        prefixCache: 'enabled',
        pflash: false,
        multiTokenPrediction: false,
        kvCacheDtype: 'int4',
      },
      candidates: [],
    };

    writeResult(outputPath, result);
    for (const target of targets) {
      result.candidates.push(
        await probeTarget({
          target,
          rapidMlxBin,
          rapidMlxWorkingDirectory,
          port: Number(process.env.ECHO_EVAL_PORT ?? '8134'),
          outputDirectory,
        })
      );
      writeResult(outputPath, result);
    }
    result.completedAt = new Date().toISOString();
    writeResult(outputPath, result);

    expect(result.candidates).toHaveLength(targets.length);
    for (const candidateResult of result.candidates) {
      assertCandidateProbeResult(candidateResult);
    }
  },
  4 * 60 * 60_000
);
