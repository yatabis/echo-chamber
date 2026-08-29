import {
  ParallelCognitiveModuleOrchestrator,
  type CognitiveModuleCommittedState,
  type CognitiveModuleName,
  type CognitiveModuleDomainPort,
  type CognitiveModuleOrchestrator,
  type CognitiveModuleOutcome,
  type CognitiveModulePhaseResult,
} from '@echo-chamber/core/agent/cognitive-module-orchestrator';
import {
  createEmotionCognitiveModuleOutputFormat,
  createMemoryRecallCognitiveModuleOutputFormat,
  createMemoryStoreCognitiveModuleOutputFormat,
  parseEmotionCognitiveModuleOutput,
  parseMemoryRecallCognitiveModuleOutput,
  parseMemoryStoreCognitiveModuleOutput,
  type MemoryCognitiveModuleOutput,
  type MemoryRecallCognitiveModuleOutput,
} from '@echo-chamber/core/agent/cognitive-module-schema';
import { ModelCognitiveModuleRunner } from '@echo-chamber/core/agent/model-cognitive-module';
import {
  buildEmotionCognitiveModuleSystemPrompt,
  buildMemoryCognitiveModuleSystemPrompt,
} from '@echo-chamber/core/agent/prompt-builder';
import type { EchoInstanceDefinition } from '@echo-chamber/core/echo/instance-definitions';
import type {
  EchoEvent,
  EchoEventPort,
} from '@echo-chamber/core/ports/echo-event';
import type {
  ModelInputItem,
  ModelPort,
  ModelResponse,
} from '@echo-chamber/core/ports/model';
import {
  OpenAIResponsesModel,
  type OpenAIResponsesModelOptions,
} from '@echo-chamber/openai-adapter/openai-responses-model';

import {
  resolveCognitiveModuleConfig,
  type CognitiveModuleEnv,
} from '../config/cognitive-module-config';

const COGNITIVE_MODULE_MAX_ATTEMPTS = 2;
const COGNITIVE_MODULE_REQUEST_TIMEOUT_MS = 30_000;
const UPDATE_EMOTION_TOOL_NAME = 'update_emotion';

/** Mainの一会話内で一意かつprovider制限に収まる擬似tool call IDを作る。 */
function createCognitiveHandoffCallId(
  sequence: number,
  toolName: 'search_memory' | typeof UPDATE_EMOTION_TOOL_NAME
): string {
  return `cognitive:${sequence}:${toolName}`;
}

const RETRYABLE_ERROR_NAMES = new Set([
  'AbortError',
  'APIConnectionError',
  'APIConnectionTimeoutError',
  'APIUserAbortError',
  'TimeoutError',
]);

const RETRYABLE_ERROR_CODES = new Set([
  'EAI_AGAIN',
  'ECONNRESET',
  'ENETUNREACH',
  'ETIMEDOUT',
]);

const RETRYABLE_WORKERS_AI_ERROR_CODES = new Set(['3007', '3008', '3040']);

/** Hosted Memory / Emotion coordinator の構築入力。 */
export interface CreateCognitiveModuleOrchestratorInput {
  env: CognitiveModuleEnv;
  instance: EchoInstanceDefinition;
  events?: EchoEventPort;
  domain: CognitiveModuleDomainPort;
  createActivationId?(): string;
  createModel?(options: OpenAIResponsesModelOptions): ModelPort;
  beforeModelRequest?(
    module: CognitiveModuleName,
    request: Parameters<ModelPort['generate']>[0]
  ): void;
}

/** unknown error を安全に record として読む。 */
function toErrorRecord(error: unknown): Record<string, unknown> | null {
  return typeof error === 'object' && error !== null
    ? (error as Record<string, unknown>)
    : null;
}

/** OpenAI error record の数値 status を読む。 */
function getErrorStatus(record: Record<string, unknown>): number | undefined {
  if (typeof record.status === 'number') {
    return record.status;
  }
  return typeof record.statusCode === 'number' ? record.statusCode : undefined;
}

/** HTTP status が bounded retry の対象か判定する。 */
function isRetryableStatus(status: number): boolean {
  return (
    status === 408 ||
    status === 409 ||
    status === 429 ||
    (status >= 500 && status <= 599)
  );
}

/** Workers AI binding errorから公開internal codeを読み取る。 */
function getWorkersAIErrorCode(
  record: Record<string, unknown>
): string | undefined {
  if (typeof record.code === 'number') {
    return String(record.code);
  }
  if (typeof record.code === 'string' && /^\d+$/.test(record.code)) {
    return record.code;
  }
  if (typeof record.message !== 'string') {
    return undefined;
  }
  return /^(\d+):/.exec(record.message)?.[1];
}

/** SDK / network error identity が bounded retry の対象か判定する。 */
function hasRetryableErrorIdentity(record: Record<string, unknown>): boolean {
  const hasRetryableName =
    typeof record.name === 'string' && RETRYABLE_ERROR_NAMES.has(record.name);
  const hasRetryableCode =
    typeof record.code === 'string' && RETRYABLE_ERROR_CODES.has(record.code);
  return hasRetryableName || hasRetryableCode;
}

/**
 * OpenAI / network error が、同一 phase snapshot から再試行できる一時失敗か判定する。
 */
export function isRetryableCognitiveModuleError(error: unknown): boolean {
  const visited = new Set<unknown>();
  let currentError: unknown = error;

  while (currentError !== undefined && !visited.has(currentError)) {
    visited.add(currentError);
    const record = toErrorRecord(currentError);
    if (record === null) {
      return false;
    }

    const status = getErrorStatus(record);
    if (status !== undefined) {
      return isRetryableStatus(status);
    }
    const workersAIErrorCode = getWorkersAIErrorCode(record);
    if (
      workersAIErrorCode !== undefined &&
      RETRYABLE_WORKERS_AI_ERROR_CODES.has(workersAIErrorCode)
    ) {
      return true;
    }
    if (hasRetryableErrorIdentity(record)) {
      return true;
    }
    currentError = record.cause;
  }

  return false;
}

/** Main と同じ event payload policy のまま module attribution を付ける。 */
function createCognitiveModuleEventPort(
  module: CognitiveModuleName,
  events: EchoEventPort | undefined
): EchoEventPort | undefined {
  if (events === undefined) {
    return undefined;
  }

  return {
    emit: async (event: EchoEvent): Promise<void> => {
      await events.emit({
        ...event,
        payload: {
          ...event.payload,
          cognitiveModule: module,
        },
      });
    },
  };
}

/** Main handoff に使える成功値だけを取り出す。 */
function getReadyValue<T>(outcome: CognitiveModuleOutcome<T>): T {
  if (outcome.status === 'failed') {
    throw new Error('Failed cognitive outcome cannot become a Main handoff');
  }
  return outcome.value;
}

/** Memory成功値がpre_main用queryであることを保証する。 */
function getRecallOutput(
  outcome: CognitiveModuleOutcome<MemoryCognitiveModuleOutput>
): MemoryRecallCognitiveModuleOutput {
  const value = getReadyValue(outcome);
  if (!('query' in value)) {
    throw new Error('Memory pre_main output must contain query');
  }
  return value;
}

/** 両moduleの確定結果をsystem-owned tool exchangeとしてMainへ渡す。 */
function formatCognitiveModuleHandoff(
  result: CognitiveModulePhaseResult,
  committed: CognitiveModuleCommittedState
): readonly ModelInputItem[] {
  if (result.phase !== 'pre_main') {
    return [];
  }

  const memory = getRecallOutput(result.memory);
  const emotion = getReadyValue(result.emotion);
  const memoryCallId = createCognitiveHandoffCallId(
    result.sequence,
    'search_memory'
  );
  const emotionCallId = createCognitiveHandoffCallId(
    result.sequence,
    UPDATE_EMOTION_TOOL_NAME
  );

  return [
    {
      type: 'tool_call',
      callId: memoryCallId,
      toolName: 'search_memory',
      input: JSON.stringify(memory),
    },
    {
      type: 'tool_result',
      callId: memoryCallId,
      output: JSON.stringify({
        success: true,
        results: committed.recalledMemories,
      }),
    },
    {
      type: 'tool_call',
      callId: emotionCallId,
      toolName: UPDATE_EMOTION_TOOL_NAME,
      input: JSON.stringify(emotion),
    },
    {
      type: 'tool_result',
      callId: emotionCallId,
      output: JSON.stringify({ success: true }),
    },
  ];
}

/** 前sessionで確定したEmotionをmodule専用の初期状態として復元する。 */
function formatInitialCognitiveModuleContext(
  committed: CognitiveModuleCommittedState
): readonly ModelInputItem[] {
  if (committed.emotion === null) {
    return [];
  }

  return [
    {
      role: 'developer',
      content: [
        '現在の感情状態です。前回の思考セッション終了時に確定しました。',
        JSON.stringify(committed.emotion),
      ].join('\n'),
    },
  ];
}

/** Test injection または実 OpenAI adapter から module model を作る。 */
function createCognitiveModuleModel(
  input: CreateCognitiveModuleOrchestratorInput,
  module: CognitiveModuleName,
  options: OpenAIResponsesModelOptions
): ModelPort {
  const model =
    input.createModel === undefined
      ? new OpenAIResponsesModel(options)
      : input.createModel(options);
  if (input.beforeModelRequest === undefined) {
    return model;
  }

  return {
    generate: async (request): Promise<ModelResponse> => {
      input.beforeModelRequest?.(module, request);
      return await model.generate(request);
    },
  };
}

/** Test injection または runtime UUID から activation identifier を作る。 */
function createCognitiveModuleActivationId(
  input: CreateCognitiveModuleOrchestratorInput
): string {
  if (input.createActivationId !== undefined) {
    return input.createActivationId();
  }
  return `${input.instance.id}:${crypto.randomUUID()}`;
}

/**
 * instance 設定と既存 OpenAI key から、同じ共有contextを読む
 * Memory / Emotion coordinatorを作る。
 */
export function createCognitiveModuleOrchestrator(
  input: CreateCognitiveModuleOrchestratorInput
): CognitiveModuleOrchestrator {
  const config = resolveCognitiveModuleConfig(input.env, input.instance);
  const memoryModel = createCognitiveModuleModel(input, 'memory', {
    apiKey: config.apiKey,
    model: config.model,
    reasoningEffort: config.reasoningEffort,
    maxRetries: 0,
    events: createCognitiveModuleEventPort('memory', input.events),
  });
  const emotionModel = createCognitiveModuleModel(input, 'emotion', {
    apiKey: config.apiKey,
    model: config.model,
    reasoningEffort: config.reasoningEffort,
    maxRetries: 0,
    events: createCognitiveModuleEventPort('emotion', input.events),
  });
  const memoryRecallFormat = createMemoryRecallCognitiveModuleOutputFormat();
  const memoryStoreFormat = createMemoryStoreCognitiveModuleOutputFormat();
  const emotionFormat = createEmotionCognitiveModuleOutputFormat();

  return new ParallelCognitiveModuleOrchestrator({
    createActivationId: (): string => createCognitiveModuleActivationId(input),
    memory: new ModelCognitiveModuleRunner<MemoryCognitiveModuleOutput>({
      model: memoryModel,
      resolveSystemPrompt: ({ phase }) =>
        buildMemoryCognitiveModuleSystemPrompt(input.instance.name, phase),
      resolveOutputContract: ({ phase }) =>
        phase === 'pre_main'
          ? {
              format: memoryRecallFormat,
              parse: parseMemoryRecallCognitiveModuleOutput,
            }
          : {
              format: memoryStoreFormat,
              parse: parseMemoryStoreCognitiveModuleOutput,
            },
    }),
    emotion: new ModelCognitiveModuleRunner({
      model: emotionModel,
      resolveSystemPrompt: () =>
        buildEmotionCognitiveModuleSystemPrompt(input.instance.name),
      resolveOutputContract: () => ({
        format: emotionFormat,
        parse: parseEmotionCognitiveModuleOutput,
      }),
    }),
    retryPolicy: {
      maxAttempts: COGNITIVE_MODULE_MAX_ATTEMPTS,
      shouldRetry: ({ error }) => isRetryableCognitiveModuleError(error),
    },
    domain: input.domain,
    formatInitialContext: formatInitialCognitiveModuleContext,
    createRequestSignal: () =>
      AbortSignal.timeout(COGNITIVE_MODULE_REQUEST_TIMEOUT_MS),
    formatHandoff: formatCognitiveModuleHandoff,
  });
}
