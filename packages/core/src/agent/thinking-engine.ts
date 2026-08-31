import { emitEchoEvent } from '../ports/echo-event';
import { getErrorMessage } from '../utils/error';

import { CognitiveModulePhaseError } from './cognitive-module-orchestrator';
import { buildAgentPromptMessages } from './prompt-builder';
import {
  accumulateModelUsage,
  AgentSessionExecutionError,
  runAgentSession,
  sanitizeToolOutputForModel,
  ZERO_MODEL_USAGE,
} from './session';
import { checkNotificationsToolSpec } from './tools/chat';

import type {
  CognitiveModuleActivation,
  CognitiveModuleActivationResult,
  CognitiveModuleOrchestrator,
} from './cognitive-module-orchestrator';
import type {
  AgentSessionResult,
  AgentSessionTool,
  AgentSessionTurnBoundaryHandler,
} from './session';
import type { EchoEventPort } from '../ports/echo-event';
import type {
  ModelInputItem,
  ModelPort,
  ModelToolCall,
  ModelToolResult,
  ModelUsage,
} from '../ports/model';

/**
 * Thinking engine の構築入力。
 * prompt に必要な data と provider/runtime service を一箇所に集約する。
 */
export interface ThinkingEngineInput {
  model: ModelPort;
  events?: EchoEventPort;
  tools: readonly AgentSessionTool[];
  systemPrompt: string;
  cognitiveModules: CognitiveModuleOrchestrator;
}

/**
 * 1 回の思考実行が返す集約結果。
 * usage に加え、次回起動時刻と committed Cognitive Module record を返す。
 */
export interface ThinkingEngineResult {
  usage: ModelUsage;
  mainUsage: ModelUsage;
  nextWakeAt: string | null;
  cognitiveModules: CognitiveModuleActivationResult;
}

/**
 * Main または Cognitive Module の失敗後も課金済み usage を失わない実行結果。
 */
export class ThinkingEngineExecutionError extends Error {
  override readonly name = 'ThinkingEngineExecutionError';
  readonly code = 'thinking_engine_execution_failed';
  readonly cause: unknown;

  /** @param cause session を fail closed にした根本原因 */
  constructor(
    cause: unknown,
    readonly mainUsage: ModelUsage,
    readonly cognitiveUsage: ModelUsage
  ) {
    super(`Thinking engine failed: ${getErrorMessage(cause)}`);
    this.cause = cause;
    this.usage = accumulateModelUsage(mainUsage, cognitiveUsage);
  }

  readonly usage: ModelUsage;
}

/** Worker isolate / test realm を跨いでも typed failure を識別する。 */
export function isThinkingEngineExecutionError(
  error: unknown
): error is ThinkingEngineExecutionError {
  if (error instanceof ThinkingEngineExecutionError) {
    return true;
  }
  if (typeof error !== 'object' || error === null) {
    return false;
  }
  const record = error as Record<string, unknown>;
  return (
    record.code === 'thinking_engine_execution_failed' &&
    isModelUsage(record.usage) &&
    isModelUsage(record.mainUsage) &&
    isModelUsage(record.cognitiveUsage)
  );
}

/** Structural clone 後の usage が最低限の保存契約を満たすか判定する。 */
function isModelUsage(value: unknown): value is ModelUsage {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const record = value as Record<string, unknown>;
  return (
    typeof record.cachedInputTokens === 'number' &&
    typeof record.cacheWriteInputTokens === 'number' &&
    typeof record.uncachedInputTokens === 'number' &&
    typeof record.totalInputTokens === 'number' &&
    typeof record.outputTokens === 'number' &&
    typeof record.reasoningTokens === 'number' &&
    typeof record.totalTokens === 'number'
  );
}

const STARTUP_TOOL_INPUT = '{}';

/** Main専用promptを含む入力とCognitive Moduleへ渡す共有context。 */
interface ThinkingEngineInitialInputs {
  mainInput: readonly ModelInputItem[];
  sharedContext: readonly ModelInputItem[];
}

/** 必須 cognitive activation の pre-main result をMain inputへ接続する。 */
async function createCognitiveInitialInput(
  activation: CognitiveModuleActivation,
  sharedContext: readonly ModelInputItem[]
): Promise<readonly ModelInputItem[]> {
  return await activation.beforeMain(sharedContext);
}

/** 必須 cognitive activation の session boundary handler を取り出す。 */
function getCognitiveBoundaryHandler(
  activation: CognitiveModuleActivation
): AgentSessionTurnBoundaryHandler {
  return activation.onMainTurnBoundary;
}

/** Main と cognitive module の結果を保存・課金用の1結果へ集約する。 */
function createThinkingEngineResult(
  session: AgentSessionResult,
  cognitiveModules: CognitiveModuleActivationResult
): ThinkingEngineResult {
  return {
    nextWakeAt: session.nextWakeAt,
    mainUsage: session.usage,
    cognitiveModules,
    usage: accumulateModelUsage(session.usage, cognitiveModules.usage),
  };
}

/** max-turn 終了だけを運用警告へ昇格する。 */
function getSessionCompletedSeverity(
  session: AgentSessionResult
): 'info' | 'warn' {
  return session.terminationReason === 'max_turns' ? 'warn' : 'info';
}

/** Cognitive boundary failureをsession運用metadataへ変換する。 */
function createSessionFailurePayload(error: unknown): Record<string, unknown> {
  if (!(error instanceof CognitiveModulePhaseError)) {
    return { error: getErrorMessage(error) };
  }

  const failedModules: Record<string, unknown>[] = [];
  if (error.phaseResult.memory.status === 'failed') {
    failedModules.push({
      module: 'memory',
      reason: error.phaseResult.memory.reason,
      error: error.phaseResult.memory.error,
      attempts: error.phaseResult.memory.attempts,
      ...(error.phaseResult.memory.outputValidation === undefined
        ? {}
        : { outputValidation: error.phaseResult.memory.outputValidation }),
    });
  }
  if (error.phaseResult.emotion.status === 'failed') {
    failedModules.push({
      module: 'emotion',
      reason: error.phaseResult.emotion.reason,
      error: error.phaseResult.emotion.error,
      attempts: error.phaseResult.emotion.attempts,
      ...(error.phaseResult.emotion.outputValidation === undefined
        ? {}
        : { outputValidation: error.phaseResult.emotion.outputValidation }),
    });
  }

  return {
    error: error.message,
    failureSource: 'cognitive_module',
    activationId: error.phaseResult.activationId,
    boundaryId: error.phaseResult.boundaryId,
    phase: error.phaseResult.phase,
    failedModules,
  };
}

/** Session wrapper の内側にある制御判断用の根本原因を返す。 */
function unwrapAgentSessionError(error: unknown): unknown {
  return error instanceof AgentSessionExecutionError ? error.cause : error;
}

/**
 * provider/runtime 非依存の思考 orchestration。
 * prompt 構築、起動時通知チェック、session 実行を順に担う。
 */
export class ThinkingEngine {
  /**
   * @param input 実行に必要な port・tool・prompt 設定
   */
  constructor(private readonly input: ThinkingEngineInput) {}

  /**
   * 1 回分の思考 session を開始し、usage と確定済み module 結果を返す。
   *
   * @returns session 全体の usage、Cognitive Module 結果、次回起動時刻
   */
  async think(): Promise<ThinkingEngineResult> {
    await emitEchoEvent(this.input.events, {
      type: 'session.started',
      severity: 'info',
      summary: 'thinking session started',
    });

    let cognitiveActivation: CognitiveModuleActivation | undefined;
    let mainUsage = ZERO_MODEL_USAGE;

    try {
      cognitiveActivation = this.input.cognitiveModules.beginActivation();
      const initialInputs = await this.buildInitialInputs();
      const cognitiveInput = await createCognitiveInitialInput(
        cognitiveActivation,
        initialInputs.sharedContext
      );
      const session = await runAgentSession({
        model: this.input.model,
        tools: this.input.tools,
        initialInput: [...initialInputs.mainInput, ...cognitiveInput],
        events: this.input.events,
        onTurnBoundary: getCognitiveBoundaryHandler(cognitiveActivation),
      });
      mainUsage = session.usage;
      const cognitiveModules = cognitiveActivation.getResultSnapshot();
      const result = createThinkingEngineResult(session, cognitiveModules);

      await emitEchoEvent(this.input.events, {
        type: 'session.completed',
        severity: getSessionCompletedSeverity(session),
        summary: 'thinking session completed',
        payload: {
          nextWakeAt: result.nextWakeAt,
          totalTokens: result.usage.totalTokens,
          committedCognitivePhases: result.cognitiveModules.phases.length,
          terminationReason: session.terminationReason,
        },
      });
      return result;
    } catch (error) {
      if (error instanceof AgentSessionExecutionError) {
        mainUsage = error.usage;
      }
      const rootCause = unwrapAgentSessionError(error);
      const activationUsage =
        cognitiveActivation?.getResultSnapshot().usage ?? ZERO_MODEL_USAGE;
      const executionError = new ThinkingEngineExecutionError(
        rootCause,
        mainUsage,
        activationUsage
      );
      await emitEchoEvent(this.input.events, {
        type: 'session.failed',
        severity: 'error',
        summary: `thinking session failed: ${getErrorMessage(rootCause)}`,
        payload: {
          ...createSessionFailurePayload(rootCause),
          usage: executionError.usage,
          mainUsage: executionError.mainUsage,
          cognitiveUsage: executionError.cognitiveUsage,
        },
      });
      throw executionError;
    }
  }

  /**
   * Main専用promptと共有runtime contextを分けて初期 input を作る。
   * 起動時の通知チェックは通常 turn に入る前に必ず 1 度実行する。
   *
   * @returns Main初期入力とCognitive Module共有context
   */
  private async buildInitialInputs(): Promise<ThinkingEngineInitialInputs> {
    const promptMessages = buildAgentPromptMessages({
      systemPrompt: this.input.systemPrompt,
      currentDatetime: new Date(),
      toolContracts: this.input.tools.map((tool) => tool.contract),
    });
    const sharedContext: ModelInputItem[] = [
      promptMessages.sharedRuntimeContext,
      this.createStartupToolCallInput(),
      await this.createStartupToolResultInput(),
    ];

    return {
      mainInput: [promptMessages.mainSystemPrompt, ...sharedContext],
      sharedContext,
    };
  }

  /**
   * startup tool を既に呼び出した扱いにするための擬似 tool_call を生成する。
   *
   * @returns startup sequence 先頭に挿入する tool_call input
   */
  private createStartupToolCallInput(): ModelToolCall {
    const startupTool = this.getStartupToolOrThrow();

    return {
      type: 'tool_call',
      callId: startupTool.name,
      toolName: startupTool.name,
      input: STARTUP_TOOL_INPUT,
    };
  }

  /**
   * startup tool の実行結果を次ターン入力へ接続する tool_result を生成する。
   *
   * @returns startup tool の実行結果を表す tool_result input
   */
  private async createStartupToolResultInput(): Promise<ModelToolResult> {
    const startupTool = this.getStartupToolOrThrow();

    return {
      type: 'tool_result',
      callId: startupTool.name,
      output: sanitizeToolOutputForModel(
        await startupTool.execute(STARTUP_TOOL_INPUT)
      ),
    };
  }

  /**
   * 起動シーケンスで必須の通知確認 tool を executable tools から取り出す。
   *
   * @returns `check_notifications` に対応する executable tool
   */
  private getStartupToolOrThrow(): AgentSessionTool {
    const startupTool = this.input.tools.find(
      (tool) => tool.name === checkNotificationsToolSpec.name
    );
    if (startupTool === undefined) {
      throw new Error(
        `Required startup tool '${checkNotificationsToolSpec.name}' is not registered`
      );
    }

    return startupTool;
  }
}
