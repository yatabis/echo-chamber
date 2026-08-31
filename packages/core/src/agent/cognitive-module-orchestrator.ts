import { getErrorMessage } from '../utils/error';

import {
  ZERO_MODEL_USAGE,
  accumulateModelUsage,
  type AgentSessionTurnBoundary,
  type AgentSessionTurnBoundaryHandler,
} from './session';

import type {
  EmotionCognitiveModuleOutput,
  MemoryCognitiveModuleOutput,
} from './cognitive-module-schema';
import type { Emotion, MemoryType } from '../echo/types';
import type { ModelInputItem, ModelUsage } from '../ports/model';

/** Cognitive Moduleが実行されるMain session境界。 */
export type CognitiveModulePhase = 'pre_main' | 'post_main';

/** 同一phase内で独立に実行されるmodule名。 */
export type CognitiveModuleName = 'memory' | 'emotion';

/** search_memoryがMainへ返すboundedなMemory。 */
export interface CognitiveModuleRecalledMemory {
  content: string;
  type: MemoryType;
  emotion: Emotion;
  createdAt: string;
}

/** 直前の思考session終了時にMemory Moduleが確定したMemory。 */
export type CognitiveModulePreviousSessionMemory =
  CognitiveModuleRecalledMemory;

/** 確定済みEmotion、前sessionのMemory、および直前のMemory検索結果。 */
export interface CognitiveModuleCommittedState {
  version: number;
  emotion: EmotionCognitiveModuleOutput | null;
  previousSessionMemory: CognitiveModulePreviousSessionMemory | null;
  recalledMemories: readonly CognitiveModuleRecalledMemory[];
}

/** 1 phaseの実行とcommitを識別するruntime情報。 */
export interface CognitiveModulePhaseInput {
  activationId: string;
  boundaryId: string;
  sequence: number;
  phase: CognitiveModulePhase;
  committed: CognitiveModuleCommittedState;
}

/** 1回のmodule model呼び出しが返す結果。 */
export interface CognitiveModuleRunResult<T> {
  value: T;
  usage: ModelUsage;
}

/** Provider後validationのboundedな診断情報。 */
export interface CognitiveModuleOutputDiagnostic {
  code: string;
  issues: readonly { path: string; code: string }[];
}

/** Provider後validation failureの分類と任意のschema診断。 */
export interface CognitiveModuleOutputValidationDetails {
  code: 'empty_output' | 'invalid_json' | 'schema_mismatch' | 'refusal';
  diagnostic?: CognitiveModuleOutputDiagnostic;
}

/** Provider response受領後のvalidation failureと課金metadataを保持する。 */
export class CognitiveModuleOutputValidationError extends Error {
  override readonly name = 'CognitiveModuleOutputValidationError';

  /**
   * @param message boundaryを特定できるfailure message
   * @param usage provider responseに含まれたusage
   * @param details failure codeとschema診断
   */
  constructor(
    message: string,
    readonly usage: ModelUsage,
    details: CognitiveModuleOutputValidationDetails = {
      code: 'schema_mismatch',
    }
  ) {
    super(message);
    this.code = details.code;
    this.diagnostic = details.diagnostic;
  }

  readonly code: CognitiveModuleOutputValidationDetails['code'];
  readonly diagnostic: CognitiveModuleOutputDiagnostic | undefined;
}

/** 現在phaseまでの共有contextとrequest制御。 */
export interface CognitiveModuleRunContext {
  sharedContext: readonly ModelInputItem[];
  signal?: AbortSignal;
}

/** Providerやdomain実装を隠蔽するmodule port。 */
export interface CognitiveModuleRunner<TOutput> {
  run(
    input: CognitiveModulePhaseInput,
    context: CognitiveModuleRunContext
  ): Promise<CognitiveModuleRunResult<TOutput>>;
}

/** Retry可否を、失敗したmoduleだけについて判定する入力。 */
export interface CognitiveModuleRetryInput {
  module: CognitiveModuleName;
  error: unknown;
  failedAttempt: number;
}

/** Retry回数と一時失敗分類を外部から注入する契約。 */
export interface CognitiveModuleRetryPolicy {
  maxAttempts: number;
  shouldRetry(input: CognitiveModuleRetryInput): boolean | Promise<boolean>;
}

/** 1 moduleの成功結果。 */
export interface CognitiveModuleReadyOutcome<T> {
  status: 'ready';
  value: T;
  attempts: number;
}

/** Mainを先へ進めてはならない、1 moduleの確定失敗。 */
export interface CognitiveModuleFailedOutcome {
  status: 'failed';
  reason: 'non_retryable' | 'retry_exhausted';
  error: string;
  attempts: number;
  outputValidation?: {
    code: CognitiveModuleOutputValidationError['code'];
    diagnostic?: CognitiveModuleOutputDiagnostic;
  };
}

/** 1 moduleの成功またはfail-closed terminal result。 */
export type CognitiveModuleOutcome<T> =
  | CognitiveModuleReadyOutcome<T>
  | CognitiveModuleFailedOutcome;

/** Memory / Emotionが両方確定した1 phaseの記録。 */
export interface CognitiveModulePhaseResult extends CognitiveModulePhaseInput {
  memory: CognitiveModuleOutcome<MemoryCognitiveModuleOutput>;
  emotion: CognitiveModuleOutcome<EmotionCognitiveModuleOutput>;
  usage: ModelUsage;
}

/** Domain commitが受理する、schema validation済みの両結果。 */
export interface CognitiveModuleDomainCommitInput {
  phase: CognitiveModulePhaseInput;
  memory: CognitiveModuleReadyOutcome<MemoryCognitiveModuleOutput>;
  emotion: CognitiveModuleReadyOutcome<EmotionCognitiveModuleOutput>;
}

/** Module出力と検索・保存実装を分離するphase commit port。 */
export interface CognitiveModuleDomainPort {
  beginActivation(activationId: string): Promise<CognitiveModuleCommittedState>;
  startPhase(phase: CognitiveModulePhaseInput): Promise<void>;
  commitPhase(
    input: CognitiveModuleDomainCommitInput
  ): Promise<CognitiveModuleCommittedState>;
  failPhase(result: CognitiveModulePhaseResult, cause?: unknown): Promise<void>;
}

/** Memory / Emotionの一方以上が確定失敗した結果。 */
export class CognitiveModulePhaseError extends Error {
  override readonly name = 'CognitiveModulePhaseError';

  /** @param phaseResult 両moduleがsettleした後のexact phase result */
  constructor(readonly phaseResult: CognitiveModulePhaseResult) {
    const failures: string[] = [];
    if (phaseResult.memory.status === 'failed') {
      failures.push(`memory:${phaseResult.memory.reason}`);
    }
    if (phaseResult.emotion.status === 'failed') {
      failures.push(`emotion:${phaseResult.emotion.reason}`);
    }
    super(
      `Cognitive module phase failed at ${phaseResult.boundaryId} (${failures.join(', ')})`
    );
  }
}

/** 両module結果のcommitに失敗しMain advancementを拒否する。 */
export class CognitiveModuleCommitError extends Error {
  override readonly name = 'CognitiveModuleCommitError';

  /** @param cause phase resultの検索または永続化に失敗した原因 */
  constructor(
    readonly phaseResult: CognitiveModulePhaseResult,
    options: { cause: unknown }
  ) {
    super(
      `Cognitive module commit failed at ${phaseResult.boundaryId}: ${getErrorMessage(options.cause)}`
    );
    (this as Error & { cause?: unknown }).cause = options.cause;
  }
}

/** 1 thinking session全体のCognitive Module結果。 */
export interface CognitiveModuleActivationResult {
  activationId: string;
  phases: readonly CognitiveModulePhaseResult[];
  usage: ModelUsage;
}

/** ThinkingEngineが1 activationのlifecycleを操作する最小契約。 */
export interface CognitiveModuleActivation {
  beforeMain(
    sharedInitialContext: readonly ModelInputItem[]
  ): Promise<readonly ModelInputItem[]>;
  onMainTurnBoundary: AgentSessionTurnBoundaryHandler;
  getResultSnapshot(): CognitiveModuleActivationResult;
}

/** ThinkingEngineからactivationごとのcoordinatorを開始するport。 */
export interface CognitiveModuleOrchestrator {
  beginActivation(): CognitiveModuleActivation;
}

/** Memory / Emotion pair coordinatorの構築入力。 */
export interface ParallelCognitiveModuleOrchestratorOptions {
  createActivationId(): string;
  memory: CognitiveModuleRunner<MemoryCognitiveModuleOutput>;
  emotion: CognitiveModuleRunner<EmotionCognitiveModuleOutput>;
  retryPolicy: CognitiveModuleRetryPolicy;
  domain: CognitiveModuleDomainPort;
  createRequestSignal?(): AbortSignal;
  formatInitialContext(
    committed: CognitiveModuleCommittedState
  ): readonly ModelInputItem[] | Promise<readonly ModelInputItem[]>;
  formatHandoff(
    result: CognitiveModulePhaseResult,
    committed: CognitiveModuleCommittedState
  ): readonly ModelInputItem[] | Promise<readonly ModelInputItem[]>;
}

interface ModuleReadyExecutionResult<T> {
  status: 'ready';
  outcome: CognitiveModuleReadyOutcome<T>;
  usage: ModelUsage;
}

interface ModuleFailedExecutionResult {
  status: 'failed';
  outcome: CognitiveModuleFailedOutcome;
  usage: ModelUsage;
}

type ModuleExecutionResult<T> =
  | ModuleReadyExecutionResult<T>
  | ModuleFailedExecutionResult;

const MAIN_THINK_TOOL_NAME = 'think';
const MAIN_THINK_SUCCESS_OUTPUT = '{"success":true}';

/**
 * Mainの自然言語出力を、Cognitive共有履歴だけで使うthink exchangeへ変換する。
 * 実際のtool callはproviderが生成した形を保ち、出力順も変更しない。
 */
function formatMainOutputForCognitiveContext(
  boundary: AgentSessionTurnBoundary
): ModelInputItem[] {
  const context: ModelInputItem[] = [];

  boundary.responseOutput.forEach((item, index) => {
    if (item.type !== 'message') {
      context.push(item);
      return;
    }

    const callId = `cognitive:main:${boundary.turnIndex}:think:${index + 1}`;
    context.push(
      {
        type: 'tool_call',
        callId,
        toolName: MAIN_THINK_TOOL_NAME,
        input: JSON.stringify({ thought: item.content }),
      },
      {
        type: 'tool_result',
        callId,
        output: MAIN_THINK_SUCCESS_OUTPUT,
      }
    );
  });

  return context;
}

/** Provider後validation errorから失敗metadataを作る。 */
function createOutputValidationMetadata(
  error: unknown
): CognitiveModuleFailedOutcome['outputValidation'] | undefined {
  if (!(error instanceof CognitiveModuleOutputValidationError)) {
    return undefined;
  }
  return {
    code: error.code,
    ...(error.diagnostic === undefined ? {} : { diagnostic: error.diagnostic }),
  };
}

/** Attempt errorをterminal outcomeの共通表現へ変換する。 */
function createFailedOutcome(
  error: unknown,
  reason: CognitiveModuleFailedOutcome['reason'],
  attempts: number
): CognitiveModuleFailedOutcome {
  const outputValidation = createOutputValidationMetadata(error);
  return {
    status: 'failed',
    reason,
    error: getErrorMessage(error),
    attempts,
    ...(outputValidation === undefined ? {} : { outputValidation }),
  };
}

type ActivationLifecycle = 'created' | 'active' | 'terminal' | 'failed';

/**
 * MemoryとEmotionを同じ共有contextから並列実行するcoordinator。
 *
 * 同一phaseの結果は互いへ渡さず、両方の確定後にだけMainへ返す。
 */
export class ParallelCognitiveModuleOrchestrator
  implements CognitiveModuleOrchestrator
{
  /** @param options module port、retry policy、Main handoff formatter */
  constructor(
    private readonly options: ParallelCognitiveModuleOrchestratorOptions
  ) {
    if (!Number.isInteger(options.retryPolicy.maxAttempts)) {
      throw new Error('Cognitive module maxAttempts must be an integer');
    }
    if (options.retryPolicy.maxAttempts < 1) {
      throw new Error('Cognitive module maxAttempts must be at least 1');
    }
  }

  /** 新しいthinking session用の独立coordinatorを開始する。 */
  beginActivation(): CognitiveModuleActivation {
    const activationId = this.options.createActivationId().trim();
    if (activationId === '') {
      throw new Error('Cognitive module activationId must not be empty');
    }
    return new ParallelCognitiveModuleActivation(activationId, this.options);
  }
}

/** 1 activationに閉じた共有context、phase、usageの所有者。 */
class ParallelCognitiveModuleActivation implements CognitiveModuleActivation {
  private readonly phases: CognitiveModulePhaseResult[] = [];
  private readonly sharedContext: ModelInputItem[] = [];
  private lifecycle: ActivationLifecycle = 'created';
  private phaseInProgress = false;
  private sequence = 0;
  private committed: CognitiveModuleCommittedState | null = null;
  private totalUsage: ModelUsage = ZERO_MODEL_USAGE;

  /**
   * @param activationId 1 thinking sessionのstable identifier
   * @param options 親orchestratorのmodule、retry、handoff設定
   */
  constructor(
    private readonly activationId: string,
    private readonly options: ParallelCognitiveModuleOrchestratorOptions
  ) {}

  /** 最初のMain model turn前にrecallとEmotion更新を実行する。 */
  async beforeMain(
    sharedInitialContext: readonly ModelInputItem[]
  ): Promise<readonly ModelInputItem[]> {
    if (this.lifecycle !== 'created') {
      throw new Error('Cognitive module pre_main phase has already started');
    }
    this.lifecycle = 'active';
    this.committed = await this.options.domain.beginActivation(
      this.activationId
    );
    const initialContext = await this.options.formatInitialContext(
      this.committed
    );
    this.sharedContext.push(...initialContext, ...sharedInitialContext);

    return await this.runPhase('pre_main');
  }

  /** 各Main turn境界で次turn前またはsession終了後のmodule処理を行う。 */
  readonly onMainTurnBoundary: AgentSessionTurnBoundaryHandler = async (
    boundary
  ): Promise<readonly ModelInputItem[]> => {
    if (this.lifecycle !== 'active') {
      throw new Error(
        `Cognitive module boundary received while activation is ${this.lifecycle}`
      );
    }

    const phase =
      boundary.terminationReason === null ? 'pre_main' : 'post_main';
    this.sharedContext.push(
      ...formatMainOutputForCognitiveContext(boundary),
      ...boundary.resolvedInput
    );
    const handoff = await this.runPhase(phase);

    if (phase === 'post_main') {
      this.lifecycle = 'terminal';
      return [];
    }
    return handoff;
  };

  /** Terminal phaseまでに確定したactivation recordを返す。 */
  getResultSnapshot(): CognitiveModuleActivationResult {
    return {
      activationId: this.activationId,
      phases: [...this.phases],
      usage: this.totalUsage,
    };
  }

  /** Memory / Emotion pairを同じ共有contextから並列実行して確定する。 */
  private async runPhase(
    phase: CognitiveModulePhase
  ): Promise<readonly ModelInputItem[]> {
    if (this.phaseInProgress) {
      throw new Error('Cognitive module phase is already in progress');
    }
    this.phaseInProgress = true;

    try {
      this.sequence += 1;
      const snapshot: CognitiveModulePhaseInput = {
        activationId: this.activationId,
        boundaryId: `${this.activationId}:${this.sequence}:${phase}`,
        sequence: this.sequence,
        phase,
        committed: this.getCommittedStateOrThrow(),
      };
      const sharedContext = [...this.sharedContext];
      await this.options.domain.startPhase(snapshot);
      const [memory, emotion] = await Promise.all([
        this.runModule('memory', this.options.memory, snapshot, sharedContext),
        this.runModule(
          'emotion',
          this.options.emotion,
          snapshot,
          sharedContext
        ),
      ]);

      const usage = accumulateModelUsage(memory.usage, emotion.usage);
      const result: CognitiveModulePhaseResult = {
        ...snapshot,
        memory: memory.outcome,
        emotion: emotion.outcome,
        usage,
      };
      this.phases.push(result);
      this.totalUsage = accumulateModelUsage(this.totalUsage, usage);

      if (memory.status === 'failed' || emotion.status === 'failed') {
        this.lifecycle = 'failed';
        await this.options.domain.failPhase(result);
        throw new CognitiveModulePhaseError(result);
      }

      try {
        this.committed = await this.options.domain.commitPhase({
          phase: snapshot,
          memory: memory.outcome,
          emotion: emotion.outcome,
        });
      } catch (error) {
        this.lifecycle = 'failed';
        try {
          await this.options.domain.failPhase(result, error);
        } catch {
          // Original commit failure remains the primary fail-closed cause.
        }
        throw new CognitiveModuleCommitError(result, { cause: error });
      }

      if (phase === 'post_main') {
        return [];
      }
      const handoff = await this.options.formatHandoff(result, this.committed);
      this.sharedContext.push(...handoff);
      return handoff;
    } finally {
      this.phaseInProgress = false;
    }
  }

  /** beginActivationで取得済みの確定状態を返す。 */
  private getCommittedStateOrThrow(): CognitiveModuleCommittedState {
    if (this.committed === null) {
      throw new Error('Cognitive module committed state was not initialized');
    }
    return this.committed;
  }

  /** 1 moduleだけを、同じ共有context snapshotからretryする。 */
  private async runModule<TOutput>(
    module: CognitiveModuleName,
    runner: CognitiveModuleRunner<TOutput>,
    snapshot: CognitiveModulePhaseInput,
    sharedContext: readonly ModelInputItem[]
  ): Promise<ModuleExecutionResult<TOutput>> {
    let attempt = 0;
    let totalUsage = ZERO_MODEL_USAGE;

    while (attempt < this.options.retryPolicy.maxAttempts) {
      attempt += 1;
      try {
        const signal = this.options.createRequestSignal?.();
        const context: CognitiveModuleRunContext = {
          sharedContext,
          ...(signal === undefined ? {} : { signal }),
        };
        // Retry is sequential for one module; its sibling stays independent.
        // eslint-disable-next-line no-await-in-loop
        const result = await runner.run(snapshot, context);
        totalUsage = accumulateModelUsage(totalUsage, result.usage);
        return {
          status: 'ready',
          outcome: {
            status: 'ready',
            value: result.value,
            attempts: attempt,
          },
          usage: totalUsage,
        };
      } catch (error) {
        if (error instanceof CognitiveModuleOutputValidationError) {
          totalUsage = accumulateModelUsage(totalUsage, error.usage);
        }
        // Retry policy may own bounded asynchronous backoff.
        // eslint-disable-next-line no-await-in-loop
        const shouldRetry = await this.options.retryPolicy.shouldRetry({
          module,
          error,
          failedAttempt: attempt,
        });
        if (!shouldRetry) {
          return {
            status: 'failed',
            outcome: createFailedOutcome(error, 'non_retryable', attempt),
            usage: totalUsage,
          };
        }
        if (attempt >= this.options.retryPolicy.maxAttempts) {
          return {
            status: 'failed',
            outcome: createFailedOutcome(error, 'retry_exhausted', attempt),
            usage: totalUsage,
          };
        }
      }
    }

    throw new Error('Cognitive module retry loop ended unexpectedly');
  }
}
