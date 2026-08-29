import type {
  MemorySystem,
  PreparedMemoryWrite,
} from '@echo-chamber/cloudflare-runtime/memory-system';
import type {
  CognitiveModuleCommittedState,
  CognitiveModuleDomainCommitInput,
  CognitiveModuleDomainPort,
  CognitiveModuleOutcome,
  CognitiveModulePhaseInput,
  CognitiveModulePhaseResult,
  CognitiveModuleRecalledMemory,
} from '@echo-chamber/core/agent/cognitive-module-orchestrator';
import type {
  EmotionCognitiveModuleOutput,
  MemoryCognitiveModuleOutput,
  MemoryRecallCognitiveModuleOutput,
  MemoryStoreCognitiveModuleOutput,
} from '@echo-chamber/core/agent/cognitive-module-schema';
import { emitEchoEvent } from '@echo-chamber/core/ports/echo-event';
import type { EchoEventPort } from '@echo-chamber/core/ports/echo-event';
import { getErrorMessage } from '@echo-chamber/core/utils/error';

type CognitiveMemorySystem = Pick<
  MemorySystem,
  | 'searchMemory'
  | 'prepareMemoryWrite'
  | 'commitPreparedMemoryWrites'
  | 'emitMemoryCommitEvents'
>;

const DOMAIN_STATE_KEY = 'cognitive:domain-state';
const MAX_RECALLED_MEMORIES = 5;

interface StoredDomainState {
  version: number;
  emotion: EmotionCognitiveModuleOutput | null;
  recalledMemories?: readonly CognitiveModuleRecalledMemory[];
  lastBoundaryId: string | null;
  updatedAt: string;
}

interface PreparedPhaseOperation {
  recalledMemories: readonly CognitiveModuleRecalledMemory[];
  memoryWrites: readonly PreparedMemoryWrite[];
}

/** DashboardがCognitive Moduleの確定済み状態だけを表示するread model。 */
export interface CognitiveModuleDashboardState {
  domainVersion: number;
  lastBoundaryId: string | null;
  updatedAt: string | null;
}

/** Cognitive Moduleの検索・更新を確定するdomain入力。 */
export interface CognitiveModuleDomainStoreOptions {
  storage: DurableObjectStorage;
  memory: CognitiveMemorySystem;
  events?: EchoEventPort;
  now?(): Date;
  isRetryable?(error: unknown): boolean;
  maxAttempts?: number;
}

/**
 * pre_mainではMemory検索とEmotion更新、post_mainではMemory保存とEmotion更新を
 * 確定する。外部I/Oをtransaction前に完了し、永続化はphase単位で行う。
 */
export class CognitiveModuleDomainStore implements CognitiveModuleDomainPort {
  private readonly now: () => Date;
  private readonly maxAttempts: number;

  /** @param options DO storage、Memory store、retry/event dependencies */
  constructor(private readonly options: CognitiveModuleDomainStoreOptions) {
    this.now = (): Date => options.now?.() ?? new Date();
    this.maxAttempts = options.maxAttempts ?? 2;
    if (!Number.isInteger(this.maxAttempts) || this.maxAttempts < 1) {
      throw new Error('Cognitive domain maxAttempts must be at least 1');
    }
  }

  /** 保存済みEmotionを読み、新しいsessionでは検索結果を空にする。 */
  async beginActivation(
    _activationId: string
  ): Promise<CognitiveModuleCommittedState> {
    const stored =
      await this.options.storage.get<StoredDomainState>(DOMAIN_STATE_KEY);
    return toRuntimeState(stored, []);
  }

  /** Mainが明示的にstore_memoryを使う際の確定済みEmotionを返す。 */
  async getCurrentEmotion(): Promise<EmotionCognitiveModuleOutput> {
    const stored =
      await this.options.storage.get<StoredDomainState>(DOMAIN_STATE_KEY);
    if (stored?.emotion === null || stored?.emotion === undefined) {
      throw new Error('Current cognitive emotion is not available');
    }
    return stored.emotion;
  }

  /** Phase開始を通常のsession eventとして記録する。 */
  async startPhase(phase: CognitiveModulePhaseInput): Promise<void> {
    await emitEchoEvent(this.options.events, {
      type: 'cognitive.phase.started',
      severity: 'debug',
      summary: `cognitive phase started: ${phase.phase}`,
      payload: createPhaseMetadata(phase),
    });
  }

  /** Module出力が指定した検索または保存を実行し、Emotionと一括確定する。 */
  async commitPhase(
    input: CognitiveModuleDomainCommitInput
  ): Promise<CognitiveModuleCommittedState> {
    const stored =
      await this.options.storage.get<StoredDomainState>(DOMAIN_STATE_KEY);
    if (stored?.lastBoundaryId === input.phase.boundaryId) {
      return toRuntimeState(stored, stored.recalledMemories ?? []);
    }

    const storedVersion = stored?.version ?? 0;
    if (storedVersion !== input.phase.committed.version) {
      throw new Error('Cognitive domain state version conflict');
    }

    const operation = await this.preparePhaseOperation(input);
    const now = this.now().toISOString();
    const nextStored: StoredDomainState = {
      version: storedVersion + 1,
      emotion: input.emotion.value,
      recalledMemories: operation.recalledMemories,
      lastBoundaryId: input.phase.boundaryId,
      updatedAt: now,
    };

    const memoryReceipt = await this.options.storage.transaction(
      async (transaction) => {
        const concurrent =
          await transaction.get<StoredDomainState>(DOMAIN_STATE_KEY);
        if (concurrent?.lastBoundaryId === input.phase.boundaryId) {
          return {
            storedIds: [],
            existingIds: operation.memoryWrites.map((write) => write.id),
            evicted: [],
          };
        }
        if ((concurrent?.version ?? 0) !== storedVersion) {
          throw new Error('Cognitive domain state changed during commit');
        }

        const receipt =
          operation.memoryWrites.length === 0
            ? { storedIds: [], existingIds: [], evicted: [] }
            : this.options.memory.commitPreparedMemoryWrites(
                operation.memoryWrites
              );
        await transaction.put(DOMAIN_STATE_KEY, nextStored);
        return receipt;
      }
    );

    if (operation.memoryWrites.length > 0) {
      await this.options.memory.emitMemoryCommitEvents(memoryReceipt);
    }
    await emitEchoEvent(this.options.events, {
      type: 'cognitive.phase.committed',
      severity: 'info',
      summary: `cognitive phase committed: ${input.phase.phase}`,
      payload: {
        ...createPhaseMetadata(input.phase),
        committedVersion: nextStored.version,
        recalledMemories: operation.recalledMemories.length,
        memoryUpdates: operation.memoryWrites.length,
      },
    });

    return toRuntimeState(nextStored, operation.recalledMemories);
  }

  /** 現在のphaseを保存せず、失敗内容を通常のsession logへ記録する。 */
  async failPhase(
    result: CognitiveModulePhaseResult,
    cause?: unknown
  ): Promise<void> {
    await emitEchoEvent(this.options.events, {
      type: 'cognitive.phase.failed',
      severity: 'error',
      summary: `cognitive phase failed: ${result.phase}`,
      payload: {
        ...createPhaseMetadata(result),
        memory: createOutcomeMetadata(result.memory),
        emotion: createOutcomeMetadata(result.emotion),
        ...(cause === undefined ? {} : { commitError: getErrorMessage(cause) }),
      },
    });
  }

  /** Dashboard用に確定済みdomain stateの位置だけを返す。 */
  async getDashboardState(): Promise<CognitiveModuleDashboardState> {
    const state =
      await this.options.storage.get<StoredDomainState>(DOMAIN_STATE_KEY);
    return {
      domainVersion: state?.version ?? 0,
      lastBoundaryId: state?.lastBoundaryId ?? null,
      updatedAt: state?.updatedAt ?? null,
    };
  }

  /** Phaseに対応するMemory操作を検証し、transaction外で準備する。 */
  private async preparePhaseOperation(
    input: CognitiveModuleDomainCommitInput
  ): Promise<PreparedPhaseOperation> {
    if (input.phase.phase === 'pre_main') {
      const recall = getRecallOutput(input.memory.value);
      return {
        recalledMemories: await this.loadRecall(recall.query),
        memoryWrites: [],
      };
    }

    const memory = getStoreOutput(input.memory.value);
    const write = await this.prepareMemoryWriteWithRetry(
      `${input.phase.boundaryId}:store_memory`,
      memory,
      input.emotion.value
    );
    return {
      recalledMemories: [],
      memoryWrites: [write],
    };
  }

  /** Memory Moduleのqueryで検索し、一時失敗だけを再試行する。 */
  private async loadRecall(
    query: string
  ): Promise<CognitiveModuleRecalledMemory[]> {
    let attempt = 0;
    while (attempt < this.maxAttempts) {
      attempt += 1;
      try {
        // eslint-disable-next-line no-await-in-loop
        const results = await this.options.memory.searchMemory(query);
        return results.slice(0, MAX_RECALLED_MEMORIES).map((memory) => ({
          content: memory.content,
          type: memory.type,
          emotion: memory.emotion,
          createdAt: memory.createdAt,
        }));
      } catch (error) {
        if (
          attempt >= this.maxAttempts ||
          this.options.isRetryable?.(error) !== true
        ) {
          throw error;
        }
      }
    }

    throw new Error('Cognitive memory recall retry loop ended unexpectedly');
  }

  /** 1件のsession Memoryを同じIDとEmotionで準備し、一時失敗だけ再試行する。 */
  private async prepareMemoryWriteWithRetry(
    id: string,
    memory: MemoryStoreCognitiveModuleOutput,
    emotion: EmotionCognitiveModuleOutput
  ): Promise<PreparedMemoryWrite> {
    let attempt = 0;
    while (attempt < this.maxAttempts) {
      attempt += 1;
      try {
        // eslint-disable-next-line no-await-in-loop
        return await this.options.memory.prepareMemoryWrite(
          id,
          memory.content,
          emotion,
          memory.type
        );
      } catch (error) {
        if (
          attempt >= this.maxAttempts ||
          this.options.isRetryable?.(error) !== true
        ) {
          throw error;
        }
      }
    }

    throw new Error(
      'Cognitive memory preparation retry loop ended unexpectedly'
    );
  }
}

/** Memory出力がpre_main用queryであることを保証する。 */
function getRecallOutput(
  value: MemoryCognitiveModuleOutput
): MemoryRecallCognitiveModuleOutput {
  if (!('query' in value)) {
    throw new Error('Memory pre_main output must contain query');
  }
  return value;
}

/** Memory出力がpost_main用store inputであることを保証する。 */
function getStoreOutput(
  value: MemoryCognitiveModuleOutput
): MemoryStoreCognitiveModuleOutput {
  if ('query' in value) {
    throw new Error('Memory post_main output must contain content and type');
  }
  return value;
}

/** Stored stateをactivation-local stateへ変換する。 */
function toRuntimeState(
  stored: StoredDomainState | undefined,
  recalledMemories: readonly CognitiveModuleRecalledMemory[]
): CognitiveModuleCommittedState {
  return {
    version: stored?.version ?? 0,
    emotion: stored?.emotion ?? null,
    recalledMemories: [...recalledMemories],
  };
}

/** Phase eventへcorrelation metadataを付ける。 */
function createPhaseMetadata(
  phase: Pick<
    CognitiveModulePhaseInput,
    'activationId' | 'boundaryId' | 'phase' | 'sequence'
  >
): Record<string, unknown> {
  return {
    activationId: phase.activationId,
    boundaryId: phase.boundaryId,
    phase: phase.phase,
    sequence: phase.sequence,
  };
}

/** Module outcomeをMain sessionと同じerror情報を含むevent payloadにする。 */
function createOutcomeMetadata(
  outcome: CognitiveModuleOutcome<unknown>
): Record<string, unknown> {
  if (outcome.status === 'ready') {
    return { status: outcome.status, attempts: outcome.attempts };
  }
  return {
    status: outcome.status,
    reason: outcome.reason,
    error: outcome.error,
    attempts: outcome.attempts,
    ...(outcome.outputValidation === undefined
      ? {}
      : { outputValidation: outcome.outputValidation }),
  };
}
