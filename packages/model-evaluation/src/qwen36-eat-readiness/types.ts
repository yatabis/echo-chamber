import type { Emotion, Note } from '@echo-chamber/core/echo/types';
import type { MemoryRecord } from '@echo-chamber/core/ports/memory';

/** Qwen runtime 評価内で session 間の継続性を表す snapshot。 */
export interface RuntimeContextSnapshot {
  content: string;
  emotion: Emotion;
  createdAt: string;
}

/** Rapid-MLXで直接起動し、同じ評価条件を適用するローカルモデル。 */
export interface LocalEvaluationTarget {
  /** ログ名、結果識別子、評価セッションIDに使う一意なID。 */
  id: string;
  /** 実行ログへ表示する人間向けモデル名。 */
  displayName: string;
  /** Rapid-MLXが直接起動できる自己完結したモデルディレクトリ。 */
  modelPath: string;
  /** Rapid-MLXの公開名とChat Completions要求で共通して使うモデル名。 */
  servedModelName: string;
}

export type RuntimeInstructionMode = 'explicit' | 'implicit';

/**
 * 一つのruntime生成条件。productionは実運用のsampling値を写し、
 * controlledはモデル間・学習前後の再現可能な比較に使う。
 */
export interface RuntimeGenerationProfile {
  id: 'controlled-greedy' | 'production-sampling';
  description: string;
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  repetitionPenalty: number;
  presencePenalty: number;
  maxTokensPerTurn: number;
  enableThinking: false;
}

/**
 * 評価中に観測した副作用またはモデル操作。
 */
export interface TraceCall {
  kind:
    | 'notification_summary'
    | 'read_chat'
    | 'send_chat'
    | 'add_reaction'
    | 'search_memory'
    | 'store_memory'
    | 'startup_memory_search'
    | 'list_notes'
    | 'get_note'
    | 'search_notes'
    | 'create_note'
    | 'update_note'
    | 'delete_note'
    | 'list_zenn'
    | 'get_zenn';
  elapsedMs: number;
  input: Record<string, unknown>;
  output?: unknown;
}

/**
 * モデルとセッションループが発火したイベントのうち、評価証拠に必要な部分。
 */
export interface TraceEvent {
  type: string;
  severity: string;
  elapsedMs: number;
  summary: string;
  payload?: Record<string, unknown>;
}

export type EvaluationCategory =
  | 'outcome'
  | 'protocol'
  | 'completion'
  | 'safety';

/**
 * 一つの採点条件と、その根拠。
 *
 * `timeComparable` が真の条件だけを同一所要時間の到達曲線に含める。
 * 「誤送信がなかった」のような否定条件は終了時にしか確定しないため、
 * 時間曲線へ混ぜない。
 */
export interface EvaluationCheck {
  id: string;
  description: string;
  category: EvaluationCategory;
  weight: number;
  passed: boolean;
  timeComparable: boolean;
  firstSatisfiedMs: number | null;
  evidence: string;
}

export interface ScoreSummary {
  earned: number;
  possible: number;
  byCategory: Record<EvaluationCategory, { earned: number; possible: number }>;
  timeComparableEarned: Record<string, number>;
  timeComparablePossible: number;
}

export interface RuntimeScenarioResult {
  scenarioId: string;
  title: string;
  instructionMode: RuntimeInstructionMode;
  generationProfile: RuntimeGenerationProfile;
  repetition: number;
  elapsedMs: number;
  terminationReason: 'finish_thinking' | 'max_turns' | 'error';
  error?: string;
  usage: {
    cachedInputTokens: number;
    cacheWriteInputTokens: number;
    uncachedInputTokens: number;
    totalInputTokens: number;
    outputTokens: number;
    reasoningTokens: number;
    totalTokens: number;
  };
  calls: TraceCall[];
  events: TraceEvent[];
  checks: EvaluationCheck[];
  score: ScoreSummary;
}

export interface RuntimeSessionTrace {
  sessionId: string;
  title: string;
  elapsedMs: number;
  terminationReason: 'finish_thinking' | 'max_turns' | 'error';
  error?: string;
  usage: RuntimeScenarioResult['usage'];
  calls: TraceCall[];
  events: TraceEvent[];
  contextBefore: RuntimeContextSnapshot | null;
  contextAfter: RuntimeContextSnapshot | null;
  memoryCountBefore: number;
  memoryCountAfter: number;
}

export interface RuntimeWorkflowResult {
  workflowId: string;
  title: string;
  instructionMode: RuntimeInstructionMode;
  generationProfile: RuntimeGenerationProfile;
  repetition: number;
  elapsedMs: number;
  sessions: RuntimeSessionTrace[];
  finalMemories: MemoryRecord[];
  finalNotes: Note[];
  checks: EvaluationCheck[];
  score: ScoreSummary;
}
