import { getErrorMessage } from '../utils/error';

import { buildAgentPromptMessages } from './prompt-builder';
import { runAgentSession } from './session';
import { checkNotificationsToolSpec } from './tools/chat';

import type {
  PromptContextSnapshot,
  PromptRelatedMemorySnapshot,
} from './prompt-builder';
import type { AgentSessionTool } from './session';
import type { ContextPort, ContextSnapshot } from '../ports/context';
import type { LoggerPort } from '../ports/logger';
import type { MemoryPort, MemorySearchResult } from '../ports/memory';
import type {
  ModelInputItem,
  ModelPort,
  ModelToolCall,
  ModelToolResult,
  ModelUsage,
} from '../ports/model';
import type { ThoughtLogPort } from '../ports/thought-log';

/**
 * Thinking engine の構築入力。
 * prompt に必要な data と provider/runtime service を一箇所に集約する。
 */
export interface ThinkingEngineInput {
  model: ModelPort;
  thoughtLog: ThoughtLogPort;
  logger: LoggerPort;
  context: Pick<ContextPort, 'load'>;
  memory: Pick<MemoryPort, 'search'>;
  tools: readonly AgentSessionTool[];
  systemPrompt: string;
}

/**
 * 1 回の思考実行が返す集約結果。
 * usage に加え、次回起動用の最新 context と次回起動時刻を
 * caller が保存できる形で返す。
 */
export interface ThinkingEngineResult {
  usage: ModelUsage;
  context: ContextSnapshot | null;
  nextWakeAt: string | null;
}

const STARTUP_TOOL_INPUT = '{}';
const THINKING_STARTED_MESSAGE = '*Thinking started...*';
const THINKING_COMPLETED_MESSAGE = '*Thinking completed.*';

/**
 * 永続化された context snapshot を prompt builder 用の最小表現へ変換する。
 *
 * @param context ContextPort から取得した最新 context
 * @returns prompt builder に渡せる context。未保存なら `null`
 */
function toPromptContext(
  context: ContextSnapshot | null
): PromptContextSnapshot | null {
  if (context === null) {
    return null;
  }

  return {
    content: context.content,
    createdAt: context.createdAt,
    emotion: context.emotion,
  };
}

/**
 * memory search 結果を prompt builder 用の最小表現へ変換する。
 */
function toPromptRelatedMemories(
  memories: MemorySearchResult[]
): PromptRelatedMemorySnapshot[] {
  return memories.map((memory) => ({
    content: memory.content,
    type: memory.type,
    createdAt: memory.createdAt,
    emotion: memory.emotion,
  }));
}

/**
 * provider/runtime 非依存の思考 orchestration。
 * prompt 構築、起動時通知チェック、session 実行、thought log 送信を順に担う。
 */
export class ThinkingEngine {
  /**
   * @param input 実行に必要な port・tool・prompt 設定
   */
  constructor(private readonly input: ThinkingEngineInput) {}

  /**
   * 1 回分の思考 session を開始し、usage と次回再開用 state を返す。
   *
   * @returns session 全体で集計した usage、保存可能な最新 context、
   * 次回起動時刻
   */
  async think(): Promise<ThinkingEngineResult> {
    await this.input.thoughtLog.send(THINKING_STARTED_MESSAGE);

    const session = await runAgentSession({
      model: this.input.model,
      tools: this.input.tools,
      initialInput: await this.buildInitialInput(),
      logger: this.input.logger,
    });
    const completedAt = new Date().toISOString();

    await this.input.thoughtLog.send(THINKING_COMPLETED_MESSAGE);
    return {
      context:
        session.context === undefined
          ? null
          : {
              content: session.context.content,
              createdAt: completedAt,
              emotion: session.context.emotion,
              updatedAt: completedAt,
            },
      nextWakeAt: session.nextWakeAt,
      usage: session.usage,
    };
  }

  /**
   * developer prompt と startup tool の往復を初期 input にまとめる。
   * 起動時の通知チェックは通常 turn に入る前に必ず 1 度実行する。
   *
   * @returns `runAgentSession()` に渡す初期 input 一式
   */
  private async buildInitialInput(): Promise<ModelInputItem[]> {
    const latestContext = await this.input.context.load();
    const relatedMemories = await this.loadRelatedMemories(latestContext);
    const promptMessages = buildAgentPromptMessages({
      systemPrompt: this.input.systemPrompt,
      currentDatetime: new Date(),
      latestContext: toPromptContext(latestContext),
      relatedMemories: toPromptRelatedMemories(relatedMemories),
    });
    const promptInputs: ModelInputItem[] = promptMessages.map((message) => ({
      role: message.role,
      content: message.content,
    }));

    return [
      ...promptInputs,
      this.createStartupToolCallInput(),
      await this.createStartupToolResultInput(),
    ];
  }

  /**
   * 最新 context があれば、それをクエリに関連メモリを読み出す。
   * 検索に失敗しても起動自体は継続する。
   */
  private async loadRelatedMemories(
    latestContext: ContextSnapshot | null
  ): Promise<MemorySearchResult[]> {
    if (latestContext === null) {
      return [];
    }

    try {
      return await this.input.memory.search(latestContext.content);
    } catch (error) {
      await this.input.logger.warn(
        `Failed to load related memories for startup context: ${getErrorMessage(error)}`
      );
      return [];
    }
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
      output: await startupTool.execute(STARTUP_TOOL_INPUT),
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
