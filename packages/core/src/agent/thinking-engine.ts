import { buildAgentPromptMessages } from './prompt-builder';
import { runAgentSession } from './session';
import { checkNotificationsToolSpec } from './tools/chat';

import type {
  PromptContextSnapshot,
  PromptMemoryContext,
} from './prompt-builder';
import type { AgentSessionTool } from './session';
import type { ContextPort, ContextSnapshot } from '../ports/context';
import type { LoggerPort } from '../ports/logger';
import type { MemoryPort, MemoryRecord } from '../ports/memory';
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
  memory: Pick<MemoryPort, 'getLatest'>;
  tools: readonly AgentSessionTool[];
  systemPrompt: string;
}

/**
 * 1 回の思考実行が返す集約結果。
 * usage に加え、次回起動用の最新 context を caller が保存できる形で返す。
 */
export interface ThinkingEngineResult {
  usage: ModelUsage;
  context: ContextSnapshot | null;
}

const STARTUP_TOOL_INPUT = '{}';
const THINKING_STARTED_MESSAGE = '*Thinking started...*';
const THINKING_COMPLETED_MESSAGE = '*Thinking completed.*';

/**
 * MemoryPort のレコードを prompt builder が受け取る最小表現へ変換する。
 * 永続化用の余分な項目は持ち込まず、起動時の文脈再開に必要な情報だけ残す。
 *
 * @param memory MemoryPort から取得した最新 memory
 * @returns prompt builder に渡せる memory context。memory がなければ `null`
 */
function toPromptMemoryContext(
  memory: MemoryRecord | null
): PromptMemoryContext | null {
  if (memory === null) {
    return null;
  }

  return {
    content: memory.content,
    createdAt: memory.createdAt,
    emotion: memory.emotion,
  };
}

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
 * provider/runtime 非依存の思考 orchestration。
 * prompt 構築、起動時通知チェック、session 実行、thought log 送信を順に担う。
 */
export class ThinkingEngine {
  /**
   * @param input 実行に必要な port・tool・prompt 設定
   */
  constructor(private readonly input: ThinkingEngineInput) {}

  /**
   * 1 回分の思考 session を開始し、usage と次回再開用 context を返す。
   *
   * @returns session 全体で集計した usage と、保存可能な最新 context
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
    const [latestMemory, latestContext] = await Promise.all([
      this.input.memory.getLatest(),
      this.input.context.load(),
    ]);
    const promptMessages = buildAgentPromptMessages({
      systemPrompt: this.input.systemPrompt,
      currentDatetime: new Date(),
      latestContext: toPromptContext(latestContext),
      latestMemory: toPromptMemoryContext(latestMemory),
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
