import { MemorySystem } from '@echo-chamber/cloudflare-runtime/memory-system';
import { NoteSystem } from '@echo-chamber/cloudflare-runtime/note-system';
import { getTodayUsageKey } from '@echo-chamber/core';
import type { EchoInstanceConfig, UsageRecord } from '@echo-chamber/core';
import { buildAgentPromptMessages } from '@echo-chamber/core/agent/prompt-builder';
import { runAgentSession } from '@echo-chamber/core/agent/session';
import type {
  ModelInputItem,
  ModelToolCall,
  ModelToolResult,
  ModelUsage,
} from '@echo-chamber/core/ports/model';
import { OpenAIResponsesModel } from '@echo-chamber/openai-adapter/openai-responses-model';

import { DiscordThoughtLog } from '../../discord/client';
import { createEmbeddingService } from '../../llm/embedding-factory';
import {
  addReactionToChatMessageFunction,
  checkNotificationsFunction,
  readChatMessagesFunction,
  sendChatMessageFunction,
} from '../../llm/openai/functions/chat';
import { finishThinkingFunction } from '../../llm/openai/functions/finish';
import {
  storeMemoryFunction,
  searchMemoryFunction,
} from '../../llm/openai/functions/memory';
import {
  createNoteFunction,
  deleteNoteFunction,
  getNoteFunction,
  listNotesFunction,
  searchNotesFunction,
  updateNoteFunction,
} from '../../llm/openai/functions/note';
import { thinkDeeplyFunction } from '../../llm/openai/functions/think';
import { createToolExecutionContext } from '../../llm/openai/functions/tool-context';

import type { ITool, ToolContext } from '../../llm/openai/functions';
import type { Logger } from '../../utils/logger';

/**
 * Echo の思考エンジン
 */
export class ThinkingEngine {
  private readonly env: Env;
  private readonly storage: DurableObjectStorage;
  private readonly toolContext: ToolContext;
  private readonly instanceConfig: EchoInstanceConfig;
  private readonly memorySystem: MemorySystem;

  constructor(options: {
    env: Env;
    storage: DurableObjectStorage;
    sql: SqlStorage;
    logger: Logger;
    instanceConfig: EchoInstanceConfig;
  }) {
    this.env = options.env;
    this.storage = options.storage;
    this.instanceConfig = options.instanceConfig;
    const embeddingService = createEmbeddingService(
      options.env,
      options.instanceConfig.embeddingConfig
    );
    this.memorySystem = new MemorySystem({
      sql: options.sql,
      embeddingService,
      logger: options.logger,
    });
    const noteSystem = new NoteSystem({
      storage: options.storage,
      logger: options.logger,
    });
    this.toolContext = createToolExecutionContext({
      instanceConfig: options.instanceConfig,
      memorySystem: this.memorySystem,
      noteSystem,
      logger: options.logger,
    });
  }

  async initialize(): Promise<void> {
    await this.memorySystem.reEmbedStaleMemories();
  }

  async think(): Promise<ModelUsage> {
    const thoughtLog = new DiscordThoughtLog({
      token: this.instanceConfig.discordBotToken,
      channelId: this.instanceConfig.thinkingChannelId,
    });
    await thoughtLog.send('*Thinking started...*');
    const tools = this.createTools();
    const session = await runAgentSession({
      model: this.createOpenAIClient(thoughtLog),
      tools: tools.map((tool) => ({
        name: tool.name,
        contract: tool.contract,
        execute: async (input: string): Promise<string> =>
          await tool.execute(input, this.toolContext),
      })),
      initialInput: await this.buildInitialInput(),
      logger: this.toolContext.logger,
    });
    const usage = session.usage;
    await thoughtLog.send(
      `*Thinking completed.*\nUsage: ${usage.totalTokens} tokens (Total: ${
        (await this.getCurrentUsage()) + usage.totalTokens
      } tokens)`
    );
    return usage;
  }

  private createOpenAIClient(
    thoughtLog: DiscordThoughtLog
  ): OpenAIResponsesModel {
    return new OpenAIResponsesModel({
      apiKey: this.env.OPENAI_API_KEY,
      logger: this.toolContext.logger,
      thoughtLog,
    });
  }

  private createTools(): ITool[] {
    return [
      checkNotificationsFunction,
      readChatMessagesFunction,
      sendChatMessageFunction,
      addReactionToChatMessageFunction,
      storeMemoryFunction,
      searchMemoryFunction,
      createNoteFunction,
      listNotesFunction,
      getNoteFunction,
      searchNotesFunction,
      updateNoteFunction,
      deleteNoteFunction,
      thinkDeeplyFunction,
      finishThinkingFunction,
    ];
  }

  private async buildInitialInput(): Promise<ModelInputItem[]> {
    const latestMemory = this.memorySystem.getLatestMemory();
    const promptMessages = buildAgentPromptMessages({
      systemPrompt: this.instanceConfig.systemPrompt,
      currentDatetime: new Date(),
      latestMemory:
        latestMemory === null
          ? null
          : {
              content: latestMemory.content,
              createdAt: latestMemory.createdAt,
              emotion: latestMemory.emotion,
            },
    });
    const promptInputs: ModelInputItem[] = promptMessages.map((message) => ({
      role: message.role,
      content: message.content,
    }));

    return [
      ...promptInputs,
      this.createToolCallInput(checkNotificationsFunction),
      await this.createToolResultInput(checkNotificationsFunction),
    ];
  }

  private createToolCallInput(tool: ITool): ModelToolCall {
    return {
      type: 'tool_call',
      callId: tool.name,
      toolName: tool.name,
      input: '{}',
    };
  }

  private async createToolResultInput(tool: ITool): Promise<ModelToolResult> {
    return {
      type: 'tool_result',
      callId: tool.name,
      output: await tool.execute('{}', this.toolContext),
    };
  }

  private async getCurrentUsage(): Promise<number> {
    const usage = await this.storage.get<UsageRecord>('usage');
    if (!usage) {
      return 0;
    }

    const todayUsage = usage[getTodayUsageKey()];
    if (!todayUsage) {
      return 0;
    }

    return todayUsage.total_tokens;
  }
}
