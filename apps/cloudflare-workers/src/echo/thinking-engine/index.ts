import { formatJapaneseDatetime, getTodayUsageKey } from '@echo-chamber/core';
import type { EchoInstanceConfig, UsageRecord } from '@echo-chamber/core';
import { ThinkingStream } from '@echo-chamber/core/utils/thinking-stream';

import { createEmbeddingService } from '../../llm/embedding-factory';
import { OpenAIClient } from '../../llm/openai/client';
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
import { MemorySystem } from '../memory-system';
import { NoteSystem } from '../note-system';

import type { ITool, ToolContext } from '../../llm/openai/functions';
import type { Logger } from '../../utils/logger';
import type {
  ResponseFunctionToolCall,
  ResponseInput,
  ResponseInputItem,
  ResponseUsage,
} from 'openai/resources/responses/responses';

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

  async think(): Promise<ResponseUsage> {
    const thinkingStream = new ThinkingStream(this.instanceConfig);
    await thinkingStream.send('*Thinking started...*');
    const openai = this.createOpenAIClient(thinkingStream);
    const messages = await this.buildInitialMessages();
    const usage = await openai.call(messages);
    await thinkingStream.send(
      `*Thinking completed.*\nUsage: ${usage.total_tokens} tokens (Total: ${
        (await this.getCurrentUsage()) + usage.total_tokens
      } tokens)`
    );
    return usage;
  }

  private createOpenAIClient(thinkingStream: ThinkingStream): OpenAIClient {
    return new OpenAIClient(
      this.env,
      [
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
      ],
      this.toolContext,
      thinkingStream
    );
  }

  private async buildInitialMessages(): Promise<ResponseInput> {
    const now = new Date();
    const currentDatetime = formatJapaneseDatetime(now);
    const latestMemory = this.memorySystem.getLatestMemory();
    const context = latestMemory
      ? `context loaded: ${JSON.stringify(
          {
            content: latestMemory.content,
            created_at: latestMemory.createdAt,
            emotion: {
              valence: latestMemory.emotion.valence,
              arousal: latestMemory.emotion.arousal,
              labels: latestMemory.emotion.labels,
            },
          },
          null,
          2
        )}`
      : 'No context loaded.';

    return [
      {
        role: 'developer',
        content: this.instanceConfig.systemPrompt,
      },
      {
        role: 'developer',
        content: `${context}\nCurrent datetime: ${currentDatetime}`,
      },
      this.createFunctionCallMessage(checkNotificationsFunction),
      await this.createFunctionCallOutputMessage(checkNotificationsFunction),
    ];
  }

  private createFunctionCallMessage(tool: ITool): ResponseFunctionToolCall {
    return {
      type: 'function_call',
      call_id: tool.name,
      name: tool.name,
      arguments: '{}',
    };
  }

  private async createFunctionCallOutputMessage(
    tool: ITool
  ): Promise<ResponseInputItem.FunctionCallOutput> {
    return {
      type: 'function_call_output',
      call_id: tool.name,
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
