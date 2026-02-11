import { OpenAIClient } from '../../llm/openai/client';
import { OpenAIEmbeddingService } from '../../llm/openai/embedding';
import {
  addReactionToChatMessageFunction,
  checkNotificationsFunction,
  readChatMessagesFunction,
  sendChatMessageFunction,
} from '../../llm/openai/functions/chat';
import {
  storeMemoryFunction,
  searchMemoryFunction,
} from '../../llm/openai/functions/memory';
import { thinkDeeplyFunction } from '../../llm/openai/functions/think';
import { formatDatetime } from '../../utils/datetime';
import { ThinkingStream } from '../../utils/thinking-stream';
import { MemorySystem } from '../memory-system';
import { getTodayUsageKey } from '../usage';

import type { ITool, ToolContext } from '../../llm/openai/functions';
import type { EchoInstanceConfig } from '../../types/echo-config';
import type { Logger } from '../../utils/logger';
import type { UsageRecord } from '../types';
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
  private readonly toolContext: ToolContext;
  private readonly instanceConfig: EchoInstanceConfig;

  constructor(options: {
    env: Env;
    storage: DurableObjectStorage;
    sql: SqlStorage;
    logger: Logger;
    instanceConfig: EchoInstanceConfig;
  }) {
    this.env = options.env;
    this.instanceConfig = options.instanceConfig;
    const embeddingService = new OpenAIEmbeddingService(options.env);
    const memorySystem = new MemorySystem({
      sql: options.sql,
      embeddingService,
      logger: options.logger,
    });
    this.toolContext = {
      instanceConfig: options.instanceConfig,
      storage: options.storage,
      memorySystem,
      logger: options.logger,
    };
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
        thinkDeeplyFunction,
      ],
      this.toolContext,
      thinkingStream
    );
  }

  private async buildInitialMessages(): Promise<ResponseInput> {
    const currentDatetime = formatDatetime(new Date());
    const latestMemory = this.toolContext.memorySystem.getLatestMemory();
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
    const usage = await this.toolContext.storage.get<UsageRecord>('usage');
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
