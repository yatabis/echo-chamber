import OpenAI from 'openai';

import type { LoggerPort } from '@echo-chamber/core/ports/logger';
import type {
  ModelInputItem,
  ModelMessage,
  ModelOutputItem,
  ModelPort,
  ModelRequest,
  ModelResponse,
  ModelToolCall,
  ModelToolContract,
  ModelUsage,
} from '@echo-chamber/core/ports/model';
import type { ThoughtLogPort } from '@echo-chamber/core/ports/thought-log';

import { toFunctionParameters } from './openai-response-mappers';

import type {
  ChatCompletion,
  ChatCompletionAssistantMessageParam,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionMessageFunctionToolCall,
  ChatCompletionMessageParam,
  ChatCompletionTool,
} from 'openai/resources/chat/completions';
import type { CompletionUsage } from 'openai/resources/completions';
import type { ReasoningEffort } from 'openai/resources/shared';

export type OpenAIChatCompletionsExtraBody = Record<string, unknown>;

export interface OpenAIChatCompletionsModelOptions {
  apiKey: string;
  model: string;
  baseURL?: string;
  logger?: Pick<LoggerPort, 'debug' | 'warn'>;
  thoughtLog?: Pick<ThoughtLogPort, 'send'>;
  reasoningEffort?: ReasoningEffort;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  extraBody?: OpenAIChatCompletionsExtraBody;
}

const EMPTY_CHAT_USAGE: CompletionUsage = {
  completion_tokens: 0,
  prompt_tokens: 0,
  total_tokens: 0,
};

/**
 * Chat Completions API を `ModelPort` として扱う adapter。
 *
 * Responses API の `previous_response_id` は使えないため、この adapter は
 * instance 内に会話履歴を保持し、agent session の turn 間で messages を引き継ぐ。
 */
export class OpenAIChatCompletionsModel implements ModelPort {
  private readonly client: OpenAI;
  private readonly messages: ChatCompletionMessageParam[] = [];
  private readonly logger: Pick<LoggerPort, 'debug' | 'warn'> | undefined;
  private readonly thoughtLog: Pick<ThoughtLogPort, 'send'> | undefined;

  /**
   * Chat Completions API を使う `ModelPort` adapter を構築する。
   *
   * @param options API キー、モデル名、base URL、任意の推論・sampling 設定
   */
  constructor(private readonly options: OpenAIChatCompletionsModelOptions) {
    this.client = new OpenAI({
      apiKey: options.apiKey,
      baseURL: options.baseURL,
    });
    this.logger = options.logger;
    this.thoughtLog = options.thoughtLog;
  }

  /**
   * provider-neutral request を Chat Completions API の入力へ変換して実行する。
   *
   * @param request `core` が定義する provider 非依存の 1 ターン分リクエスト
   * @returns Chat Completions API が返した生の `ChatCompletion`
   */
  async createChatCompletion(request: ModelRequest): Promise<ChatCompletion> {
    this.messages.push(...this.toChatMessages(request.input));

    const params = this.createParams(request);
    await this.logger?.debug(JSON.stringify(params.messages, null, 2));

    const response = await this.client.chat.completions.create(params);
    const assistantMessage = response.choices[0]?.message;

    if (assistantMessage === undefined) {
      await this.logger?.warn('Chat completion returned no choices');
      return response;
    }

    this.messages.push(toAssistantMessageParam(assistantMessage));

    if (!response.usage) {
      await this.logger?.warn('Chat completion usage information is undefined');
    }

    return response;
  }

  /**
   * provider-neutral request を Chat Completions API に投げ、`ModelResponse` へ正規化して返す。
   *
   * @param request `core` の session loop から渡される 1 ターン分リクエスト
   * @returns provider 非依存の output / usage
   */
  async generate(request: ModelRequest): Promise<ModelResponse> {
    const response = await this.createChatCompletion(request);
    const message = response.choices[0]?.message;
    const output = message === undefined ? [] : toModelOutput(message);

    await this.logger?.debug(JSON.stringify(message ?? null, null, 2));
    await this.thoughtLog?.send(formatChatLogOutput(output));

    return {
      output,
      usage: toChatModelUsage(response.usage),
      responseToken: response.id,
    };
  }

  /**
   * Chat Completions API に渡す request body を組み立てる。
   *
   * @param request provider-neutral request
   * @returns non-streaming Chat Completions request
   */
  private createParams(
    request: ModelRequest
  ): ChatCompletionCreateParamsNonStreaming & OpenAIChatCompletionsExtraBody {
    return {
      ...this.options.extraBody,
      messages: [...this.messages],
      model: this.options.model,
      frequency_penalty: this.options.frequencyPenalty,
      max_tokens: this.options.maxTokens,
      n: 1,
      parallel_tool_calls: true,
      presence_penalty: this.options.presencePenalty,
      reasoning_effort: this.options.reasoningEffort,
      stream: false,
      temperature: this.options.temperature,
      tool_choice: 'auto',
      tools: request.tools.map(toChatCompletionTool),
      top_p: this.options.topP,
    };
  }

  /**
   * provider-neutral input item を Chat Completions messages に変換する。
   *
   * @param input provider-neutral input item
   * @returns Chat Completions message param
   */
  private toChatMessages(
    input: readonly ModelInputItem[]
  ): ChatCompletionMessageParam[] {
    return input.map((item) => {
      if (isModelMessage(item)) {
        return toChatMessage(item);
      }

      if (item.type === 'tool_call') {
        return {
          role: 'assistant',
          content: null,
          tool_calls: [toChatToolCall(item)],
        };
      }

      return {
        role: 'tool',
        tool_call_id: item.callId,
        content: item.output,
      };
    });
  }
}

/**
 * `core` の tool contract を Chat Completions function tool 定義へ変換する。
 *
 * @param tool provider 非依存の tool contract
 * @returns Chat Completions API に渡せる function tool 定義
 */
export function toChatCompletionTool(
  tool: ModelToolContract
): ChatCompletionTool {
  const parameters = toFunctionParameters(tool.inputSchema);

  return {
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      ...(parameters === null ? {} : { parameters }),
      strict: tool.strict ?? false,
    },
  };
}

/**
 * Chat Completions usage を `core` の `ModelUsage` に正規化する。
 *
 * @param usage Chat Completions usage。未定義なら 0 usage 扱い
 * @returns provider 非依存の usage 集計値
 */
export function toChatModelUsage(
  usage: CompletionUsage | undefined
): ModelUsage {
  const safeUsage = usage ?? EMPTY_CHAT_USAGE;
  const cachedInputTokens = safeUsage.prompt_tokens_details?.cached_tokens ?? 0;
  const totalInputTokens = safeUsage.prompt_tokens;

  return {
    cachedInputTokens,
    uncachedInputTokens: totalInputTokens - cachedInputTokens,
    totalInputTokens,
    outputTokens: safeUsage.completion_tokens,
    reasoningTokens: safeUsage.completion_tokens_details?.reasoning_tokens ?? 0,
    totalTokens: safeUsage.total_tokens,
  };
}

/**
 * Chat completion assistant message を provider-neutral output に変換する。
 *
 * @param message Chat Completions API の assistant message
 * @returns `core` の session loop が扱える output item 配列
 */
export function toModelOutput(
  message: ChatCompletion.Choice['message']
): ModelOutputItem[] {
  const output: ModelOutputItem[] = [];

  if (message.content !== null && message.content !== '') {
    output.push({
      type: 'message',
      role: 'assistant',
      content: message.content,
    });
  }

  for (const toolCall of message.tool_calls ?? []) {
    if (toolCall.type !== 'function') {
      continue;
    }

    output.push({
      type: 'tool_call',
      callId: toolCall.id,
      toolName: toolCall.function.name,
      input: toolCall.function.arguments,
    });
  }

  return output;
}

/**
 * provider-neutral message を Chat Completions message に変換する。
 *
 * Chat Completions 互換 endpoint の chat template 互換性を優先し、`developer`
 * は常に `user` として渡す。
 */
function toChatMessage(message: ModelMessage): ChatCompletionMessageParam {
  if (message.role === 'developer') {
    return {
      role: 'user',
      content: message.content,
    };
  }

  return {
    role: message.role,
    content: message.content,
  };
}

/**
 * provider-neutral tool call input を assistant tool call message に変換する。
 */
function toChatToolCall(
  toolCall: ModelToolCall
): ChatCompletionMessageFunctionToolCall {
  return {
    id: toolCall.callId,
    type: 'function',
    function: {
      name: toolCall.toolName,
      arguments: toolCall.input,
    },
  };
}

/**
 * Chat Completions API の assistant message を履歴へ保存できる param に変換する。
 */
function toAssistantMessageParam(
  message: ChatCompletion.Choice['message']
): ChatCompletionAssistantMessageParam {
  return {
    role: 'assistant',
    content: message.content,
    refusal: message.refusal,
    tool_calls: message.tool_calls,
  };
}

/**
 * thought log 向けに chat output を短く整形する。
 */
function formatChatLogOutput(output: readonly ModelOutputItem[]): string {
  return output
    .map((item) => {
      if (item.type === 'message') {
        return `*thinking: ${item.content}*`;
      }

      return `*${item.toolName}: ${item.input}*`;
    })
    .join('\n\n');
}

/**
 * input item が通常メッセージかどうかを判定する。
 */
function isModelMessage(item: ModelInputItem): item is ModelMessage {
  return 'role' in item;
}
