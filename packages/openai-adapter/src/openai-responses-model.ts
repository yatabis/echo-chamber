import OpenAI from 'openai';

import type { LoggerPort } from '@echo-chamber/core/ports/logger';
import type {
  ModelPort,
  ModelRequest,
  ModelResponse,
} from '@echo-chamber/core/ports/model';
import type { ThoughtLogPort } from '@echo-chamber/core/ports/thought-log';

import {
  toFunctionToolDefinition,
  toModelOutputItem,
  toModelUsage,
  toResponseInputItem,
} from './openai-response-mappers';

import type {
  EasyInputMessage,
  Response,
  ResponseFunctionCallOutputItemList,
  ResponseFunctionToolCall,
  ResponseInputItem,
  ResponseOutputItem,
  ResponseOutputMessage,
} from 'openai/resources/responses/responses';

export interface OpenAIResponsesModelOptions {
  apiKey: string;
  model?: string;
  logger?: Pick<LoggerPort, 'debug' | 'warn'>;
  thoughtLog?: Pick<ThoughtLogPort, 'send'>;
}

/**
 * OpenAI Responses API を `ModelPort` として扱う adapter。
 * provider 固有の request / response 型変換と thought log 送信だけを担当する。
 */
export class OpenAIResponsesModel implements ModelPort {
  private readonly client: OpenAI;
  private readonly model: string;
  private readonly logger: Pick<LoggerPort, 'debug' | 'warn'> | undefined;
  private readonly thoughtLog: Pick<ThoughtLogPort, 'send'> | undefined;

  /**
   * OpenAI Responses API を使う `ModelPort` adapter を構築する。
   *
   * @param options API キー、モデル名、任意の logger / thought log 送信先
   */
  constructor(options: OpenAIResponsesModelOptions) {
    this.client = new OpenAI({
      apiKey: options.apiKey,
    });
    this.model = options.model ?? 'gpt-5.4';
    this.logger = options.logger;
    this.thoughtLog = options.thoughtLog;
  }

  /**
   * provider-neutral request を OpenAI Responses API の入力へ変換して実行する。
   *
   * @param request `core` が定義する provider 非依存の 1 ターン分リクエスト
   * @returns OpenAI Responses API が返した生の `Response`
   */
  async createResponse(request: ModelRequest): Promise<Response> {
    const response = await this.client.responses.create({
      input: request.input.map(toResponseInputItem),
      model: this.model,
      parallel_tool_calls: true,
      previous_response_id: request.previousResponseToken,
      reasoning: {
        effort: 'none',
      },
      store: true,
      stream: false,
      text: {
        format: {
          type: 'text',
        },
        verbosity: 'medium',
      },
      tool_choice: 'auto',
      tools: request.tools.map(toFunctionToolDefinition),
      truncation: 'auto',
    });

    if (!response.usage) {
      await this.logger?.warn('Response usage information is undefined');
    }

    return response;
  }

  /**
   * provider-neutral request を OpenAI に投げ、`ModelResponse` へ正規化して返す。
   *
   * @param request `core` の session loop から渡される 1 ターン分リクエスト
   * @returns provider 非依存の output / usage / response token
   */
  async generate(request: ModelRequest): Promise<ModelResponse> {
    const responseInput = request.input.map(toResponseInputItem);
    await this.logger?.debug(responseInput.map(formatInputItem).join('\n\n'));

    const response = await this.createResponse(request);

    await this.logger?.debug(
      response.output.map(formatOutputItem).join('\n\n')
    );
    await this.thoughtLog?.send(formatLogOutput(response.output));

    return {
      output: response.output.flatMap(toModelOutputItem),
      usage: toModelUsage(response.usage),
      responseToken: response.id,
    };
  }
}

type FunctionFormatterArgs = Record<string, unknown>;

/**
 * `read_chat_messages` 呼び出しを thought log 向けに整形する。
 *
 * @param args function call の引数
 * @returns thought log に流す短い説明文
 */
function formatReadChatMessagesCall(args: FunctionFormatterArgs): string {
  const channelKey = args.channelKey as string | undefined;
  const target =
    channelKey !== undefined && channelKey !== '' ? ` [${channelKey}]` : '';
  return `*read_chat_messages${target}: ${args.limit as number}*`;
}

/**
 * `send_chat_message` 呼び出しを thought log 向けに整形する。
 *
 * @param args function call の引数
 * @returns thought log に流す短い説明文
 */
function formatSendChatMessageCall(args: FunctionFormatterArgs): string {
  const channelKey = args.channelKey as string | undefined;
  const target =
    channelKey !== undefined && channelKey !== '' ? ` [${channelKey}]` : '';
  return `*send_chat_message${target}: ${args.message as string}*`;
}

/**
 * `add_reaction_to_chat_message` 呼び出しを thought log 向けに整形する。
 *
 * @param args function call の引数
 * @returns thought log に流す短い説明文
 */
function formatAddReactionToChatMessageCall(
  args: FunctionFormatterArgs
): string {
  const channelKey = args.channelKey as string | undefined;
  const target =
    channelKey !== undefined && channelKey !== '' ? ` [${channelKey}]` : '';
  return `*add_reaction_to_chat_message${target}: ${args.messageId as string} ${args.reaction as string}*`;
}

/**
 * `think_deeply` 呼び出しを thought log 向けに整形する。
 *
 * @param args function call の引数
 * @returns thought log に流す短い説明文
 */
function formatThinkDeeplyCall(args: FunctionFormatterArgs): string {
  return `*think_deeply: ${args.thought as string}*`;
}

/**
 * `store_memory` 呼び出しを thought log 向けに整形する。
 *
 * @param args function call の引数
 * @returns thought log に流す短い説明文
 */
function formatStoreMemoryCall(args: FunctionFormatterArgs): string {
  const content = args.content as string;
  const type = args.type as string;
  const emotion = args.emotion as {
    valence: number;
    arousal: number;
    labels: string[];
  };

  return `*store_memory [${type}]: ${content}\n(${emotion.valence}, ${emotion.arousal}) [${emotion.labels.join(', ')}]*`;
}

/**
 * `search_memory` 呼び出しを thought log 向けに整形する。
 *
 * @param args function call の引数
 * @returns thought log に流す短い説明文
 */
function formatSearchMemoryCall(args: FunctionFormatterArgs): string {
  const query = args.query as string;
  const type = args.type as string | undefined;
  return type !== undefined && type !== ''
    ? `*search_memory [${type}]: ${query}*`
    : `*search_memory: ${query}*`;
}

/**
 * `create_note` 呼び出しを thought log 向けに整形する。
 *
 * @param args function call の引数
 * @returns thought log に流す短い説明文
 */
function formatCreateNoteCall(args: FunctionFormatterArgs): string {
  return `*create_note: ${args.title as string}*`;
}

/**
 * `list_notes` 呼び出しを thought log 向けに整形する。
 *
 * @returns thought log に流す短い説明文
 */
function formatListNotesCall(): string {
  return '*list_notes*';
}

/**
 * `get_note` 呼び出しを thought log 向けに整形する。
 *
 * @param args function call の引数
 * @returns thought log に流す短い説明文
 */
function formatGetNoteCall(args: FunctionFormatterArgs): string {
  return `*get_note: ${args.id as string}*`;
}

/**
 * `search_notes` 呼び出しを thought log 向けに整形する。
 *
 * @param args function call の引数
 * @returns thought log に流す短い説明文
 */
function formatSearchNotesCall(args: FunctionFormatterArgs): string {
  return `*search_notes: ${args.query as string}*`;
}

/**
 * `update_note` 呼び出しを thought log 向けに整形する。
 *
 * @param args function call の引数
 * @returns thought log に流す短い説明文
 */
function formatUpdateNoteCall(args: FunctionFormatterArgs): string {
  return `*update_note: ${args.id as string}*`;
}

/**
 * `delete_note` 呼び出しを thought log 向けに整形する。
 *
 * @param args function call の引数
 * @returns thought log に流す短い説明文
 */
function formatDeleteNoteCall(args: FunctionFormatterArgs): string {
  return `*delete_note: ${args.id as string}*`;
}

/**
 * `list_trending_zenn_articles` 呼び出しを thought log 向けに整形する。
 *
 * @param args function call の引数
 * @returns thought log に流す短い説明文
 */
function formatListTrendingZennArticlesCall(
  args: FunctionFormatterArgs
): string {
  const articleType = args.articleType as string | undefined;
  const target =
    articleType !== undefined && articleType !== '' ? ` [${articleType}]` : '';
  return `*list_trending_zenn_articles${target}*`;
}

/**
 * `get_zenn_article` 呼び出しを thought log 向けに整形する。
 *
 * @param args function call の引数
 * @returns thought log に流す短い説明文
 */
function formatGetZennArticleCall(args: FunctionFormatterArgs): string {
  const maxCharacters = args.maxCharacters as number | undefined;
  const truncation =
    maxCharacters !== undefined ? ` (maxCharacters: ${maxCharacters})` : '';
  const slug =
    typeof args.slug === 'string' ? args.slug.trim() : (args.slug as string);
  return `*get_zenn_article: ${slug}${truncation}*`;
}

/**
 * `finish_thinking` 呼び出しを thought log 向けに整形する。
 *
 * @param args function call の引数
 * @returns thought log に流す短い説明文
 */
function formatFinishThinkingCall(args: FunctionFormatterArgs): string {
  const nextWakeAt = args.next_wake_at as string | undefined;
  const sessionRecord = args.session_record as
    | {
        content: string;
        emotion: {
          valence: number;
          arousal: number;
          labels: string[];
        };
      }
    | undefined;

  return `*finish_thinking: ${args.reason as string}(next_wake_at: ${nextWakeAt})\nsession_record: ${sessionRecord?.content}\n(${sessionRecord?.emotion.valence}, ${sessionRecord?.emotion.arousal}) [${sessionRecord?.emotion.labels.join(', ')}]*`;
}

const functionCallFormatters: Record<
  string,
  (args: FunctionFormatterArgs) => string
> = {
  read_chat_messages: formatReadChatMessagesCall,
  send_chat_message: formatSendChatMessageCall,
  add_reaction_to_chat_message: formatAddReactionToChatMessageCall,
  think_deeply: formatThinkDeeplyCall,
  store_memory: formatStoreMemoryCall,
  search_memory: formatSearchMemoryCall,
  create_note: formatCreateNoteCall,
  list_notes: formatListNotesCall,
  get_note: formatGetNoteCall,
  search_notes: formatSearchNotesCall,
  update_note: formatUpdateNoteCall,
  delete_note: formatDeleteNoteCall,
  list_trending_zenn_articles: formatListTrendingZennArticlesCall,
  get_zenn_article: formatGetZennArticleCall,
  finish_thinking: formatFinishThinkingCall,
};

/**
 * OpenAI output item 列を thought log 向けの簡潔なテキストへ整形する。
 *
 * @param output OpenAI Responses API が返した output item 列
 * @returns thought log に送る 1 つの文字列
 */
export function formatLogOutput(output: ResponseOutputItem[]): string {
  return output
    .map((item) => {
      if (item.type === 'message') {
        return item.content
          .map((content) => {
            const contentType = content.type;
            switch (contentType) {
              case 'output_text':
                return `*thinking: ${content.text}*`;
              case 'refusal':
                return `*refusal: ${content.refusal}*`;
              default:
                throw new Error(
                  `Unexpected contentType: ${contentType satisfies never}`
                );
            }
          })
          .join('\n\n');
      }

      if (item.type === 'reasoning') {
        const content = (item.content ?? item.summary)
          .map(({ text }) => text)
          .join('\n');
        if (!content) {
          return '*reasoning*';
        }
        return `*reasoning: ${content}*`;
      }

      if (item.type === 'function_call') {
        const args = JSON.parse(item.arguments) as Record<string, unknown>;
        const formatter = functionCallFormatters[item.name];

        if (formatter) {
          return formatter(args);
        }

        return `*${item.name}*`;
      }

      return undefined;
    })
    .filter((message) => message !== undefined)
    .join('\n\n')
    .trim();
}

/**
 * OpenAI Responses API の input item をログ表示用テキストへ整形する。
 *
 * @param item OpenAI Responses API の input item
 * @returns logger に出すための人間可読テキスト
 */
export function formatInputItem(item: ResponseInputItem): string {
  const itemType = item.type;
  if (!itemType) {
    if ('content' in item) {
      return formatMessage(item);
    }

    return `<item_reference>${item.id}</item_reference>`;
  }

  if (itemType === 'message') {
    return formatMessage(item);
  }

  if (itemType === 'function_call') {
    return formatFunctionCall(item);
  }

  if (itemType === 'function_call_output') {
    return `[function call output] ${item.call_id} (${item.status})\n${formatFunctionCallOutput(item.output)}`;
  }

  return `<${itemType} />`;
}

/**
 * OpenAI Responses API の output item をログ表示用テキストへ整形する。
 *
 * @param item OpenAI Responses API の output item
 * @returns logger に出すための人間可読テキスト
 */
export function formatOutputItem(item: ResponseOutputItem): string {
  const itemType = item.type;
  if (itemType === 'message') {
    return formatMessage(item);
  }

  if (itemType === 'function_call') {
    return formatFunctionCall(item);
  }

  return `<${item.type} />`;
}

/**
 * OpenAI の message item を role 付きテキストブロックへ整形する。
 *
 * @param item 入力または出力の message item
 * @returns role を含むログ表示用テキスト
 */
export function formatMessage(
  item: EasyInputMessage | ResponseInputItem.Message | ResponseOutputMessage
): string {
  const { role, content } = item;
  if (typeof content === 'string') {
    return formatBlock(role, content);
  }

  return content
    .map((contentItem) => {
      const contentType = contentItem.type;
      switch (contentType) {
        case 'input_text':
        case 'output_text':
          return formatBlock(role, contentItem.text);
        case 'input_image':
          return formatBlock(role, `<image>${contentItem.image_url}</image>`);
        case 'input_file':
          return formatBlock(
            role,
            `<file>${contentItem.file_url ?? contentItem.filename}</file>`
          );
        case 'refusal':
          return formatBlock(role, `<refusal>${contentItem.refusal}</refusal>`);
        default:
          throw new Error(
            `Unexpected contentType: ${contentType satisfies never}`
          );
      }
    })
    .join('\n\n');
}

/**
 * role と content を共通のブロック表現へ整形する。
 *
 * @param role メッセージの role
 * @param content 表示対象の本文
 * @returns `[role]:` 形式のブロック文字列
 */
export function formatBlock(role: string, content: string): string {
  return `[${role}]:\n${content}`;
}

/**
 * function call item をログ表示用テキストへ整形する。
 *
 * @param item OpenAI Responses API の function call item
 * @returns call id / status / 関数名 / 引数を含む文字列
 */
export function formatFunctionCall(item: ResponseFunctionToolCall): string {
  return `[function call] ${item.call_id} (${item.status})\n${item.name}(${item.arguments})`;
}

/**
 * function call output を見やすい JSON 文字列へ整形する。
 *
 * @param output tool 実行結果の文字列または structured output
 * @returns pretty print された文字列。JSON でなければ元の文字列
 */
export function formatFunctionCallOutput(
  output: string | ResponseFunctionCallOutputItemList
): string {
  if (typeof output !== 'string') {
    return JSON.stringify(output, null, 2);
  }

  try {
    return JSON.stringify(JSON.parse(output), null, 2);
  } catch {
    return output;
  }
}
