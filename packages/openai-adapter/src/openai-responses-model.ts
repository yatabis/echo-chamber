import OpenAI from 'openai';

import { emitEchoEvent } from '@echo-chamber/core/ports/echo-event';
import type { EchoEventPort } from '@echo-chamber/core/ports/echo-event';
import type {
  ModelPort,
  ModelRequest,
  ModelResponse,
} from '@echo-chamber/core/ports/model';

import {
  toFunctionToolDefinition,
  toModelOutputItem,
  toModelUsage,
  toResponseInputItem,
} from './openai-response-mappers';
import { projectResponsesWebToolExchange } from './web-tool-audit';

import type {
  EasyInputMessage,
  Response,
  ResponseCreateParamsNonStreaming,
  ResponseFunctionCallOutputItemList,
  ResponseFunctionToolCall,
  ResponseInputItem,
  ResponseOutputItem,
  ResponseOutputMessage,
} from 'openai/resources/responses/responses';
import type { ReasoningEffort } from 'openai/resources/shared';

export interface OpenAIResponsesModelOptions {
  apiKey: string;
  model?: string;
  events?: EchoEventPort;
  reasoningEffort?: ReasoningEffort;
}

/**
 * OpenAI Responses API を `ModelPort` として扱う adapter。
 * provider 固有の request / response 型変換と model event 送信だけを担当する。
 */
export class OpenAIResponsesModel implements ModelPort {
  private readonly client: OpenAI;
  private readonly model: string;
  private readonly events: EchoEventPort | undefined;
  private readonly reasoningEffort: ReasoningEffort;
  private webToolCallIds = new Set<string>();

  /**
   * OpenAI Responses API を使う `ModelPort` adapter を構築する。
   *
   * @param options API キー、モデル名、任意の event 送信先
   */
  constructor(options: OpenAIResponsesModelOptions) {
    this.client = new OpenAI({
      apiKey: options.apiKey,
    });
    this.model = options.model ?? 'gpt-5.6';
    this.events = options.events;
    this.reasoningEffort = options.reasoningEffort ?? 'none';
  }

  /**
   * provider-neutral request を OpenAI Responses API の入力へ変換して実行する。
   *
   * @param request `core` が定義する provider 非依存の 1 ターン分リクエスト
   * @returns OpenAI Responses API が返した生の `Response`
   */
  async createResponse(request: ModelRequest): Promise<Response> {
    const response = await this.client.responses.create(
      this.createResponseParams(request)
    );

    if (!response.usage) {
      await this.emitProviderWarning(request, {
        code: 'missing_usage',
        message: 'Response usage information is undefined',
      });
    }

    return response;
  }

  /**
   * provider-neutral request から Responses API の request body を組み立てる。
   *
   * @param request `core` が定義する provider 非依存の 1 ターン分リクエスト
   * @returns Responses API に渡す non-streaming request body
   */
  private createResponseParams(
    request: ModelRequest
  ): ResponseCreateParamsNonStreaming {
    return {
      input: request.input.map(toResponseInputItem),
      model: this.model,
      parallel_tool_calls: true,
      previous_response_id: request.previousResponseToken,
      reasoning: {
        effort: this.reasoningEffort,
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
    };
  }

  /**
   * provider-neutral request を OpenAI に投げ、`ModelResponse` へ正規化して返す。
   *
   * @param request `core` の session loop から渡される 1 ターン分リクエスト
   * @returns provider 非依存の output / usage / response token
   */
  async generate(request: ModelRequest): Promise<ModelResponse> {
    const responseParams = this.createResponseParams(request);

    const response = await this.client.responses.create(responseParams);

    if (!response.usage) {
      await this.emitProviderWarning(request, {
        code: 'missing_usage',
        message: 'Response usage information is undefined',
      });
    }

    await this.emitModelExchangeRecorded(request, responseParams, response);
    await this.emitModelOutputEmitted(request, response.output);

    return {
      output: response.output.flatMap(toModelOutputItem),
      usage: toModelUsage(response.usage),
      responseToken: response.id,
    };
  }

  /**
   * API payloadの監査用コピーをdebug eventとして記録する。
   * read_web_pageだけはURLと取得本文をmetadataへ置換する。
   *
   * @param request provider-neutral request
   * @param responseParams Responses API request body
   * @param response Responses API response body
   */
  private async emitModelExchangeRecorded(
    request: ModelRequest,
    responseParams: ResponseCreateParamsNonStreaming,
    response: Response
  ): Promise<void> {
    const audit = projectResponsesWebToolExchange(
      responseParams,
      response,
      this.webToolCallIds
    );
    this.webToolCallIds = audit.webToolCallIds;

    await emitEchoEvent(this.events, {
      type: 'model.exchange.recorded',
      severity: 'debug',
      summary: `model exchange recorded: ${this.model}`,
      payload: {
        provider: 'openai.responses',
        model: this.model,
        turnIndex: request.turnIndex,
        request: audit.request,
        response: audit.response,
      },
    });
  }

  /**
   * model output のうち thought stream に載せる本文だけをイベント化する。
   * function_call は `tool.*` event に寄せるため、ここでは emit しない。
   *
   * @param request provider-neutral request
   * @param output Responses API output item
   */
  private async emitModelOutputEmitted(
    request: ModelRequest,
    output: ResponseOutputItem[]
  ): Promise<void> {
    const content = formatModelOutputContent(output);
    if (content === '') {
      return;
    }

    await emitEchoEvent(this.events, {
      type: 'model.output.emitted',
      severity: 'info',
      summary: 'model output emitted',
      payload: {
        provider: 'openai.responses',
        model: this.model,
        turnIndex: request.turnIndex,
        content,
      },
    });
  }

  /**
   * provider 応答の欠落など、adapter 内で検知した警告をイベント化する。
   *
   * @param request provider-neutral request
   * @param warning 警告コードと説明
   */
  private async emitProviderWarning(
    request: ModelRequest,
    warning: { code: string; message: string }
  ): Promise<void> {
    await emitEchoEvent(this.events, {
      type: 'model.provider.warning',
      severity: 'warn',
      summary: warning.message,
      payload: {
        provider: 'openai.responses',
        model: this.model,
        turnIndex: request.turnIndex,
        code: warning.code,
      },
    });
  }
}

/**
 * OpenAI output item 列から、model category の thought 表示に載せる本文だけを整形する。
 *
 * function call は `tool.called` / `tool.completed` と重複するため含めない。
 *
 * @param output OpenAI Responses API が返した output item 列
 * @returns thought stream に送る model output 本文
 */
export function formatModelOutputContent(output: ResponseOutputItem[]): string {
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
 * @returns 人間可読な表示用テキスト
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
 * @returns 人間可読な表示用テキスト
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
