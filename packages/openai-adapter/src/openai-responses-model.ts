import OpenAI from 'openai';

import type { LoggerPort } from '@echo-chamber/core/ports/logger';
import type {
  ModelInputItem,
  ModelMessage,
  ModelOutputItem,
  ModelPort,
  ModelRequest,
  ModelResponse,
  ModelToolContract,
  ModelUsage,
} from '@echo-chamber/core/ports/model';
import type { ThoughtLogPort } from '@echo-chamber/core/ports/thought-log';

import type {
  EasyInputMessage,
  Response,
  ResponseFunctionCallOutputItemList,
  ResponseFunctionToolCall,
  ResponseInputItem,
  ResponseOutputItem,
  ResponseOutputMessage,
  ResponseUsage,
} from 'openai/resources/responses/responses';

interface FunctionToolDefinition {
  type: 'function';
  name: string;
  description: string;
  parameters: Record<string, unknown> | null;
  strict: boolean;
}

const EMPTY_RESPONSE_USAGE: ResponseUsage = {
  input_tokens: 0,
  input_tokens_details: { cached_tokens: 0 },
  output_tokens: 0,
  output_tokens_details: { reasoning_tokens: 0 },
  total_tokens: 0,
};

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

function toFunctionToolDefinition(
  tool: ModelToolContract
): FunctionToolDefinition {
  return {
    type: 'function',
    name: tool.name,
    description: tool.description,
    parameters: toFunctionParameters(tool.inputSchema),
    strict: tool.strict ?? false,
  };
}

function toFunctionParameters(
  inputSchema: unknown
): Record<string, unknown> | null {
  if (inputSchema === null) {
    return null;
  }

  if (typeof inputSchema === 'object' && !Array.isArray(inputSchema)) {
    return Object.fromEntries(Object.entries(inputSchema));
  }

  return null;
}

function toResponseInputItem(item: ModelInputItem): ResponseInputItem {
  if (isModelMessage(item)) {
    return {
      role: item.role,
      content: item.content,
    };
  }

  if (item.type === 'tool_call') {
    return {
      type: 'function_call',
      call_id: item.callId,
      name: item.toolName,
      arguments: item.input,
    };
  }

  return {
    type: 'function_call_output',
    call_id: item.callId,
    output: item.output,
  };
}

function isModelMessage(item: ModelInputItem): item is ModelMessage {
  return 'role' in item;
}

function toModelUsage(usage: ResponseUsage | undefined): ModelUsage {
  const safeUsage = usage ?? EMPTY_RESPONSE_USAGE;
  const cachedInputTokens = safeUsage.input_tokens_details.cached_tokens;
  const totalInputTokens = safeUsage.input_tokens;

  return {
    cachedInputTokens,
    uncachedInputTokens: totalInputTokens - cachedInputTokens,
    totalInputTokens,
    outputTokens: safeUsage.output_tokens,
    reasoningTokens: safeUsage.output_tokens_details.reasoning_tokens,
    totalTokens: safeUsage.total_tokens,
  };
}

function toModelOutputItem(item: ResponseOutputItem): ModelOutputItem[] {
  if (item.type === 'function_call') {
    return [
      {
        type: 'tool_call',
        callId: item.call_id,
        toolName: item.name,
        input: item.arguments,
      },
    ];
  }

  if (item.type === 'message') {
    return [
      {
        type: 'message',
        role: 'assistant',
        content: extractOutputMessageText(item),
      },
    ];
  }

  return [];
}

function extractOutputMessageText(message: ResponseOutputMessage): string {
  return message.content
    .map((content) => {
      const contentType = content.type;
      switch (contentType) {
        case 'output_text':
          return content.text;
        case 'refusal':
          return `<refusal>${content.refusal}</refusal>`;
        default:
          throw new Error(
            `Unexpected contentType: ${contentType satisfies never}`
          );
      }
    })
    .join('\n\n');
}

const functionCallFormatters: Record<
  string,
  (args: Record<string, unknown>) => string
> = {
  read_chat_messages: (args) => `*read_chat_messages: ${args.limit as number}*`,
  think_deeply: (args) => `*think_deeply: ${args.thought as string}*`,
  store_memory: (args) => {
    const content = args.content as string;
    const type = args.type as string;
    const emotion = args.emotion as {
      valence: number;
      arousal: number;
      labels: string[];
    };

    return `*store_memory [${type}]: ${content}\n(${emotion.valence}, ${emotion.arousal}) [${emotion.labels.join(', ')}]*`;
  },
  search_memory: (args) => {
    const query = args.query as string;
    const type = args.type as string | undefined;
    return type !== undefined && type !== ''
      ? `*search_memory [${type}]: ${query}*`
      : `*search_memory: ${query}*`;
  },
  create_note: (args) => `*create_note: ${args.title as string}*`,
  list_notes: () => `*list_notes*`,
  get_note: (args) => `*get_note: ${args.id as string}*`,
  search_notes: (args) => `*search_notes: ${args.query as string}*`,
  update_note: (args) => `*update_note: ${args.id as string}*`,
  delete_note: (args) => `*delete_note: ${args.id as string}*`,
  finish_thinking: (args) => {
    const nextWakeAt = args.next_wake_at as string | undefined;
    return `*finish_thinking: ${args.reason as string}(next_wake_at: ${nextWakeAt})*`;
  },
};

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

export function formatBlock(role: string, content: string): string {
  return `[${role}]:\n${content}`;
}

export function formatFunctionCall(item: ResponseFunctionToolCall): string {
  return `[function call] ${item.call_id} (${item.status})\n${item.name}(${item.arguments})`;
}

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
