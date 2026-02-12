import OpenAI from 'openai';

import { getErrorMessage } from '../../utils/error';
import { createLogger } from '../../utils/logger';

import type { ITool, ToolContext } from './functions';
import type { Emotion } from '../../echo/types';
import type { Logger } from '../../utils/logger';
import type { ThinkingStream } from '../../utils/thinking-stream';
import type {
  Response,
  ResponseInput,
  ResponseInputItem,
  ResponseOutputItem,
  ResponseFunctionToolCall,
  EasyInputMessage,
  ResponseOutputMessage,
  ResponseUsage,
} from 'openai/resources/responses/responses';

const MAX_TURNS = 10;

export class OpenAIClient {
  private readonly client: OpenAI;
  private readonly tools: ITool[];
  private readonly toolContext: ToolContext;
  private readonly logger: Logger;
  private readonly thinkingStream: ThinkingStream;
  private previousResponseId: string | undefined;

  constructor(
    env: Env,
    tools: ITool[],
    toolContext: ToolContext,
    thinkingStream: ThinkingStream
  ) {
    this.client = new OpenAI({
      apiKey: env.OPENAI_API_KEY,
    });
    this.tools = tools;
    this.toolContext = toolContext;
    this.logger = createLogger(env);
    this.thinkingStream = thinkingStream;
  }

  /**
   * Responses APIを実行
   */
  async createResponse(input: ResponseInput): Promise<Response> {
    const response = await this.client.responses.create({
      input,

      model: 'gpt-5.1',
      parallel_tool_calls: true,
      previous_response_id: this.previousResponseId,
      reasoning: {
        effort: 'low',
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
      tools: this.tools.map((tool) => tool.definition),
      truncation: 'auto',
    });

    // Usage情報の確認（undefinedの場合のみwarnログ）
    if (!response.usage) {
      await this.logger.warn('Response usage information is undefined');
    }

    return response;
  }

  /**
   * Function Callを実行
   */
  async executeFunction(
    functionCall: ResponseFunctionToolCall
  ): Promise<string> {
    const tool = this.tools.find(({ name }) => functionCall.name === name);
    if (!tool) {
      return JSON.stringify({
        error: `Function '${functionCall.name}' is not registered`,
        available_functions: this.tools.map(({ name }) => name),
      });
    }

    try {
      return await tool.execute(functionCall.arguments, this.toolContext);
    } catch (error) {
      return JSON.stringify({
        success: false,
        error: getErrorMessage(error),
      });
    }
  }

  /**
   * 完全な会話フロー（Function Calling含む）を実行
   */
  async call(input: ResponseInput, turn = 1): Promise<ResponseUsage> {
    if (turn > MAX_TURNS) {
      await this.logger.warn('Maximum turns exceeded');
      return {
        input_tokens: 0,
        input_tokens_details: { cached_tokens: 0 },
        output_tokens: 0,
        output_tokens_details: { reasoning_tokens: 0 },
        total_tokens: 0,
      };
    }

    await this.logger.debug(input.map(formatInputItem).join('\n\n'));

    const response = await this.createResponse(input);

    if (response.id) {
      this.previousResponseId = response.id;
    }

    await this.logger.debug(response.output.map(formatOutputItem).join('\n\n'));

    // 思考ログをThinkingStreamで送信（Loggerとは独立）
    await this.thinkingStream.send(formatLogOutput(response.output));

    // 現在のレスポンスのusageを取得
    let totalUsage: ResponseUsage = response.usage ?? {
      input_tokens: 0,
      input_tokens_details: { cached_tokens: 0 },
      output_tokens: 0,
      output_tokens_details: { reasoning_tokens: 0 },
      total_tokens: 0,
    };

    const functionCalls = response.output.filter(
      (item) => item.type === 'function_call'
    );

    // finish_thinking が呼ばれた場合はループを終了
    const hasFinishThinking = functionCalls.some(
      (item) => item.name === 'finish_thinking'
    );

    const nextInput: ResponseInput = await Promise.all(
      functionCalls.map(async (item) => ({
        type: 'function_call_output',
        call_id: item.call_id,
        output: await this.executeFunction(item),
      }))
    );

    if (nextInput.length > 0 && !hasFinishThinking) {
      const recursiveUsage = await this.call(nextInput, turn + 1);
      totalUsage = accumulateUsage(totalUsage, recursiveUsage);
    }

    return totalUsage;
  }
}

/**
 * 複数のUsageオブジェクトを累積する
 */
export function accumulateUsage(
  total: ResponseUsage,
  additional: ResponseUsage
): ResponseUsage {
  return {
    input_tokens: total.input_tokens + additional.input_tokens,
    input_tokens_details: {
      cached_tokens:
        total.input_tokens_details.cached_tokens +
        additional.input_tokens_details.cached_tokens,
    },
    output_tokens: total.output_tokens + additional.output_tokens,
    output_tokens_details: {
      reasoning_tokens:
        total.output_tokens_details.reasoning_tokens +
        additional.output_tokens_details.reasoning_tokens,
    },
    total_tokens: total.total_tokens + additional.total_tokens,
  };
}

export function formatLogOutput(output: ResponseOutputItem[]): string {
  return output
    .map((item) => {
      if (item.type === 'message') {
        return item.content
          .map((c) => {
            const contentType = c.type;
            switch (contentType) {
              case 'output_text':
                return `*thinking: ${c.text}*`;
              case 'refusal':
                return `*refusal: ${c.refusal}*`;
              default:
                throw new Error(
                  `Unexpected contentType: ${contentType satisfies never}`
                );
            }
          })
          .join('\n\n');
      } else if (item.type === 'reasoning') {
        const content = (item.content ?? item.summary)
          .map(({ text }) => text)
          .join('\n');
        if (!content) {
          return '*reasoning*';
        }
        return `*reasoning: ${content}*`;
      } else if (item.type === 'function_call') {
        const args = JSON.parse(item.arguments) as Record<string, unknown>;
        const formatter = functionCallFormatters[item.name];

        if (formatter) {
          return formatter(args);
        }

        return `*${item.name}*`;
      }
    })
    .filter((msg) => msg !== undefined)
    .join('\n\n')
    .trim();
}

const functionCallFormatters: Record<
  string,
  (args: Record<string, unknown>) => string
> = {
  read_chat_messages: (args) => `*read_chat_messages: ${args.limit as number}*`,
  think_deeply: (args) => `*think_deeply: ${args.thought as string}*`,
  store_memory: (args) => {
    const content = args.content as string;
    const { valence, arousal, labels } = args.emotion as Emotion;
    return `*store_memory: ${content}\n(${valence}, ${arousal}) [${labels.join(', ')}]*`;
  },
  search_memory: (args) => `*search_memory: ${args.query as string}*`,
  finish_thinking: (args) => {
    const nextWakeAt = args.next_wake_at as string | undefined;
    return `*finish_thinking: ${args.reason as string}(next_wake_at: ${nextWakeAt})*`;
  },
};

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
    return `[function call output] ${item.call_id} (${item.status})\n${JSON.stringify(JSON.parse(item.output), null, 2)}`;
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
    .map((c) => {
      const contentType = c.type;
      switch (contentType) {
        case 'input_text':
        case 'output_text':
          return formatBlock(role, c.text);
        case 'input_image':
          return formatBlock(role, `<image>${c.image_url}</image>`);
        case 'input_file':
          return formatBlock(role, `<file>${c.file_url ?? c.filename}</file>`);
        case 'input_audio':
          return formatBlock(role, `<audio type="${c.input_audio.format}" />`);
        case 'refusal':
          return formatBlock(role, `<refusal>${c.refusal}</refusal>`);
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
