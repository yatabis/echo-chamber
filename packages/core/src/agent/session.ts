import { getErrorMessage } from '../utils/error';

import type { LoggerPort } from '../ports/logger';
import type {
  ModelInputItem,
  ModelPort,
  ModelResponse,
  ModelToolCall,
  ModelToolContract,
  ModelUsage,
} from '../ports/model';

/**
 * agent session が実行できる tool の最小契約。
 * session loop は tool handler の中身を知らず、
 * `contract` と `execute` だけを使って turn を進める。
 */
export interface AgentSessionTool {
  name: string;
  contract: ModelToolContract;
  execute(input: string): Promise<string>;
}

/**
 * provider 非依存 session を実行するための入力。
 */
export interface RunAgentSessionInput {
  model: ModelPort;
  tools: readonly AgentSessionTool[];
  initialInput: ModelInputItem[];
  logger?: Pick<LoggerPort, 'warn'>;
  maxTurns?: number;
}

/**
 * session 全体の実行結果。
 * 現時点では usage 集計と provider 側の継続 token を返す。
 */
export interface AgentSessionResult {
  usage: ModelUsage;
  responseToken?: string;
}

/**
 * usage 累積の初期値。
 */
export const ZERO_MODEL_USAGE: ModelUsage = {
  cachedInputTokens: 0,
  uncachedInputTokens: 0,
  totalInputTokens: 0,
  outputTokens: 0,
  reasoningTokens: 0,
  totalTokens: 0,
};

/**
 * 各ターンの usage を session 全体の usage に加算する。
 */
export function accumulateModelUsage(
  total: ModelUsage,
  additional: ModelUsage
): ModelUsage {
  return {
    cachedInputTokens: total.cachedInputTokens + additional.cachedInputTokens,
    uncachedInputTokens:
      total.uncachedInputTokens + additional.uncachedInputTokens,
    totalInputTokens: total.totalInputTokens + additional.totalInputTokens,
    outputTokens: total.outputTokens + additional.outputTokens,
    reasoningTokens: total.reasoningTokens + additional.reasoningTokens,
    totalTokens: total.totalTokens + additional.totalTokens,
  };
}

function getToolContracts(
  tools: readonly AgentSessionTool[]
): ModelToolContract[] {
  return tools.map((tool) => tool.contract);
}

function findTool(
  tools: readonly AgentSessionTool[],
  toolName: string
): AgentSessionTool | undefined {
  return tools.find((tool) => tool.name === toolName);
}

/**
 * 1件の tool call を実行し、そのまま次ターンへ返せる文字列結果に変換する。
 * 未登録 tool や handler 例外も JSON 文字列へ正規化して返す。
 */
export async function executeAgentToolCall(
  toolCall: ModelToolCall,
  tools: readonly AgentSessionTool[]
): Promise<string> {
  const tool = findTool(tools, toolCall.toolName);
  if (tool === undefined) {
    return JSON.stringify({
      error: `Function '${toolCall.toolName}' is not registered`,
      available_functions: tools.map((candidate) => candidate.name),
    });
  }

  try {
    return await tool.execute(toolCall.input);
  } catch (error) {
    return JSON.stringify({
      success: false,
      error: getErrorMessage(error),
    });
  }
}

function getToolCalls(response: ModelResponse): ModelToolCall[] {
  return response.output.filter((item) => item.type === 'tool_call');
}

function shouldFinish(toolCalls: readonly ModelToolCall[]): boolean {
  return toolCalls.some((toolCall) => toolCall.toolName === 'finish_thinking');
}

async function createNextInput(
  toolCalls: readonly ModelToolCall[],
  tools: readonly AgentSessionTool[]
): Promise<ModelInputItem[]> {
  return await Promise.all(
    toolCalls.map(async (toolCall) => ({
      type: 'tool_result' as const,
      callId: toolCall.callId,
      output: await executeAgentToolCall(toolCall, tools),
    }))
  );
}

/**
 * provider 非依存の agent session loop。
 * モデル出力に tool call があれば実行結果を次ターン input に変換し、
 * `finish_thinking` が現れるか tool call が尽きるまで turn を繰り返す。
 */
export async function runAgentSession(
  input: RunAgentSessionInput
): Promise<AgentSessionResult> {
  const maxTurns = input.maxTurns ?? 10;
  let currentInput = input.initialInput;
  let previousResponseToken: string | undefined;
  let totalUsage = ZERO_MODEL_USAGE;

  for (let turn = 1; turn <= maxTurns; turn += 1) {
    // Agent turns are inherently sequential because each model response
    // depends on the previous turn's tool outputs.
    // eslint-disable-next-line no-await-in-loop
    const response = await input.model.generate({
      input: currentInput,
      tools: getToolContracts(input.tools),
      previousResponseToken,
    });

    totalUsage = accumulateModelUsage(totalUsage, response.usage);
    previousResponseToken = response.responseToken;

    const toolCalls = getToolCalls(response);
    if (toolCalls.length === 0) {
      return {
        usage: totalUsage,
        responseToken: previousResponseToken,
      };
    }

    // Tool results become the next model input for the following turn.
    // eslint-disable-next-line no-await-in-loop
    currentInput = await createNextInput(toolCalls, input.tools);
    if (shouldFinish(toolCalls)) {
      return {
        usage: totalUsage,
        responseToken: previousResponseToken,
      };
    }
  }

  await input.logger?.warn('Maximum turns exceeded');
  return {
    usage: totalUsage,
    responseToken: previousResponseToken,
  };
}
