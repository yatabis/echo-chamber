import { emitEchoEvent } from '../ports/echo-event';
import { getErrorMessage } from '../utils/error';

import {
  finishThinkingInputSchema,
  type FinishThinkingInput,
  type FinishThinkingSessionRecord,
} from './tools/thinking';

import type { EchoEventPort } from '../ports/echo-event';
import type {
  ModelInputItem,
  ModelMessage,
  ModelMessageContentPart,
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
  events?: EchoEventPort;
  maxTurns?: number;
}

/**
 * session 全体の実行結果。
 * usage 集計に加えて、`finish_thinking` が返した session record と
 * 次回起動時刻、provider 側の継続 token、終了理由を返す。
 */
export interface AgentSessionResult {
  usage: ModelUsage;
  context?: FinishThinkingSessionRecord;
  nextWakeAt: string | null;
  responseToken?: string;
  terminationReason: 'finish_thinking' | 'max_turns';
}

/**
 * usage 累積の初期値。
 */
export const ZERO_MODEL_USAGE: ModelUsage = {
  cachedInputTokens: 0,
  cacheWriteInputTokens: 0,
  uncachedInputTokens: 0,
  totalInputTokens: 0,
  outputTokens: 0,
  reasoningTokens: 0,
  totalTokens: 0,
};

const MAX_TOOL_VISION_IMAGES = 20;

interface ChatToolVisionImage {
  messageId: string;
  user: string;
  createdAt: string;
  url: string;
  filename?: string | null;
  contentType?: string | null;
  width?: number | null;
  height?: number | null;
  description?: string | null;
}

/**
 * 各ターンの usage を session 全体の usage に加算する。
 */
export function accumulateModelUsage(
  total: ModelUsage,
  additional: ModelUsage
): ModelUsage {
  return {
    cachedInputTokens: total.cachedInputTokens + additional.cachedInputTokens,
    cacheWriteInputTokens:
      total.cacheWriteInputTokens + additional.cacheWriteInputTokens,
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
  tools: readonly AgentSessionTool[],
  events?: EchoEventPort,
  turnIndex?: number
): Promise<string> {
  await emitEchoEvent(events, {
    type: 'tool.called',
    severity: 'info',
    summary: `${toolCall.toolName} called`,
    payload: {
      callId: toolCall.callId,
      toolName: toolCall.toolName,
      turnIndex,
      input: toolCall.input,
    },
  });

  const startedAt = Date.now();
  const tool = findTool(tools, toolCall.toolName);
  if (tool === undefined) {
    await emitEchoEvent(events, {
      type: 'tool.failed',
      severity: 'warn',
      summary: `${toolCall.toolName} is not registered`,
      payload: {
        callId: toolCall.callId,
        toolName: toolCall.toolName,
        turnIndex,
        durationMs: Date.now() - startedAt,
        availableFunctions: tools.map((candidate) => candidate.name),
      },
    });
    return JSON.stringify({
      error: `Function '${toolCall.toolName}' is not registered`,
      available_functions: tools.map((candidate) => candidate.name),
    });
  }

  try {
    const output = await tool.execute(toolCall.input);
    const parsedOutput = parseToolOutput(output);
    const sanitizedOutput = sanitizeToolOutputForModel(output);
    const success = parsedOutput.success !== false;
    const metadata = createToolEventMetadata(toolCall, parsedOutput);
    await emitEchoEvent(events, {
      type: success ? 'tool.completed' : 'tool.failed',
      severity: success ? 'info' : 'warn',
      summary: `${toolCall.toolName} ${success ? 'completed' : 'failed'}`,
      payload: {
        callId: toolCall.callId,
        toolName: toolCall.toolName,
        turnIndex,
        durationMs: Date.now() - startedAt,
        success,
        error: parsedOutput.error,
        ...(parsedOutput.diagnostics === undefined
          ? {}
          : { diagnostics: parsedOutput.diagnostics }),
        outputLength: sanitizedOutput.length,
        ...metadata,
      },
    });
    return sanitizedOutput;
  } catch (error) {
    const message = getErrorMessage(error);
    await emitEchoEvent(events, {
      type: 'tool.failed',
      severity: 'warn',
      summary: `${toolCall.toolName} failed`,
      payload: {
        callId: toolCall.callId,
        toolName: toolCall.toolName,
        turnIndex,
        durationMs: Date.now() - startedAt,
        success: false,
        error: message,
      },
    });
    return JSON.stringify({
      success: false,
      error: message,
    });
  }
}

function parseToolOutput(output: string): {
  success?: boolean;
  error?: unknown;
  diagnostics?: unknown;
  record?: Record<string, unknown>;
} {
  try {
    const parsed: unknown = JSON.parse(output);
    if (typeof parsed !== 'object' || parsed === null) {
      return {};
    }

    const record = parsed as Record<string, unknown>;
    return {
      success: typeof record.success === 'boolean' ? record.success : undefined,
      error: record.error,
      diagnostics: record.diagnostics,
      record,
    };
  } catch {
    return {};
  }
}

function sanitizeToolOutput(
  output: string,
  parsedOutput: { diagnostics?: unknown; record?: Record<string, unknown> }
): string {
  if (
    parsedOutput.diagnostics === undefined ||
    parsedOutput.record === undefined
  ) {
    return output;
  }

  const sanitized = { ...parsedOutput.record };
  delete sanitized.diagnostics;
  return JSON.stringify(sanitized);
}

/**
 * EventPort 用の診断情報を、model に返す tool output から取り除く。
 *
 * @param output tool handler が返した JSON 文字列
 * @returns model input として渡せる JSON 文字列
 */
export function sanitizeToolOutputForModel(output: string): string {
  return sanitizeToolOutput(output, parseToolOutput(output));
}

function createToolEventMetadata(
  toolCall: ModelToolCall,
  parsedOutput: { record?: Record<string, unknown> }
): Record<string, unknown> {
  const operation = getToolOperation(toolCall.toolName);
  if (operation === undefined) {
    return {};
  }

  return {
    operation,
    ...getToolEntityMetadata(toolCall, parsedOutput),
  };
}

function getToolOperation(toolName: string): string | undefined {
  switch (toolName) {
    case 'create_note':
      return 'note.create';
    case 'list_notes':
      return 'note.list';
    case 'get_note':
      return 'note.get';
    case 'search_notes':
      return 'note.search';
    case 'update_note':
      return 'note.update';
    case 'delete_note':
      return 'note.delete';
    default:
      return undefined;
  }
}

function getToolEntityMetadata(
  toolCall: ModelToolCall,
  parsedOutput: { record?: Record<string, unknown> }
): Record<string, unknown> {
  const entityId =
    getNoteIdFromOutput(parsedOutput.record) ??
    getStringProperty(parseToolInput(toolCall.input), 'id');

  return {
    entityType: 'note',
    ...(entityId === undefined ? {} : { entityId }),
  };
}

function getNoteIdFromOutput(
  output: Record<string, unknown> | undefined
): string | undefined {
  const note = output?.note;
  if (typeof note !== 'object' || note === null) {
    return undefined;
  }

  return getStringProperty(note as Record<string, unknown>, 'id');
}

function parseToolInput(input: string): Record<string, unknown> | undefined {
  try {
    const parsed: unknown = JSON.parse(input);
    if (typeof parsed !== 'object' || parsed === null) {
      return undefined;
    }

    return parsed as Record<string, unknown>;
  } catch {
    return undefined;
  }
}

function getStringProperty(
  record: Record<string, unknown> | undefined,
  key: string
): string | undefined {
  const value = record?.[key];
  return typeof value === 'string' ? value : undefined;
}

function getToolCalls(response: ModelResponse): ModelToolCall[] {
  return response.output.filter((item) => item.type === 'tool_call');
}

/**
 * `finish_thinking` 呼び出し列から、有効な入力 payload を抜き出す。
 * tool 名だけでは終了扱いにせず、schema に合致した入力を持つ場合だけ完了とみなす。
 *
 * @param toolCalls 現在ターンでモデルが返した tool call 一覧
 * @returns 正常終了に使える finish_thinking 入力。見つからない、または不正なら `null`
 */
function parseFinishThinkingInput(
  toolCalls: readonly ModelToolCall[]
): FinishThinkingInput | null {
  for (const toolCall of toolCalls) {
    if (toolCall.toolName !== 'finish_thinking') {
      continue;
    }

    try {
      const parsed = finishThinkingInputSchema.safeParse(
        JSON.parse(toolCall.input)
      );
      if (parsed.success) {
        return parsed.data;
      }
    } catch {
      continue;
    }
  }

  return null;
}

/**
 * 現在ターンの tool call 群を、次ターンへ渡す tool_result input 列へ変換する。
 * tool が 1 件も無い場合は空配列を返し、そのまま次ターンを継続できるようにする。
 *
 * @param toolCalls 現在ターンで実行対象になった tool call
 * @param tools 実行可能な tool 群
 * @returns 次ターン input にそのまま渡せる tool_result 列
 */
async function createNextInput(
  toolCalls: readonly ModelToolCall[],
  tools: readonly AgentSessionTool[],
  events: EchoEventPort | undefined,
  turnIndex: number
): Promise<ModelInputItem[]> {
  if (toolCalls.length === 0) {
    return [];
  }

  const nextInput: ModelInputItem[] = [];
  const visionInputs: ModelMessage[] = [];
  const toolResults = await Promise.all(
    toolCalls.map(async (toolCall) => ({
      toolCall,
      output: await executeAgentToolCall(toolCall, tools, events, turnIndex),
    }))
  );

  for (const { toolCall, output } of toolResults) {
    nextInput.push({
      type: 'tool_result',
      callId: toolCall.callId,
      output,
    });

    const visionInput = createToolVisionInput(toolCall, output);
    if (visionInput !== null) {
      visionInputs.push(visionInput);
    }
  }

  return [...nextInput, ...visionInputs];
}

/**
 * tool output に含まれる画像参照を、モデルへ直接渡せる vision input に変換する。
 * 現時点では Discord チャット取得結果だけを対象にする。
 *
 * @param toolCall 実行済み tool call
 * @param output model へ返す sanitization 済み tool output
 * @returns 画像があれば追加入力する user message。なければ `null`
 */
function createToolVisionInput(
  toolCall: ModelToolCall,
  output: string
): ModelMessage | null {
  if (toolCall.toolName !== 'read_chat_messages') {
    return null;
  }

  const images = extractReadChatMessageImages(output);
  if (images.length === 0) {
    return null;
  }

  const selectedImages = images.slice(0, MAX_TOOL_VISION_IMAGES);
  const content: ModelMessageContentPart[] = [
    {
      type: 'text',
      text: `Discord image attachments from read_chat_messages (${selectedImages.length}/${images.length}).`,
    },
  ];

  for (const [index, image] of selectedImages.entries()) {
    content.push(
      {
        type: 'text',
        text: formatVisionImageContext(image, index + 1),
      },
      {
        type: 'image',
        imageUrl: image.url,
        detail: 'auto',
      }
    );
  }

  if (images.length > selectedImages.length) {
    content.push({
      type: 'text',
      text: `${images.length - selectedImages.length} additional image attachment(s) were omitted from direct vision input to keep the model request bounded.`,
    });
  }

  return {
    role: 'user',
    content,
  };
}

/**
 * read_chat_messages の JSON 結果から画像添付を抽出する。
 *
 * @param output tool output JSON
 * @returns モデルへ画像入力として渡せる画像参照
 */
function extractReadChatMessageImages(output: string): ChatToolVisionImage[] {
  const parsed = parseJsonObject(output);
  if (parsed?.success !== true || !Array.isArray(parsed.messages)) {
    return [];
  }

  return parsed.messages.flatMap((message): ChatToolVisionImage[] => {
    if (!isRecord(message) || !Array.isArray(message.images)) {
      return [];
    }

    const messageId = getStringProperty(message, 'messageId');
    const user = getStringProperty(message, 'user');
    const createdAt = getStringProperty(message, 'created_at');
    if (
      messageId === undefined ||
      user === undefined ||
      createdAt === undefined
    ) {
      return [];
    }

    return message.images.flatMap((image): ChatToolVisionImage[] => {
      if (!isRecord(image)) {
        return [];
      }

      const url = getStringProperty(image, 'url');
      if (url === undefined) {
        return [];
      }

      return [
        {
          messageId,
          user,
          createdAt,
          url,
          filename: getNullableStringProperty(image, 'filename'),
          contentType: getNullableStringProperty(image, 'content_type'),
          width: getNullableNumberProperty(image, 'width'),
          height: getNullableNumberProperty(image, 'height'),
          description: getNullableStringProperty(image, 'description'),
        },
      ];
    });
  });
}

/**
 * 画像の出所をモデルへ伝えるための短い説明文を作る。
 */
function formatVisionImageContext(
  image: ChatToolVisionImage,
  index: number
): string {
  const metadata: string[] = [
    `Image ${index}`,
    `messageId=${image.messageId}`,
    `user=${image.user}`,
    `created_at=${image.createdAt}`,
  ];
  appendNullableMetadata(metadata, 'filename', image.filename);
  appendNullableMetadata(metadata, 'content_type', image.contentType);
  appendImageSizeMetadata(metadata, image);
  appendNullableMetadata(metadata, 'description', image.description);

  return metadata.join(', ');
}

/**
 * nullable なメタデータを説明文配列へ追加する。
 */
function appendNullableMetadata(
  metadata: string[],
  key: string,
  value: string | null | undefined
): void {
  if (value !== undefined && value !== null) {
    metadata.push(`${key}=${value}`);
  }
}

/**
 * 画像の width / height が揃っている場合だけ size メタデータを追加する。
 */
function appendImageSizeMetadata(
  metadata: string[],
  image: Pick<ChatToolVisionImage, 'width' | 'height'>
): void {
  if (
    image.width !== undefined &&
    image.width !== null &&
    image.height !== undefined &&
    image.height !== null
  ) {
    metadata.push(`size=${image.width}x${image.height}`);
  }
}

/**
 * JSON 文字列を record として読む。失敗時は `null` を返す。
 */
function parseJsonObject(output: string): Record<string, unknown> | null {
  try {
    const parsed: unknown = JSON.parse(output);
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

/**
 * 値が object record かを判定する。
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * nullable string property を取得する。
 */
function getNullableStringProperty(
  record: Record<string, unknown>,
  key: string
): string | null | undefined {
  const value = record[key];
  if (value === null) {
    return null;
  }

  return typeof value === 'string' ? value : undefined;
}

/**
 * nullable number property を取得する。
 */
function getNullableNumberProperty(
  record: Record<string, unknown>,
  key: string
): number | null | undefined {
  const value = record[key];
  if (value === null) {
    return null;
  }

  return typeof value === 'number' ? value : undefined;
}

/**
 * provider 非依存の agent session loop。
 * モデル出力に tool call があれば実行結果を次ターン input に変換し、
 * 有効な `finish_thinking` が現れるか maxTurns に達するまで turn を繰り返す。
 */
export async function runAgentSession(
  input: RunAgentSessionInput
): Promise<AgentSessionResult> {
  const maxTurns = input.maxTurns ?? 10;
  let currentInput = input.initialInput;
  let previousResponseToken: string | undefined;
  let totalUsage = ZERO_MODEL_USAGE;

  for (let turn = 1; turn <= maxTurns; turn += 1) {
    // Agent turns are sequential; event emission belongs to the same turn.
    // eslint-disable-next-line no-await-in-loop
    await emitEchoEvent(input.events, {
      type: 'model.turn.started',
      severity: 'debug',
      summary: `model turn ${turn} started`,
      payload: {
        turnIndex: turn,
        inputItemCount: currentInput.length,
      },
    });

    const turnStartedAt = Date.now();
    // Agent turns are inherently sequential because each model response
    // depends on the previous turn's tool outputs.
    // eslint-disable-next-line no-await-in-loop
    const response = await input.model.generate({
      input: currentInput,
      tools: getToolContracts(input.tools),
      previousResponseToken,
      turnIndex: turn,
    });

    totalUsage = accumulateModelUsage(totalUsage, response.usage);
    previousResponseToken = response.responseToken;

    const toolCalls = getToolCalls(response);
    const warnings = toolCalls.length === 0 ? ['no_tool_calls'] : [];
    // Agent turns are sequential; event emission belongs to the same turn.
    // eslint-disable-next-line no-await-in-loop
    await emitEchoEvent(input.events, {
      type: 'model.turn.completed',
      severity: warnings.length > 0 ? 'warn' : 'debug',
      summary: `model turn ${turn} completed`,
      payload: {
        turnIndex: turn,
        durationMs: Date.now() - turnStartedAt,
        outputItemCount: response.output.length,
        toolCallCount: toolCalls.length,
        warnings,
        usage: response.usage,
      },
    });

    // The loop stays alive until finish_thinking appears explicitly,
    // even when the model returned no tool calls in this turn.

    const finishThinking = parseFinishThinkingInput(toolCalls);

    // Tool results, or an empty carry-over when no tools were used,
    // become the next model input for the following turn.
    // eslint-disable-next-line no-await-in-loop
    currentInput = await createNextInput(
      toolCalls,
      input.tools,
      input.events,
      turn
    );
    if (finishThinking !== null) {
      return {
        context: finishThinking.session_record,
        nextWakeAt: finishThinking.next_wake_at ?? null,
        usage: totalUsage,
        responseToken: previousResponseToken,
        terminationReason: 'finish_thinking',
      };
    }
  }

  return {
    nextWakeAt: null,
    usage: totalUsage,
    responseToken: previousResponseToken,
    terminationReason: 'max_turns',
  };
}
