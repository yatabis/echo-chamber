import { READ_WEB_PAGE_TOOL_NAME } from '@echo-chamber/core/agent/tools/web';
import { WEB_PAGE_READER_ERROR_CODES } from '@echo-chamber/core/ports/web-page-reader';

type UnknownRecord = Record<string, unknown>;

/**
 * provider exchangeの監査用コピーと、次turnへ持ち越すWeb call ID。
 */
export interface WebToolAuditProjection<TRequest, TResponse> {
  request: TRequest;
  response: TResponse;
  webToolCallIds: Set<string>;
}

const WEB_ERROR_CODES = new Set<string>(WEB_PAGE_READER_ERROR_CODES);
const WEB_CONTENT_TYPES = new Set([
  'text/html',
  'application/xhtml+xml',
  'text/plain',
]);

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function parseRecord(value: unknown): UnknownRecord | undefined {
  if (isRecord(value)) {
    return value;
  }
  if (typeof value !== 'string') {
    return undefined;
  }

  try {
    const parsed: unknown = JSON.parse(value);
    return isRecord(parsed) ? parsed : undefined;
  } catch {
    return undefined;
  }
}

function getRecord(
  record: UnknownRecord | undefined,
  key: string
): UnknownRecord | undefined {
  const value = record?.[key];
  return isRecord(value) ? value : undefined;
}

function getSafeNumber(
  record: UnknownRecord | undefined,
  key: string
): number | undefined {
  const value = record?.[key];
  return typeof value === 'number' && Number.isFinite(value) && value >= 0
    ? value
    : undefined;
}

function getBoolean(
  record: UnknownRecord | undefined,
  key: string
): boolean | undefined {
  const value = record?.[key];
  return typeof value === 'boolean' ? value : undefined;
}

function getAllowedString(
  record: UnknownRecord | undefined,
  key: string,
  allowedValues: ReadonlySet<string>
): string | undefined {
  const value = record?.[key];
  return typeof value === 'string' && allowedValues.has(value)
    ? value
    : undefined;
}

function getArrayLength(
  record: UnknownRecord | undefined,
  key: string
): number | undefined {
  const value = record?.[key];
  return Array.isArray(value) ? value.length : undefined;
}

function compactRecord(record: UnknownRecord): UnknownRecord {
  return Object.fromEntries(
    Object.entries(record).filter(([, value]) => value !== undefined)
  );
}

function hasQuery(url: string): boolean {
  try {
    return new URL(url).search !== '';
  } catch {
    return url.includes('?');
  }
}

/**
 * Web tool引数をURL値を含まない監査metadataへ変換する。
 */
function serializeWebArguments(value: unknown): string {
  const parsed = parseRecord(value);
  const url = parsed?.url;
  const maxCharacters = getSafeNumber(parsed, 'maxCharacters');

  return JSON.stringify(
    compactRecord({
      redacted: true,
      urlLength: typeof url === 'string' ? url.length : 0,
      hasQuery: typeof url === 'string' ? hasQuery(url) : false,
      maxCharacters,
    })
  );
}

/**
 * Web tool結果から本文・title・URL・linkを除き、安全な集計値だけを残す。
 */
function serializeWebResult(value: unknown): string {
  const parsed = parseRecord(value);
  const source = getRecord(parsed, 'source');
  const document = getRecord(parsed, 'document');

  return JSON.stringify(
    compactRecord({
      redacted: true,
      success: getBoolean(parsed, 'success'),
      code: getAllowedString(parsed, 'code', WEB_ERROR_CODES),
      retryable: getBoolean(parsed, 'retryable'),
      httpStatus: getSafeNumber(source, 'httpStatus'),
      contentType: getAllowedString(source, 'contentType', WEB_CONTENT_TYPES),
      redirectCount: getSafeNumber(source, 'redirectCount'),
      returnedCharacters: getSafeNumber(document, 'returnedCharacters'),
      extractedCharacters: getSafeNumber(document, 'extractedCharacters'),
      truncated: getBoolean(document, 'truncated'),
      linkCount: getArrayLength(document, 'links'),
    })
  );
}

function updateCallId(
  callIds: Set<string>,
  callId: unknown,
  toolName: unknown
): void {
  if (typeof callId !== 'string' || typeof toolName !== 'string') {
    return;
  }

  if (toolName === READ_WEB_PAGE_TOOL_NAME) {
    callIds.add(callId);
  } else {
    callIds.delete(callId);
  }
}

function projectResponseItems(value: unknown, callIds: Set<string>): unknown {
  if (!Array.isArray(value)) {
    return value;
  }

  const items: readonly unknown[] = value;
  const projected = items.map((item) => {
    if (!isRecord(item)) {
      return item;
    }

    if (item.type === 'function_call') {
      updateCallId(callIds, item.call_id, item.name);
      if (item.name === READ_WEB_PAGE_TOOL_NAME) {
        return {
          ...item,
          arguments: serializeWebArguments(item.arguments),
        };
      }
      return item;
    }

    if (
      item.type === 'function_call_output' &&
      typeof item.call_id === 'string' &&
      callIds.has(item.call_id)
    ) {
      callIds.delete(item.call_id);
      return {
        ...item,
        output: serializeWebResult(item.output),
      };
    }

    return item;
  });

  return projected.some((item, index) => item !== items[index])
    ? projected
    : value;
}

function projectChatToolCalls(value: unknown, callIds: Set<string>): unknown {
  if (!Array.isArray(value)) {
    return value;
  }

  const toolCalls: readonly unknown[] = value;
  const projected = toolCalls.map((toolCall) => {
    if (!isRecord(toolCall) || !isRecord(toolCall.function)) {
      return toolCall;
    }

    updateCallId(callIds, toolCall.id, toolCall.function.name);
    if (toolCall.function.name !== READ_WEB_PAGE_TOOL_NAME) {
      return toolCall;
    }

    return {
      ...toolCall,
      function: {
        ...toolCall.function,
        arguments: serializeWebArguments(toolCall.function.arguments),
      },
    };
  });

  return projected.some((toolCall, index) => toolCall !== toolCalls[index])
    ? projected
    : value;
}

function projectChatMessage(value: unknown, callIds: Set<string>): unknown {
  if (!isRecord(value)) {
    return value;
  }

  const toolCalls = projectChatToolCalls(value.tool_calls, callIds);
  const isWebResult =
    value.role === 'tool' &&
    typeof value.tool_call_id === 'string' &&
    callIds.has(value.tool_call_id);
  if (toolCalls === value.tool_calls && !isWebResult) {
    return value;
  }

  const projected = {
    ...value,
    ...(toolCalls === value.tool_calls ? {} : { tool_calls: toolCalls }),
    ...(isWebResult ? { content: serializeWebResult(value.content) } : {}),
  };
  if (isWebResult && typeof value.tool_call_id === 'string') {
    callIds.delete(value.tool_call_id);
  }
  return projected;
}

function projectChatMessages(value: unknown, callIds: Set<string>): unknown {
  if (!Array.isArray(value)) {
    return value;
  }

  const messages: readonly unknown[] = value;
  const projected = messages.map((message) =>
    projectChatMessage(message, callIds)
  );
  return projected.some((message, index) => message !== messages[index])
    ? projected
    : value;
}

function projectChatChoices(value: unknown, callIds: Set<string>): unknown {
  if (!Array.isArray(value)) {
    return value;
  }

  const choices: readonly unknown[] = value;
  const projected = choices.map((choice) => {
    if (!isRecord(choice)) {
      return choice;
    }
    const message = projectChatMessage(choice.message, callIds);
    if (message === choice.message) {
      return choice;
    }

    return { ...choice, message };
  });
  return projected.some((choice, index) => choice !== choices[index])
    ? projected
    : value;
}

/**
 * Responses APIの既知function call shapeだけをWeb用監査コピーへ写像する。
 * live request/responseと非Web itemは変更しない。
 */
export function projectResponsesWebToolExchange<
  TRequest extends object,
  TResponse extends object,
>(
  request: TRequest,
  response: TResponse,
  knownWebToolCallIds: ReadonlySet<string>
): WebToolAuditProjection<TRequest, TResponse> {
  const webToolCallIds = new Set(knownWebToolCallIds);
  const requestRecord = request as UnknownRecord;
  const responseRecord = response as UnknownRecord;
  const input = projectResponseItems(requestRecord.input, webToolCallIds);
  const output = projectResponseItems(responseRecord.output, webToolCallIds);

  return {
    request:
      input === requestRecord.input
        ? request
        : ({ ...request, input } as TRequest),
    response:
      output === responseRecord.output
        ? response
        : ({ ...response, output } as TResponse),
    webToolCallIds,
  };
}

/**
 * Chat Completionsの既知tool_calls/role:tool shapeだけをWeb用監査コピーへ写像する。
 * live request/responseと非Web messageは変更しない。
 */
export function projectChatCompletionsWebToolExchange<
  TRequest extends object,
  TResponse extends object,
>(
  request: TRequest,
  response: TResponse,
  knownWebToolCallIds: ReadonlySet<string>
): WebToolAuditProjection<TRequest, TResponse> {
  const webToolCallIds = new Set(knownWebToolCallIds);
  const requestRecord = request as UnknownRecord;
  const responseRecord = response as UnknownRecord;
  const messages = projectChatMessages(requestRecord.messages, webToolCallIds);
  const choices = projectChatChoices(responseRecord.choices, webToolCallIds);

  return {
    request:
      messages === requestRecord.messages
        ? request
        : ({ ...request, messages } as TRequest),
    response:
      choices === responseRecord.choices
        ? response
        : ({ ...response, choices } as TResponse),
    webToolCallIds,
  };
}
