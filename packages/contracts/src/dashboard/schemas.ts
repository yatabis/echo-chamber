import { z } from 'zod';

import { emotionSchema as coreEmotionSchema } from '@echo-chamber/core/echo/schemas';
import { MEMORY_TYPES } from '@echo-chamber/core/echo/types';
import { ECHO_INSTANCE_IDS } from '@echo-chamber/core/types/echo-config';

const finiteNumber = z.number();

/**
 * Echo 本体の state 文字列。
 */
export const echoStateSchema = z.enum(['Idling', 'Running', 'Sleeping']);

/**
 * Dashboard 一覧が扱う state 文字列。
 *
 * 一覧 API では、個別インスタンスの取得失敗時に `Unknown` へフォールバックする。
 */
export const dashboardSummaryStateSchema = z.union([
  echoStateSchema,
  z.literal('Unknown'),
]);

/**
 * token usage 1 件分の payload。
 */
export const tokenUsageSchema = z
  .object({
    cached_input_tokens: finiteNumber,
    uncached_input_tokens: finiteNumber,
    total_input_tokens: finiteNumber,
    output_tokens: finiteNumber,
    reasoning_tokens: finiteNumber,
    total_tokens: finiteNumber,
  })
  .strict();

/**
 * provider / model ごとの usage 内訳。
 */
export const usageModelBreakdownSchema = tokenUsageSchema
  .extend({
    provider: z.string(),
    model: z.string(),
  })
  .strict();

/**
 * 日次 usage 1 件分の payload。
 */
export const usageSchema = tokenUsageSchema
  .extend({
    by_model: z.array(usageModelBreakdownSchema).min(1),
  })
  .strict();

/**
 * `YYYY-MM-DD` キーごとの usage 集計。
 */
export const usageRecordSchema = z.record(z.string(), usageSchema);

/**
 * Dashboard に表示するメイン LLM 設定。
 */
export const dashboardMainLlmConfigSchema = z
  .object({
    provider: z.string(),
    model: z.string(),
  })
  .strict();

/**
 * Dashboard に表示する token limit 設定。
 */
export const dashboardTokenLimitConfigSchema = z
  .object({
    dailyHardLimit: finiteNumber,
    dailySoftLimit: finiteNumber,
    hardLimitBufferFactor: finiteNumber,
  })
  .strict();

/**
 * Dashboard に表示する runtime 設定。
 */
export const dashboardRuntimeConfigSchema = z
  .object({
    mainLlm: dashboardMainLlmConfigSchema,
    tokenLimits: dashboardTokenLimitConfigSchema,
  })
  .strict();

/**
 * dashboard event timeline が扱う severity。
 */
export const dashboardEchoEventSeveritySchema = z.enum([
  'debug',
  'info',
  'warn',
  'error',
]);

/**
 * dashboard event timeline が扱う stream。
 */
export const dashboardEchoEventStreamSchema = z.enum([
  'thought',
  'system',
  'analysis',
]);

/**
 * Dashboard に返す Echo event payload。
 */
export const dashboardEchoEventSchema = z
  .object({
    id: z.string(),
    createdAt: z.string(),
    archiveDay: z.string(),
    sessionId: z.string().nullable(),
    type: z.string(),
    category: z.string(),
    severity: dashboardEchoEventSeveritySchema,
    streams: z.array(dashboardEchoEventStreamSchema),
    summary: z.string(),
    payload: z.record(z.string(), z.unknown()).nullable(),
  })
  .strict();

/**
 * `/\:instanceId/events` の payload。
 */
export const dashboardEchoEventsResponseSchema = z
  .object({
    archiveDay: z.string(),
    events: z.array(dashboardEchoEventSchema),
  })
  .strict();

/**
 * Dashboard に返す note payload。
 */
export const noteSchema = z
  .object({
    id: z.string(),
    title: z.string(),
    content: z.string(),
    createdAt: z.string(),
    updatedAt: z.string(),
  })
  .strict();

/**
 * Dashboard に返す memory payload。
 */
export const echoMemorySchema = z
  .object({
    content: z.string(),
    type: z.enum(MEMORY_TYPES),
    emotion: coreEmotionSchema.strict(),
    embedding_model: z.string(),
    createdAt: z.string(),
    updatedAt: z.string(),
  })
  .strict();

/**
 * `/\:instanceId` 詳細 API の payload。
 */
export const echoStatusSchema = z
  .object({
    id: z.enum(ECHO_INSTANCE_IDS),
    name: z.string(),
    state: echoStateSchema,
    nextAlarm: z.string().nullable(),
    runtime: dashboardRuntimeConfigSchema,
    memories: z.array(echoMemorySchema),
    notes: z.array(noteSchema),
    usage: usageRecordSchema,
  })
  .strict();

/**
 * `/\:instanceId/summary` の payload。
 */
export const dashboardInstanceSummarySchema = z
  .object({
    id: z.enum(ECHO_INSTANCE_IDS),
    name: z.string(),
    state: dashboardSummaryStateSchema,
    nextAlarm: z.string().nullable(),
    noteCount: z.number().int().nonnegative(),
    memoryCount: z.number().int().nonnegative(),
    todayUsageTokens: finiteNumber,
    sevenDayUsageTokens: finiteNumber,
    thirtyDayUsageTokens: finiteNumber,
    runtime: dashboardRuntimeConfigSchema,
    latestNoteUpdatedAt: z.string().nullable(),
    latestMemoryUpdatedAt: z.string().nullable(),
  })
  .strict();

/**
 * `/instances` の payload。
 */
export const dashboardInstancesResponseSchema = z
  .object({
    instances: z.array(dashboardInstanceSummarySchema),
  })
  .strict();

/**
 * `/\:instanceId/events` の unknown payload を契約型へ変換する。
 */
export function parseDashboardEchoEventsResponse(
  value: unknown
): z.infer<typeof dashboardEchoEventsResponseSchema> {
  return dashboardEchoEventsResponseSchema.parse(value);
}

/**
 * `/\:instanceId/summary` の unknown payload を契約型へ変換する。
 */
export function parseDashboardInstanceSummary(
  value: unknown
): z.infer<typeof dashboardInstanceSummarySchema> {
  return dashboardInstanceSummarySchema.parse(value);
}

/**
 * `/instances` の unknown payload を契約型へ変換する。
 */
export function parseDashboardInstancesResponse(
  value: unknown
): z.infer<typeof dashboardInstancesResponseSchema> {
  return dashboardInstancesResponseSchema.parse(value);
}

/**
 * `/\:instanceId` の unknown payload を契約型へ変換する。
 */
export function parseEchoStatus(
  value: unknown
): z.infer<typeof echoStatusSchema> {
  return echoStatusSchema.parse(value);
}
