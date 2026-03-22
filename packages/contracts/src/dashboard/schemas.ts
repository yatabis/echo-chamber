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
 * 日次 usage 1 件分の payload。
 */
export const usageSchema = z
  .object({
    cached_input_tokens: finiteNumber,
    uncached_input_tokens: finiteNumber,
    total_input_tokens: finiteNumber,
    output_tokens: finiteNumber,
    reasoning_tokens: finiteNumber,
    total_tokens: finiteNumber,
    total_cost: finiteNumber,
  })
  .strict();

/**
 * `YYYY-MM-DD` キーごとの usage 集計。
 */
export const usageRecordSchema = z.record(z.string(), usageSchema);

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
