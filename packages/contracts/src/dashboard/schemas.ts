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
    cache_write_input_tokens: finiteNumber,
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

/** Cognitive Module の確定位置を表示するread model。 */
export const dashboardCognitiveModuleStatusSchema = z
  .object({
    domainVersion: z.number().int().nonnegative(),
    lastBoundaryId: z.string().nullable(),
    updatedAt: z.string().nullable(),
  })
  .strict();

/**
 * Dashboard session log builder が archive から読む Echo event severity。
 */
export const dashboardEchoEventSeveritySchema = z.enum([
  'debug',
  'info',
  'warn',
  'error',
]);

/**
 * Dashboard session log builder が archive から読む Echo event stream。
 */
export const dashboardEchoEventStreamSchema = z.enum([
  'thought',
  'system',
  'analysis',
]);

/**
 * Dashboard session log builder が archive から読む Echo event payload。
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
 * Dashboard activity log の分類。
 */
export const dashboardActivityKindSchema = z.enum([
  'session',
  'thought',
  'action',
  'decision',
  'knowledge',
  'system',
  'issue',
]);

/**
 * Dashboard activity log の視覚的な強調度。
 */
export const dashboardActivityToneSchema = z.enum([
  'critical',
  'warning',
  'neutral',
  'positive',
]);

/**
 * Dashboard に返す Echo activity payload。
 *
 * raw event stream ではなく、Dashboard がそのまま描画できる言動ログに射影済みの形。
 */
export const dashboardActivitySchema = z
  .object({
    id: z.string(),
    body: z.string(),
    createdAt: z.string(),
    details: z.record(z.string(), z.unknown()).nullable(),
    kind: dashboardActivityKindSchema,
    meta: z.array(z.string()),
    tone: dashboardActivityToneSchema,
    title: z.string(),
  })
  .strict();

/**
 * Dashboard に返す session 単位の activity log。
 */
export const dashboardSessionLogSchema = z
  .object({
    id: z.string(),
    activities: z.array(dashboardActivitySchema),
    activityCount: z.number().int().nonnegative(),
    latestActivityAt: z.string(),
    meta: z.array(z.string()),
    sessionId: z.string(),
    startedAt: z.string(),
    title: z.string(),
    warningCount: z.number().int().nonnegative(),
  })
  .strict();

/**
 * `/\:instanceId/session-logs` の payload。
 */
export const dashboardSessionLogsResponseSchema = z
  .object({
    archiveDay: z.string(),
    sessionLogs: z.array(dashboardSessionLogSchema),
  })
  .strict();

/**
 * Dashboard action analysis が集計する期間。
 */
export const dashboardActionAnalysisPeriodDaysSchema = z.union([
  z.literal(1),
  z.literal(7),
  z.literal(30),
]);

/**
 * Dashboard action analysis に返す tool 別集計。
 */
export const dashboardActionAnalysisToolSummarySchema = z
  .object({
    toolName: z.string(),
    calledCount: z.number().int().nonnegative(),
    completedCount: z.number().int().nonnegative(),
    failedCount: z.number().int().nonnegative(),
    failureRate: finiteNumber,
  })
  .strict();

/**
 * Dashboard action analysis に返す期間別の行動集計。
 */
export const dashboardActionAnalysisPeriodSchema = z
  .object({
    days: dashboardActionAnalysisPeriodDaysSchema,
    startArchiveDay: z.string(),
    endArchiveDay: z.string(),
    eventCount: z.number().int().nonnegative(),
    sessionCount: z.number().int().nonnegative(),
    completedSessionCount: z.number().int().nonnegative(),
    failedSessionCount: z.number().int().nonnegative(),
    warningSessionCount: z.number().int().nonnegative(),
    maxTurnsSessionCount: z.number().int().nonnegative(),
    totalTokens: finiteNumber,
    averageTokensPerCompletedSession: finiteNumber,
    averageSessionDurationMs: finiteNumber,
    totalTurns: z.number().int().nonnegative(),
    noToolCallTurns: z.number().int().nonnegative(),
    toolCallCount: z.number().int().nonnegative(),
    toolCompletedCount: z.number().int().nonnegative(),
    toolFailedCount: z.number().int().nonnegative(),
    toolFailureRate: finiteNumber,
    topTools: z.array(dashboardActionAnalysisToolSummarySchema),
    memorySearchCompletedCount: z.number().int().nonnegative(),
    memorySearchFailedCount: z.number().int().nonnegative(),
    memorySearchZeroResultCount: z.number().int().nonnegative(),
    memorySearchAverageFinalResultCount: finiteNumber,
    storeMemoryCompletedCount: z.number().int().nonnegative(),
  })
  .strict();

/**
 * `/\:instanceId/action-analysis` の payload。
 */
export const dashboardActionAnalysisResponseSchema = z
  .object({
    archiveDay: z.string(),
    generatedAt: z.string(),
    periods: z.array(dashboardActionAnalysisPeriodSchema),
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

/** Dashboard に返す既存の永続 Context snapshot。 */
export const dashboardContextSnapshotSchema = z
  .object({
    content: z.string(),
    emotion: coreEmotionSchema.strict(),
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
    nextWakeAt: z.string().nullable(),
    context: dashboardContextSnapshotSchema.nullable(),
    cognitive: dashboardCognitiveModuleStatusSchema,
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
    nextWakeAt: z.string().nullable(),
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
 * `/\:instanceId/session-logs` の unknown payload を契約型へ変換する。
 */
export function parseDashboardSessionLogsResponse(
  value: unknown
): z.infer<typeof dashboardSessionLogsResponseSchema> {
  return dashboardSessionLogsResponseSchema.parse(value);
}

/**
 * `/\:instanceId/action-analysis` の unknown payload を契約型へ変換する。
 */
export function parseDashboardActionAnalysisResponse(
  value: unknown
): z.infer<typeof dashboardActionAnalysisResponseSchema> {
  return dashboardActionAnalysisResponseSchema.parse(value);
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
