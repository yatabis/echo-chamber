import type {
  dashboardActivityKindSchema,
  dashboardActivitySchema,
  dashboardActivityToneSchema,
  dashboardActionAnalysisPeriodDaysSchema,
  dashboardActionAnalysisPeriodSchema,
  dashboardActionAnalysisResponseSchema,
  dashboardActionAnalysisToolSummarySchema,
  dashboardCognitiveMemorySchema,
  dashboardCognitiveModuleStatusSchema,
  dashboardInstanceSummarySchema,
  dashboardEchoEventSchema,
  dashboardEchoEventSeveritySchema,
  dashboardEchoEventStreamSchema,
  dashboardInstancesResponseSchema,
  dashboardMainLlmConfigSchema,
  dashboardRuntimeConfigSchema,
  dashboardSessionLogsResponseSchema,
  dashboardSessionLogSchema,
  dashboardSummaryStateSchema,
  dashboardTokenLimitConfigSchema,
  echoMemorySchema,
  echoStateSchema,
  echoStatusSchema,
  noteSchema,
  tokenUsageSchema,
  usageModelBreakdownSchema,
  usageRecordSchema,
  usageSchema,
} from './schemas';
import type { z } from 'zod';

export type EchoState = z.infer<typeof echoStateSchema>;
export type DashboardSummaryState = z.infer<typeof dashboardSummaryStateSchema>;
export type TokenUsage = z.infer<typeof tokenUsageSchema>;
export type UsageModelBreakdown = z.infer<typeof usageModelBreakdownSchema>;
export type Usage = z.infer<typeof usageSchema>;
export type UsageRecord = z.infer<typeof usageRecordSchema>;
export type DashboardMainLlmConfig = z.infer<
  typeof dashboardMainLlmConfigSchema
>;
export type DashboardTokenLimitConfig = z.infer<
  typeof dashboardTokenLimitConfigSchema
>;
export type DashboardRuntimeConfig = z.infer<
  typeof dashboardRuntimeConfigSchema
>;
export type Note = z.infer<typeof noteSchema>;
export type EchoMemory = z.infer<typeof echoMemorySchema>;
export type DashboardCognitiveMemory = z.infer<
  typeof dashboardCognitiveMemorySchema
>;
export type DashboardCognitiveModuleStatus = z.infer<
  typeof dashboardCognitiveModuleStatusSchema
>;
export type EchoStatus = z.infer<typeof echoStatusSchema>;
export type DashboardEchoEventSeverity = z.infer<
  typeof dashboardEchoEventSeveritySchema
>;
export type DashboardEchoEventStream = z.infer<
  typeof dashboardEchoEventStreamSchema
>;
export type DashboardEchoEvent = z.infer<typeof dashboardEchoEventSchema>;
export type DashboardActivityKind = z.infer<typeof dashboardActivityKindSchema>;
export type DashboardActivityTone = z.infer<typeof dashboardActivityToneSchema>;
export type DashboardActivity = z.infer<typeof dashboardActivitySchema>;
export type DashboardSessionLog = z.infer<typeof dashboardSessionLogSchema>;
export type DashboardSessionLogsResponse = z.infer<
  typeof dashboardSessionLogsResponseSchema
>;
export type DashboardActionAnalysisPeriodDays = z.infer<
  typeof dashboardActionAnalysisPeriodDaysSchema
>;
export type DashboardActionAnalysisToolSummary = z.infer<
  typeof dashboardActionAnalysisToolSummarySchema
>;
export type DashboardActionAnalysisPeriod = z.infer<
  typeof dashboardActionAnalysisPeriodSchema
>;
export type DashboardActionAnalysisResponse = z.infer<
  typeof dashboardActionAnalysisResponseSchema
>;
export type DashboardInstanceSummary = z.infer<
  typeof dashboardInstanceSummarySchema
>;
export type DashboardInstancesResponse = z.infer<
  typeof dashboardInstancesResponseSchema
>;

export type DashboardUsageDays = 7 | 30;

export interface DashboardUsageStackedPoint {
  dateKey: string;
  cachedInputTokens: number;
  cacheWriteInputTokens: number;
  uncachedInputTokens: number;
  normalOutputTokens: number;
  reasoningOutputTokens: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalTokens: number;
  estimatedCostUsd: number | null;
  estimatedCostIsPartial: boolean;
}

export interface DashboardUsageBreakdownTotals {
  cachedInputTokens: number;
  cacheWriteInputTokens: number;
  uncachedInputTokens: number;
  normalOutputTokens: number;
  reasoningOutputTokens: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalTokens: number;
  estimatedCostUsd: number | null;
  estimatedCostIsPartial: boolean;
}

export interface DashboardUsageRatioMetrics {
  cacheRateInInput: number;
  cacheWriteRateInInput: number;
  uncachedRateInInput: number;
  inputRateInTotal: number;
  outputRateInTotal: number;
}
