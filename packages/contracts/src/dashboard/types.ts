import type {
  dashboardInstanceSummarySchema,
  dashboardInstancesResponseSchema,
  dashboardMainLlmConfigSchema,
  dashboardRuntimeConfigSchema,
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
export type EchoStatus = z.infer<typeof echoStatusSchema>;
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
  uncachedRateInInput: number;
  inputRateInTotal: number;
  outputRateInTotal: number;
}
