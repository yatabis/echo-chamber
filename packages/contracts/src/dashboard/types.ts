import type {
  dashboardInstanceSummarySchema,
  dashboardInstancesResponseSchema,
  dashboardSummaryStateSchema,
  echoMemorySchema,
  echoStateSchema,
  echoStatusSchema,
  noteSchema,
  usageRecordSchema,
  usageSchema,
} from './schemas';
import type { z } from 'zod';

export type EchoState = z.infer<typeof echoStateSchema>;
export type DashboardSummaryState = z.infer<typeof dashboardSummaryStateSchema>;
export type Usage = z.infer<typeof usageSchema>;
export type UsageRecord = z.infer<typeof usageRecordSchema>;
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
  totalCost: number;
}

export interface DashboardUsageBreakdownTotals {
  cachedInputTokens: number;
  uncachedInputTokens: number;
  normalOutputTokens: number;
  reasoningOutputTokens: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalTokens: number;
  totalCost: number;
}

export interface DashboardUsageRatioMetrics {
  cacheRateInInput: number;
  uncachedRateInInput: number;
  inputRateInTotal: number;
  outputRateInTotal: number;
}
