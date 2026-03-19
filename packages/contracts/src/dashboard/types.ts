import type {
  EchoState,
  Note,
  UsageRecord,
} from '@echo-chamber/core/echo/types';
import type { EchoInstanceId } from '@echo-chamber/core/types/echo-config';

export interface EchoMemory {
  content: string;
  type: 'semantic' | 'episode';
  emotion: {
    valence: number;
    arousal: number;
    labels: string[];
  };
  embedding_model: string;
  createdAt: string;
  updatedAt: string;
}

export interface EchoStatus {
  id: string;
  name: string;
  state: EchoState;
  nextAlarm: string | null;
  memories: EchoMemory[];
  notes: Note[];
  usage: UsageRecord;
}

export interface DashboardInstanceSummary {
  id: EchoInstanceId;
  name: string;
  state: EchoState | 'Unknown';
  nextAlarm: string | null;
}

export interface DashboardInstancesResponse {
  instances: DashboardInstanceSummary[];
}

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
