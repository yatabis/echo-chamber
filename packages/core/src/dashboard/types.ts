import type { EchoState, Note, Usage, UsageRecord } from '../echo/types';
import type { EchoInstanceId } from '../types/echo-config';

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

export interface DashboardUsagePoint {
  dateKey: string;
  usage: Usage | null;
}

export interface DashboardUsageTotals {
  totalTokens: number;
  totalCost: number;
}
