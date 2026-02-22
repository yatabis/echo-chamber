export type EchoState = 'Idling' | 'Running' | 'Sleeping';

export interface Usage {
  cached_input_tokens: number;
  uncached_input_tokens: number;
  total_input_tokens: number;
  output_tokens: number;
  reasoning_tokens: number;
  total_tokens: number;
  total_cost: number;
}

export type UsageRecord = Record<string, Usage>;

export interface Note {
  id: string;
  title: string;
  content: string;
  createdAt: string;
  updatedAt: string;
}

export interface Emotion {
  valence: number; // [-1.0, 1.0] - negative to positive
  arousal: number; // [0.0, 1.0] - calm to excited
  labels: string[]; // e.g., ["intellectual-engagement", "mild-anticipation"]
}

export const MEMORY_TYPES = ['semantic', 'episode'] as const;
export type MemoryType = (typeof MEMORY_TYPES)[number];

export interface Memory {
  content: string; // Max 500 characters
  embedding: number[]; // 1536-dim vector from content
  emotion: Emotion;
  createdAt: string; // ISO 8601
  updatedAt: string; // ISO 8601
}
