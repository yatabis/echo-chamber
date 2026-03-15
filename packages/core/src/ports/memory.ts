import type { Emotion, MemoryType } from '../echo/types';

export interface MemoryRecord {
  content: string;
  type: MemoryType;
  emotion: Emotion;
  createdAt: string;
  updatedAt: string;
}

export interface MemorySearchResult extends MemoryRecord {
  similarity: number;
}

export interface MemoryPort {
  getLatest(): Promise<MemoryRecord | null> | MemoryRecord | null;
  list(): Promise<MemoryRecord[]> | MemoryRecord[];
  store(content: string, emotion: Emotion, type: MemoryType): Promise<void>;
  search(query: string, type?: MemoryType): Promise<MemorySearchResult[]>;
}
