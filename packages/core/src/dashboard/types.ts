import type { EchoState, Note, UsageRecord } from '../echo/types';

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
