export interface ContextSnapshot {
  summary: string;
  createdAt: string;
  updatedAt: string;
  metadata?: Record<string, unknown>;
}

export interface ContextPort {
  load(): Promise<ContextSnapshot | null>;
  save(snapshot: ContextSnapshot): Promise<void>;
  clear(): Promise<void>;
}
