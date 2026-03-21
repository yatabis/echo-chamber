import type { AgentSessionTool } from './session';
import type { LoggerPort } from '../ports/logger';
import type { MemoryPort } from '../ports/memory';
import type { ModelPort } from '../ports/model';
import type { ThoughtLogPort } from '../ports/thought-log';

/**
 * Thinking engine の構築入力。
 * prompt に必要な data と provider/runtime service を一箇所に集約する。
 */
export interface ThinkingEngineInput {
  model: ModelPort;
  thoughtLog: ThoughtLogPort;
  logger: LoggerPort;
  memory: Pick<MemoryPort, 'getLatest'>;
  tools: readonly AgentSessionTool[];
  systemPrompt: string;
}
