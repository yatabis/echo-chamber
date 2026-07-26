import systemPromptMarie from '../llm/prompts/marie';
import systemPromptRin from '../llm/prompts/rin';

import { TOKEN_LIMITS } from './constants';

import type { EchoInstanceId } from '../types/echo-config';

export type EchoMainLLMProvider = 'openai' | 'openai-compatible';
export type EchoMainLLMReasoningEffort =
  | 'none'
  | 'minimal'
  | 'low'
  | 'medium'
  | 'high'
  | 'xhigh';

export interface EchoMainLLMDefinition {
  provider?: EchoMainLLMProvider;
  model?: string;
  baseURL?: string;
  reasoningEffort?: EchoMainLLMReasoningEffort;
}

export interface EchoTokenLimitDefinition {
  dailyHardLimit?: number;
  dailySoftLimit?: number;
  hardLimitBufferFactor?: number;
}

export interface EchoInstanceDefinition {
  id: EchoInstanceId;
  name: string;
  systemPrompt: string;
  mainLlm: EchoMainLLMDefinition;
  tokenLimits: EchoTokenLimitDefinition;
}

export const ECHO_INSTANCE_DEFINITIONS = {
  rin: {
    id: 'rin',
    name: 'リン',
    systemPrompt: systemPromptRin,
    mainLlm: {
      provider: 'openai',
      model: 'gpt-5.6-sol',
      reasoningEffort: 'none',
    },
    tokenLimits: {
      dailyHardLimit: TOKEN_LIMITS.DAILY_HARD_LIMIT,
      dailySoftLimit: TOKEN_LIMITS.DAILY_SOFT_LIMIT,
      hardLimitBufferFactor: TOKEN_LIMITS.HARD_LIMIT_BUFFER_FACTOR,
    },
  },
  marie: {
    id: 'marie',
    name: 'マリー',
    systemPrompt: systemPromptMarie,
    mainLlm: {
      provider: 'openai',
      model: 'gpt-5.6-terra',
      reasoningEffort: 'low',
    },
    tokenLimits: {
      dailyHardLimit: TOKEN_LIMITS.DAILY_HARD_LIMIT * 5,
      dailySoftLimit: TOKEN_LIMITS.DAILY_SOFT_LIMIT * 2,
      hardLimitBufferFactor: TOKEN_LIMITS.HARD_LIMIT_BUFFER_FACTOR,
    },
  },
} satisfies Record<EchoInstanceId, EchoInstanceDefinition>;

export function getEchoInstanceDefinition(
  instanceId: EchoInstanceId
): EchoInstanceDefinition {
  return ECHO_INSTANCE_DEFINITIONS[instanceId];
}
