import systemPromptRin from '../llm/prompts/rin';

import type { EchoInstanceId } from '../types/echo-config';

export interface EchoInstanceDefinition {
  id: EchoInstanceId;
  name: string;
  systemPrompt: string;
}

export const ECHO_INSTANCE_DEFINITIONS = {
  rin: {
    id: 'rin',
    name: 'リン',
    systemPrompt: systemPromptRin,
  },
  marie: {
    id: 'marie',
    name: 'マリー',
    systemPrompt: '', // TODO: マリーのシステムプロンプト割り当てを確定する
  },
} satisfies Record<EchoInstanceId, EchoInstanceDefinition>;

export function getEchoInstanceDefinition(
  instanceId: EchoInstanceId
): EchoInstanceDefinition {
  return ECHO_INSTANCE_DEFINITIONS[instanceId];
}
