import { describe, expect, it } from 'vitest';

import { ECHO_INSTANCE_IDS } from '../types/echo-config';

import {
  ECHO_INSTANCE_DEFINITIONS,
  getEchoInstanceDefinition,
} from './instance-definitions';

const expectedInstanceRuntimeSettings = {
  rin: {
    mainLlm: {
      provider: 'openai',
      model: 'gpt-5.6-sol',
      reasoningEffort: 'none',
    },
    cognitiveModules: {
      model: 'gpt-5.6-luna',
      reasoningEffort: 'low',
    },
    tokenLimits: {
      dailyHardLimit: 500_000,
      dailySoftLimit: 300_000,
      hardLimitBufferFactor: 1.5,
    },
  },
  marie: {
    mainLlm: {
      provider: 'openai',
      model: 'gpt-5.6-terra',
      reasoningEffort: 'low',
    },
    cognitiveModules: {
      model: 'gpt-5.6-luna',
      reasoningEffort: 'low',
    },
    tokenLimits: {
      dailyHardLimit: 2_500_000,
      dailySoftLimit: 600_000,
      hardLimitBufferFactor: 1.5,
    },
  },
} as const;

/** prompt の `<tag>...</tag>` 形式のセクションと、その tag 名。 */
const PROMPT_SECTION_PATTERN = /<([a-z_]+)>[\s\S]*?<\/\1>/g;

describe('echo instance definitions', () => {
  it('全 instance id を definition catalogue がカバーしている', () => {
    expect(Object.keys(ECHO_INSTANCE_DEFINITIONS).sort()).toEqual(
      [...ECHO_INSTANCE_IDS].sort()
    );
  });

  it.each(ECHO_INSTANCE_IDS)(
    '%s の definition を id 一致で取得できる',
    (instanceId) => {
      const definition = getEchoInstanceDefinition(instanceId);

      expect(definition.id).toBe(instanceId);
      expect(definition.tokenLimits.dailyHardLimit).toBeGreaterThan(0);
      expect(definition.tokenLimits.dailySoftLimit).toBeGreaterThan(0);
      expect(definition.tokenLimits.hardLimitBufferFactor).toBeGreaterThan(0);
    }
  );

  it.each(ECHO_INSTANCE_IDS)(
    '%s の runtime 設定を catalogue から取得できる',
    (instanceId) => {
      const definition = getEchoInstanceDefinition(instanceId);

      expect({
        mainLlm: definition.mainLlm,
        cognitiveModules: definition.cognitiveModules,
        tokenLimits: definition.tokenLimits,
      }).toEqual(expectedInstanceRuntimeSettings[instanceId]);
    }
  );

  it.each(ECHO_INSTANCE_IDS)(
    '%s の Cognitive 用の人物像は、Main と同じ記述のうち個体を理解するための部分だけを持つ',
    (instanceId) => {
      const { personaProfile, systemPrompt } =
        getEchoInstanceDefinition(instanceId);
      const sections = [...personaProfile.matchAll(PROMPT_SECTION_PATTERN)];

      expect(sections.map((section) => section[1])).toEqual([
        'persona_definition',
        'attention_and_cognition',
        'emotional_tendency',
        'relationship_with_yatabis',
        'other_echoes',
      ]);
      // Main と別の写しを持たず、同じ記述を共有していることを確かめる。
      for (const [section] of sections) {
        expect(systemPrompt).toContain(section);
      }
      expect(personaProfile).toBe(
        sections.map(([section]) => section).join('\n\n')
      );
    }
  );
});
