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
    tokenLimits: {
      dailyHardLimit: 2_500_000,
      dailySoftLimit: 600_000,
      hardLimitBufferFactor: 1.5,
    },
  },
} as const;

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
        tokenLimits: definition.tokenLimits,
      }).toEqual(expectedInstanceRuntimeSettings[instanceId]);
    }
  );
});
