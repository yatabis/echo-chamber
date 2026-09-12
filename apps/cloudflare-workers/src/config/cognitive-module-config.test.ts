import { describe, expect, it } from 'vitest';

import {
  getEchoInstanceDefinition,
  type EchoInstanceDefinition,
} from '@echo-chamber/core/echo/instance-definitions';

import {
  DEFAULT_COGNITIVE_MODULE_MODEL,
  DEFAULT_COGNITIVE_MODULE_REASONING_EFFORT,
  resolveCognitiveModuleConfig,
  type CognitiveModuleEnv,
} from './cognitive-module-config';

type CognitiveModuleTestEnvOverrides = Partial<CognitiveModuleEnv> &
  Record<string, string | undefined>;

function createEnv(
  overrides: CognitiveModuleTestEnvOverrides = {}
): CognitiveModuleEnv {
  return {
    OPENAI_API_KEY: 'existing-openai-key',
    ...overrides,
  };
}

function createDefinition(
  overrides: Partial<EchoInstanceDefinition> = {}
): EchoInstanceDefinition {
  return {
    ...getEchoInstanceDefinition('rin'),
    ...overrides,
  };
}

describe('resolveCognitiveModuleConfig', () => {
  it('既存 OpenAI key と instance definition の設定を返す', () => {
    expect(
      resolveCognitiveModuleConfig(
        createEnv(),
        getEchoInstanceDefinition('rin')
      )
    ).toEqual({
      provider: 'openai',
      api: 'responses',
      apiKey: 'existing-openai-key',
      model: DEFAULT_COGNITIVE_MODULE_MODEL,
      reasoningEffort: DEFAULT_COGNITIVE_MODULE_REASONING_EFFORT,
    });
    expect(DEFAULT_COGNITIVE_MODULE_MODEL).toBe('gpt-5.6-luna');
    expect(DEFAULT_COGNITIVE_MODULE_REASONING_EFFORT).toBe('low');
  });

  it('model と reasoning effort は instance 環境変数で上書きできる', () => {
    expect(
      resolveCognitiveModuleConfig(
        createEnv({
          RIN_COGNITIVE_MODULE_MODEL: ' GPT-5.6-LUNA-OVERRIDE ',
          RIN_COGNITIVE_MODULE_REASONING_EFFORT: 'MEDIUM',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toEqual({
      provider: 'openai',
      api: 'responses',
      apiKey: 'existing-openai-key',
      model: 'GPT-5.6-LUNA-OVERRIDE',
      reasoningEffort: 'medium',
    });
  });

  it('instance definition は global 環境変数より優先する', () => {
    expect(
      resolveCognitiveModuleConfig(
        createEnv({
          COGNITIVE_MODULE_MODEL: 'global-cognitive-model',
          COGNITIVE_MODULE_REASONING_EFFORT: 'high',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toMatchObject({
      model: 'gpt-5.6-luna',
      reasoningEffort: 'low',
    });
  });

  it('definition が空の項目は global 環境変数で補完できる', () => {
    expect(
      resolveCognitiveModuleConfig(
        createEnv({
          COGNITIVE_MODULE_MODEL: 'global-cognitive-model',
          COGNITIVE_MODULE_REASONING_EFFORT: 'HIGH',
        }),
        createDefinition({ cognitiveModules: {} })
      )
    ).toMatchObject({
      model: 'global-cognitive-model',
      reasoningEffort: 'high',
    });
  });

  it('空の instance 環境変数は definition へ戻す', () => {
    expect(
      resolveCognitiveModuleConfig(
        createEnv({
          RIN_COGNITIVE_MODULE_MODEL: ' ',
          RIN_COGNITIVE_MODULE_REASONING_EFFORT: '',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toMatchObject({
      model: 'gpt-5.6-luna',
      reasoningEffort: 'low',
    });
  });

  it('全設定が未指定なら Worker の既定値へ戻す', () => {
    expect(
      resolveCognitiveModuleConfig(
        createEnv(),
        createDefinition({ cognitiveModules: {} })
      )
    ).toMatchObject({
      model: DEFAULT_COGNITIVE_MODULE_MODEL,
      reasoningEffort: DEFAULT_COGNITIVE_MODULE_REASONING_EFFORT,
    });
  });

  it('未対応 reasoning effort は拒否する', () => {
    expect(() =>
      resolveCognitiveModuleConfig(
        createEnv({
          RIN_COGNITIVE_MODULE_REASONING_EFFORT: 'extreme',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toThrow('Unsupported COGNITIVE_MODULE_REASONING_EFFORT: extreme');
  });

  it('既存 OpenAI key が空なら拒否する', () => {
    expect(() =>
      resolveCognitiveModuleConfig(
        createEnv({ OPENAI_API_KEY: ' ' }),
        getEchoInstanceDefinition('rin')
      )
    ).toThrow('OPENAI_API_KEY must not be empty');
  });
});
