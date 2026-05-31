import { describe, expect, it } from 'vitest';

import {
  getEchoInstanceDefinition,
  type EchoInstanceDefinition,
} from '@echo-chamber/core/echo/instance-definitions';

import {
  DEFAULT_OPENAI_RESPONSES_MODEL,
  MAX_TOKENS,
  PRESENCE_PENALTY,
  TEMPERATURE,
  TOP_K,
  TOP_P,
  resolveMainLLMConfig,
  type MainLLMEnv,
} from './main-llm-config';

type MainLLMTestEnvOverrides = Partial<MainLLMEnv> &
  Record<string, string | undefined>;

function createEnv(overrides: MainLLMTestEnvOverrides = {}): MainLLMEnv {
  return {
    OPENAI_API_KEY: 'openai-key',
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

describe('resolveMainLLMConfig', () => {
  it('instance definition の OpenAI 設定を返す', () => {
    expect(
      resolveMainLLMConfig(createEnv(), getEchoInstanceDefinition('rin'))
    ).toEqual({
      provider: 'openai',
      api: 'responses',
      apiKey: 'openai-key',
      model: DEFAULT_OPENAI_RESPONSES_MODEL,
      baseURL: undefined,
    });
  });

  it('OpenAI のモデルと API key を instance 環境変数で上書きできる', () => {
    expect(
      resolveMainLLMConfig(
        createEnv({
          RIN_MAIN_LLM_PROVIDER: 'openai',
          RIN_MAIN_LLM_API_KEY: 'main-Key',
          RIN_MAIN_LLM_MODEL: 'GPT-5.4',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toEqual({
      provider: 'openai',
      api: 'responses',
      apiKey: 'main-Key',
      model: 'GPT-5.4',
      baseURL: undefined,
    });
  });

  it('definition が空の項目は global 環境変数で補完できる', () => {
    expect(
      resolveMainLLMConfig(
        createEnv({
          MAIN_LLM_PROVIDER: 'openai',
          MAIN_LLM_MODEL: 'GPT-5.2',
        }),
        createDefinition({
          mainLlm: {},
        })
      )
    ).toMatchObject({
      provider: 'openai',
      model: 'GPT-5.2',
    });
  });

  it('LM Studio は Chat Completions API 用の固定パラメータを返す', () => {
    expect(
      resolveMainLLMConfig(
        createEnv({
          RIN_MAIN_LLM_API_KEY: 'sk-lm-AbC123',
        }),
        createDefinition({
          mainLlm: {
            provider: 'lmstudio',
            model: 'openai/gpt-oss-20b',
            baseURL: 'http://localhost:1234/V1',
          },
        })
      )
    ).toEqual({
      provider: 'lmstudio',
      api: 'chat_completions',
      apiKey: 'sk-lm-AbC123',
      model: 'openai/gpt-oss-20b',
      baseURL: 'http://localhost:1234/V1',
      maxTokens: MAX_TOKENS,
      temperature: TEMPERATURE,
      topP: TOP_P,
      presencePenalty: PRESENCE_PENALTY,
      extraBody: {
        top_k: TOP_K,
        chat_template_kwargs: { enable_thinking: false },
      },
    });
  });

  it('LM Studio の baseURL と API key を instance 環境変数で上書きできる', () => {
    expect(
      resolveMainLLMConfig(
        createEnv({
          MARIE_MAIN_LLM_PROVIDER: 'lm-studio',
          MARIE_MAIN_LLM_API_KEY: 'local-key',
          MARIE_MAIN_LLM_MODEL: 'local-model',
          MARIE_MAIN_LLM_BASE_URL: 'http://127.0.0.1:4321/v1',
        }),
        getEchoInstanceDefinition('marie')
      )
    ).toEqual({
      provider: 'lmstudio',
      api: 'chat_completions',
      apiKey: 'local-key',
      model: 'local-model',
      baseURL: 'http://127.0.0.1:4321/v1',
      maxTokens: MAX_TOKENS,
      temperature: TEMPERATURE,
      topP: TOP_P,
      presencePenalty: PRESENCE_PENALTY,
      extraBody: {
        top_k: TOP_K,
        chat_template_kwargs: { enable_thinking: false },
      },
    });
  });

  it('provider を LM Studio に切り替えた場合は OpenAI definition model を流用しない', () => {
    expect(() =>
      resolveMainLLMConfig(
        createEnv({
          RIN_MAIN_LLM_PROVIDER: 'lmstudio',
          RIN_MAIN_LLM_BASE_URL: 'http://localhost:1234/v1',
          RIN_MAIN_LLM_API_KEY: 'local-key',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toThrow(
      'MAIN_LLM_MODEL is required when MAIN_LLM_PROVIDER is "lmstudio".'
    );
  });

  it('provider を LM Studio に切り替えた場合は global model fallback を使える', () => {
    expect(
      resolveMainLLMConfig(
        createEnv({
          RIN_MAIN_LLM_PROVIDER: 'lmstudio',
          RIN_MAIN_LLM_BASE_URL: 'http://localhost:1234/v1',
          RIN_MAIN_LLM_API_KEY: 'local-key',
          MAIN_LLM_MODEL: 'local-global-model',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toMatchObject({
      provider: 'lmstudio',
      model: 'local-global-model',
      baseURL: 'http://localhost:1234/v1',
      apiKey: 'local-key',
    });
  });

  it('LM Studio 指定でモデルが無い場合はエラーにする', () => {
    expect(() =>
      resolveMainLLMConfig(
        createEnv({
          MAIN_LLM_PROVIDER: 'lmstudio',
        }),
        createDefinition({ mainLlm: {} })
      )
    ).toThrow(
      'MAIN_LLM_MODEL is required when MAIN_LLM_PROVIDER is "lmstudio".'
    );
  });

  it('LM Studio 指定で API key が無い場合はエラーにする', () => {
    expect(() =>
      resolveMainLLMConfig(
        createEnv({
          RIN_MAIN_LLM_PROVIDER: 'lmstudio',
          RIN_MAIN_LLM_MODEL: 'local-model',
          RIN_MAIN_LLM_BASE_URL: 'http://localhost:1234/v1',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toThrow(
      'MAIN_LLM_API_KEY is required when MAIN_LLM_PROVIDER is "lmstudio".'
    );
  });

  it('LM Studio 指定で base URL が無い場合はエラーにする', () => {
    expect(() =>
      resolveMainLLMConfig(
        createEnv({
          RIN_MAIN_LLM_PROVIDER: 'lmstudio',
          RIN_MAIN_LLM_MODEL: 'local-model',
          RIN_MAIN_LLM_API_KEY: 'local-key',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toThrow(
      'MAIN_LLM_BASE_URL is required when MAIN_LLM_PROVIDER is "lmstudio".'
    );
  });

  it('未対応 provider はエラーにする', () => {
    expect(() =>
      resolveMainLLMConfig(
        createEnv({
          RIN_MAIN_LLM_PROVIDER: 'anthropic',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toThrow(
      'Unsupported MAIN_LLM_PROVIDER: anthropic. Use "openai" or "lmstudio".'
    );
  });
});
