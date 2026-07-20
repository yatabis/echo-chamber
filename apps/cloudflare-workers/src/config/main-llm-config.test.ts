import { describe, expect, it } from 'vitest';

import {
  getEchoInstanceDefinition,
  type EchoInstanceDefinition,
} from '@echo-chamber/core/echo/instance-definitions';

import {
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
      model: 'gpt-5.5',
      baseURL: undefined,
      reasoningEffort: undefined,
      runtimeProfile: 'standard',
    });
  });

  it('model 未指定時は GPT-5.6 を既定値として返す', () => {
    expect(
      resolveMainLLMConfig(
        createEnv(),
        createDefinition({
          mainLlm: {},
        })
      )
    ).toMatchObject({
      provider: 'openai',
      api: 'responses',
      model: 'gpt-5.6',
    });
  });

  it('instance definition の OpenAI reasoning effort を返す', () => {
    expect(
      resolveMainLLMConfig(createEnv(), getEchoInstanceDefinition('marie'))
    ).toMatchObject({
      provider: 'openai',
      api: 'responses',
      model: 'gpt-5.6-luna',
      reasoningEffort: 'low',
    });
  });

  it('OpenAI のモデル、API key、reasoning effort を instance 環境変数で上書きできる', () => {
    expect(
      resolveMainLLMConfig(
        createEnv({
          RIN_MAIN_LLM_PROVIDER: 'openai',
          RIN_MAIN_LLM_API_KEY: 'main-Key',
          RIN_MAIN_LLM_MODEL: 'GPT-5.4',
          RIN_MAIN_LLM_REASONING_EFFORT: 'HIGH',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toEqual({
      provider: 'openai',
      api: 'responses',
      apiKey: 'main-Key',
      model: 'GPT-5.4',
      baseURL: undefined,
      reasoningEffort: 'high',
      runtimeProfile: 'standard',
    });
  });

  it('definition が空の項目は global 環境変数で補完できる', () => {
    expect(
      resolveMainLLMConfig(
        createEnv({
          MAIN_LLM_PROVIDER: 'openai',
          MAIN_LLM_MODEL: 'GPT-5.2',
          MAIN_LLM_REASONING_EFFORT: 'medium',
        }),
        createDefinition({
          mainLlm: {},
        })
      )
    ).toMatchObject({
      provider: 'openai',
      model: 'GPT-5.2',
      reasoningEffort: 'medium',
    });
  });

  it('OpenAI-compatible definition は Chat Completions API 用の固定パラメータを返す', () => {
    expect(
      resolveMainLLMConfig(
        createEnv({
          RIN_MAIN_LLM_API_KEY: 'sk-lm-AbC123',
        }),
        createDefinition({
          mainLlm: {
            provider: 'openai-compatible',
            model: 'openai/gpt-oss-20b',
            baseURL: 'http://localhost:1234/V1',
          },
        })
      )
    ).toEqual({
      provider: 'openai-compatible',
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
      runtimeProfile: 'standard',
    });
  });

  it('OpenAI-compatible の接続設定を instance 環境変数で上書きできる', () => {
    expect(
      resolveMainLLMConfig(
        createEnv({
          MARIE_MAIN_LLM_PROVIDER: 'openai-compatible',
          MARIE_MAIN_LLM_API_KEY: 'local-key',
          MARIE_MAIN_LLM_MODEL: 'local-model',
          MARIE_MAIN_LLM_BASE_URL: 'http://127.0.0.1:4321/v1',
        }),
        getEchoInstanceDefinition('marie')
      )
    ).toEqual({
      provider: 'openai-compatible',
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
      runtimeProfile: 'standard',
    });
  });

  it('OpenAI-compatible 指定だけでは cache profile を有効にしない', () => {
    expect(
      resolveMainLLMConfig(
        createEnv({
          RIN_MAIN_LLM_PROVIDER: 'openai-compatible',
          RIN_MAIN_LLM_MODEL: 'qwen3.6-27b',
          RIN_MAIN_LLM_BASE_URL: 'http://localhost:8000/v1',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toEqual({
      provider: 'openai-compatible',
      api: 'chat_completions',
      apiKey: 'not-needed',
      model: 'qwen3.6-27b',
      baseURL: 'http://localhost:8000/v1',
      maxTokens: MAX_TOKENS,
      temperature: TEMPERATURE,
      topP: TOP_P,
      presencePenalty: PRESENCE_PENALTY,
      extraBody: {
        top_k: TOP_K,
        chat_template_kwargs: { enable_thinking: false },
      },
      runtimeProfile: 'standard',
    });
  });

  it('provider を OpenAI-compatible に切り替えた場合は OpenAI definition model を流用しない', () => {
    expect(() =>
      resolveMainLLMConfig(
        createEnv({
          RIN_MAIN_LLM_PROVIDER: 'openai-compatible',
          RIN_MAIN_LLM_BASE_URL: 'http://localhost:1234/v1',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toThrow(
      'MAIN_LLM_MODEL is required when MAIN_LLM_PROVIDER is "openai-compatible".'
    );
  });

  it('provider を OpenAI-compatible に切り替えた場合は global model fallback を使える', () => {
    expect(
      resolveMainLLMConfig(
        createEnv({
          RIN_MAIN_LLM_PROVIDER: 'openai-compatible',
          RIN_MAIN_LLM_BASE_URL: 'http://localhost:1234/v1',
          MAIN_LLM_MODEL: 'local-global-model',
          RIN_MAIN_LLM_REASONING_EFFORT: 'deep',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toMatchObject({
      provider: 'openai-compatible',
      model: 'local-global-model',
      baseURL: 'http://localhost:1234/v1',
      apiKey: 'not-needed',
    });
  });

  it('OpenAI-compatible 指定でモデルが無い場合はエラーにする', () => {
    expect(() =>
      resolveMainLLMConfig(
        createEnv({
          MAIN_LLM_PROVIDER: 'openai-compatible',
        }),
        createDefinition({ mainLlm: {} })
      )
    ).toThrow(
      'MAIN_LLM_MODEL is required when MAIN_LLM_PROVIDER is "openai-compatible".'
    );
  });

  it('OpenAI-compatible 指定で API key が無ければ dummy key を使う', () => {
    expect(
      resolveMainLLMConfig(
        createEnv({
          RIN_MAIN_LLM_PROVIDER: 'openai-compatible',
          RIN_MAIN_LLM_MODEL: 'local-model',
          RIN_MAIN_LLM_BASE_URL: 'http://localhost:1234/v1',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toMatchObject({
      provider: 'openai-compatible',
      apiKey: 'not-needed',
      runtimeProfile: 'standard',
    });
  });

  it('OpenAI-compatible 指定で base URL が無い場合はエラーにする', () => {
    expect(() =>
      resolveMainLLMConfig(
        createEnv({
          RIN_MAIN_LLM_PROVIDER: 'openai-compatible',
          RIN_MAIN_LLM_MODEL: 'local-model',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toThrow(
      'MAIN_LLM_BASE_URL is required when MAIN_LLM_PROVIDER is "openai-compatible".'
    );
  });

  it('E.C.H.O. session cache runtime profile は明示指定した場合だけ返す', () => {
    expect(
      resolveMainLLMConfig(
        createEnv({
          RIN_MAIN_LLM_PROVIDER: 'openai-compatible',
          RIN_MAIN_LLM_MODEL: 'qwen3.6-27b',
          RIN_MAIN_LLM_BASE_URL: 'http://localhost:8000/v1',
          RIN_MAIN_LLM_RUNTIME_PROFILE: 'ECHO-SESSION-CACHE-V1',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toMatchObject({
      provider: 'openai-compatible',
      runtimeProfile: 'echo-session-cache-v1',
    });
  });

  it('OpenAI Responses API では session cache runtime profile を拒否する', () => {
    expect(() =>
      resolveMainLLMConfig(
        createEnv({
          RIN_MAIN_LLM_RUNTIME_PROFILE: 'echo-session-cache-v1',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toThrow(
      'MAIN_LLM_RUNTIME_PROFILE "echo-session-cache-v1" requires MAIN_LLM_PROVIDER "openai-compatible".'
    );
  });

  it('未対応 runtime profile はエラーにする', () => {
    expect(() =>
      resolveMainLLMConfig(
        createEnv({
          RIN_MAIN_LLM_RUNTIME_PROFILE: 'custom-cache',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toThrow(
      'Unsupported MAIN_LLM_RUNTIME_PROFILE: custom-cache. Use "standard" or "echo-session-cache-v1".'
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
      'Unsupported MAIN_LLM_PROVIDER: anthropic. Use "openai" or "openai-compatible".'
    );
  });

  it('未対応 reasoning effort はエラーにする', () => {
    expect(() =>
      resolveMainLLMConfig(
        createEnv({
          RIN_MAIN_LLM_REASONING_EFFORT: 'deep',
        }),
        getEchoInstanceDefinition('rin')
      )
    ).toThrow(
      'Unsupported MAIN_LLM_REASONING_EFFORT: deep. Use "none", "minimal", "low", "medium", "high", or "xhigh".'
    );
  });
});
