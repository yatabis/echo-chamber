import { describe, expect, it } from 'vitest';

import {
  MAX_TOKENS,
  PRESENCE_PENALTY,
  TEMPERATURE,
  TOP_K,
  TOP_P,
  resolveMainLLMConfig,
} from './main-llm-config';

function createEnv(overrides: Partial<Env> = {}): Env {
  return {
    OPENAI_API_KEY: 'openai-key',
    ...overrides,
  } as Env;
}

describe('resolveMainLLMConfig', () => {
  it('未指定時は OpenAI 設定を返す', () => {
    expect(resolveMainLLMConfig(createEnv())).toEqual({
      provider: 'openai',
      api: 'responses',
      apiKey: 'openai-key',
      model: undefined,
      baseURL: undefined,
    });
  });

  it('OpenAI のモデルと API key を環境変数で上書きできる', () => {
    expect(
      resolveMainLLMConfig(
        createEnv({
          MAIN_LLM_PROVIDER: 'openai',
          MAIN_LLM_API_KEY: 'main-Key',
          MAIN_LLM_MODEL: 'GPT-5.4',
        })
      )
    ).toEqual({
      provider: 'openai',
      api: 'responses',
      apiKey: 'main-Key',
      model: 'GPT-5.4',
      baseURL: undefined,
    });
  });

  it('LM Studio は Chat Completions API 用の固定パラメータを返す', () => {
    expect(
      resolveMainLLMConfig(
        createEnv({
          MAIN_LLM_PROVIDER: 'lmstudio',
          MAIN_LLM_MODEL: 'openai/gpt-oss-20b',
          MAIN_LLM_API_KEY: 'sk-lm-AbC123',
          MAIN_LLM_BASE_URL: 'http://localhost:1234/V1',
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

  it('LM Studio の baseURL と API key を環境変数で上書きできる', () => {
    expect(
      resolveMainLLMConfig(
        createEnv({
          MAIN_LLM_PROVIDER: 'lm-studio',
          MAIN_LLM_API_KEY: 'local-key',
          MAIN_LLM_MODEL: 'local-model',
          MAIN_LLM_BASE_URL: 'http://127.0.0.1:4321/v1',
        })
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

  it('LM Studio 指定でモデルが無い場合はエラーにする', () => {
    expect(() =>
      resolveMainLLMConfig(
        createEnv({
          MAIN_LLM_PROVIDER: 'lmstudio',
        })
      )
    ).toThrow(
      'MAIN_LLM_MODEL is required when MAIN_LLM_PROVIDER is "lmstudio".'
    );
  });

  it('LM Studio 指定で API key が無い場合はエラーにする', () => {
    expect(() =>
      resolveMainLLMConfig(
        createEnv({
          MAIN_LLM_PROVIDER: 'lmstudio',
          MAIN_LLM_MODEL: 'local-model',
          MAIN_LLM_BASE_URL: 'http://localhost:1234/v1',
        })
      )
    ).toThrow(
      'MAIN_LLM_API_KEY is required when MAIN_LLM_PROVIDER is "lmstudio".'
    );
  });

  it('LM Studio 指定で base URL が無い場合はエラーにする', () => {
    expect(() =>
      resolveMainLLMConfig(
        createEnv({
          MAIN_LLM_PROVIDER: 'lmstudio',
          MAIN_LLM_MODEL: 'local-model',
          MAIN_LLM_API_KEY: 'local-key',
        })
      )
    ).toThrow(
      'MAIN_LLM_BASE_URL is required when MAIN_LLM_PROVIDER is "lmstudio".'
    );
  });

  it('未対応 provider はエラーにする', () => {
    expect(() =>
      resolveMainLLMConfig(
        createEnv({
          MAIN_LLM_PROVIDER: 'anthropic',
        })
      )
    ).toThrow(
      'Unsupported MAIN_LLM_PROVIDER: anthropic. Use "openai" or "lmstudio".'
    );
  });
});
