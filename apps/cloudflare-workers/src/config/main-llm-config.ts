export const MAX_TOKENS = 32768;
export const TEMPERATURE = 0.7;
export const TOP_P = 0.8;
export const PRESENCE_PENALTY = 1.5;
export const TOP_K = 20;

export type MainLLMProvider = 'openai' | 'lmstudio';
export type MainLLMApi = 'responses' | 'chat_completions';
export type MainLLMExtraBody = Record<string, unknown>;

export interface MainLLMConfig {
  provider: MainLLMProvider;
  api: MainLLMApi;
  apiKey: string;
  model?: string;
  baseURL?: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  presencePenalty?: number;
  extraBody?: MainLLMExtraBody;
}

interface MainLLMEnv {
  OPENAI_API_KEY: string;
  MAIN_LLM_PROVIDER?: string;
  MAIN_LLM_MODEL?: string;
  MAIN_LLM_BASE_URL?: string;
  MAIN_LLM_API_KEY?: string;
}

/**
 * 空文字の環境変数を未設定として扱う。
 *
 * @param value 環境変数の生値
 * @returns trim 済みの値。空なら `undefined`
 */
function normalizeOptionalEnv(value: string | undefined): string | undefined {
  const normalized = value?.trim();
  return normalized === '' ? undefined : normalized;
}

/**
 * メイン LLM provider 名を正規化する。
 *
 * @param provider 環境変数の provider 指定
 * @returns 利用可能な provider
 */
function resolveProvider(provider: string | undefined): MainLLMProvider {
  const normalized = normalizeOptionalEnv(provider)?.toLowerCase();

  if (normalized === undefined || normalized === 'openai') {
    return 'openai';
  }
  if (normalized === 'lmstudio' || normalized === 'lm-studio') {
    return 'lmstudio';
  }

  throw new Error(
    `Unsupported MAIN_LLM_PROVIDER: ${provider}. Use "openai" or "lmstudio".`
  );
}

/**
 * Worker 環境変数からメイン LLM の接続設定を解決する。
 *
 * OpenAI は既存の `OPENAI_API_KEY` と adapter のデフォルトモデルを使い、
 * LM Studio は OpenAI 互換 Chat Completions endpoint として接続する。
 *
 * @param env Worker の環境変数
 * @returns OpenAI-compatible model adapter に渡す設定
 */
export function resolveMainLLMConfig(env: MainLLMEnv): MainLLMConfig {
  const provider = resolveProvider(env.MAIN_LLM_PROVIDER);
  const model = normalizeOptionalEnv(env.MAIN_LLM_MODEL);

  if (provider === 'openai') {
    return {
      provider,
      api: 'responses',
      apiKey: normalizeOptionalEnv(env.MAIN_LLM_API_KEY) ?? env.OPENAI_API_KEY,
      model,
      baseURL: normalizeOptionalEnv(env.MAIN_LLM_BASE_URL),
    };
  }

  if (model === undefined) {
    throw new Error(
      'MAIN_LLM_MODEL is required when MAIN_LLM_PROVIDER is "lmstudio".'
    );
  }

  const baseURL = normalizeOptionalEnv(env.MAIN_LLM_BASE_URL);
  if (baseURL === undefined) {
    throw new Error(
      'MAIN_LLM_BASE_URL is required when MAIN_LLM_PROVIDER is "lmstudio".'
    );
  }

  const apiKey = normalizeOptionalEnv(env.MAIN_LLM_API_KEY);
  if (apiKey === undefined) {
    throw new Error(
      'MAIN_LLM_API_KEY is required when MAIN_LLM_PROVIDER is "lmstudio".'
    );
  }

  return {
    provider,
    api: 'chat_completions',
    apiKey,
    model,
    baseURL,
    maxTokens: MAX_TOKENS,
    temperature: TEMPERATURE,
    topP: TOP_P,
    presencePenalty: PRESENCE_PENALTY,
    extraBody: {
      top_k: TOP_K,
      chat_template_kwargs: { enable_thinking: false },
    },
  };
}
