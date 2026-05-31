import type {
  EchoInstanceDefinition,
  EchoMainLLMProvider,
} from '@echo-chamber/core/echo/instance-definitions';

export const MAX_TOKENS = 32768;
export const TEMPERATURE = 0.7;
export const TOP_P = 0.8;
export const PRESENCE_PENALTY = 1.5;
export const TOP_K = 20;
export const DEFAULT_OPENAI_RESPONSES_MODEL = 'gpt-5.5';

export type MainLLMProvider = EchoMainLLMProvider;
export type MainLLMApi = 'responses' | 'chat_completions';
export type MainLLMExtraBody = Record<string, unknown>;

type MainLLMConfigKey = 'provider' | 'model' | 'baseURL' | 'apiKey';

const MAIN_LLM_ENV_KEYS = {
  provider: 'PROVIDER',
  model: 'MODEL',
  baseURL: 'BASE_URL',
  apiKey: 'API_KEY',
} as const satisfies Record<MainLLMConfigKey, string>;

interface ResolveMainLLMValueOptions {
  skipDefinition?: boolean;
}

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

/**
 * main LLM 設定解決で参照する Worker 環境変数。
 *
 * instance 固有キーは `readEnv` で動的に参照するため、
 * ここでは共通 fallback と OpenAI の既定 API key だけを明示する。
 */
export interface MainLLMEnv {
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
 * instance 固有の環境変数 prefix を返す。
 *
 * @param definition Echo instance 定義
 * @returns `RIN` や `MARIE` のような環境変数 prefix
 */
function getInstanceEnvPrefix(definition: EchoInstanceDefinition): string {
  return definition.id.toUpperCase();
}

/**
 * Worker env から任意キーの値を読み出す。
 *
 * @param env Worker 環境変数
 * @param key 読み出す環境変数名
 * @returns 環境変数値。未設定なら `undefined`
 */
function readEnv(env: MainLLMEnv, key: string): string | undefined {
  return (env as unknown as Record<string, string | undefined>)[key];
}

/**
 * instance 定義に含まれる main LLM 設定値を返す。
 *
 * @param definition Echo instance 定義
 * @param key 解決対象の設定キー
 * @returns instance 定義の設定値。未設定なら `undefined`
 */
function resolveMainLLMDefinitionValue(
  definition: EchoInstanceDefinition,
  key: MainLLMConfigKey
): string | undefined {
  if (key === 'apiKey') {
    return undefined;
  }

  return definition.mainLlm[key];
}

/**
 * instance 固有 env、instance 定義、global env の順に main LLM 設定値を解決する。
 *
 * @param env Worker 環境変数
 * @param definition Echo instance 定義
 * @param key 解決対象の設定キー
 * @param options definition fallback の扱い
 * @returns 解決した設定値。未設定なら `undefined`
 */
function resolveMainLLMValue(
  env: MainLLMEnv,
  definition: EchoInstanceDefinition,
  key: MainLLMConfigKey,
  options: ResolveMainLLMValueOptions = {}
): string | undefined {
  const envKey = MAIN_LLM_ENV_KEYS[key];
  const instanceEnvValue = normalizeOptionalEnv(
    readEnv(env, `${getInstanceEnvPrefix(definition)}_MAIN_LLM_${envKey}`)
  );
  if (instanceEnvValue !== undefined) {
    return instanceEnvValue;
  }

  if (options.skipDefinition !== true) {
    const definitionValue = resolveMainLLMDefinitionValue(definition, key);
    if (definitionValue !== undefined) {
      return definitionValue;
    }
  }

  const globalEnvValue = normalizeOptionalEnv(
    readEnv(env, `MAIN_LLM_${envKey}`)
  );
  if (globalEnvValue !== undefined) {
    return globalEnvValue;
  }

  return undefined;
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
 * provider 固有の model/baseURL を instance 定義から流用できるかを返す。
 *
 * @param definition Echo instance 定義
 * @param provider 解決済み provider
 * @returns definition の provider と一致する場合は `true`
 */
function canUseDefinitionProviderDetails(
  definition: EchoInstanceDefinition,
  provider: MainLLMProvider
): boolean {
  return (
    definition.mainLlm.provider === undefined ||
    definition.mainLlm.provider === provider
  );
}

/**
 * Worker 環境変数と instance 定義からメイン LLM の接続設定を解決する。
 *
 * provider/model/baseURL は instance 固有 env、instance 定義、global env の順に解決する。
 * API key は instance 固有 env、global env、`OPENAI_API_KEY` の順に解決する。
 * LM Studio は OpenAI 互換 Chat Completions endpoint として接続する。
 *
 * @param env Worker の環境変数
 * @param definition Echo instance 定義
 * @returns OpenAI-compatible model adapter に渡す設定
 */
export function resolveMainLLMConfig(
  env: MainLLMEnv,
  definition: EchoInstanceDefinition
): MainLLMConfig {
  const provider = resolveProvider(
    resolveMainLLMValue(env, definition, 'provider')
  );
  const skipDefinitionProviderDetails = !canUseDefinitionProviderDetails(
    definition,
    provider
  );
  const model = resolveMainLLMValue(env, definition, 'model', {
    skipDefinition: skipDefinitionProviderDetails,
  });
  const baseURL = resolveMainLLMValue(env, definition, 'baseURL', {
    skipDefinition: skipDefinitionProviderDetails,
  });
  const providerApiKey = resolveMainLLMValue(env, definition, 'apiKey');

  if (provider === 'openai') {
    return {
      provider,
      api: 'responses',
      apiKey: providerApiKey ?? env.OPENAI_API_KEY,
      model: model ?? DEFAULT_OPENAI_RESPONSES_MODEL,
      baseURL,
    };
  }

  if (model === undefined) {
    throw new Error(
      'MAIN_LLM_MODEL is required when MAIN_LLM_PROVIDER is "lmstudio".'
    );
  }

  if (baseURL === undefined) {
    throw new Error(
      'MAIN_LLM_BASE_URL is required when MAIN_LLM_PROVIDER is "lmstudio".'
    );
  }

  if (providerApiKey === undefined) {
    throw new Error(
      'MAIN_LLM_API_KEY is required when MAIN_LLM_PROVIDER is "lmstudio".'
    );
  }

  return {
    provider,
    api: 'chat_completions',
    apiKey: providerApiKey,
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
