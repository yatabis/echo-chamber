import type {
  EchoInstanceDefinition,
  EchoModelReasoningEffort,
} from '@echo-chamber/core/echo/instance-definitions';

/** 現行 Hosted backend で Memory / Emotion が共有する最終 fallback model。 */
export const DEFAULT_COGNITIVE_MODULE_MODEL = 'gpt-5.6-luna';
export const DEFAULT_COGNITIVE_MODULE_REASONING_EFFORT =
  'low' as const satisfies EchoModelReasoningEffort;

type CognitiveModuleConfigKey = 'model' | 'reasoningEffort';

const COGNITIVE_MODULE_ENV_KEYS = {
  model: 'MODEL',
  reasoningEffort: 'REASONING_EFFORT',
} as const satisfies Record<CognitiveModuleConfigKey, string>;

const COGNITIVE_MODULE_REASONING_EFFORTS = [
  'none',
  'minimal',
  'low',
  'medium',
  'high',
  'xhigh',
  'max',
] as const satisfies readonly EchoModelReasoningEffort[];

/** Cognitive submodule が OpenAI Responses API へ接続する設定。 */
export interface CognitiveModuleConfig {
  provider: 'openai';
  api: 'responses';
  apiKey: string;
  model: string;
  reasoningEffort: EchoModelReasoningEffort;
}

/** 既存 OpenAI secret と任意の全 instance 共通 fallback。 */
export interface CognitiveModuleEnv {
  OPENAI_API_KEY: string;
  COGNITIVE_MODULE_MODEL?: string;
  COGNITIVE_MODULE_REASONING_EFFORT?: string;
}

/**
 * 空文字の optional env を未設定として扱う。
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
 * Worker env から動的な instance 固有キーを読み出す。
 *
 * @param env Worker 環境変数
 * @param key 読み出す環境変数名
 * @returns 環境変数値。未設定なら `undefined`
 */
function readEnv(env: CognitiveModuleEnv, key: string): string | undefined {
  return (env as unknown as Record<string, string | undefined>)[key];
}

/**
 * instance 固有 env、instance 定義、global env の順に設定値を解決する。
 *
 * @param env Worker 環境変数
 * @param definition Echo instance 定義
 * @param key 解決対象の設定キー
 * @returns 解決した設定値。すべて未設定なら `undefined`
 */
function resolveCognitiveModuleValue(
  env: CognitiveModuleEnv,
  definition: EchoInstanceDefinition,
  key: CognitiveModuleConfigKey
): string | undefined {
  const envKey = COGNITIVE_MODULE_ENV_KEYS[key];
  const instanceEnvValue = normalizeOptionalEnv(
    readEnv(
      env,
      `${getInstanceEnvPrefix(definition)}_COGNITIVE_MODULE_${envKey}`
    )
  );
  if (instanceEnvValue !== undefined) {
    return instanceEnvValue;
  }

  const definitionValue = definition.cognitiveModules[key];
  if (definitionValue !== undefined) {
    return definitionValue;
  }

  return normalizeOptionalEnv(readEnv(env, `COGNITIVE_MODULE_${envKey}`));
}

/**
 * Cognitive Module の reasoning effort 設定値へ正規化する。
 *
 * @param value instance 定義または環境変数から解決した値
 * @returns 正規化済み reasoning effort
 */
function resolveReasoningEffort(
  value: string | undefined
): EchoModelReasoningEffort {
  const normalized = normalizeOptionalEnv(value)?.toLowerCase();
  if (normalized === undefined) {
    return DEFAULT_COGNITIVE_MODULE_REASONING_EFFORT;
  }

  if (
    COGNITIVE_MODULE_REASONING_EFFORTS.includes(
      normalized as EchoModelReasoningEffort
    )
  ) {
    return normalized as EchoModelReasoningEffort;
  }

  const supportedValues = COGNITIVE_MODULE_REASONING_EFFORTS.map(
    (effort) => `"${effort}"`
  ).join(', ');
  throw new Error(
    `Unsupported COGNITIVE_MODULE_REASONING_EFFORT: ${value}. Use one of: ${supportedValues}.`
  );
}

/**
 * Cognitive submodule 用の OpenAI 設定を解決する。
 *
 * API key は専用 secret を増やさず既存 `OPENAI_API_KEY` を再利用する。
 * model / reasoning effort は instance 固有 env、instance 定義、global env、
 * Worker 既定値の順に解決する。
 *
 * @param env Worker 環境変数
 * @param definition Echo instance 定義
 * @returns Memory / Emotion model adapter に渡す設定
 */
export function resolveCognitiveModuleConfig(
  env: CognitiveModuleEnv,
  definition: EchoInstanceDefinition
): CognitiveModuleConfig {
  if (env.OPENAI_API_KEY.trim() === '') {
    throw new Error('OPENAI_API_KEY must not be empty');
  }

  const model =
    resolveCognitiveModuleValue(env, definition, 'model') ??
    DEFAULT_COGNITIVE_MODULE_MODEL;
  const reasoningEffort = resolveReasoningEffort(
    resolveCognitiveModuleValue(env, definition, 'reasoningEffort')
  );

  return {
    provider: 'openai',
    api: 'responses',
    apiKey: env.OPENAI_API_KEY,
    model,
    reasoningEffort,
  };
}
