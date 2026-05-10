import type { EchoInstanceDefinition } from '@echo-chamber/core/echo/instance-definitions';

export interface TokenLimitConfig {
  dailyHardLimit: number;
  dailySoftLimit: number;
  hardLimitBufferFactor: number;
}

type TokenLimitEnv = object;

type TokenLimitConfigKey =
  | 'dailyHardLimit'
  | 'dailySoftLimit'
  | 'hardLimitBufferFactor';

const TOKEN_LIMIT_ENV_KEYS = {
  dailyHardLimit: 'DAILY_HARD_TOKEN_LIMIT',
  dailySoftLimit: 'DAILY_SOFT_TOKEN_LIMIT',
  hardLimitBufferFactor: 'HARD_TOKEN_LIMIT_BUFFER_FACTOR',
} as const satisfies Record<TokenLimitConfigKey, string>;

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
function readEnv(env: TokenLimitEnv, key: string): string | undefined {
  return (env as unknown as Record<string, string | undefined>)[key];
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
 * token limit の環境変数値を正の有限数へ変換する。
 *
 * @param name 環境変数名
 * @param value 環境変数の正規化済み文字列
 * @returns 正の有限数
 */
function parsePositiveNumberEnv(name: string, value: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive number.`);
  }

  return parsed;
}

/**
 * instance 固有 env、instance 定義、global env の順に token limit 値を解決する。
 *
 * @param env Worker 環境変数
 * @param definition Echo instance 定義
 * @param key 解決対象の token limit key
 * @returns 解決した正の有限数
 */
function resolveTokenLimitValue(
  env: TokenLimitEnv,
  definition: EchoInstanceDefinition,
  key: TokenLimitConfigKey
): number {
  const envKey = TOKEN_LIMIT_ENV_KEYS[key];
  const instanceEnvKey = `${getInstanceEnvPrefix(definition)}_${envKey}`;
  const instanceEnvValue = normalizeOptionalEnv(readEnv(env, instanceEnvKey));
  if (instanceEnvValue !== undefined) {
    return parsePositiveNumberEnv(instanceEnvKey, instanceEnvValue);
  }

  const definitionValue = definition.tokenLimits[key];
  if (
    definitionValue !== undefined &&
    Number.isFinite(definitionValue) &&
    definitionValue > 0
  ) {
    return definitionValue;
  }

  const globalEnvValue = normalizeOptionalEnv(readEnv(env, envKey));
  if (globalEnvValue !== undefined) {
    return parsePositiveNumberEnv(envKey, globalEnvValue);
  }

  throw new Error(`Token limit is not configured: ${key}.`);
}

/**
 * token limit 設定同士の整合性を検証する。
 *
 * @param config 解決済み token limit 設定
 */
function assertTokenLimitConfig(config: TokenLimitConfig): void {
  if (config.dailySoftLimit > config.dailyHardLimit) {
    throw new Error(
      'DAILY_SOFT_TOKEN_LIMIT must be less than or equal to DAILY_HARD_TOKEN_LIMIT.'
    );
  }
}

/**
 * Worker 環境変数と instance 定義から token limit 設定を解決する。
 *
 * @param env Worker 環境変数
 * @param definition Echo instance 定義
 * @returns 実行判定で使う token limit 設定
 */
export function resolveTokenLimitConfig(
  env: TokenLimitEnv,
  definition: EchoInstanceDefinition
): TokenLimitConfig {
  const config: TokenLimitConfig = {
    dailyHardLimit: resolveTokenLimitValue(env, definition, 'dailyHardLimit'),
    dailySoftLimit: resolveTokenLimitValue(env, definition, 'dailySoftLimit'),
    hardLimitBufferFactor: resolveTokenLimitValue(
      env,
      definition,
      'hardLimitBufferFactor'
    ),
  };

  assertTokenLimitConfig(config);

  return config;
}
