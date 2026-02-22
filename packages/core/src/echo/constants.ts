/**
 * Echo Durable Object で使用する設定定数
 */

/**
 * トークン使用量管理の設定
 */
export const TOKEN_LIMITS = {
  /** 1日あたりの最大トークン使用量 */
  DAILY_LIMIT: 900_000,
  /** 1日あたりの余剰トークン使用量 */
  DAILY_SOFT_LIMIT: 300_000,
  /** 時間按分計算での余裕係数 */
  BUFFER_FACTOR: 1.5,
} as const;

/**
 * アラーム設定
 */
export const ALARM_CONFIG = {
  /** アラーム間隔（分） */
  INTERVAL_MINUTES: 1,
} as const;
