/**
 * Echo Durable Object で使用する設定定数
 */

/**
 * トークン使用量管理の設定
 */
export const TOKEN_LIMITS = {
  /** 1日あたりのハード上限トークン使用量 */
  DAILY_HARD_LIMIT: 500_000,
  /** 時間按分で使う1日あたりのソフト上限トークン使用量 */
  DAILY_SOFT_LIMIT: 300_000,
  /** ハードリミットの時間按分計算で使う余裕係数 */
  HARD_LIMIT_BUFFER_FACTOR: 1.5,
} as const;

/**
 * 起動判定と再開タイミングに関する設定
 */
export const SCHEDULING_CONFIG = {
  /** soft limit 未満でも next_wake_at が近すぎる場合は待機する閾値（分） */
  SOFT_LIMIT_NEXT_WAKE_AT_WINDOW_MINUTES: 10,
} as const;

/**
 * アラーム設定
 */
export const ALARM_CONFIG = {
  /** アラーム間隔（分） */
  INTERVAL_MINUTES: 1,
} as const;
