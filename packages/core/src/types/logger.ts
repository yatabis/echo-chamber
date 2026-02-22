export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export type LogContext = Record<string, unknown>;

export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  message: string;
  context?: LogContext;
}

export interface LoggerConfig {
  /**
   * Discord通知の閾値
   * このレベル以上のログのみDiscordに送信
   * コンソール出力には影響しない（全レベル出力）
   */
  discordNotifyLevel: LogLevel;

  /**
   * Discord送信設定
   */
  discord?: {
    token: string;
    channelId: string;
  };
}
