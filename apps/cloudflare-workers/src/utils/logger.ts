import type { LoggerPort } from '@echo-chamber/core/ports/logger';
import type {
  LogContext,
  LogEntry,
  LogLevel,
} from '@echo-chamber/core/types/logger';
import { sendChannelMessage } from '@echo-chamber/discord-adapter/api';

export interface LoggerOptions {
  /**
   * Discord 通知のしきい値。
   *
   * このレベル以上のログだけを Discord に送る。
   * console 出力は全レベルで行う。
   */
  discordNotifyLevel: LogLevel;

  /**
   * Discord 通知先の設定。
   */
  discord?: {
    token: string;
    channelId: string;
  };

  /**
   * Discord への実送信を行う callback。
   *
   * @param token Discord bot token
   * @param channelId 通知先 channel ID
   * @param content 送信するログ本文
   * @returns 送信完了
   */
  sendDiscordMessage?(
    token: string,
    channelId: string,
    content: string
  ): Promise<void>;
}

const LOG_LEVELS: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

/**
 * Worker runtime で使う logger 実装。
 *
 * Discord 送信自体は callback で受け取り、logger 実装を transport から切り離す。
 */
export class Logger implements LoggerPort {
  private readonly config: LoggerOptions;

  /**
   * Worker runtime 用 logger を構築する。
   *
   * @param config ログレベル設定と任意の Discord 通知設定
   */
  constructor(config: LoggerOptions) {
    this.config = config;
  }

  /**
   * Discord 通知に使うしきい値を返す。
   *
   * @returns Discord 通知対象となる最小ログレベル
   */
  get discordNotifyLevel(): LogLevel {
    return this.config.discordNotifyLevel;
  }

  /**
   * debug ログを出力する。
   *
   * @param message ログ本文
   * @param context 追加の構造化コンテキスト
   * @returns 出力完了
   */
  async debug(message: string, context?: LogContext): Promise<void> {
    await this.log('debug', message, context);
  }

  /**
   * info ログを出力する。
   *
   * @param message ログ本文
   * @param context 追加の構造化コンテキスト
   * @returns 出力完了
   */
  async info(message: string, context?: LogContext): Promise<void> {
    await this.log('info', message, context);
  }

  /**
   * warn ログを出力する。
   *
   * @param message ログ本文
   * @param context 追加の構造化コンテキスト
   * @returns 出力完了
   */
  async warn(message: string, context?: LogContext): Promise<void> {
    await this.log('warn', message, context);
  }

  /**
   * error オブジェクト付きの error ログを出力する。
   *
   * @param message ログ本文
   * @param error 任意の Error オブジェクト
   * @returns 出力完了
   */
  async error(message: string, error?: Error): Promise<void> {
    const errorContext = error
      ? {
          error: {
            name: error.name,
            message: error.message,
            stack: error.stack,
          },
        }
      : undefined;

    await this.log('error', message, errorContext);
  }

  /**
   * 指定レベルの構造化ログを console と必要に応じて Discord へ出力する。
   *
   * @param level ログレベル
   * @param message ログ本文
   * @param context 追加の構造化コンテキスト
   * @returns 出力完了
   */
  async log(
    level: LogLevel,
    message: string,
    context?: LogContext
  ): Promise<void> {
    const logEntry: LogEntry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      context,
    };

    const logString = JSON.stringify(logEntry);
    switch (level) {
      case 'debug':
        console.debug(logString);
        break;
      case 'info':
        console.info(logString);
        break;
      case 'warn':
        console.warn(logString);
        break;
      case 'error':
        console.error(logString);
        break;
    }

    if (
      this.config.discord &&
      this.config.sendDiscordMessage &&
      LOG_LEVELS[level] >= LOG_LEVELS[this.config.discordNotifyLevel]
    ) {
      try {
        await this.sendToDiscord(logEntry);
      } catch (error: unknown) {
        console.error('Failed to send log to Discord:', error);
      }
    }
  }

  /**
   * 1 件のログエントリを Discord 通知用メッセージへ整形して送る。
   *
   * @param entry 送信対象のログエントリ
   * @returns 送信完了
   */
  private async sendToDiscord(entry: LogEntry): Promise<void> {
    const discord = this.config.discord;
    if (!discord || !this.config.sendDiscordMessage) {
      return;
    }

    const levelEmoji: Record<LogLevel, string> = {
      debug: '🔍',
      info: 'ℹ️',
      warn: '⚠️',
      error: '🚨',
    };

    const contextStr =
      entry.context && Object.keys(entry.context).length > 0
        ? `\n\`\`\`json\n${JSON.stringify(entry.context, null, 2)}\n\`\`\``
        : '';

    const baseMessage = `${levelEmoji[entry.level]} **[${entry.level.toUpperCase()}]** ${entry.message}${contextStr}`;
    const discordMessage =
      baseMessage.length > 2000
        ? `${baseMessage.substring(0, 1980)}...(truncated)`
        : baseMessage;

    await this.config.sendDiscordMessage(
      discord.token,
      discord.channelId,
      discordMessage
    );
  }
}

/**
 * 環境変数からLoggerを生成
 */
export function createLogger(env: Env): Logger {
  const isLocal = env.ENVIRONMENT === 'local';

  return new Logger({
    discordNotifyLevel: isLocal ? 'debug' : 'info',
    discord: {
      token: env.DISCORD_BOT_TOKEN,
      channelId: env.LOG_CHANNEL_ID,
    },
    sendDiscordMessage: async (
      token: string,
      channelId: string,
      content: string
    ): Promise<void> => {
      await sendChannelMessage(token, channelId, { content });
    },
  });
}
