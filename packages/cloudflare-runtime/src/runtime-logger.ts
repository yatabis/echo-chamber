import type { LoggerPort } from '@echo-chamber/core/ports/logger';
import type {
  LogContext,
  LogEntry,
  LogLevel,
  LoggerConfig,
} from '@echo-chamber/core/types/logger';

export interface RuntimeLoggerOptions extends LoggerConfig {
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
 * Cloudflare runtime で使う logger 実装。
 *
 * Discord 送信自体は callback で受け取り、runtime package から
 * Discord adapter への直接依存を避ける。
 */
export class CloudflareRuntimeLogger implements LoggerPort {
  private readonly config: RuntimeLoggerOptions;

  constructor(config: RuntimeLoggerOptions) {
    this.config = config;
  }

  get discordNotifyLevel(): LogLevel {
    return this.config.discordNotifyLevel;
  }

  async debug(message: string, context?: LogContext): Promise<void> {
    await this.log('debug', message, context);
  }

  async info(message: string, context?: LogContext): Promise<void> {
    await this.log('info', message, context);
  }

  async warn(message: string, context?: LogContext): Promise<void> {
    await this.log('warn', message, context);
  }

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
