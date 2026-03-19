import { CloudflareRuntimeLogger } from '@echo-chamber/cloudflare-runtime/runtime-logger';
import type { LogContext, LogLevel } from '@echo-chamber/core';

import { sendChannelMessage } from '../discord/client';

export class Logger extends CloudflareRuntimeLogger {
  async debug(message: string, context?: LogContext): Promise<void> {
    await super.debug(message, context);
  }

  async info(message: string, context?: LogContext): Promise<void> {
    await super.info(message, context);
  }

  async warn(message: string, context?: LogContext): Promise<void> {
    await super.warn(message, context);
  }

  async log(
    level: LogLevel,
    message: string,
    context?: LogContext
  ): Promise<void> {
    await super.log(level, message, context);
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
