import type { ChatChannel } from '@echo-chamber/core/ports/chat';
import type { NotificationPort } from '@echo-chamber/core/ports/notification';

import { getNotificationDetails } from './notification-utils';

interface DiscordNotificationChannel extends ChatChannel {
  discordChannelId: string;
}

export interface DiscordNotificationPortOptions {
  token: string;
  channels: readonly DiscordNotificationChannel[];
}

/**
 * Discord channel を `NotificationPort` として扱う adapter。
 *
 * @param options Discord token と対象 channel 定義
 * @returns `NotificationPort` 実装
 */
export function createDiscordNotificationPort(
  options: DiscordNotificationPortOptions
): NotificationPort {
  return {
    async getNotificationSummary(): ReturnType<
      NotificationPort['getNotificationSummary']
    > {
      return await Promise.all(
        options.channels.map(async (channel) => ({
          channel: {
            key: channel.key,
            displayName: channel.displayName,
            description: channel.description,
          },
          ...(await getNotificationDetails(
            options.token,
            channel.discordChannelId
          )),
        }))
      );
    },
  };
}
