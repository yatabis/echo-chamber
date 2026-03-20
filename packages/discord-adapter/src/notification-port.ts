import type { NotificationPort } from '@echo-chamber/core/ports/notification';

import { getNotificationDetails } from './notification-utils';

export interface DiscordNotificationPortOptions {
  token: string;
  channelId: string;
}

/**
 * Discord channel を `NotificationPort` として扱う adapter。
 *
 * @param options Discord token と対象 channel ID
 * @returns `NotificationPort` 実装
 */
export function createDiscordNotificationPort(
  options: DiscordNotificationPortOptions
): NotificationPort {
  return {
    async getNotificationSummary(): ReturnType<
      NotificationPort['getNotificationSummary']
    > {
      return await getNotificationDetails(options.token, options.channelId);
    },
  };
}
