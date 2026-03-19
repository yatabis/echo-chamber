import type { ThoughtLogPort } from '@echo-chamber/core/ports/thought-log';

import { sendChannelMessage } from './api';

export interface DiscordThoughtLogOptions {
  token: string;
  channelId: string;
}

/**
 * Discord channel へ思考ログを送る `ThoughtLogPort` 実装。
 */
export class DiscordThoughtLog implements ThoughtLogPort {
  private readonly token: string;
  private readonly channelId: string;

  constructor(options: DiscordThoughtLogOptions) {
    this.token = options.token;
    this.channelId = options.channelId;
  }

  async send(content: string): Promise<void> {
    if (content === '') {
      return;
    }

    const message = truncateForDiscord(content);

    try {
      await sendChannelMessage(this.token, this.channelId, {
        content: message,
      });
    } catch (error) {
      console.error('Failed to send thinking to Discord:', error);
    }
  }
}

function truncateForDiscord(content: string): string {
  const maxLength = 2000;
  if (content.length <= maxLength) {
    return content;
  }

  return `${content.substring(0, maxLength - 15)}...(truncated)`;
}
