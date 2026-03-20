import type { ThoughtLogPort } from '@echo-chamber/core/ports/thought-log';

import { sendChannelMessage } from './api';
import { truncateForDiscord } from './discord-message-utils';

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

  /**
   * Discord channel に思考ログを送る adapter を構築する。
   *
   * @param options Discord token と送信先 channel ID
   */
  constructor(options: DiscordThoughtLogOptions) {
    this.token = options.token;
    this.channelId = options.channelId;
  }

  /**
   * 思考ログを Discord へ送信する。
   *
   * @param content Discord へ流す thought log 本文
   * @returns 送信完了。Discord 送信失敗は握りつぶす
   */
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
