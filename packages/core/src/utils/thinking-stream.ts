import type { EchoInstanceConfig } from '../types/echo-config';

async function sendThinkingMessage(
  token: string,
  channelId: string,
  content: string
): Promise<void> {
  const { sendChannelMessage } = await import('../discord');
  await sendChannelMessage(token, channelId, { content });
}

/**
 * 思考ログ専用のストリームクラス
 * OpenAIの推論結果をTHINKING_CHANNEL_IDに送信
 * Loggerとは完全に独立
 */
export class ThinkingStream {
  private readonly discordToken: string;
  private readonly channelId: string;

  constructor(instanceConfig: EchoInstanceConfig) {
    this.discordToken = instanceConfig.discordBotToken;
    this.channelId = instanceConfig.thinkingChannelId;
  }

  /**
   * 思考ログを送信
   * 呼び出し元で既にフォーマット済みの内容をそのまま送信
   */
  async send(content: string): Promise<void> {
    if (content === '') {
      return;
    }

    const message = this.truncateForDiscord(content);

    try {
      await sendThinkingMessage(this.discordToken, this.channelId, message);
    } catch (error) {
      // 思考ログ送信失敗はコンソールにのみ出力（無限ループ防止）
      console.error('Failed to send thinking to Discord:', error);
    }
  }

  private truncateForDiscord(content: string): string {
    const maxLength = 2000;
    if (content.length <= maxLength) {
      return content;
    }
    return `${content.substring(0, maxLength - 15)}...(truncated)`;
  }
}
