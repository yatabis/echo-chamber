const DISCORD_MESSAGE_LIMIT = 2000;
const TRUNCATED_SUFFIX = '...(truncated)';

/**
 * Discord の投稿上限に合わせてメッセージを切り詰める。
 *
 * @param content Discord へ送信したい文字列
 * @returns 2000 文字以内に収まる文字列
 */
export function truncateForDiscord(content: string): string {
  if (content.length <= DISCORD_MESSAGE_LIMIT) {
    return content;
  }

  return `${content.substring(0, DISCORD_MESSAGE_LIMIT - TRUNCATED_SUFFIX.length)}${TRUNCATED_SUFFIX}`;
}
