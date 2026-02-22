import { REST } from '@discordjs/rest';

import type {
  RESTGetAPIChannelMessagesQuery,
  RESTGetAPIChannelMessagesResult,
  RESTGetAPICurrentUserResult,
  RESTPostAPIChannelMessageJSONBody,
  RESTPostAPIChannelMessageResult,
} from 'discord-api-types/v10';

type DiscordRoute = `/${string}`;

function routeChannelMessages(channelId: string): DiscordRoute {
  return `/channels/${channelId}/messages` as DiscordRoute;
}

function routeChannelMessageOwnReaction(
  channelId: string,
  messageId: string,
  reaction: string
): DiscordRoute {
  const encodedReaction = encodeURIComponent(reaction);
  return `/channels/${channelId}/messages/${messageId}/reactions/${encodedReaction}/@me` as DiscordRoute;
}

function routeCurrentUser(): DiscordRoute {
  return '/users/@me';
}

/**
 * チャンネルからメッセージを取得
 * @param token Discord Bot Token
 * @param channelId チャンネルID
 * @param options オプション（limitなど）
 * @returns メッセージの配列
 */
export async function getChannelMessages(
  token: string,
  channelId: string,
  options: RESTGetAPIChannelMessagesQuery = {}
): Promise<RESTGetAPIChannelMessagesResult> {
  const rest = new REST().setToken(token);
  return rest.get(routeChannelMessages(channelId), {
    query: new URLSearchParams(
      Object.entries(options).map(([key, value]) => [key, String(value)])
    ),
  }) as Promise<RESTGetAPIChannelMessagesResult>;
}

/**
 * チャンネルにメッセージを送信
 * @param token Discord Bot Token
 * @param channelId チャンネルID
 * @param options 送信するメッセージの内容
 * @returns 送信したメッセージの結果
 */
export async function sendChannelMessage(
  token: string,
  channelId: string,
  options: RESTPostAPIChannelMessageJSONBody
): Promise<RESTPostAPIChannelMessageResult> {
  const rest = new REST().setToken(token);
  return rest.post(routeChannelMessages(channelId), {
    body: options,
  }) as Promise<RESTPostAPIChannelMessageResult>;
}

/**
 * メッセージにリアクションを追加
 * @param token Discord Bot Token
 * @param channelId チャンネルID
 * @param messageId メッセージID
 * @param reaction 追加するリアクション（絵文字など）
 */
export async function addReactionToMessage(
  token: string,
  channelId: string,
  messageId: string,
  reaction: string
): Promise<void> {
  const rest = new REST().setToken(token);
  await rest.put(
    routeChannelMessageOwnReaction(channelId, messageId, reaction)
  );
}

/**
 * ボットの情報を取得（認証テスト用）
 * @param token Discord Bot Token
 * @returns ボットのユーザー情報
 */
export async function getCurrentUser(
  token: string
): Promise<RESTGetAPICurrentUserResult> {
  const rest = new REST().setToken(token);
  return rest.get(routeCurrentUser()) as Promise<RESTGetAPICurrentUserResult>;
}
