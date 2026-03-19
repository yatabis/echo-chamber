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
 * チャンネルからメッセージを取得する。
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
 * チャンネルにメッセージを送信する。
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
 * メッセージにリアクションを追加する。
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
 * Bot 自身のユーザー情報を取得する。
 */
export async function getCurrentUser(
  token: string
): Promise<RESTGetAPICurrentUserResult> {
  const rest = new REST().setToken(token);
  return rest.get(routeCurrentUser()) as Promise<RESTGetAPICurrentUserResult>;
}
