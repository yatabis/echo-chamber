import { REST } from '@discordjs/rest';

import type {
  RESTGetAPIChannelMessagesQuery,
  RESTGetAPIChannelMessagesResult,
  RESTGetAPICurrentUserResult,
  RESTPostAPIChannelMessageJSONBody,
  RESTPostAPIChannelMessageResult,
} from 'discord-api-types/v10';

type DiscordRoute = `/${string}`;

const DISCORD_REST_OPTIONS = {
  handlerSweepInterval: 0,
  hashSweepInterval: 0,
} as const;

/**
 * Durable Object を non-hibernateable にしやすい internal sweeper を無効化した
 * Discord REST client を生成する。
 *
 * @param token Discord bot token
 * @returns 認証済み REST client
 */
function createDiscordRestClient(token: string): REST {
  return new REST(DISCORD_REST_OPTIONS).setToken(token);
}

/**
 * チャンネルメッセージ取得・投稿に使う route を組み立てる。
 *
 * @param channelId Discord channel ID
 * @returns `/channels/:id/messages` 形式の route
 */
function routeChannelMessages(channelId: string): DiscordRoute {
  return `/channels/${channelId}/messages` as DiscordRoute;
}

/**
 * 自分自身のリアクション追加に使う route を組み立てる。
 *
 * @param channelId Discord channel ID
 * @param messageId Discord message ID
 * @param reaction リアクション文字列
 * @returns `/channels/:channelId/messages/:messageId/reactions/:reaction/@me`
 */
function routeChannelMessageOwnReaction(
  channelId: string,
  messageId: string,
  reaction: string
): DiscordRoute {
  const encodedReaction = encodeURIComponent(reaction);
  return `/channels/${channelId}/messages/${messageId}/reactions/${encodedReaction}/@me` as DiscordRoute;
}

/**
 * 現在の bot ユーザー取得に使う route を返す。
 *
 * @returns `/users/@me`
 */
function routeCurrentUser(): DiscordRoute {
  return '/users/@me';
}

/**
 * チャンネルからメッセージを取得する。
 *
 * @param token Discord bot token
 * @param channelId Discord channel ID
 * @param options Discord API の query parameter
 * @returns Discord API が返すメッセージ配列
 */
export async function getChannelMessages(
  token: string,
  channelId: string,
  options: RESTGetAPIChannelMessagesQuery = {}
): Promise<RESTGetAPIChannelMessagesResult> {
  const rest = createDiscordRestClient(token);
  return rest.get(routeChannelMessages(channelId), {
    query: new URLSearchParams(
      Object.entries(options).map(([key, value]) => [key, String(value)])
    ),
  }) as Promise<RESTGetAPIChannelMessagesResult>;
}

/**
 * チャンネルにメッセージを送信する。
 *
 * @param token Discord bot token
 * @param channelId Discord channel ID
 * @param options Discord API に渡す投稿 body
 * @returns Discord API が返す投稿結果
 */
export async function sendChannelMessage(
  token: string,
  channelId: string,
  options: RESTPostAPIChannelMessageJSONBody
): Promise<RESTPostAPIChannelMessageResult> {
  const rest = createDiscordRestClient(token);
  return rest.post(routeChannelMessages(channelId), {
    body: options,
  }) as Promise<RESTPostAPIChannelMessageResult>;
}

/**
 * メッセージにリアクションを追加する。
 *
 * @param token Discord bot token
 * @param channelId Discord channel ID
 * @param messageId Discord message ID
 * @param reaction 追加するリアクション
 * @returns Discord API への反映完了
 */
export async function addReactionToMessage(
  token: string,
  channelId: string,
  messageId: string,
  reaction: string
): Promise<void> {
  const rest = createDiscordRestClient(token);
  await rest.put(
    routeChannelMessageOwnReaction(channelId, messageId, reaction)
  );
}

/**
 * Bot 自身のユーザー情報を取得する。
 *
 * @param token Discord bot token
 * @returns 現在の bot ユーザー情報
 */
export async function getCurrentUser(
  token: string
): Promise<RESTGetAPICurrentUserResult> {
  const rest = createDiscordRestClient(token);
  return rest.get(routeCurrentUser()) as Promise<RESTGetAPICurrentUserResult>;
}
