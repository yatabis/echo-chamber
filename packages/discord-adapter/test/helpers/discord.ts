import type {
  APIMessage,
  APIReaction,
  APIUser,
  RESTGetAPIChannelMessagesResult,
  RESTGetAPICurrentUserResult,
} from 'discord-api-types/v10';

interface MessageInput {
  message: string;
  user: string;
  userId?: string;
  timestamp: string;
  reactions?: {
    emoji: string;
    me: boolean;
  }[];
}

export function createDiscordMessagesResponse(
  messages: MessageInput[]
): RESTGetAPIChannelMessagesResult {
  return messages.map((message, index) => {
    const author: APIUser = {
      id: message.userId ?? `user-${index + 1}`,
      username: message.user,
      discriminator: '0000',
      global_name: message.user,
      avatar: null,
      bot: false,
      system: false,
      mfa_enabled: false,
    };

    const reactions: APIReaction[] | undefined = message.reactions?.map(
      ({ emoji, me }) => ({
        count: 1,
        count_details: {
          burst: 0,
          normal: 1,
        },
        me,
        me_burst: false,
        emoji: {
          id: null,
          name: emoji,
          animated: false,
        },
        burst_colors: [],
      })
    );

    const discordMessage: APIMessage = {
      id: `message-${index + 1}`,
      type: 0,
      content: message.message,
      channel_id: 'test-channel-id',
      author,
      timestamp: message.timestamp,
      edited_timestamp: null,
      tts: false,
      mention_everyone: false,
      mentions: [],
      mention_roles: [],
      mention_channels: [],
      attachments: [],
      embeds: [],
      reactions,
      pinned: false,
    };

    return discordMessage;
  });
}

export function createDiscordCurrentUserResponse(
  user: string
): RESTGetAPICurrentUserResult {
  return {
    id: user,
    username: user,
    discriminator: '0000',
    global_name: user,
    avatar: null,
    bot: true,
    system: false,
    mfa_enabled: false,
    banner: null,
    accent_color: null,
    locale: 'en-US',
    verified: true,
    email: null,
    premium_type: 0,
  };
}
