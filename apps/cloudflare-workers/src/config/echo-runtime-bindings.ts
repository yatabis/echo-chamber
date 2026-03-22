import type { ChatChannel } from '@echo-chamber/core/ports/chat';
import type {
  EchoInstanceId,
  EmbeddingConfig,
} from '@echo-chamber/core/types/echo-config';

export interface EchoChatChannelBinding extends ChatChannel {
  discordChannelId: string;
}

export interface EchoRuntimeBindings {
  discordBotToken: string;
  chatChannels: EchoChatChannelBinding[];
  thinkingChannelId: string;
  embeddingConfig?: EmbeddingConfig;
}

export type EchoChatRuntimeBindings = Pick<
  EchoRuntimeBindings,
  'discordBotToken' | 'chatChannels'
>;

interface RuntimeBindingSource {
  getDiscordBotToken(env: Env): string;
  chatChannels: readonly EchoChatChannelBinding[];
  thinkingChannelKvKey: string;
  embeddingConfig?: EmbeddingConfig;
}

const RUNTIME_BINDING_SOURCES: Record<EchoInstanceId, RuntimeBindingSource> = {
  rin: {
    getDiscordBotToken: (env) => env.DISCORD_BOT_TOKEN_RIN,
    chatChannels: [
      {
        key: 'yatabis',
        displayName: 'yatabis',
        description: 'yatabisとのチャンネル',
        discordChannelId: '1371143541526499470',
      },
      {
        key: 'all',
        displayName: 'all',
        description: '全体チャンネル',
        discordChannelId: '1485185701468176524',
      },
    ],
    thinkingChannelKvKey: 'thinking_channel_discord_rin',
    embeddingConfig: {
      provider: 'workersai',
      model: '@cf/pfnet/plamo-embedding-1b',
    },
  },
  marie: {
    getDiscordBotToken: (env) => env.DISCORD_BOT_TOKEN_MARIE,
    chatChannels: [
      {
        key: 'yatabis',
        displayName: 'yatabis',
        description: 'yatabisとのチャンネル',
        discordChannelId: '1460280463443624020',
      },
      {
        key: 'all',
        displayName: 'all',
        description: '全体チャンネル',
        discordChannelId: '1485185701468176524',
      },
    ],
    thinkingChannelKvKey: 'thinking_channel_discord_marie',
  },
};

export async function resolveEchoRuntimeBindings(
  env: Env,
  store: Pick<KVNamespace, 'get'>,
  instanceId: EchoInstanceId
): Promise<EchoRuntimeBindings> {
  const source = RUNTIME_BINDING_SOURCES[instanceId];
  const thinkingChannelId = await store.get(source.thinkingChannelKvKey);

  if (thinkingChannelId === null) {
    throw new Error(
      `Thinking channel ID not found in KV for instance "${instanceId}" (key: ${source.thinkingChannelKvKey})`
    );
  }

  return {
    discordBotToken: source.getDiscordBotToken(env),
    chatChannels: source.chatChannels.map((channel) => ({ ...channel })),
    thinkingChannelId,
    embeddingConfig: source.embeddingConfig,
  };
}
