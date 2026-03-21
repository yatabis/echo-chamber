import type {
  EchoInstanceId,
  EmbeddingConfig,
} from '@echo-chamber/core/types/echo-config';

export interface EchoRuntimeBindings {
  discordBotToken: string;
  chatChannelId: string;
  thinkingChannelId: string;
  embeddingConfig?: EmbeddingConfig;
}

export type EchoChatRuntimeBindings = Pick<
  EchoRuntimeBindings,
  'discordBotToken' | 'chatChannelId'
>;

interface RuntimeBindingSource {
  getDiscordBotToken(env: Env): string;
  chatChannelKvKey: string;
  thinkingChannelKvKey: string;
  embeddingConfig?: EmbeddingConfig;
}

const RUNTIME_BINDING_SOURCES: Record<EchoInstanceId, RuntimeBindingSource> = {
  rin: {
    getDiscordBotToken: (env) => env.DISCORD_BOT_TOKEN_RIN,
    chatChannelKvKey: 'chat_channel_discord_rin',
    thinkingChannelKvKey: 'thinking_channel_discord_rin',
    embeddingConfig: {
      provider: 'workersai',
      model: '@cf/pfnet/plamo-embedding-1b',
    },
  },
  marie: {
    getDiscordBotToken: (env) => env.DISCORD_BOT_TOKEN_MARIE,
    chatChannelKvKey: 'chat_channel_discord_marie',
    thinkingChannelKvKey: 'thinking_channel_discord_marie',
  },
};

export async function resolveEchoRuntimeBindings(
  env: Env,
  store: Pick<KVNamespace, 'get'>,
  instanceId: EchoInstanceId
): Promise<EchoRuntimeBindings> {
  const source = RUNTIME_BINDING_SOURCES[instanceId];
  const [chatChannelId, thinkingChannelId] = await Promise.all([
    store.get(source.chatChannelKvKey),
    store.get(source.thinkingChannelKvKey),
  ]);

  if (chatChannelId === null) {
    throw new Error(
      `Chat channel ID not found in KV for instance "${instanceId}" (key: ${source.chatChannelKvKey})`
    );
  }

  if (thinkingChannelId === null) {
    throw new Error(
      `Thinking channel ID not found in KV for instance "${instanceId}" (key: ${source.thinkingChannelKvKey})`
    );
  }

  return {
    discordBotToken: source.getDiscordBotToken(env),
    chatChannelId,
    thinkingChannelId,
    embeddingConfig: source.embeddingConfig,
  };
}
