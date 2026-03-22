import { describe, expect, it } from 'vitest';

import { resolveEchoRuntimeBindings } from './echo-runtime-bindings';

const RIN_CHAT_CHANNELS = [
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
] as const;

function createMockEnv(): Env {
  return {
    DISCORD_BOT_TOKEN_RIN: 'rin-token',
    DISCORD_BOT_TOKEN_MARIE: 'marie-token',
  } as Env;
}

function createMockStore(
  values: Record<string, string | null>
): Pick<KVNamespace, 'get'> {
  return {
    get: (async (key: string) =>
      Promise.resolve(values[key] ?? null)) as KVNamespace['get'],
  };
}

describe('resolveEchoRuntimeBindings', () => {
  it('rin の runtime bindings を Env と KV から解決する', async () => {
    const bindings = await resolveEchoRuntimeBindings(
      createMockEnv(),
      createMockStore({
        thinking_channel_discord_rin: 'thinking-rin',
      }),
      'rin'
    );

    expect(bindings).toEqual({
      discordBotToken: 'rin-token',
      chatChannels: RIN_CHAT_CHANNELS,
      thinkingChannelId: 'thinking-rin',
      embeddingConfig: {
        provider: 'workersai',
        model: '@cf/pfnet/plamo-embedding-1b',
      },
    });
  });

  it('chat channels はコード定義から解決する', async () => {
    const bindings = await resolveEchoRuntimeBindings(
      createMockEnv(),
      createMockStore({
        thinking_channel_discord_rin: 'thinking-rin',
      }),
      'rin'
    );

    expect(bindings.chatChannels).toEqual(RIN_CHAT_CHANNELS);
  });

  it('thinking channel id が無い場合はエラーにする', async () => {
    await expect(
      resolveEchoRuntimeBindings(createMockEnv(), createMockStore({}), 'marie')
    ).rejects.toThrow(
      'Thinking channel ID not found in KV for instance "marie" (key: thinking_channel_discord_marie)'
    );
  });
});
