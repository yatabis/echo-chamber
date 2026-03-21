import { describe, expect, it } from 'vitest';

import { resolveEchoRuntimeBindings } from './echo-runtime-bindings';

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
        chat_channel_discord_rin: 'chat-rin',
        thinking_channel_discord_rin: 'thinking-rin',
      }),
      'rin'
    );

    expect(bindings).toEqual({
      discordBotToken: 'rin-token',
      chatChannelId: 'chat-rin',
      thinkingChannelId: 'thinking-rin',
      embeddingConfig: {
        provider: 'workersai',
        model: '@cf/pfnet/plamo-embedding-1b',
      },
    });
  });

  it('chat channel id が無い場合はエラーにする', async () => {
    await expect(
      resolveEchoRuntimeBindings(
        createMockEnv(),
        createMockStore({
          thinking_channel_discord_rin: 'thinking-rin',
        }),
        'rin'
      )
    ).rejects.toThrow(
      'Chat channel ID not found in KV for instance "rin" (key: chat_channel_discord_rin)'
    );
  });

  it('thinking channel id が無い場合はエラーにする', async () => {
    await expect(
      resolveEchoRuntimeBindings(
        createMockEnv(),
        createMockStore({
          chat_channel_discord_marie: 'chat-marie',
        }),
        'marie'
      )
    ).rejects.toThrow(
      'Thinking channel ID not found in KV for instance "marie" (key: thinking_channel_discord_marie)'
    );
  });
});
