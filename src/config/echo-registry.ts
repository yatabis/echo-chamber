/**
 * Echo インスタンス設定レジストリ
 *
 * 各 Echo インスタンスの設定を一元管理する
 * 新しいインスタンスを追加する場合はこのファイルのみを更新する
 */

import systemPromptRin from '../llm/prompts/rin';

import type {
  EchoInstanceConfig,
  EchoInstanceId,
  EmbeddingConfig,
} from '../types/echo-config';

/**
 * インスタンスごとの静的設定
 */
interface StaticInstanceConfig {
  name: string;
  systemPrompt: string;
  getDiscordBotToken(env: Env): string;
  chatChannelKvKey: string;
  thinkingChannelKvKey: string;
  embeddingConfig?: EmbeddingConfig;
}

const INSTANCE_CONFIGS: Record<EchoInstanceId, StaticInstanceConfig> = {
  rin: {
    name: 'リン',
    systemPrompt: systemPromptRin,
    getDiscordBotToken: (env) => env.DISCORD_BOT_TOKEN_RIN,
    chatChannelKvKey: 'chat_channel_discord_rin',
    thinkingChannelKvKey: 'thinking_channel_discord_rin',
    embeddingConfig: {
      provider: 'workersai',
      model: '@cf/pfnet/plamo-embedding-1b',
    },
  },
  marie: {
    name: 'マリー',
    systemPrompt: '', // TODO: マリーのシステムプロンプトを設定
    getDiscordBotToken: (env) => env.DISCORD_BOT_TOKEN_MARIE,
    chatChannelKvKey: 'chat_channel_discord_marie',
    thinkingChannelKvKey: 'thinking_channel_discord_marie',
  },
};

/**
 * 指定されたインスタンス ID の設定を取得する
 *
 * KV からチャンネル ID を取得し、完全なインスタンス設定を構築する
 *
 * @param env - Cloudflare Workers 環境変数
 * @param store - KV Namespace（チャンネルID取得用）
 * @param instanceId - Echo インスタンス ID
 * @returns インスタンス設定
 */
export async function getInstanceConfig(
  env: Env,
  store: KVNamespace,
  instanceId: EchoInstanceId
): Promise<EchoInstanceConfig> {
  const staticConfig = INSTANCE_CONFIGS[instanceId];

  const [chatChannelId, thinkingChannelId] = await Promise.all([
    store.get(staticConfig.chatChannelKvKey),
    store.get(staticConfig.thinkingChannelKvKey),
  ]);

  if (chatChannelId === null) {
    throw new Error(
      `Chat channel ID not found in KV for instance "${instanceId}" (key: ${staticConfig.chatChannelKvKey})`
    );
  }

  if (thinkingChannelId === null) {
    throw new Error(
      `Thinking channel ID not found in KV for instance "${instanceId}" (key: ${staticConfig.thinkingChannelKvKey})`
    );
  }

  return {
    id: instanceId,
    name: staticConfig.name,
    systemPrompt: staticConfig.systemPrompt,
    discordBotToken: staticConfig.getDiscordBotToken(env),
    chatChannelId,
    thinkingChannelId,
    embeddingConfig: staticConfig.embeddingConfig,
  };
}
