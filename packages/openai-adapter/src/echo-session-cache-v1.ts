import type { OpenAIChatCompletionsRequestBodyExtension } from './openai-chat-completions-model';

export const ECHO_SESSION_CACHE_RUNTIME_PROFILE =
  'echo-session-cache-v1' as const;

/**
 * E.C.H.O. 向け runtime の version 1 session-cache request 拡張を作る。
 *
 * この独自 body は通常の OpenAI-compatible endpoint には送らず、
 * 対応 runtime profile が明示された場合だけ composition root から注入する。
 *
 * @param sessionId Echo instance ごとに状態を分離する論理 session ID
 * @returns exchange 状態に応じて pinned / rolling slot を選ぶ request 拡張
 */
export function createEchoSessionCacheRequestBodyExtension(
  sessionId: string
): OpenAIChatCompletionsRequestBodyExtension {
  return ({ hasCompletedExchange }) => ({
    cache: {
      mode: 'auto',
      session_id: sessionId,
      session_slot: hasCompletedExchange ? 'rolling' : 'pinned',
    },
  });
}
