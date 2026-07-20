import { OpenAIChatCompletionsModel } from '@echo-chamber/openai-adapter/openai-chat-completions-model';
import type { OpenAIChatCompletionsRequestBodyExtension } from '@echo-chamber/openai-adapter/openai-chat-completions-model';

import type { RuntimeModelFactory } from '../qwen36-eat-readiness/runtime-harness';

/** OpenAI互換Chat Completions評価ランナーの接続設定。 */
export interface OpenAICompatibleModelFactoryOptions {
  apiKey: string;
  baseURL: string;
  servedModelName: string;
  createSessionRequestBodyExtension?(
    sessionId: string
  ): OpenAIChatCompletionsRequestBodyExtension;
}

/**
 * OpenAI互換Chat Completions serverをE.C.H.O.評価用`ModelPort`へ接続する。
 *
 * 通常の互換serverでは`createSessionRequestBodyExtension`を省略する。
 * 専用ランタイムだけが、抽象的な評価session IDを独自request bodyへ変換する。
 *
 * @param options API接続情報と、必要な場合だけ使うsession拡張生成関数
 * @returns 評価セッションごとに独立したChat Completions modelを作る関数
 */
export function createOpenAICompatibleModelFactory(
  options: OpenAICompatibleModelFactoryOptions
): RuntimeModelFactory {
  return ({ events, generationProfile, sessionId }) =>
    new OpenAIChatCompletionsModel({
      apiKey: options.apiKey,
      baseURL: options.baseURL,
      model: options.servedModelName,
      events,
      maxTokens: generationProfile.maxTokensPerTurn,
      temperature: generationProfile.temperature,
      topP: generationProfile.topP,
      presencePenalty: generationProfile.presencePenalty,
      frequencyPenalty: 0,
      extraBody: {
        top_k: generationProfile.topK,
        min_p: generationProfile.minP,
        repetition_penalty: generationProfile.repetitionPenalty,
        chat_template_kwargs: {
          enable_thinking: generationProfile.enableThinking,
        },
      },
      requestBodyExtension:
        sessionId === undefined ||
        options.createSessionRequestBodyExtension === undefined
          ? undefined
          : options.createSessionRequestBodyExtension(sessionId),
    });
}
