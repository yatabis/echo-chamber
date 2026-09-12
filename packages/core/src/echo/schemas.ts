import { z } from 'zod';

import { MEMORY_TYPES } from './types';

export const MAX_MEMORY_CONTENT_LENGTH = 500;
export const MAX_MEMORY_QUERY_LENGTH = 500;
export const MAX_EMOTION_LABEL_LENGTH = 12;
export const MAX_EMOTION_LABELS = 5;

/** E.C.H.O. Chamber が共有する感情状態のcontract。 */
export const emotionSchema = z
  .object({
    valence: z
      .number()
      .min(-1.0)
      .max(1.0)
      .describe('感情価（-1.0：ネガティブ 〜 1.0：ポジティブ）'),
    arousal: z
      .number()
      .min(0.0)
      .max(1.0)
      .describe('覚醒度（0.0：穏やか 〜 1.0：興奮）'),
    labels: z
      .array(z.string().trim().min(1).max(MAX_EMOTION_LABEL_LENGTH))
      .max(MAX_EMOTION_LABELS)
      .describe('感情ラベル（例: "楽しい", "悲しい", "驚き", "知的好奇心"）'),
  })
  .strict();

/** Memory検索で使用するqueryのcontract。 */
export const memoryQuerySchema = z
  .string()
  .trim()
  .min(1)
  .max(MAX_MEMORY_QUERY_LENGTH)
  .describe(
    `検索クエリ。埋め込み化され、保存済みMemoryとの類似性検索に使用される。最大${MAX_MEMORY_QUERY_LENGTH}文字。`
  );

/** E.C.H.O. Chamber が共有するMemory種別。 */
export const memoryTypeSchema = z.enum(MEMORY_TYPES);

/** Memoryへ保存する本文のcontract。 */
export const memoryContentSchema = z
  .string()
  .trim()
  .min(1)
  .max(MAX_MEMORY_CONTENT_LENGTH)
  .describe(
    `関連するすべての詳細を含む記憶の完全な内容。最大${MAX_MEMORY_CONTENT_LENGTH}文字。`
  );

/** Memory Moduleとruntime toolが共有するstore_memory入力。 */
export const memoryStoreInputSchema = z
  .object({
    content: memoryContentSchema,
    type: memoryTypeSchema.describe(
      '記憶のタイプ。semantic: 事実や一般的な知識（「東京は日本の首都」など）。episode: 特定の体験や出来事（「今日ユーザーと楽しい会話をした」など）。'
    ),
  })
  .strict();
