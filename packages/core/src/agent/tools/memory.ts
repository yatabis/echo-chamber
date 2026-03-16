import { z } from 'zod';

import { MEMORY_TYPES } from '../../echo/types';

import { createToolResultSchema, defineToolSpecification } from './shared';

const MAX_MEMORY_CONTENT_LENGTH = 500;
const MAX_MEMORY_QUERY_LENGTH = 500;

const memoryTypeSchema = z.enum(MEMORY_TYPES);

const emotionSchema = z.object({
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
    .array(z.string())
    .describe('感情ラベル（例: "楽しい", "悲しい", "驚き", "知的好奇心"）'),
});

const memorySearchResultSchema = z.object({
  content: z.string(),
  type: memoryTypeSchema,
  emotion: emotionSchema,
  createdAt: z.string(),
});

export const storeMemoryToolSpec = defineToolSpecification({
  name: 'store_memory',
  description:
    '将来のセマンティック検索のために、感情的コンテキストを伴う記憶を保存する。意味のある体験、会話、または感情的な重要性を持つ瞬間を保存するために使用せよ。システムはセマンティック検索のためにエンベディングを使用し、容量がいっぱいになると最も古い記憶を削除して自動的に管理する。',
  parameters: {
    content: z
      .string()
      .min(1)
      .max(MAX_MEMORY_CONTENT_LENGTH)
      .trim()
      .describe(
        `関連するすべての詳細を含む記憶の完全な内容。最大${MAX_MEMORY_CONTENT_LENGTH}文字。`
      ),
    type: memoryTypeSchema.describe(
      '記憶のタイプ。semantic: 事実や一般的な知識（「東京は日本の首都」など）。episode: 特定の体験や出来事（「今日ユーザーと楽しい会話をした」など）。'
    ),
    emotion: emotionSchema.describe('この記憶に付随する感情'),
  },
  outputSchema: createToolResultSchema({}),
});

export const searchMemoryToolSpec = defineToolSpecification({
  name: 'search_memory',
  description:
    'セマンティック類似性を使用して関連する記憶を検索する。過去の経験を思い出したり、関連する記憶を見つけたり、正確なキーワードではなく概念的にクエリに一致する記憶を取得するために使用せよ。セマンティック類似性でソートされた最大5件の最も関連性の高い記憶を返す。',
  parameters: {
    query: z
      .string()
      .min(1)
      .max(MAX_MEMORY_QUERY_LENGTH)
      .trim()
      .describe(
        '検索クエリ。埋め込み化され、コサイン類似度を使用して保存された記憶と比較される。'
      ),
    type: memoryTypeSchema
      .optional()
      .describe(
        '検索対象の記憶タイプ。指定しない場合は全タイプを検索する。semantic: 事実や一般的な知識。episode: 特定の体験や出来事。'
      ),
  },
  outputSchema: createToolResultSchema({
    results: z.array(memorySearchResultSchema),
  }),
});
