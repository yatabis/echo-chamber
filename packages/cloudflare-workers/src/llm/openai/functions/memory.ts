import { z } from 'zod';

import { MEMORY_TYPES, getErrorMessage } from '@echo-chamber/core';

import { Tool } from './index';

const MAX_CONTENT_LENGTH = 500;
const MAX_QUERY_LENGTH = 500;

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

export const storeMemoryFunction = new Tool(
  'store_memory',
  '将来のセマンティック検索のために、感情的コンテキストを伴う記憶を保存する。意味のある体験、会話、または感情的な重要性を持つ瞬間を保存するために使用せよ。システムはセマンティック検索のためにエンベディングを使用し、容量がいっぱいになると最も古い記憶を削除して自動的に管理する。',
  {
    content: z
      .string()
      .min(1)
      .max(MAX_CONTENT_LENGTH)
      .trim()
      .describe(
        `関連するすべての詳細を含む記憶の完全な内容。最大${MAX_CONTENT_LENGTH}文字。`
      ),
    type: memoryTypeSchema.describe(
      '記憶のタイプ。semantic: 事実や一般的な知識（「東京は日本の首都」など）。episode: 特定の体験や出来事（「今日ユーザーと楽しい会話をした」など）。'
    ),
    emotion: emotionSchema.describe('この記憶に付随する感情'),
  },
  async ({ content, type, emotion }, ctx) => {
    try {
      await ctx.memorySystem.storeMemory(content, emotion, type);
      return { success: true };
    } catch (error) {
      await ctx.logger.error(`Error storing memory: ${getErrorMessage(error)}`);
      return {
        success: false,
        error: 'Failed to store memory',
      };
    }
  }
);

export const searchMemoryFunction = new Tool(
  'search_memory',
  'セマンティック類似性を使用して関連する記憶を検索する。過去の経験を思い出したり、関連する記憶を見つけたり、正確なキーワードではなく概念的にクエリに一致する記憶を取得するために使用せよ。セマンティック類似性でソートされた最大5件の最も関連性の高い記憶を返す。',
  {
    query: z
      .string()
      .min(1)
      .max(MAX_QUERY_LENGTH)
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
  async ({ query, type }, ctx) => {
    try {
      const results = await ctx.memorySystem.searchMemory(query, type);
      return {
        success: true,
        results: results.map(({ content, type, emotion, createdAt }) => ({
          content,
          type,
          emotion,
          createdAt,
        })),
      };
    } catch (error) {
      await ctx.logger.error(
        `Error searching memory: ${getErrorMessage(error)}`
      );
      return {
        success: false,
        error: 'Failed to search memory',
      };
    }
  }
);
