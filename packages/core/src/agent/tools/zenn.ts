import { z } from 'zod';

import { createToolResultSchema, defineToolSpecification } from './shared';

const MAX_TRENDING_ZENN_ARTICLE_LIMIT = 20;
const MIN_ZENN_ARTICLE_MAX_CHARACTERS = 500;
const MAX_ZENN_ARTICLE_MAX_CHARACTERS = 50_000;

const zennAuthorSchema = z.object({
  username: z.string(),
  name: z.string(),
});

const zennPublicationSchema = z
  .object({
    displayName: z.string(),
  })
  .nullable();

const zennTrendingArticleSchema = z.object({
  slug: z.string(),
  url: z.string(),
  title: z.string(),
  articleType: z.enum(['tech', 'idea']),
  likedCount: z.number(),
  bookmarkedCount: z.number(),
  publishedAt: z.string(),
  author: zennAuthorSchema,
  publication: zennPublicationSchema,
});

const zennArticleTocItemSchema = z.object({
  id: z.string(),
  text: z.string(),
  level: z.number(),
});

export const listTrendingZennArticlesToolSpec = defineToolSpecification({
  name: 'list_trending_zenn_articles',
  description: `Zenn の日次トレンド記事一覧を取得する。記事選定に必要な最小限のメタデータを最大${MAX_TRENDING_ZENN_ARTICLE_LIMIT}件返す。まず候補を絞り、その後 get_zenn_article に slug を渡して本文を読む想定。`,
  parameters: {
    articleType: z
      .enum(['tech', 'idea'])
      .optional()
      .describe(
        '記事タイプで絞り込む。指定しない場合は tech と idea の両方を返す。'
      ),
  },
  outputSchema: createToolResultSchema({
    articles: z.array(zennTrendingArticleSchema),
  }),
});

export const getZennArticleToolSpec = defineToolSpecification({
  name: 'get_zenn_article',
  description:
    '指定した Zenn 記事の本文を取得する。list_trending_zenn_articles の結果に含まれる slug を使う。maxCharacters を指定した場合だけ本文をその長さまで切り詰め、途中で切った場合は truncated を true にする。',
  parameters: {
    slug: z
      .string()
      .min(1)
      .trim()
      .describe(
        '取得対象の記事 slug。list_trending_zenn_articles の結果を使う。'
      ),
    maxCharacters: z
      .int()
      .min(MIN_ZENN_ARTICLE_MAX_CHARACTERS)
      .max(MAX_ZENN_ARTICLE_MAX_CHARACTERS)
      .optional()
      .describe(
        `返却する本文の最大文字数。${MIN_ZENN_ARTICLE_MAX_CHARACTERS}〜${MAX_ZENN_ARTICLE_MAX_CHARACTERS}。省略時は全文を返す。`
      ),
  },
  outputSchema: createToolResultSchema({
    slug: z.string(),
    url: z.string(),
    title: z.string(),
    author: zennAuthorSchema,
    topics: z.array(z.string()),
    tableOfContents: z.array(zennArticleTocItemSchema),
    content: z.string(),
    truncated: z.boolean(),
  }),
});
