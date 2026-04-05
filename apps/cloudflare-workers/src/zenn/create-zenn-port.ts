import { z } from 'zod';

import type {
  ZennArticle,
  ZennArticlePublication,
  ZennArticleTocItem,
  ZennPort,
  ZennTrendingArticleSummary,
} from '@echo-chamber/core/ports/zenn';

const ZENN_BASE_URL = 'https://zenn.dev';
const ZENN_TRENDING_ARTICLES_API_PATH = '/api/articles?order=daily';

const zennUserSchema = z.object({
  username: z.string(),
  name: z.string(),
});

const zennPublicationSchema = z
  .object({
    display_name: z.string(),
  })
  .nullable();

const zennApiArticleSchema = z.object({
  slug: z.string(),
  path: z.string(),
  title: z.string(),
  article_type: z.enum(['tech', 'idea']),
  liked_count: z.number(),
  bookmarked_count: z.number(),
  published_at: z.string(),
  body_updated_at: z.string().nullable(),
  user: zennUserSchema,
  publication: zennPublicationSchema,
});

interface ZennApiTocItem {
  id: string;
  text: string;
  level: number;
  children: ZennApiTocItem[];
}

const zennApiTocItemSchema: z.ZodType<ZennApiTocItem> = z.lazy(() =>
  z.object({
    id: z.string(),
    text: z.string(),
    level: z.number(),
    children: z.array(zennApiTocItemSchema),
  })
);

const zennArticleListResponseSchema = z.object({
  articles: z.array(zennApiArticleSchema),
});

const zennArticleDetailResponseSchema = z.object({
  article: zennApiArticleSchema.extend({
    body_html: z.string(),
    toc: z.array(zennApiTocItemSchema),
    topics: z.array(
      z.object({
        display_name: z.string(),
      })
    ),
  }),
});

/**
 * 変換時に扱う主要な HTML エンティティ。
 */
const HTML_ENTITY_MAP: Record<string, string> = {
  '&amp;': '&',
  '&lt;': '<',
  '&gt;': '>',
  '&quot;': '"',
  '&#39;': "'",
  '&#x27;': "'",
  '&#x60;': '`',
  '&nbsp;': ' ',
};

/**
 * Publication 情報を core の型へ詰め替える。
 *
 * @param publication Zenn API の publication
 * @returns core 側の publication 型
 */
function mapPublication(
  publication: z.infer<typeof zennPublicationSchema>
): ZennArticlePublication | null {
  if (publication === null) {
    return null;
  }

  return {
    displayName: publication.display_name,
  };
}

/**
 * path から Zenn の記事 URL を組み立てる。
 *
 * @param path Zenn API が返す path
 * @returns 絶対 URL
 */
function buildArticleUrl(path: string): string {
  return new URL(path, ZENN_BASE_URL).toString();
}

/**
 * HTML エンティティを平文へ復号する。
 *
 * @param value 復号対象の文字列
 * @returns 復号済み文字列
 */
function decodeHtmlEntities(value: string): string {
  return value.replace(
    /&(#x?[0-9a-fA-F]+|[a-zA-Z]+);/g,
    (match: string, entity: string) => {
      if (HTML_ENTITY_MAP[match] != null) {
        return HTML_ENTITY_MAP[match];
      }

      if (entity.startsWith('#x') || entity.startsWith('#X')) {
        const codePoint = Number.parseInt(entity.slice(2), 16);
        return Number.isNaN(codePoint)
          ? match
          : String.fromCodePoint(codePoint);
      }

      if (entity.startsWith('#')) {
        const codePoint = Number.parseInt(entity.slice(1), 10);
        return Number.isNaN(codePoint)
          ? match
          : String.fromCodePoint(codePoint);
      }

      return match;
    }
  );
}

/**
 * body_html を LLM が読みやすいプレーンテキストへ崩す。
 *
 * @param html Zenn API が返す body_html
 * @returns 整形済み本文テキスト
 */
function convertZennHtmlToPlainText(html: string): string {
  const withStructureMarkers = html
    .replace(/\r\n/g, '\n')
    .replace(
      /<img\b[^>]*alt="([^"]*)"[^>]*>/gi,
      (_, altText: string) => `[Image: ${altText}]`
    )
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<li\b[^>]*>/gi, '- ')
    .replace(/<\/li>/gi, '\n')
    .replace(/<pre\b[^>]*>/gi, '\n```\n')
    .replace(/<\/pre>/gi, '\n```\n')
    .replace(
      /<\/(p|div|section|article|aside|blockquote|h[1-6]|ul|ol)>/gi,
      '\n\n'
    )
    .replace(/<\/tr>/gi, '\n')
    .replace(/<\/(td|th)>/gi, '\t');

  const withoutTags = withStructureMarkers.replace(/<[^>]+>/g, '');
  const decoded = decodeHtmlEntities(withoutTags);

  return decoded
    .split('\n')
    .map((line) => line.replace(/[ \t]+/g, ' ').trimEnd())
    .join('\n')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

/**
 * ネストした TOC をフラットな配列へ展開する。
 *
 * @param items Zenn API の TOC
 * @returns フラット化済み TOC
 */
function flattenTableOfContents(
  items: readonly ZennApiTocItem[]
): ZennArticleTocItem[] {
  return items.flatMap((item) => [
    {
      id: item.id,
      text: item.text,
      level: item.level,
    },
    ...flattenTableOfContents(item.children),
  ]);
}

/**
 * 一覧 API の記事を core 側の要約型へ変換する。
 *
 * @param article Zenn API の記事オブジェクト
 * @returns core 側の要約
 */
function mapTrendingArticle(
  article: z.infer<typeof zennApiArticleSchema>
): ZennTrendingArticleSummary {
  return {
    slug: article.slug,
    url: buildArticleUrl(article.path),
    title: article.title,
    articleType: article.article_type,
    likedCount: article.liked_count,
    bookmarkedCount: article.bookmarked_count,
    publishedAt: article.published_at,
    author: {
      username: article.user.username,
      name: article.user.name,
    },
    publication: mapPublication(article.publication),
  };
}

/**
 * 詳細 API の記事を core 側の詳細型へ変換する。
 *
 * @param article Zenn API の記事オブジェクト
 * @returns core 側の詳細
 */
function mapArticleDetail(
  article: z.infer<typeof zennArticleDetailResponseSchema>['article']
): ZennArticle {
  return {
    slug: article.slug,
    url: buildArticleUrl(article.path),
    title: article.title,
    author: {
      username: article.user.username,
      name: article.user.name,
    },
    topics: article.topics.map((topic) => topic.display_name),
    tableOfContents: flattenTableOfContents(article.toc),
    content: convertZennHtmlToPlainText(article.body_html),
  };
}

/**
 * Zenn の JSON エンドポイントからレスポンスを取得して検証する。
 *
 * @param fetchImpl 実際に使う fetch
 * @param url 取得 URL
 * @param schema 想定レスポンス schema
 * @returns schema で検証済みのレスポンス
 */
async function fetchZennJson<Schema extends z.ZodType>(
  fetchImpl: typeof fetch,
  url: string,
  schema: Schema
): Promise<z.infer<Schema>> {
  const response = await fetchImpl(url, {
    headers: {
      accept: 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error(`Zenn API request failed: ${response.status}`);
  }

  const data: unknown = await response.json();
  return z.parse(schema, data);
}

/**
 * Zenn 記事参照ポートを生成する。
 *
 * @param fetchImpl 実行時に使う fetch 実装
 * @returns ZennPort 実装
 */
export function createZennPort(fetchImpl: typeof fetch = fetch): ZennPort {
  return {
    async listTrendingArticles(): Promise<
      readonly ZennTrendingArticleSummary[]
    > {
      const response = await fetchZennJson(
        fetchImpl,
        new URL(ZENN_TRENDING_ARTICLES_API_PATH, ZENN_BASE_URL).toString(),
        zennArticleListResponseSchema
      );

      return response.articles.map((article) => mapTrendingArticle(article));
    },

    async getArticleBySlug(slug: string): Promise<ZennArticle> {
      const url = new URL(
        `/api/articles/${encodeURIComponent(slug)}`,
        ZENN_BASE_URL
      );
      const response = await fetchZennJson(
        fetchImpl,
        url.toString(),
        zennArticleDetailResponseSchema
      );

      return mapArticleDetail(response.article);
    },
  };
}
