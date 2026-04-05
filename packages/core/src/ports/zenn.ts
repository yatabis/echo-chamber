/**
 * Zenn 記事の種別。
 */
export const ZENN_ARTICLE_TYPES = ['tech', 'idea'] as const;

/**
 * Zenn 記事の種別。
 */
export type ZennArticleType = (typeof ZENN_ARTICLE_TYPES)[number];

/**
 * Zenn 上の著者情報。
 */
export interface ZennArticleAuthor {
  username: string;
  name: string;
}

/**
 * Zenn Publication の最小表現。
 */
export interface ZennArticlePublication {
  displayName: string;
}

/**
 * Zenn 記事の目次項目。
 */
export interface ZennArticleTocItem {
  id: string;
  text: string;
  level: number;
}

/**
 * トレンド一覧に表示する Zenn 記事の要約。
 */
export interface ZennTrendingArticleSummary {
  slug: string;
  url: string;
  title: string;
  articleType: ZennArticleType;
  likedCount: number;
  bookmarkedCount: number;
  publishedAt: string;
  author: ZennArticleAuthor;
  publication: ZennArticlePublication | null;
}

/**
 * 本文まで展開した Zenn 記事の詳細。
 */
export interface ZennArticle {
  slug: string;
  url: string;
  title: string;
  author: ZennArticleAuthor;
  topics: readonly string[];
  tableOfContents: readonly ZennArticleTocItem[];
  content: string;
}

/**
 * Zenn の公開記事を参照するためのポート。
 */
export interface ZennPort {
  /**
   * 日次トレンドの記事一覧を取得する。
   *
   * @returns トレンド記事の要約一覧
   */
  listTrendingArticles(): Promise<readonly ZennTrendingArticleSummary[]>;

  /**
   * slug を指定して公開記事の詳細を取得する。
   *
   * @param slug Zenn 記事の slug
   * @returns 本文を含む記事詳細
   */
  getArticleBySlug(slug: string): Promise<ZennArticle>;
}
