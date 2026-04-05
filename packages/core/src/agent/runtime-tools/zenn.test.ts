import { beforeEach, describe, expect, it, vi } from 'vitest';

import { mockToolContext } from './mock-tool-context';
import { getZennArticleTool, listTrendingZennArticlesTool } from './zenn';

const ZENN_API_ERROR = 'Zenn API error';
const TEST_ZENN_ARTICLE_SLUG = 'example-zenn-article';

const mockedListTrendingArticles = vi.mocked(
  // eslint-disable-next-line @typescript-eslint/unbound-method
  mockToolContext.zenn.listTrendingArticles
);
// eslint-disable-next-line @typescript-eslint/unbound-method
const mockedGetArticleBySlug = vi.mocked(mockToolContext.zenn.getArticleBySlug);

const trendingArticle = {
  slug: TEST_ZENN_ARTICLE_SLUG,
  url: `https://zenn.dev/example_author/articles/${TEST_ZENN_ARTICLE_SLUG}`,
  title: 'ダミーの Zenn 記事タイトル',
  articleType: 'tech' as const,
  likedCount: 42,
  bookmarkedCount: 12,
  publishedAt: '2026-04-01T10:00:00.000+09:00',
  author: {
    username: 'example_author',
    name: 'Example Author',
  },
  publication: {
    displayName: 'Example Publication',
  },
};

const ideaArticle = {
  ...trendingArticle,
  slug: 'example-idea-article',
  url: 'https://zenn.dev/example_author/articles/example-idea-article',
  articleType: 'idea' as const,
};

const articleDetail = {
  slug: TEST_ZENN_ARTICLE_SLUG,
  url: `https://zenn.dev/example_author/articles/${TEST_ZENN_ARTICLE_SLUG}`,
  title: 'ダミーの Zenn 記事タイトル',
  author: {
    username: 'example_author',
    name: 'Example Author',
  },
  topics: ['Testing', 'Examples'],
  tableOfContents: [
    {
      id: 'intro',
      text: 'はじめに',
      level: 2,
    },
  ],
  content: '段落1\n\n段落2\n\n段落3',
};

beforeEach(() => {
  vi.resetAllMocks();
  mockedListTrendingArticles.mockResolvedValue([]);
  mockedGetArticleBySlug.mockResolvedValue(articleDetail);
});

describe('listTrendingZennArticlesTool', () => {
  it('name', () => {
    expect(listTrendingZennArticlesTool.name).toBe(
      'list_trending_zenn_articles'
    );
  });

  it('一覧を返す', async () => {
    mockedListTrendingArticles.mockResolvedValue([
      trendingArticle,
      ideaArticle,
    ]);

    const result = await listTrendingZennArticlesTool.handler(
      {},
      mockToolContext
    );

    expect(mockedListTrendingArticles).toHaveBeenCalledWith();
    expect(result).toEqual({
      success: true,
      articles: [trendingArticle, ideaArticle],
    });
  });

  it('一覧は最大20件まで返す', async () => {
    mockedListTrendingArticles.mockResolvedValue(
      Array.from({ length: 25 }, (_, index) => ({
        ...trendingArticle,
        slug: `example-zenn-article-${index + 1}`,
        url: `https://zenn.dev/example_author/articles/example-zenn-article-${index + 1}`,
      }))
    );

    const result = await listTrendingZennArticlesTool.handler(
      {},
      mockToolContext
    );

    if (result.success) {
      const successResult = result as {
        success: true;
        articles: (typeof trendingArticle)[];
      };

      expect(successResult.articles).toHaveLength(20);
      expect(successResult.articles[0]).toEqual(
        expect.objectContaining({ slug: 'example-zenn-article-1' })
      );
    } else {
      expect.unreachable('expected success result');
    }
  });

  it('articleType で絞り込む', async () => {
    mockedListTrendingArticles.mockResolvedValue([
      trendingArticle,
      ideaArticle,
    ]);

    const result = await listTrendingZennArticlesTool.handler(
      { articleType: 'idea' },
      mockToolContext
    );

    expect(result).toEqual({
      success: true,
      articles: [ideaArticle],
    });
  });

  it('ZennPort エラー時は失敗を返す', async () => {
    mockedListTrendingArticles.mockRejectedValue(new Error(ZENN_API_ERROR));

    const result = await listTrendingZennArticlesTool.handler(
      {},
      mockToolContext
    );

    expect(result).toEqual({
      success: false,
      error: 'Failed to list Zenn trending articles',
    });
  });
});

describe('getZennArticleTool', () => {
  it('name', () => {
    expect(getZennArticleTool.name).toBe('get_zenn_article');
  });

  it('slug で記事詳細を返す', async () => {
    const result = await getZennArticleTool.handler(
      { slug: TEST_ZENN_ARTICLE_SLUG },
      mockToolContext
    );

    expect(mockedGetArticleBySlug).toHaveBeenCalledWith(TEST_ZENN_ARTICLE_SLUG);
    expect(result).toEqual({
      success: true,
      ...articleDetail,
      truncated: false,
    });
  });

  it('maxCharacters を超える場合は段落境界で切り詰める', async () => {
    const result = await getZennArticleTool.handler(
      { slug: TEST_ZENN_ARTICLE_SLUG, maxCharacters: 10 },
      mockToolContext
    );

    expect(result).toEqual({
      success: true,
      slug: TEST_ZENN_ARTICLE_SLUG,
      url: `https://zenn.dev/example_author/articles/${TEST_ZENN_ARTICLE_SLUG}`,
      title: 'ダミーの Zenn 記事タイトル',
      author: {
        username: 'example_author',
        name: 'Example Author',
      },
      topics: ['Testing', 'Examples'],
      tableOfContents: [
        {
          id: 'intro',
          text: 'はじめに',
          level: 2,
        },
      ],
      content: '段落1\n\n段落2',
      truncated: true,
    });
  });

  it('ZennPort エラー時は失敗を返す', async () => {
    mockedGetArticleBySlug.mockRejectedValue(new Error(ZENN_API_ERROR));

    const result = await getZennArticleTool.handler(
      { slug: TEST_ZENN_ARTICLE_SLUG },
      mockToolContext
    );

    expect(result).toEqual({
      success: false,
      error: 'Failed to fetch Zenn article',
    });
  });
});
