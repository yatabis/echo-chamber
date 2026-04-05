import { describe, expect, it, vi } from 'vitest';

import { createZennPort } from './create-zenn-port';

const TEST_ZENN_ARTICLE_SLUG = 'example-zenn-article';
const TEST_ZENN_ARTICLE_PATH = `/example_author/articles/${TEST_ZENN_ARTICLE_SLUG}`;
const TEST_ZENN_ARTICLE_URL = `https://zenn.dev${TEST_ZENN_ARTICLE_PATH}`;

function resolveRequestUrl(input: RequestInfo | URL): string {
  if (typeof input === 'string') {
    return input;
  }

  if (input instanceof URL) {
    return input.toString();
  }

  return input.url;
}

describe('createZennPort', () => {
  it('日次トレンド記事一覧を軽量 payload へ変換する', async () => {
    const fetchMock = vi.fn<typeof fetch>(async (input) => {
      const url = resolveRequestUrl(input);

      if (url === 'https://zenn.dev/api/articles?order=daily') {
        return await Promise.resolve(
          new Response(
            JSON.stringify({
              articles: [
                {
                  slug: TEST_ZENN_ARTICLE_SLUG,
                  path: TEST_ZENN_ARTICLE_PATH,
                  title: 'ダミーの Zenn 記事タイトル',
                  article_type: 'tech',
                  liked_count: 42,
                  bookmarked_count: 12,
                  published_at: '2026-04-01T10:00:00.000+09:00',
                  body_updated_at: '2026-04-02T11:30:00.000+09:00',
                  user: {
                    username: 'example_author',
                    name: 'Example Author',
                  },
                  publication: {
                    display_name: 'Example Publication',
                  },
                },
              ],
            }),
            {
              status: 200,
              headers: {
                'content-type': 'application/json',
              },
            }
          )
        );
      }

      throw new Error(`Unexpected URL: ${url}`);
    });

    const port = createZennPort(fetchMock);
    const result = await port.listTrendingArticles();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(result).toEqual([
      {
        slug: TEST_ZENN_ARTICLE_SLUG,
        url: TEST_ZENN_ARTICLE_URL,
        title: 'ダミーの Zenn 記事タイトル',
        articleType: 'tech',
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
      },
    ]);
  });

  it('記事本文を取得して詳細 payload へ変換する', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(
        JSON.stringify({
          article: {
            slug: TEST_ZENN_ARTICLE_SLUG,
            path: TEST_ZENN_ARTICLE_PATH,
            title: 'ダミーの Zenn 記事タイトル',
            article_type: 'tech',
            liked_count: 42,
            bookmarked_count: 12,
            published_at: '2026-04-01T10:00:00.000+09:00',
            body_updated_at: '2026-04-02T11:30:00.000+09:00',
            body_html:
              '<h2>はじめに</h2><p>Hello &amp; welcome<br>world</p><ul><li>one</li><li>two</li></ul><p><img alt="図" src="https://example.com/image.png"></p>',
            toc: [
              {
                id: 'intro',
                text: 'はじめに',
                level: 2,
                children: [
                  {
                    id: 'child',
                    text: '詳細',
                    level: 3,
                    children: [],
                  },
                ],
              },
            ],
            topics: [
              {
                display_name: 'Testing',
              },
              {
                display_name: 'Examples',
              },
            ],
            user: {
              username: 'example_author',
              name: 'Example Author',
            },
            publication: null,
          },
        }),
        {
          status: 200,
          headers: {
            'content-type': 'application/json',
          },
        }
      )
    );

    const port = createZennPort(fetchMock);
    const result = await port.getArticleBySlug(TEST_ZENN_ARTICLE_SLUG);

    expect(fetchMock).toHaveBeenCalledWith(
      `https://zenn.dev/api/articles/${TEST_ZENN_ARTICLE_SLUG}`,
      {
        headers: {
          accept: 'application/json',
        },
      }
    );
    expect(result).toEqual({
      slug: TEST_ZENN_ARTICLE_SLUG,
      url: TEST_ZENN_ARTICLE_URL,
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
        {
          id: 'child',
          text: '詳細',
          level: 3,
        },
      ],
      content:
        'はじめに\n\nHello & welcome\nworld\n\n- one\n- two\n\n[Image: 図]',
    });
  });

  it('Zenn API エラー時は例外を投げる', async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response('not found', {
        status: 404,
      })
    );

    const port = createZennPort(fetchMock);

    await expect(port.getArticleBySlug(TEST_ZENN_ARTICLE_SLUG)).rejects.toThrow(
      'Zenn API request failed: 404'
    );
  });
});
