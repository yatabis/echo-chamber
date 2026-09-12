import { describe, expect, it, vi } from 'vitest';

import { readWebPageTool } from './web';

import type { ToolContext } from './tool';
import type {
  WebPageDocument,
  WebPageReaderPort,
} from '../../ports/web-page-reader';

function createDocument(
  overrides: Partial<WebPageDocument> = {}
): WebPageDocument {
  return {
    requestedUrl: 'https://example.com/article?lang=ja',
    finalUrl: 'https://www.example.com/article?lang=ja',
    retrievedAt: '2026-08-10T00:00:00.000Z',
    httpStatus: 200,
    redirectCount: 1,
    contentType: 'text/html',
    title: 'Example article',
    contentFormat: 'markdown',
    rendering: 'static',
    content: '# Example article\n\n本文',
    extractedCharacters: 21,
    documentTruncated: false,
    truncationReasons: [],
    links: [
      {
        text: 'Primary source',
        url: 'https://example.org/source',
      },
    ],
    ...overrides,
  };
}

function createContext(webPageReader: WebPageReaderPort): ToolContext {
  return {
    chat: {
      readMessages: vi.fn().mockResolvedValue([]),
      sendMessage: vi.fn().mockResolvedValue(undefined),
      addReaction: vi.fn().mockResolvedValue(undefined),
    },
    notifications: {
      getNotificationSummary: vi.fn().mockResolvedValue([]),
    },
    memory: {
      store: vi.fn().mockResolvedValue(undefined),
      search: vi.fn().mockResolvedValue([]),
    },
    notes: {
      list: vi.fn().mockResolvedValue([]),
      get: vi.fn().mockResolvedValue(null),
      search: vi.fn().mockResolvedValue([]),
      create: vi.fn(),
      update: vi.fn().mockResolvedValue(null),
      delete: vi.fn().mockResolvedValue(false),
    },
    zenn: {
      listTrendingArticles: vi.fn().mockResolvedValue([]),
      getArticleBySlug: vi.fn(),
    },
    webPageReader,
  };
}

describe('readWebPageTool', () => {
  it('port結果をsource/document/trustへ写像する', async () => {
    const readPage = vi.fn<WebPageReaderPort['readPage']>().mockResolvedValue({
      success: true,
      document: createDocument(),
    });

    const output = JSON.parse(
      await readWebPageTool.execute(
        JSON.stringify({ url: 'https://example.com/article?lang=ja' }),
        createContext({ readPage })
      )
    ) as unknown;

    expect(readPage).toHaveBeenCalledWith(
      'https://example.com/article?lang=ja'
    );
    expect(output).toEqual({
      success: true,
      source: {
        requestedUrl: 'https://example.com/article?lang=ja',
        finalUrl: 'https://www.example.com/article?lang=ja',
        retrievedAt: '2026-08-10T00:00:00.000Z',
        httpStatus: 200,
        redirectCount: 1,
        contentType: 'text/html',
        title: 'Example article',
      },
      document: {
        format: 'markdown',
        rendering: 'static',
        text: '# Example article\n\n本文',
        returnedCharacters: 21,
        extractedCharacters: 21,
        truncated: false,
        truncationReasons: [],
        links: [
          {
            text: 'Primary source',
            url: 'https://example.org/source',
          },
        ],
      },
      trust: 'untrusted_external_content',
    });
  });

  it('maxCharactersを意味ブロック境界で適用する', async () => {
    const content = `${'あ'.repeat(400)}\n\n${'い'.repeat(400)}`;
    const readPage = vi.fn<WebPageReaderPort['readPage']>().mockResolvedValue({
      success: true,
      document: createDocument({
        content,
        extractedCharacters: content.length,
      }),
    });

    const output = JSON.parse(
      await readWebPageTool.execute(
        JSON.stringify({
          url: 'https://example.com/article',
          maxCharacters: 500,
        }),
        createContext({ readPage })
      )
    ) as {
      document: {
        text: string;
        returnedCharacters: number;
        extractedCharacters: number;
        truncated: boolean;
        truncationReasons: string[];
      };
    };

    expect(output.document).toMatchObject({
      text: 'あ'.repeat(400),
      returnedCharacters: 400,
      extractedCharacters: content.length,
      truncated: true,
      truncationReasons: ['output_character_limit'],
    });
  });

  it('portの安定errorをそのまま返す', async () => {
    const readPage = vi.fn<WebPageReaderPort['readPage']>().mockResolvedValue({
      success: false,
      code: 'timeout',
      error: 'The page request timed out.',
      retryable: true,
    });

    const output = JSON.parse(
      await readWebPageTool.execute(
        JSON.stringify({ url: 'https://example.com/article' }),
        createContext({ readPage })
      )
    ) as unknown;

    expect(output).toEqual({
      success: false,
      code: 'timeout',
      error: 'The page request timed out.',
      retryable: true,
    });
  });

  it('不正な最終shapeを安全なinternal_errorへ写像する', async () => {
    const readPage = vi.fn<WebPageReaderPort['readPage']>().mockResolvedValue({
      success: true,
      document: createDocument({ retrievedAt: 'not-a-datetime' }),
    });

    const output = await readWebPageTool.execute(
      JSON.stringify({ url: 'https://example.com/article' }),
      createContext({ readPage })
    );

    expect(JSON.parse(output)).toEqual({
      success: false,
      code: 'internal_error',
      error: 'The Web page could not be read because of an internal error.',
      retryable: false,
    });
  });

  it('予期しない例外のmessageをモデルへ返さない', async () => {
    const readPage = vi
      .fn<WebPageReaderPort['readPage']>()
      .mockRejectedValue(
        new Error('failed at https://secret.example/private?token=canary')
      );

    const output = await readWebPageTool.execute(
      JSON.stringify({ url: 'https://example.com/article' }),
      createContext({ readPage })
    );

    expect(JSON.parse(output)).toEqual({
      success: false,
      code: 'internal_error',
      error: 'The Web page could not be read because of an internal error.',
      retryable: false,
    });
    expect(output).not.toContain('secret.example');
    expect(output).not.toContain('canary');
  });
});
