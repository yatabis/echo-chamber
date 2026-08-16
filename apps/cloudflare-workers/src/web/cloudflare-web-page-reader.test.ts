import { describe, expect, it, vi } from 'vitest';

import {
  MAX_WEB_PAGE_BODY_BYTES,
  createCloudflareWebPageReader,
} from './cloudflare-web-page-reader';

function createHtmlResponse(body: string, init: ResponseInit = {}): Response {
  return new Response(body, {
    status: init.status ?? 200,
    headers: {
      'content-type': 'text/html; charset=utf-8',
      ...Object.fromEntries(new Headers(init.headers)),
    },
  });
}

describe('createCloudflareWebPageReader', () => {
  it('固定されたGET条件でHTMLを取得し、制御Markdownへ変換する', async () => {
    const fetcher = vi
      .fn<(input: string, init: RequestInit) => Promise<Response>>()
      .mockResolvedValue(
        createHtmlResponse(
          `<html><head><title>公開記事</title></head><body><article>
          <h1>公開記事</h1><p>${'本文です。'.repeat(60)}</p>
        </article></body></html>`
        )
      );
    const reader = createCloudflareWebPageReader({
      fetcher,
      now: () => new Date('2026-08-10T00:00:00.000Z'),
    });

    const result = await reader.readPage(
      'https://www.wikipedia.org/article#section'
    );

    expect(result.success).toBe(true);
    if (!result.success) {
      throw new Error('Expected Web page read to succeed');
    }
    expect(fetcher).toHaveBeenCalledWith(
      'https://www.wikipedia.org/article',
      expect.objectContaining({
        method: 'GET',
        redirect: 'manual',
        cache: 'no-store',
        credentials: 'omit',
        headers: {
          Accept: 'text/html, application/xhtml+xml, text/plain;q=0.8',
          'User-Agent': 'ECHO-Chamber-Public-Web-Reader/1.0',
        },
      })
    );
    expect(fetcher.mock.calls[0]?.[1].signal).toBeInstanceOf(AbortSignal);
    expect(result.document).toMatchObject({
      requestedUrl: 'https://www.wikipedia.org/article',
      finalUrl: 'https://www.wikipedia.org/article',
      retrievedAt: '2026-08-10T00:00:00.000Z',
      httpStatus: 200,
      redirectCount: 0,
      contentType: 'text/html',
      title: '公開記事',
      contentFormat: 'markdown',
      rendering: 'static',
      documentTruncated: false,
    });
    expect(result.document.content).toContain('# 公開記事');
  });

  it('text/plainを同じMarkdown contractへ正規化する', async () => {
    const fetcher = vi.fn().mockResolvedValue(
      new Response('見出し候補\n\n本文 *canary*\u0000', {
        headers: { 'content-type': 'text/plain; charset=us-ascii' },
      })
    );
    const reader = createCloudflareWebPageReader({ fetcher });

    const result = await reader.readPage('https://www.wikipedia.org/plain');

    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.code).toBe('unsupported_content_type');
    }
  });

  it('UTF-8 text/plainを正規化しMarkdown記号をescapeする', async () => {
    const fetcher = vi.fn().mockResolvedValue(
      new Response('見出し候補\n\n本文 *canary*', {
        headers: { 'content-type': 'text/plain; charset=utf-8' },
      })
    );
    const reader = createCloudflareWebPageReader({ fetcher });

    const result = await reader.readPage('https://www.wikipedia.org/plain');

    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.document).toMatchObject({
        contentType: 'text/plain',
        title: null,
        links: [],
      });
      expect(result.document.content).toContain('本文 \\*canary\\*');
    }
  });

  it('relative redirectを追跡して最終URLと回数を返す', async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(null, {
          status: 302,
          headers: { location: '/new-page' },
        })
      )
      .mockResolvedValueOnce(
        createHtmlResponse(
          `<html><body><main><p>${'redirected content '.repeat(
            20
          )}</p></main></body></html>`
        )
      );
    const reader = createCloudflareWebPageReader({ fetcher });

    const result = await reader.readPage('https://www.wikipedia.org/old-page');

    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.document.finalUrl).toBe(
        'https://www.wikipedia.org/new-page'
      );
      expect(result.document.redirectCount).toBe(1);
    }
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it('4回目のredirectを拒否する', async () => {
    const fetcher = vi.fn().mockImplementation(async (url: string) => {
      const current = new URL(url);
      const next = Number(current.searchParams.get('hop') ?? '0') + 1;
      return await Promise.resolve(
        new Response(null, {
          status: 302,
          headers: {
            location: `https://www.wikipedia.org/page?hop=${next}`,
          },
        })
      );
    });
    const reader = createCloudflareWebPageReader({ fetcher });

    const result = await reader.readPage(
      'https://www.wikipedia.org/page?hop=0'
    );

    expect(result).toMatchObject({
      success: false,
      code: 'redirect_limit',
      retryable: false,
    });
    expect(fetcher).toHaveBeenCalledTimes(4);
  });

  it('private redirectとHTTPS downgradeを拒否する', async () => {
    const privateFetcher = vi.fn().mockResolvedValue(
      new Response(null, {
        status: 302,
        headers: { location: 'http://127.0.0.1/private' },
      })
    );
    const downgradeFetcher = vi.fn().mockResolvedValue(
      new Response(null, {
        status: 302,
        headers: { location: 'http://www.wikipedia.org/page' },
      })
    );

    await expect(
      createCloudflareWebPageReader({ fetcher: privateFetcher }).readPage(
        'https://www.wikipedia.org/page'
      )
    ).resolves.toMatchObject({
      success: false,
      code: 'redirect_not_allowed',
    });
    await expect(
      createCloudflareWebPageReader({ fetcher: downgradeFetcher }).readPage(
        'https://www.wikipedia.org/page'
      )
    ).resolves.toMatchObject({
      success: false,
      code: 'redirect_not_allowed',
    });
  });

  it.each([
    [404, false],
    [408, true],
    [425, true],
    [429, true],
    [500, true],
  ])('HTTP %sを安定errorへ変換する', async (status, retryable) => {
    const reader = createCloudflareWebPageReader({
      fetcher: vi.fn().mockResolvedValue(new Response(null, { status })),
    });

    await expect(
      reader.readPage('https://www.wikipedia.org/page')
    ).resolves.toMatchObject({
      success: false,
      code: 'http_status',
      retryable,
    });
  });

  it('許可外MIMEとcharsetを拒否する', async () => {
    const binaryReader = createCloudflareWebPageReader({
      fetcher: vi.fn().mockResolvedValue(
        new Response('binary', {
          headers: { 'content-type': 'application/octet-stream' },
        })
      ),
    });
    const charsetReader = createCloudflareWebPageReader({
      fetcher: vi.fn().mockResolvedValue(
        new Response('text', {
          headers: { 'content-type': 'text/plain; charset=shift_jis' },
        })
      ),
    });

    await expect(
      binaryReader.readPage('https://www.wikipedia.org/file')
    ).resolves.toMatchObject({
      success: false,
      code: 'unsupported_content_type',
    });
    await expect(
      charsetReader.readPage('https://www.wikipedia.org/file')
    ).resolves.toMatchObject({
      success: false,
      code: 'unsupported_charset',
    });
  });

  it('Content-Lengthと実stream byteの両方へ1 MiB上限を適用する', async () => {
    const contentLengthReader = createCloudflareWebPageReader({
      fetcher: vi.fn().mockResolvedValue(
        new Response('short', {
          headers: {
            'content-type': 'text/plain',
            'content-length': String(MAX_WEB_PAGE_BODY_BYTES + 1),
          },
        })
      ),
    });
    const streamedReader = createCloudflareWebPageReader({
      fetcher: vi.fn().mockResolvedValue(
        new Response(new Uint8Array(MAX_WEB_PAGE_BODY_BYTES + 1), {
          headers: { 'content-type': 'text/plain' },
        })
      ),
    });

    await expect(
      contentLengthReader.readPage('https://www.wikipedia.org/large')
    ).resolves.toMatchObject({
      success: false,
      code: 'response_too_large',
    });
    await expect(
      streamedReader.readPage('https://www.wikipedia.org/large')
    ).resolves.toMatchObject({
      success: false,
      code: 'response_too_large',
    });
  });

  it('不正UTF-8とbinary-like textを拒否する', async () => {
    const invalidUtf8Reader = createCloudflareWebPageReader({
      fetcher: vi.fn().mockResolvedValue(
        new Response(new Uint8Array([0xc3, 0x28]), {
          headers: { 'content-type': 'text/plain; charset=utf-8' },
        })
      ),
    });
    const binaryTextReader = createCloudflareWebPageReader({
      fetcher: vi.fn().mockResolvedValue(
        new Response(new Uint8Array([0x41, 0x00, 0x42]), {
          headers: { 'content-type': 'text/plain' },
        })
      ),
    });

    await expect(
      invalidUtf8Reader.readPage('https://www.wikipedia.org/text')
    ).resolves.toMatchObject({
      success: false,
      code: 'invalid_text_encoding',
    });
    await expect(
      binaryTextReader.readPage('https://www.wikipedia.org/text')
    ).resolves.toMatchObject({
      success: false,
      code: 'unsupported_content_type',
    });
  });

  it('timeoutと空本文を安定errorへ変換する', async () => {
    const timeoutReader = createCloudflareWebPageReader({
      fetcher: vi
        .fn()
        .mockRejectedValue(
          new DOMException('deadline exceeded', 'TimeoutError')
        ),
    });
    const emptyReader = createCloudflareWebPageReader({
      fetcher: vi
        .fn()
        .mockResolvedValue(
          createHtmlResponse('<html><body><script>only</script></body></html>')
        ),
    });

    await expect(
      timeoutReader.readPage('https://www.wikipedia.org/page')
    ).resolves.toMatchObject({
      success: false,
      code: 'timeout',
      retryable: true,
    });
    await expect(
      emptyReader.readPage('https://www.wikipedia.org/page')
    ).resolves.toMatchObject({
      success: false,
      code: 'empty_content',
      retryable: false,
    });
  });
});
