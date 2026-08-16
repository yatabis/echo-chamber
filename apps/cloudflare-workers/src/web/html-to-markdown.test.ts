import { describe, expect, it } from 'vitest';

import { extractHtmlToMarkdown } from './html-to-markdown';

describe('extractHtmlToMarkdown', () => {
  it('最も実質的なmain候補を選び、意味構造をMarkdownへ保つ', async () => {
    const articleBody = `${'これは公開記事の本文です。'.repeat(
      20
    )}最後に検証対象となる重要な結論があります。`;
    const result = await extractHtmlToMarkdown(
      `<!doctype html>
      <html>
        <head><title>  公開記事 &amp; 検証  </title></head>
        <body>
          <nav>navigation-canary</nav>
          <main>
            <article>
              <h1>公開記事</h1>
              <p>${articleBody}</p>
              <h2>要点</h2>
              <ul><li>第一の要点</li><li>第二の要点</li></ul>
              <blockquote>外部本文中の引用</blockquote>
            </article>
          </main>
          <footer>footer-canary</footer>
          <script>script-canary</script>
        </body>
      </html>`,
      new URL('https://www.wikipedia.org/wiki/Test')
    );

    expect(result.title).toBe('公開記事 & 検証');
    expect(result.content).toContain('# 公開記事');
    expect(result.content).toContain('## 要点');
    expect(result.content).toContain('- 第一の要点');
    expect(result.content).toContain('> 外部本文中の引用');
    expect(result.content).toContain('重要な結論');
    expect(result.content).not.toContain('navigation-canary');
    expect(result.content).not.toContain('footer-canary');
    expect(result.content).not.toContain('script-canary');
  });

  it('短いmain候補しかない場合はbodyへfallbackする', async () => {
    const result = await extractHtmlToMarkdown(
      `<html><body><main><p>短いmain</p></main><section><p>${'body本文。'.repeat(
        50
      )}</p></section></body></html>`,
      new URL('https://www.wikipedia.org/page')
    );

    expect(result.content).toContain('短いmain');
    expect(result.content).toContain('body本文');
  });

  it('許可リンクだけを絶対URLへ解決し、重複metadataを作らない', async () => {
    const result = await extractHtmlToMarkdown(
      `<html><body><article><p>${'本文。'.repeat(80)}</p>
        <a href="/source#section">一次資料</a>
        <a href="/source">一次資料の再掲</a>
        <a href="#local">ページ内</a>
        <a href="https://www.google.com/path?token=canary">秘密リンク</a>
        <a href="mailto:test@example.com">メール</a>
      </article></body></html>`,
      new URL('https://www.wikipedia.org/wiki/Test')
    );

    expect(result.content).toContain(
      '[一次資料](https://www.wikipedia.org/source)'
    );
    expect(result.content).not.toContain('token=canary');
    expect(result.links).toEqual([
      {
        text: '一次資料',
        url: 'https://www.wikipedia.org/source',
      },
    ]);
  });

  it('table、code、Markdown記号を制御された表現にする', async () => {
    const result = await extractHtmlToMarkdown(
      `<html><body><article><p>${'本文。'.repeat(80)}</p>
        <p># page heading *not emphasis*</p>
        <pre><code>const value = \`\`\`unsafe\`\`\`;</code></pre>
        <table><tr><th>Name</th><th>Value</th></tr><tr><td>A</td><td>1</td></tr></table>
      </article></body></html>`,
      new URL('https://www.wikipedia.org/page')
    );

    expect(result.content).toContain('\\# page heading \\*not emphasis\\*');
    expect(result.content).toContain('~~~');
    expect(result.content).toContain('| Name | Value |');
    expect(result.content).toContain('| A | 1 |');
  });

  it('64,000文字をUnicode境界で切り詰める', async () => {
    const result = await extractHtmlToMarkdown(
      `<html><body><article><p>${'長'.repeat(70_000)}😀</p></article></body></html>`,
      new URL('https://www.wikipedia.org/long')
    );

    expect(result.extractedCharacters).toBeGreaterThan(64_000);
    expect(result.content.length).toBeLessThanOrEqual(64_000);
    expect(result.documentTruncated).toBe(true);
    expect(result.truncationReasons).toEqual(['document_character_limit']);
    expect(
      result.content.charCodeAt(result.content.length - 1)
    ).not.toBeGreaterThanOrEqual(0xd800);
  });
});
