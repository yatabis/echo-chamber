import { describe, expect, it } from 'vitest';

import { admitPublicWebUrl, admitWebRedirect } from './url-policy';

describe('admitPublicWebUrl', () => {
  it('通常のHTTP(S) URLを正規化しfragmentだけを除去する', () => {
    const result = admitPublicWebUrl(
      'https://www.wikipedia.org:443/wiki/Test?lang=ja#section'
    );

    expect(result.success).toBe(true);
    if (!result.success) {
      throw new Error('Expected URL admission to succeed');
    }
    expect(result.url.href).toBe('https://www.wikipedia.org/wiki/Test?lang=ja');
  });

  it.each(['ftp:', 'file:', 'data:', ['java', 'script:'].join(''), 'ws:'])(
    '%s schemeを拒否する',
    (scheme) => {
      const result = admitPublicWebUrl(`${scheme}//www.wikipedia.org/page`);

      expect(result).toMatchObject({
        success: false,
        code: 'invalid_url',
        retryable: false,
      });
    }
  );

  it('URL credentialと非default portを拒否する', () => {
    expect(
      admitPublicWebUrl('https://user:password@www.wikipedia.org/page')
    ).toMatchObject({ success: false, code: 'sensitive_url' });
    expect(
      admitPublicWebUrl('https://www.wikipedia.org:8443/page')
    ).toMatchObject({ success: false, code: 'invalid_url' });
  });

  it.each([
    'token',
    'ACCESS_TOKEN',
    'api-key',
    'signature',
    'x-amz-credential',
    'X-Goog-Signature',
  ])('既知のcredential-like query key %s を拒否する', (key) => {
    const result = admitPublicWebUrl(
      `https://www.wikipedia.org/page?${key}=canary`
    );

    expect(result).toMatchObject({
      success: false,
      code: 'sensitive_url',
      retryable: false,
    });
  });

  it('通常のquery keyは許可する', () => {
    expect(
      admitPublicWebUrl('https://www.google.com/search?q=cloudflare&page=2')
    ).toMatchObject({ success: true });
  });

  it.each([
    'localhost',
    'api.localhost.',
    'metadata.google.internal',
    'printer.local',
    'service.test',
    'example.com',
  ])('非公開または予約hostname %s を拒否する', (hostname) => {
    expect(admitPublicWebUrl(`https://${hostname}/`)).toMatchObject({
      success: false,
      code: 'destination_not_public',
    });
  });

  it.each([
    '127.0.0.1',
    '2130706433',
    '0x7f000001',
    '0177.0.0.1',
    '127.1',
    '10.0.0.1',
    '100.64.0.1',
    '169.254.169.254',
    '172.16.0.1',
    '192.168.1.1',
    '192.0.2.1',
    '198.18.0.1',
    '198.51.100.1',
    '203.0.113.1',
    '224.0.0.1',
    '255.255.255.255',
  ])('非公開またはspecial-use IPv4 %s を拒否する', (address) => {
    expect(admitPublicWebUrl(`http://${address}/`)).toMatchObject({
      success: false,
      code: 'destination_not_public',
    });
  });

  it.each([
    '[::]',
    '[::1]',
    '[::ffff:127.0.0.1]',
    '[fc00::1]',
    '[fe80::1]',
    '[ff02::1]',
    '[2001:db8::1]',
    '[2001:2::1]',
  ])('非公開またはspecial-use IPv6 %s を拒否する', (address) => {
    expect(admitPublicWebUrl(`http://${address}/`)).toMatchObject({
      success: false,
      code: 'destination_not_public',
    });
  });

  it('public IP literalとIDNA hostnameを許可する', () => {
    expect(admitPublicWebUrl('https://8.8.8.8/')).toMatchObject({
      success: true,
    });
    const idna = admitPublicWebUrl('https://日本語.jp/');
    expect(idna).toMatchObject({ success: true });
    if (idna.success) {
      expect(idna.url.hostname).toBe('xn--wgv71a119e.jp');
    }
  });
});

describe('admitWebRedirect', () => {
  it('relative redirectを解決する', () => {
    const result = admitWebRedirect(
      new URL('https://www.wikipedia.org/old/page'),
      '../new/page'
    );

    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.url.href).toBe('https://www.wikipedia.org/new/page');
    }
  });

  it('HTTPS downgradeと非公開destinationを拒否する', () => {
    expect(
      admitWebRedirect(
        new URL('https://www.wikipedia.org/page'),
        'http://www.wikipedia.org/page'
      )
    ).toMatchObject({ success: false, code: 'redirect_not_allowed' });
    expect(
      admitWebRedirect(
        new URL('https://www.wikipedia.org/page'),
        'http://127.0.0.1/private'
      )
    ).toMatchObject({ success: false, code: 'redirect_not_allowed' });
  });
});
