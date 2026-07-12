import { Hono } from 'hono';
import { describe, expect, it, vi } from 'vitest';

import {
  createCloudflareAccessMiddleware,
  type AccessJwtVerifier,
} from './cloudflare-access';

const PRODUCTION_HOSTNAME = 'echo-chamber.yatabis.workers.dev';
const PREVIEW_HOSTNAME = 'version-123-echo-chamber.yatabis.workers.dev';
const TEAM_DOMAIN = 'https://yatabis.cloudflareaccess.com';

interface AccessTestEnvOverrides {
  ACCESS_PREVIEW_AUD?: string;
  ACCESS_PREVIEW_HOSTNAME_SUFFIX?: string;
  ACCESS_PRODUCTION_AUD?: string;
  ACCESS_PRODUCTION_HOSTNAME?: string;
  ACCESS_TEAM_DOMAIN?: string;
  ENVIRONMENT?: string;
}

/**
 * Access 認証テスト用の Worker bindings を作る。
 *
 * @param overrides 差し替える binding
 * @returns テスト用 bindings
 */
function createEnv(overrides: AccessTestEnvOverrides = {}): Env {
  return {
    ENVIRONMENT: 'production',
    ACCESS_TEAM_DOMAIN: TEAM_DOMAIN,
    ACCESS_PRODUCTION_HOSTNAME: PRODUCTION_HOSTNAME,
    ACCESS_PRODUCTION_AUD: 'production-aud',
    ACCESS_PREVIEW_HOSTNAME_SUFFIX: '-echo-chamber.yatabis.workers.dev',
    ACCESS_PREVIEW_AUD: 'preview-aud',
    ...overrides,
  } as unknown as Env;
}

/**
 * Access middleware だけを通る最小 Hono app を作る。
 *
 * @param verifyJwt JWT verifier test double
 * @returns テスト用 Hono app
 */
function createApp(verifyJwt: AccessJwtVerifier): Hono<{ Bindings: Env }> {
  const app = new Hono<{ Bindings: Env }>();
  app.use('*', createCloudflareAccessMiddleware({ verifyJwt }));
  app.get('/protected', (c) => c.text('OK'));
  return app;
}

describe('createCloudflareAccessMiddleware', () => {
  it('local 環境では Access JWT 検証を省略する', async () => {
    const verifyJwt = vi.fn<AccessJwtVerifier>();
    const app = createApp(verifyJwt);

    const response = await app.fetch(
      new Request('http://localhost/protected'),
      createEnv({ ENVIRONMENT: 'local' })
    );

    expect(response.status).toBe(200);
    expect(await response.text()).toBe('OK');
    expect(verifyJwt).not.toHaveBeenCalled();
  });

  it('production hostname は production AUD で検証する', async () => {
    const verifyJwt = vi.fn<AccessJwtVerifier>().mockResolvedValue(undefined);
    const app = createApp(verifyJwt);

    const response = await app.fetch(
      new Request(`https://${PRODUCTION_HOSTNAME}/protected`, {
        headers: {
          'Cf-Access-Jwt-Assertion': 'production-token',
        },
      }),
      createEnv()
    );

    expect(response.status).toBe(200);
    expect(verifyJwt).toHaveBeenCalledOnce();
    expect(verifyJwt).toHaveBeenCalledWith({
      audience: 'production-aud',
      issuer: TEAM_DOMAIN,
      token: 'production-token',
    });
  });

  it('preview hostname は preview AUD で検証する', async () => {
    const verifyJwt = vi.fn<AccessJwtVerifier>().mockResolvedValue(undefined);
    const app = createApp(verifyJwt);

    const response = await app.fetch(
      new Request(`https://${PREVIEW_HOSTNAME}/protected`, {
        headers: {
          'Cf-Access-Jwt-Assertion': 'preview-token',
        },
      }),
      createEnv()
    );

    expect(response.status).toBe(200);
    expect(verifyJwt).toHaveBeenCalledWith({
      audience: 'preview-aud',
      issuer: TEAM_DOMAIN,
      token: 'preview-token',
    });
  });

  it('JWT header が無ければ fail closed で 403 を返す', async () => {
    const verifyJwt = vi.fn<AccessJwtVerifier>();
    const app = createApp(verifyJwt);

    const response = await app.fetch(
      new Request(`https://${PRODUCTION_HOSTNAME}/protected`),
      createEnv()
    );

    expect(response.status).toBe(403);
    expect(response.headers.get('cache-control')).toBe('no-store');
    expect(await response.text()).toBe('Forbidden');
    expect(verifyJwt).not.toHaveBeenCalled();
  });

  it('未設定の hostname は fail closed で 403 を返す', async () => {
    const verifyJwt = vi.fn<AccessJwtVerifier>();
    const app = createApp(verifyJwt);

    const response = await app.fetch(
      new Request('https://dashboard.example.com/protected', {
        headers: {
          'Cf-Access-Jwt-Assertion': 'token',
        },
      }),
      createEnv()
    );

    expect(response.status).toBe(403);
    expect(verifyJwt).not.toHaveBeenCalled();
  });

  it('JWT 検証失敗の詳細を response に露出しない', async () => {
    const verifyJwt = vi
      .fn<AccessJwtVerifier>()
      .mockRejectedValue(new Error('signature mismatch'));
    const app = createApp(verifyJwt);

    const response = await app.fetch(
      new Request(`https://${PRODUCTION_HOSTNAME}/protected`, {
        headers: {
          'Cf-Access-Jwt-Assertion': 'invalid-token',
        },
      }),
      createEnv()
    );

    expect(response.status).toBe(403);
    expect(await response.text()).toBe('Forbidden');
  });

  it('必須の Access binding が無ければ fail closed で 403 を返す', async () => {
    const verifyJwt = vi.fn<AccessJwtVerifier>();
    const app = createApp(verifyJwt);

    const response = await app.fetch(
      new Request(`https://${PRODUCTION_HOSTNAME}/protected`, {
        headers: {
          'Cf-Access-Jwt-Assertion': 'token',
        },
      }),
      createEnv({ ACCESS_PRODUCTION_AUD: '' })
    );

    expect(response.status).toBe(403);
    expect(verifyJwt).not.toHaveBeenCalled();
  });
});
