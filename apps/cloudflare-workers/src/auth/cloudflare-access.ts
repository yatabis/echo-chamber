import { createRemoteJWKSet, jwtVerify } from 'jose';

import type { MiddlewareHandler } from 'hono';

const ACCESS_JWT_HEADER = 'Cf-Access-Jwt-Assertion';

type RemoteJwkSet = ReturnType<typeof createRemoteJWKSet>;

const remoteJwkSets = new Map<string, RemoteJwkSet>();

export interface AccessJwtVerificationInput {
  audience: string;
  issuer: string;
  token: string;
}

export type AccessJwtVerifier = (
  input: AccessJwtVerificationInput
) => Promise<void>;

export interface CloudflareAccessMiddlewareOptions {
  verifyJwt?: AccessJwtVerifier;
}

/**
 * 空ではない Worker binding を取得する。
 *
 * @param value binding value
 * @param bindingName binding name used for diagnostics
 * @returns 前後空白を除去した binding value
 * @throws binding が未設定または空の場合
 */
function getRequiredBinding(
  value: string | undefined,
  bindingName: string
): string {
  const normalized = value?.trim();

  if (normalized === undefined || normalized.length === 0) {
    throw new Error(`Missing required Worker binding: ${bindingName}`);
  }

  return normalized;
}

/**
 * Access team domain を issuer と JWKS URL に使える origin へ正規化する。
 *
 * @param value `ACCESS_TEAM_DOMAIN` binding value
 * @returns HTTPS origin
 * @throws HTTPS origin 以外が設定されている場合
 */
function normalizeTeamDomain(value: string | undefined): string {
  const teamDomain = getRequiredBinding(value, 'ACCESS_TEAM_DOMAIN');
  const url = new URL(teamDomain);

  if (
    url.protocol !== 'https:' ||
    url.username.length > 0 ||
    url.password.length > 0 ||
    url.pathname !== '/' ||
    url.search.length > 0 ||
    url.hash.length > 0
  ) {
    throw new Error('ACCESS_TEAM_DOMAIN must be an HTTPS origin');
  }

  return url.origin;
}

/**
 * request hostname に対応する Access application AUD を解決する。
 *
 * Production と Preview で別 AUD を要求し、一方の application token を
 * 他方へ replay できないようにする。未知の hostname は fail closed とする。
 *
 * @param hostname request hostname
 * @param env Worker bindings
 * @returns hostname に対応する AUD
 * @throws hostname が未設定または未知の場合
 */
function resolveAudience(hostname: string, env: Env): string {
  const normalizedHostname = hostname.toLowerCase();
  const productionHostname = getRequiredBinding(
    env.ACCESS_PRODUCTION_HOSTNAME,
    'ACCESS_PRODUCTION_HOSTNAME'
  ).toLowerCase();

  if (normalizedHostname === productionHostname) {
    return getRequiredBinding(
      env.ACCESS_PRODUCTION_AUD,
      'ACCESS_PRODUCTION_AUD'
    );
  }

  const previewHostnameSuffix = getRequiredBinding(
    env.ACCESS_PREVIEW_HOSTNAME_SUFFIX,
    'ACCESS_PREVIEW_HOSTNAME_SUFFIX'
  ).toLowerCase();

  if (
    normalizedHostname.length > previewHostnameSuffix.length &&
    normalizedHostname.endsWith(previewHostnameSuffix)
  ) {
    return getRequiredBinding(env.ACCESS_PREVIEW_AUD, 'ACCESS_PREVIEW_AUD');
  }

  throw new Error(`No Access audience configured for hostname: ${hostname}`);
}

/**
 * Access team domain ごとに jose の remote JWKS resolver を再利用する。
 *
 * `createRemoteJWKSet` が持つ JWKS cache と key rotation 処理を Worker isolate 内で
 * request 間共有し、Dashboard API request ごとの外部 fetch を避ける。
 *
 * @param issuer Access team domain
 * @returns remote JWKS resolver
 */
function getRemoteJwkSet(issuer: string): RemoteJwkSet {
  const cached = remoteJwkSets.get(issuer);
  if (cached !== undefined) {
    return cached;
  }

  const jwks = createRemoteJWKSet(
    new URL('/cdn-cgi/access/certs', `${issuer}/`)
  );
  remoteJwkSets.set(issuer, jwks);
  return jwks;
}

/**
 * Cloudflare Access application token の署名、issuer、AUD、時刻 claim を検証する。
 *
 * @param input JWT verification input
 */
async function verifyCloudflareAccessJwt(
  input: AccessJwtVerificationInput
): Promise<void> {
  await jwtVerify(input.token, getRemoteJwkSet(input.issuer), {
    audience: input.audience,
    issuer: input.issuer,
  });
}

/**
 * 認証失敗時の cache 不可 403 response を作る。
 *
 * 詳細な失敗理由は外部へ返さず、設定値や JWT 検証情報の露出を避ける。
 *
 * @returns forbidden response
 */
function createForbiddenResponse(): Response {
  return new Response('Forbidden', {
    status: 403,
    headers: {
      'Cache-Control': 'no-store',
      'Content-Type': 'text/plain; charset=UTF-8',
    },
  });
}

/**
 * Cloudflare Access JWT を検証する Hono middleware を作る。
 *
 * ローカル開発だけは Access を経由しないため bypass する。それ以外の環境では
 * binding、hostname、JWT header、署名、issuer、AUD のいずれかが不正なら
 * fail closed で 403 を返す。
 *
 * @param options dependency injection options for tests
 * @returns Hono middleware
 */
export function createCloudflareAccessMiddleware(
  options: CloudflareAccessMiddlewareOptions = {}
): MiddlewareHandler<{ Bindings: Env }> {
  const verifyJwt = options.verifyJwt ?? verifyCloudflareAccessJwt;

  return async (c, next): Promise<Response | undefined> => {
    if (c.env.ENVIRONMENT === 'local') {
      await next();
      return;
    }

    const token = c.req.header(ACCESS_JWT_HEADER);
    if (token === undefined || token.length === 0) {
      return createForbiddenResponse();
    }

    try {
      const issuer = normalizeTeamDomain(c.env.ACCESS_TEAM_DOMAIN);
      const audience = resolveAudience(new URL(c.req.url).hostname, c.env);
      await verifyJwt({ audience, issuer, token });
    } catch {
      return createForbiddenResponse();
    }

    await next();
  };
}
