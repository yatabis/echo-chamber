import { Hono, type Context } from 'hono';

import {
  parseDashboardInstanceSummary,
  parseDashboardInstancesResponse,
} from '@echo-chamber/contracts/dashboard/schemas';
import type {
  DashboardInstanceSummary,
  DashboardInstancesResponse,
} from '@echo-chamber/contracts/dashboard/types';
import {
  ECHO_INSTANCE_IDS,
  isValidInstanceId,
} from '@echo-chamber/core/types/echo-config';

import { Echo } from './echo';

const app = new Hono<{ Bindings: Env }>();

type AppContext = Context<{ Bindings: Env }>;

/**
 * Dashboard summary 取得に失敗した instance の代替値を作る。
 */
function createUnknownDashboardInstanceSummary(
  instanceId: DashboardInstanceSummary['id']
): DashboardInstanceSummary {
  return {
    id: instanceId,
    name: instanceId,
    state: 'Unknown',
    nextAlarm: null,
    noteCount: 0,
    memoryCount: 0,
    todayUsageTokens: 0,
    sevenDayUsageTokens: 0,
    thirtyDayUsageTokens: 0,
    latestNoteUpdatedAt: null,
    latestMemoryUpdatedAt: null,
  };
}

/**
 * Dashboard SPA のエントリ HTML を返す。
 *
 * `/dashboard` 直アクセス時と、`/dashboard/*` でアセットが見つからなかった時の
 * fallback の両方から利用される。
 *
 * @param c Hono context
 * @returns `public/dashboard/index.html` のレスポンス
 */
async function serveDashboardIndex(c: AppContext): Promise<Response> {
  const requestUrl = new URL(c.req.url);
  requestUrl.pathname = '/dashboard/';

  return await c.env.ASSETS.fetch(
    new Request(requestUrl.toString(), c.req.raw)
  );
}

/**
 * `/dashboard/*` パスが静的アセット要求かを判定する。
 *
 * SPA ルート（例: `/dashboard/rin`）と区別するため、`assets/` 配下または
 * 拡張子付きファイル名を持つパスを「静的アセット要求」とみなす。
 *
 * @param pathname リクエストの pathname
 * @returns 静的アセット要求であれば `true`
 */
function isDashboardAssetPath(pathname: string): boolean {
  const relativePath = pathname.replace(/^\/dashboard\/?/, '');
  if (relativePath.length === 0) {
    return false;
  }

  return (
    relativePath.startsWith('assets/') ||
    /\/[^/]+\.[a-zA-Z0-9]+$/.test(`/${relativePath}`)
  );
}

/**
 * `/:instanceId` リクエストを Durable Object へ委譲する。
 *
 * `instanceId` が不正な場合は 404 を返し、正しい場合のみ該当インスタンスの
 * Durable Object `fetch` にフォワードする。
 *
 * @param c Hono context
 * @returns Durable Object からのレスポンス
 */
async function forwardToInstance(c: AppContext): Promise<Response> {
  const instanceId = c.req.param('instanceId');

  if (!isValidInstanceId(instanceId)) {
    return c.notFound();
  }

  const id = c.env.ECHO.idFromName(instanceId);
  const echo = c.env.ECHO.get(id);

  return await echo.fetch(c.req.raw);
}

app.get('/', (c) => {
  return c.text('E.C.H.O Chamber is running.');
});

app.get('/instances', async (c) => {
  const requestUrl = new URL(c.req.url);
  const origin = requestUrl.origin;

  // 一覧は「1件失敗しても全体を落とさない」方針で、各インスタンスを個別に回収する。
  const instances = await Promise.all(
    ECHO_INSTANCE_IDS.map(
      async (instanceId): Promise<DashboardInstanceSummary> => {
        try {
          const id = c.env.ECHO.idFromName(instanceId);
          const echo = c.env.ECHO.get(id);
          const summaryResponse = await echo.fetch(
            new Request(`${origin}/${instanceId}/summary`)
          );

          if (!summaryResponse.ok) {
            throw new Error(
              `Failed to fetch summary for ${instanceId}: ${summaryResponse.status}`
            );
          }

          return parseDashboardInstanceSummary(
            await summaryResponse.json<unknown>()
          );
        } catch {
          return createUnknownDashboardInstanceSummary(instanceId);
        }
      }
    )
  );

  return c.json<DashboardInstancesResponse>(
    parseDashboardInstancesResponse({ instances })
  );
});

app.get('/dashboard', async (c) => {
  return await serveDashboardIndex(c);
});

app.get('/dashboard/*', async (c) => {
  const pathname = new URL(c.req.url).pathname;
  const directAssetResponse = await c.env.ASSETS.fetch(c.req.raw);

  if (isDashboardAssetPath(pathname)) {
    return directAssetResponse;
  }

  if (
    directAssetResponse.status === 200 ||
    directAssetResponse.status === 304
  ) {
    return directAssetResponse;
  }

  return await serveDashboardIndex(c);
});

app.all('/:instanceId', async (c) => {
  return await forwardToInstance(c);
});

app.all('/:instanceId/*', async (c) => {
  return await forwardToInstance(c);
});

export { Echo };

export default app satisfies ExportedHandler<Env>;
