import { Hono, type Context } from 'hono';

import { ECHO_INSTANCE_IDS, isValidInstanceId } from '@echo-chamber/core';
import type {
  DashboardInstanceSummary,
  DashboardInstancesResponse,
} from '@echo-chamber/core';

import { Echo } from './echo';

const app = new Hono<{ Bindings: Env }>();

type AppContext = Context<{ Bindings: Env }>;

/**
 * Dashboard で許容する state 文字列か判定する。
 *
 * @param value 検証対象の unknown 値
 * @return DashboardInstanceSummary の state として有効な文字列であれば true
 */
function isState(value: unknown): boolean {
  return (
    value === 'Idling' ||
    value === 'Running' ||
    value === 'Sleeping' ||
    value === 'Unknown'
  );
}

/**
 * `/instances` で取得した JSON が `DashboardInstanceSummary` として妥当か判定する。
 *
 * @param value 検証対象の unknown 値
 * @returns `DashboardInstanceSummary` として扱える場合 `true`
 */
function isDashboardInstanceSummary(
  value: unknown
): value is DashboardInstanceSummary {
  if (typeof value !== 'object' || value === null) {
    return false;
  }

  const record = value as Record<string, unknown>;
  return (
    (record.id === 'rin' || record.id === 'marie') &&
    typeof record.name === 'string' &&
    isState(record.state) &&
    (typeof record.nextAlarm === 'string' || record.nextAlarm === null)
  );
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
  requestUrl.pathname = '/dashboard/index.html';

  return await c.env.ASSETS.fetch(
    new Request(requestUrl.toString(), c.req.raw)
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

          const summary = await summaryResponse.json<unknown>();
          if (!isDashboardInstanceSummary(summary)) {
            throw new Error(`Invalid summary payload for ${instanceId}`);
          }

          return summary;
        } catch {
          return {
            id: instanceId,
            name: instanceId,
            state: 'Unknown',
            nextAlarm: null,
          };
        }
      }
    )
  );

  return c.json<DashboardInstancesResponse>({ instances });
});

app.get('/dashboard', async (c) => {
  return await serveDashboardIndex(c);
});

app.get('/dashboard/*', async (c) => {
  const directAssetResponse = await c.env.ASSETS.fetch(c.req.raw);
  if (directAssetResponse.status !== 404) {
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
