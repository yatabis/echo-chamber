import { describe, expect, it } from 'vitest';

import type { DashboardInstanceSummary, EchoStatus } from '@echo-chamber/core';

import worker from './index';

interface MockInstanceBehavior {
  summaryStatus: number;
  summary: DashboardInstanceSummary;
  status: EchoStatus;
}

interface MockEnvOptions {
  failSummaryFor?: 'rin' | 'marie';
  redirectSpaAssetPath?: string;
}

interface MockEnvResult {
  env: Env;
  getDurableObjectFetchCount(): number;
}

function isInstancesPayload(
  value: unknown
): value is { instances: DashboardInstanceSummary[] } {
  if (typeof value !== 'object' || value === null) {
    return false;
  }

  const record = value as Record<string, unknown>;
  return Array.isArray(record.instances);
}

function isEchoStatus(value: unknown): value is EchoStatus {
  if (typeof value !== 'object' || value === null) {
    return false;
  }

  const record = value as Record<string, unknown>;
  return (
    typeof record.id === 'string' &&
    typeof record.name === 'string' &&
    (record.state === 'Idling' ||
      record.state === 'Running' ||
      record.state === 'Sleeping') &&
    (typeof record.nextAlarm === 'string' || record.nextAlarm === null)
  );
}

function createMockEnv(options: MockEnvOptions = {}): MockEnvResult {
  const instanceBehavior: Record<'rin' | 'marie', MockInstanceBehavior> = {
    rin: {
      summaryStatus: options.failSummaryFor === 'rin' ? 500 : 200,
      summary: {
        id: 'rin',
        name: 'リン',
        state: 'Idling',
        nextAlarm: '2026-02-22T12:00:00.000Z',
      },
      status: {
        id: 'rin',
        name: 'リン',
        state: 'Idling',
        nextAlarm: '2026-02-22T12:00:00.000Z',
        memories: [],
        notes: [],
        usage: {},
      },
    },
    marie: {
      summaryStatus: options.failSummaryFor === 'marie' ? 500 : 200,
      summary: {
        id: 'marie',
        name: 'マリー',
        state: 'Sleeping',
        nextAlarm: null,
      },
      status: {
        id: 'marie',
        name: 'マリー',
        state: 'Sleeping',
        nextAlarm: null,
        memories: [],
        notes: [],
        usage: {},
      },
    },
  };

  let durableObjectFetchCount = 0;

  const env = {
    ASSETS: {
      fetch: async (request: Request): Promise<Response> => {
        const pathname = new URL(request.url).pathname;
        if (pathname === '/dashboard/assets/app.js') {
          return await Promise.resolve(
            new Response('console.log("dashboard asset");', {
              status: 200,
              headers: {
                'content-type': 'application/javascript;charset=utf-8',
              },
            })
          );
        }

        if (
          options.redirectSpaAssetPath !== undefined &&
          pathname === options.redirectSpaAssetPath
        ) {
          return await Promise.resolve(
            new Response(null, {
              status: 307,
              headers: {
                location: '/dashboard/',
              },
            })
          );
        }

        if (
          pathname === '/dashboard/' ||
          pathname === '/dashboard/index.html'
        ) {
          return await Promise.resolve(
            new Response('<!doctype html><title>dashboard</title>', {
              status: 200,
              headers: {
                'content-type': 'text/html;charset=utf-8',
              },
            })
          );
        }

        return await Promise.resolve(
          new Response('not found', { status: 404 })
        );
      },
    },
    ECHO: {
      idFromName: (name: string): DurableObjectId => {
        return {
          toString: () => name,
          equals: () => false,
          name,
        } as unknown as DurableObjectId;
      },
      get: (id: DurableObjectId): DurableObjectStub => {
        const instanceId = id.toString() as 'rin' | 'marie';
        const behavior = instanceBehavior[instanceId];

        return {
          fetch: async (request: RequestInfo | URL): Promise<Response> => {
            durableObjectFetchCount += 1;
            const req =
              request instanceof Request ? request : new Request(request);
            const pathname = new URL(req.url).pathname;

            if (pathname === `/${instanceId}/summary`) {
              if (behavior.summaryStatus !== 200) {
                return await Promise.resolve(
                  new Response('summary error', {
                    status: behavior.summaryStatus,
                  })
                );
              }
              return await Promise.resolve(Response.json(behavior.summary));
            }

            if (
              pathname === `/${instanceId}` ||
              pathname === `/${instanceId}/`
            ) {
              return await Promise.resolve(Response.json(behavior.status));
            }

            if (
              pathname === `/${instanceId}/json` ||
              pathname === `/${instanceId}/usage`
            ) {
              return await Promise.resolve(
                new Response('not found', { status: 404 })
              );
            }

            return await Promise.resolve(
              new Response('not found', { status: 404 })
            );
          },
        } as unknown as DurableObjectStub;
      },
    },
  } as unknown as Env;

  return {
    env,
    getDurableObjectFetchCount: () => durableObjectFetchCount,
  };
}

async function request(pathname: string, env: Env): Promise<Response> {
  return await worker.fetch(
    new Request(`https://example.com${pathname}`),
    env,
    {} as ExecutionContext
  );
}

describe('worker routes', () => {
  it('/instances returns summaries and fallback Unknown on per-instance failure', async () => {
    const { env } = createMockEnv({ failSummaryFor: 'marie' });

    const response = await request('/instances', env);

    expect(response.status).toBe(200);

    const body = await response.json<unknown>();
    expect(isInstancesPayload(body)).toBe(true);

    if (!isInstancesPayload(body)) {
      throw new Error('Invalid /instances payload');
    }

    expect(body.instances).toHaveLength(2);
    expect(body.instances[0]).toEqual({
      id: 'rin',
      name: 'リン',
      state: 'Idling',
      nextAlarm: '2026-02-22T12:00:00.000Z',
    });
    expect(body.instances[1]).toEqual({
      id: 'marie',
      name: 'marie',
      state: 'Unknown',
      nextAlarm: null,
    });
  });

  it('/dashboard serves index.html', async () => {
    const mock = createMockEnv();

    const response = await request('/dashboard', mock.env);

    expect(response.status).toBe(200);
    expect(response.headers.get('content-type')).toContain('text/html');
    expect(await response.text()).toContain('dashboard');
    expect(mock.getDurableObjectFetchCount()).toBe(0);
  });

  it('/dashboard/rin uses SPA fallback and does not collide with /:instanceId', async () => {
    const mock = createMockEnv();

    const response = await request('/dashboard/rin', mock.env);

    expect(response.status).toBe(200);
    expect(await response.text()).toContain('dashboard');
    expect(mock.getDurableObjectFetchCount()).toBe(0);
  });

  it('/dashboard/rin falls back to index.html when ASSETS returns 307', async () => {
    const mock = createMockEnv({ redirectSpaAssetPath: '/dashboard/rin' });

    const response = await request('/dashboard/rin', mock.env);

    expect(response.status).toBe(200);
    expect(response.headers.get('content-type')).toContain('text/html');
    expect(await response.text()).toContain('dashboard');
    expect(mock.getDurableObjectFetchCount()).toBe(0);
  });

  it('/dashboard/assets/app.js is served directly from ASSETS', async () => {
    const mock = createMockEnv();

    const response = await request('/dashboard/assets/app.js', mock.env);

    expect(response.status).toBe(200);
    expect(response.headers.get('content-type')).toContain(
      'application/javascript'
    );
    expect(await response.text()).toContain('dashboard asset');
    expect(mock.getDurableObjectFetchCount()).toBe(0);
  });

  it('/:instanceId returns EchoStatus JSON', async () => {
    const { env } = createMockEnv();

    const response = await request('/rin', env);

    expect(response.status).toBe(200);

    const body = await response.json<unknown>();
    expect(isEchoStatus(body)).toBe(true);

    if (!isEchoStatus(body)) {
      throw new Error('Invalid EchoStatus payload');
    }

    expect(body.id).toBe('rin');
    expect(body.name).toBe('リン');
    expect(body.state).toBe('Idling');
  });
});
