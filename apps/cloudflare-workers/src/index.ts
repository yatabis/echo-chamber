import { Hono } from 'hono';

import { isValidInstanceId } from '@echo-chamber/core';

import { Echo } from './echo';

const app = new Hono<{ Bindings: Env }>();

app.get('/', (c) => {
  return c.text('E.C.H.O Chamber is running.');
});

app.get('/dashboard', (c) => {
  return c.redirect('/dashboard/');
});

app.get('/dashboard/*', async (c) => {
  const directAssetResponse = await c.env.ASSETS.fetch(c.req.raw);
  if (directAssetResponse.status !== 404) {
    return directAssetResponse;
  }

  const requestUrl = new URL(c.req.url);
  requestUrl.pathname = '/dashboard/index.html';

  return await c.env.ASSETS.fetch(
    new Request(requestUrl.toString(), c.req.raw)
  );
});

app.all('/:instanceId/*', async (c) => {
  const instanceId = c.req.param('instanceId');

  if (!isValidInstanceId(instanceId)) {
    return c.notFound();
  }

  const id = c.env.ECHO.idFromName(instanceId);
  const echo = c.env.ECHO.get(id);

  return await echo.fetch(c.req.raw);
});

export { Echo };

export default app satisfies ExportedHandler<Env>;
