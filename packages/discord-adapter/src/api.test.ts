import { REST } from '@discordjs/rest';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  addReactionToMessage,
  getChannelMessages,
  getCurrentUser,
  sendChannelMessage,
} from './api';

const restInstances: MockREST[] = [];
let nextGetResult: unknown;
let nextPostResult: unknown;
let nextPutResult: unknown;

class MockREST {
  readonly get = vi.fn<(route: string, options?: unknown) => Promise<unknown>>(
    async () => await Promise.resolve(nextGetResult)
  );
  readonly post = vi.fn<(route: string, options?: unknown) => Promise<unknown>>(
    async () => await Promise.resolve(nextPostResult)
  );
  readonly put = vi.fn<(route: string) => Promise<unknown>>(
    async () => await Promise.resolve(nextPutResult)
  );
  readonly setToken = vi.fn<(token: string) => MockREST>().mockReturnThis();
}

vi.mock('@discordjs/rest', () => ({
  REST: vi.fn().mockImplementation(() => {
    const rest = new MockREST();
    restInstances.push(rest);
    return rest;
  }),
}));

function getRestInstance(): MockREST {
  const rest = restInstances[0];
  if (!rest) {
    throw new Error('Expected REST instance to be created');
  }

  return rest;
}

describe('discord api helpers', () => {
  beforeEach(() => {
    restInstances.length = 0;
    nextGetResult = undefined;
    nextPostResult = undefined;
    nextPutResult = undefined;
  });

  it('getChannelMessages は query 付きで GET する', async () => {
    const expected = [{ id: 'message-1' }];
    nextGetResult = expected;

    const result = await getChannelMessages('token', 'channel-1', {
      limit: 10,
      before: 'message-9',
    });

    const rest = getRestInstance();

    expect(result).toBe(expected);
    expect(vi.mocked(REST)).toHaveBeenCalledWith({
      handlerSweepInterval: 0,
      hashSweepInterval: 0,
    });
    expect(rest.setToken).toHaveBeenCalledWith('token');
    expect(rest.get).toHaveBeenCalledTimes(1);
    expect(rest.get.mock.calls[0]?.[0]).toBe('/channels/channel-1/messages');
    expect(
      (
        rest.get.mock.calls[0]?.[1] as { query: URLSearchParams }
      ).query.toString()
    ).toBe('limit=10&before=message-9');
  });

  it('sendChannelMessage は body 付きで POST する', async () => {
    const expected = { id: 'message-1' };
    nextPostResult = expected;

    const result = await sendChannelMessage('token', 'channel-1', {
      content: 'hello',
    });

    const rest = getRestInstance();

    expect(result).toBe(expected);
    expect(vi.mocked(REST)).toHaveBeenCalledWith({
      handlerSweepInterval: 0,
      hashSweepInterval: 0,
    });
    expect(rest.setToken).toHaveBeenCalledWith('token');
    expect(rest.post).toHaveBeenCalledWith('/channels/channel-1/messages', {
      body: { content: 'hello' },
    });
  });

  it('addReactionToMessage は URL エンコードした route へ PUT する', async () => {
    nextPutResult = undefined;

    await addReactionToMessage('token', 'channel-1', 'message-1', '👍');

    const rest = getRestInstance();

    expect(vi.mocked(REST)).toHaveBeenCalledWith({
      handlerSweepInterval: 0,
      hashSweepInterval: 0,
    });
    expect(rest.setToken).toHaveBeenCalledWith('token');
    expect(rest.put).toHaveBeenCalledWith(
      '/channels/channel-1/messages/message-1/reactions/%F0%9F%91%8D/@me'
    );
  });

  it('getCurrentUser は /users/@me を取得する', async () => {
    const expected = { id: 'bot-user-1' };
    nextGetResult = expected;

    const result = await getCurrentUser('token');

    const rest = getRestInstance();

    expect(result).toBe(expected);
    expect(vi.mocked(REST)).toHaveBeenCalledWith({
      handlerSweepInterval: 0,
      hashSweepInterval: 0,
    });
    expect(rest.setToken).toHaveBeenCalledWith('token');
    expect(rest.get).toHaveBeenCalledWith('/users/@me');
  });
});
