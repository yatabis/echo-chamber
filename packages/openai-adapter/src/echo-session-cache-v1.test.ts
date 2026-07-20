import { describe, expect, it } from 'vitest';

import {
  ECHO_SESSION_CACHE_RUNTIME_PROFILE,
  createEchoSessionCacheRequestBodyExtension,
} from './echo-session-cache-v1';

describe('E.C.H.O. session-cache runtime extension v1', () => {
  it('profile 名を version 付きで公開する', () => {
    expect(ECHO_SESSION_CACHE_RUNTIME_PROFILE).toBe('echo-session-cache-v1');
  });

  it('最初の exchange は pinned、完了後は rolling slot を選ぶ', () => {
    const extendRequest =
      createEchoSessionCacheRequestBodyExtension('echo:rin');

    expect(extendRequest({ hasCompletedExchange: false })).toEqual({
      cache: {
        mode: 'auto',
        session_id: 'echo:rin',
        session_slot: 'pinned',
      },
    });
    expect(extendRequest({ hasCompletedExchange: true })).toEqual({
      cache: {
        mode: 'auto',
        session_id: 'echo:rin',
        session_slot: 'rolling',
      },
    });
  });
});
