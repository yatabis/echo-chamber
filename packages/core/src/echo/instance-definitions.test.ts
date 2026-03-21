import { describe, expect, it } from 'vitest';

import { ECHO_INSTANCE_IDS } from '../types/echo-config';

import {
  ECHO_INSTANCE_DEFINITIONS,
  getEchoInstanceDefinition,
} from './instance-definitions';

describe('echo instance definitions', () => {
  it('全 instance id を definition catalogue がカバーしている', () => {
    expect(Object.keys(ECHO_INSTANCE_DEFINITIONS).sort()).toEqual(
      [...ECHO_INSTANCE_IDS].sort()
    );
  });

  it.each(ECHO_INSTANCE_IDS)(
    '%s の definition を id 一致で取得できる',
    (instanceId) => {
      expect(getEchoInstanceDefinition(instanceId)).toMatchObject({
        id: instanceId,
      });
    }
  );
});
