import { describe, expect, it } from 'vitest';

import { TOKEN_LIMITS } from '@echo-chamber/core/echo/constants';
import {
  getEchoInstanceDefinition,
  type EchoInstanceDefinition,
} from '@echo-chamber/core/echo/instance-definitions';

import { resolveTokenLimitConfig } from './token-limit-config';

function createEnv(overrides: Partial<Env> = {}): Env {
  return {
    OPENAI_API_KEY: 'openai-key',
    ...overrides,
  } as Env;
}

function createDefinition(
  overrides: Partial<EchoInstanceDefinition> = {}
): EchoInstanceDefinition {
  return {
    ...getEchoInstanceDefinition('rin'),
    ...overrides,
  };
}

describe('resolveTokenLimitConfig', () => {
  it('instance definition の token limit 設定を返す', () => {
    expect(
      resolveTokenLimitConfig(createEnv(), getEchoInstanceDefinition('rin'))
    ).toEqual({
      dailyHardLimit: TOKEN_LIMITS.DAILY_HARD_LIMIT,
      dailySoftLimit: TOKEN_LIMITS.DAILY_SOFT_LIMIT,
      hardLimitBufferFactor: TOKEN_LIMITS.HARD_LIMIT_BUFFER_FACTOR,
    });
  });

  it('instance 環境変数で token limit を上書きできる', () => {
    expect(
      resolveTokenLimitConfig(
        createEnv({
          RIN_DAILY_HARD_TOKEN_LIMIT: '700000',
          RIN_DAILY_SOFT_TOKEN_LIMIT: '420000',
          RIN_HARD_TOKEN_LIMIT_BUFFER_FACTOR: '1.25',
        } as Partial<Env>),
        getEchoInstanceDefinition('rin')
      )
    ).toEqual({
      dailyHardLimit: 700_000,
      dailySoftLimit: 420_000,
      hardLimitBufferFactor: 1.25,
    });
  });

  it('definition が空の項目は global 環境変数で補完できる', () => {
    expect(
      resolveTokenLimitConfig(
        createEnv({
          DAILY_HARD_TOKEN_LIMIT: '600000',
          DAILY_SOFT_TOKEN_LIMIT: '300000',
          HARD_TOKEN_LIMIT_BUFFER_FACTOR: '1.4',
        } as Partial<Env>),
        createDefinition({
          tokenLimits: {},
        })
      )
    ).toEqual({
      dailyHardLimit: 600_000,
      dailySoftLimit: 300_000,
      hardLimitBufferFactor: 1.4,
    });
  });

  it('不正な数値はエラーにする', () => {
    expect(() =>
      resolveTokenLimitConfig(
        createEnv({
          RIN_DAILY_HARD_TOKEN_LIMIT: 'invalid',
        } as Partial<Env>),
        getEchoInstanceDefinition('rin')
      )
    ).toThrow('RIN_DAILY_HARD_TOKEN_LIMIT must be a positive number.');
  });

  it('soft limit が hard limit を超える場合はエラーにする', () => {
    expect(() =>
      resolveTokenLimitConfig(
        createEnv({
          RIN_DAILY_HARD_TOKEN_LIMIT: '100',
          RIN_DAILY_SOFT_TOKEN_LIMIT: '101',
        } as Partial<Env>),
        getEchoInstanceDefinition('rin')
      )
    ).toThrow(
      'DAILY_SOFT_TOKEN_LIMIT must be less than or equal to DAILY_HARD_TOKEN_LIMIT.'
    );
  });
});
