import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

import {
  addUsage,
  convertUsage,
  calculateDynamicTokenLimit,
  getTodayUsageKey,
} from './usage';

import type { Usage, UsageRecord } from './types';
import type { ResponseUsage } from 'openai/resources/responses/responses';

describe('getTodayUsageKey', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('7時00分は当日の日付を返す', () => {
    vi.setSystemTime(new Date('2025-09-06T07:00:00+09:00'));

    const result = getTodayUsageKey();

    expect(result).toBe('2025-09-06');
  });

  it('6時59分は前日の日付を返す', () => {
    vi.setSystemTime(new Date('2025-09-06T06:59:59+09:00'));

    const result = getTodayUsageKey();

    expect(result).toBe('2025-09-05');
  });

  it('正午は当日の日付を返す', () => {
    vi.setSystemTime(new Date('2025-09-06T12:00:00+09:00'));

    const result = getTodayUsageKey();

    expect(result).toBe('2025-09-06');
  });

  it('23時59分は当日の日付を返す', () => {
    vi.setSystemTime(new Date('2025-09-06T23:59:59+09:00'));

    const result = getTodayUsageKey();

    expect(result).toBe('2025-09-06');
  });

  it('翌日0時00分は前日の日付を返す', () => {
    vi.setSystemTime(new Date('2025-09-07T00:00:00+09:00'));

    const result = getTodayUsageKey();

    expect(result).toBe('2025-09-06');
  });

  it('月跨ぎ: 3月1日の6時59分は2月末日の日付を返す', () => {
    vi.setSystemTime(new Date('2025-03-01T06:59:59+09:00'));

    const result = getTodayUsageKey();

    expect(result).toBe('2025-02-28');
  });

  it('年跨ぎ: 1月1日の6時59分は前年12月31日の日付を返す', () => {
    vi.setSystemTime(new Date('2025-01-01T06:59:59+09:00'));

    const result = getTodayUsageKey();

    expect(result).toBe('2024-12-31');
  });

  it('うるう年の月跨ぎ: 3月1日の6時59分は2月29日の日付を返す', () => {
    vi.setSystemTime(new Date('2024-03-01T06:59:59+09:00'));

    const result = getTodayUsageKey();

    expect(result).toBe('2024-02-29');
  });

  it('月跨ぎ確認: 3月1日の7時00分は当日の日付を返す', () => {
    vi.setSystemTime(new Date('2025-03-01T07:00:00+09:00'));

    const result = getTodayUsageKey();

    expect(result).toBe('2025-03-01');
  });

  it('年跨ぎ確認: 1月1日の7時00分は当日の日付を返す', () => {
    vi.setSystemTime(new Date('2025-01-01T07:00:00+09:00'));

    const result = getTodayUsageKey();

    expect(result).toBe('2025-01-01');
  });
});

describe('addUsage', () => {
  it('新しいキーを追加する', () => {
    const usageRecord: UsageRecord = {};
    const key = '2025-01-01';
    const usage: Usage = {
      cached_input_tokens: 100,
      uncached_input_tokens: 200,
      total_input_tokens: 300,
      output_tokens: 50,
      reasoning_tokens: 10,
      total_tokens: 350,
      total_cost: 0.5,
    };

    const result = addUsage(usageRecord, key, usage);

    expect(result[key]).toEqual(usage);
    expect(result).toBe(usageRecord);
  });

  it('既存のキーに使用量を累積する', () => {
    const existingUsage: Usage = {
      cached_input_tokens: 100,
      uncached_input_tokens: 200,
      total_input_tokens: 300,
      output_tokens: 50,
      reasoning_tokens: 10,
      total_tokens: 350,
      total_cost: 0.5,
    };

    const usageRecord: UsageRecord = {
      '2025-01-01': existingUsage,
    };

    const additionalUsage: Usage = {
      cached_input_tokens: 50,
      uncached_input_tokens: 100,
      total_input_tokens: 150,
      output_tokens: 25,
      reasoning_tokens: 5,
      total_tokens: 175,
      total_cost: 0.25,
    };

    const result = addUsage(usageRecord, '2025-01-01', additionalUsage);

    expect(result['2025-01-01']).toEqual({
      cached_input_tokens: 150,
      uncached_input_tokens: 300,
      total_input_tokens: 450,
      output_tokens: 75,
      reasoning_tokens: 15,
      total_tokens: 525,
      total_cost: 0.75,
    });
  });

  it('複数回の累積操作を行う', () => {
    const usageRecord: UsageRecord = {};
    const key = '2025-01-01';

    const usage1: Usage = {
      cached_input_tokens: 100,
      uncached_input_tokens: 200,
      total_input_tokens: 300,
      output_tokens: 50,
      reasoning_tokens: 10,
      total_tokens: 350,
      total_cost: 0.5,
    };

    const usage2: Usage = {
      cached_input_tokens: 50,
      uncached_input_tokens: 100,
      total_input_tokens: 150,
      output_tokens: 25,
      reasoning_tokens: 5,
      total_tokens: 175,
      total_cost: 0.25,
    };

    const usage3: Usage = {
      cached_input_tokens: 25,
      uncached_input_tokens: 50,
      total_input_tokens: 75,
      output_tokens: 10,
      reasoning_tokens: 2,
      total_tokens: 85,
      total_cost: 0.1,
    };

    addUsage(usageRecord, key, usage1);
    addUsage(usageRecord, key, usage2);
    const result = addUsage(usageRecord, key, usage3);

    expect(result[key]).toEqual({
      cached_input_tokens: 175,
      uncached_input_tokens: 350,
      total_input_tokens: 525,
      output_tokens: 85,
      reasoning_tokens: 17,
      total_tokens: 610,
      total_cost: 0.85,
    });
  });

  it('異なる複数のキーを処理する', () => {
    const usageRecord: UsageRecord = {};

    const usage1: Usage = {
      cached_input_tokens: 100,
      uncached_input_tokens: 200,
      total_input_tokens: 300,
      output_tokens: 50,
      reasoning_tokens: 10,
      total_tokens: 350,
      total_cost: 0.5,
    };

    const usage2: Usage = {
      cached_input_tokens: 50,
      uncached_input_tokens: 100,
      total_input_tokens: 150,
      output_tokens: 25,
      reasoning_tokens: 5,
      total_tokens: 175,
      total_cost: 0.25,
    };

    addUsage(usageRecord, '2025-01-01', usage1);
    const result = addUsage(usageRecord, '2025-01-02', usage2);

    expect(result['2025-01-01']).toEqual(usage1);
    expect(result['2025-01-02']).toEqual(usage2);
    expect(Object.keys(result)).toHaveLength(2);
  });
});

describe('convertUsage', () => {
  it('基本的なResponseUsageをUsageに変換する', () => {
    const responseUsage: ResponseUsage = {
      input_tokens: 1000,
      input_tokens_details: { cached_tokens: 200 },
      output_tokens: 500,
      output_tokens_details: { reasoning_tokens: 50 },
      total_tokens: 1500,
    };

    const result = convertUsage(responseUsage);

    expect(result).toEqual({
      cached_input_tokens: 200,
      uncached_input_tokens: 800, // 1000 - 200
      total_input_tokens: 1000,
      output_tokens: 500,
      reasoning_tokens: 50,
      total_tokens: 1500,
      total_cost: (200 * 0.125 + 800 * 1.25 + 500 * 10) / 1_000_000, // 0.006025
    });
  });

  it('料金計算が正確である', () => {
    const responseUsage: ResponseUsage = {
      input_tokens: 2000,
      input_tokens_details: { cached_tokens: 500 },
      output_tokens: 1000,
      output_tokens_details: { reasoning_tokens: 100 },
      total_tokens: 3000,
    };

    const result = convertUsage(responseUsage);

    // 期待される料金計算: (500 * 0.125 + 1500 * 1.25 + 1000 * 10) / 1,000,000
    const expectedCost = (500 * 0.125 + 1500 * 1.25 + 1000 * 10) / 1_000_000;

    expect(result.total_cost).toBeCloseTo(expectedCost, 10);
    // 正確な値: (62.5 + 1875 + 10000) / 1000000 = 11937.5 / 1000000 = 0.0119375
    expect(result.total_cost).toBeCloseTo(0.0119375, 10);
  });

  it('キャッシュトークンがゼロの場合', () => {
    const responseUsage: ResponseUsage = {
      input_tokens: 1000,
      input_tokens_details: { cached_tokens: 0 },
      output_tokens: 500,
      output_tokens_details: { reasoning_tokens: 25 },
      total_tokens: 1500,
    };

    const result = convertUsage(responseUsage);

    expect(result).toEqual({
      cached_input_tokens: 0,
      uncached_input_tokens: 1000,
      total_input_tokens: 1000,
      output_tokens: 500,
      reasoning_tokens: 25,
      total_tokens: 1500,
      total_cost: (0 * 0.125 + 1000 * 1.25 + 500 * 10) / 1_000_000, // 0.00625
    });
  });

  it('リーズニングトークンがゼロの場合', () => {
    const responseUsage: ResponseUsage = {
      input_tokens: 500,
      input_tokens_details: { cached_tokens: 100 },
      output_tokens: 300,
      output_tokens_details: { reasoning_tokens: 0 },
      total_tokens: 800,
    };

    const result = convertUsage(responseUsage);

    expect(result).toEqual({
      cached_input_tokens: 100,
      uncached_input_tokens: 400,
      total_input_tokens: 500,
      output_tokens: 300,
      reasoning_tokens: 0,
      total_tokens: 800,
      total_cost: (100 * 0.125 + 400 * 1.25 + 300 * 10) / 1_000_000, // 0.003512
    });
  });

  it('すべてのトークンがキャッシュされている場合', () => {
    const responseUsage: ResponseUsage = {
      input_tokens: 1000,
      input_tokens_details: { cached_tokens: 1000 },
      output_tokens: 200,
      output_tokens_details: { reasoning_tokens: 10 },
      total_tokens: 1200,
    };

    const result = convertUsage(responseUsage);

    expect(result).toEqual({
      cached_input_tokens: 1000,
      uncached_input_tokens: 0,
      total_input_tokens: 1000,
      output_tokens: 200,
      reasoning_tokens: 10,
      total_tokens: 1200,
      total_cost: (1000 * 0.125 + 0 * 1.25 + 200 * 10) / 1_000_000, // 0.002125
    });
  });
});

describe('calculateDynamicTokenLimit', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('Asia/Tokyo 午前7時における計算（開始時刻 = 0トークン）', () => {
    vi.setSystemTime(new Date('2025-01-01T07:00:00+09:00'));

    const tokenLimit = 1_000_000;
    const bufferFactor = 1.0;

    const result = calculateDynamicTokenLimit(tokenLimit, bufferFactor);

    // 7時が開始なので経過時間は0時間 / 20時間 = 0
    // 現在時刻までの按分: 1,000,000 * 0 = 0
    expect(result).toBe(0);
  });

  it('Asia/Tokyo 午後1時における計算（6時間経過 = 30%）', () => {
    vi.setSystemTime(new Date('2025-01-01T13:00:00+09:00'));

    const tokenLimit = 1_000_000;
    const bufferFactor = 1.0;

    const result = calculateDynamicTokenLimit(tokenLimit, bufferFactor);

    // 経過時間: 6時間 / 20時間 = 0.3
    // 現在時刻までの按分: 1,000,000 * 0.3 = 300,000
    expect(result).toBe(300_000);
  });

  it('Asia/Tokyo 午後5時における計算（10時間経過 = 50%）', () => {
    vi.setSystemTime(new Date('2025-01-01T17:00:00+09:00'));

    const tokenLimit = 1_000_000;
    const bufferFactor = 1.0;

    const result = calculateDynamicTokenLimit(tokenLimit, bufferFactor);

    // 経過時間: 10時間 / 20時間 = 0.5
    // 現在時刻までの按分: 1,000,000 * 0.5 = 500,000
    expect(result).toBe(500_000);
  });

  it('Asia/Tokyo 午前1時における計算（18時間経過、翌日）', () => {
    vi.setSystemTime(new Date('2025-01-02T01:00:00+09:00'));

    const tokenLimit = 1_000_000;
    const bufferFactor = 1.0;

    const result = calculateDynamicTokenLimit(tokenLimit, bufferFactor);

    // 経過時間: 18時間 / 20時間 = 0.9
    // 現在時刻までの按分: 1,000,000 * 0.9 = 900,000
    // バッファ適用: 900,000 * 1.0 = 900,000
    expect(result).toBe(900_000);
  });

  it('Asia/Tokyo 午前3時における計算（終了時刻 = 100%）', () => {
    vi.setSystemTime(new Date('2025-01-02T03:00:00+09:00'));

    const tokenLimit = 1_000_000;
    const bufferFactor = 1.0;

    const result = calculateDynamicTokenLimit(tokenLimit, bufferFactor);

    // 経過時間: 20時間 / 20時間 = 1.0
    // 現在時刻までの按分: 1,000,000 * 1.0 = 1,000,000
    expect(result).toBe(1_000_000);
  });

  it('Asia/Tokyo 午前5時における計算（計算対象外 = 0トークン）', () => {
    vi.setSystemTime(new Date('2025-01-01T05:00:00+09:00'));

    const tokenLimit = 1_000_000;
    const bufferFactor = 1.0;

    const result = calculateDynamicTokenLimit(tokenLimit, bufferFactor);

    // 対象外の時間は0トークンを返す
    expect(result).toBe(0);
  });

  it('Asia/Tokyo 午前7時0分1秒における計算（1秒経過）', () => {
    vi.setSystemTime(new Date('2025-01-01T07:00:01+09:00'));

    const tokenLimit = 1_000_000;
    const bufferFactor = 1.0;

    const result = calculateDynamicTokenLimit(tokenLimit, bufferFactor);

    // 1分未満の端数は切り捨て
    // 経過時間: 1秒 (= 0分) / 20時間 = 0 / 20 = 0
    // 現在時刻までの按分: 1,000,000 * 0 = 0
    expect(result).toBe(0);
  });

  it('Asia/Tokyo 午前7時1分0秒における計算（1分経過）', () => {
    vi.setSystemTime(new Date('2025-01-01T07:01:00+09:00'));

    const tokenLimit = 1_000_000;
    const bufferFactor = 1.0;

    const result = calculateDynamicTokenLimit(tokenLimit, bufferFactor);

    // 経過時間: 1分 / 20時間 = 1 / 1200 ≈ 0.000833333
    // 現在時刻までの按分: 1,000,000 * (1 / 1200) ≈ 833.333
    expect(result).toBe(833); // 小数点以下切り捨て
  });

  it('バッファファクターの適用', () => {
    vi.setSystemTime(new Date('2025-01-01T11:00:00+09:00'));

    const tokenLimit = 1_000_000;
    const bufferFactor = 1.5;

    const result = calculateDynamicTokenLimit(tokenLimit, bufferFactor);

    // 経過時間: 4時間 / 20時間 = 0.2
    // 現在時刻までの按分: 1,000,000 * 0.2 = 200,000
    // バッファ適用: 200,000 * 1.5 = 300,000
    expect(result).toBe(300_000);
  });

  it('デフォルトバッファファクター', () => {
    vi.setSystemTime(new Date('2025-01-01T15:30:00+09:00'));

    const tokenLimit = 500_000;

    const result = calculateDynamicTokenLimit(tokenLimit);

    // 経過時間: 8.5時間 / 20時間 = 0.425
    // 現在時刻までの按分: 500,000 * 0.425 = 212,500
    // バッファ適用: 212,500 * 1.0 = 212,500
    expect(result).toBe(212_500);
  });

  it('上限値適用', () => {
    vi.setSystemTime(new Date('2025-01-01T23:00:00+09:00'));

    const tokenLimit = 1_000_000;
    const bufferFactor = 2.0;

    const result = calculateDynamicTokenLimit(tokenLimit, bufferFactor);

    // 経過時間: 16時間 / 20時間 = 0.8
    // 現在時刻までの按分: 1,000,000 * 0.8 = 800,000
    // バッファ適用: 800,000 * 2.0 = 1,600,000
    // 上限適用: min(1,600,000, 1,000,000) = 1,000,000
    expect(result).toBe(1_000_000);
  });
});
