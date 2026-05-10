import { describe, expect, it } from 'vitest';

import {
  buildUsageRatioMetrics,
  buildUsageStackedSeries,
  filterNotes,
  sumUsageBreakdown,
} from './utils';

import type { Note, UsageRecord } from './types';

describe('buildUsageStackedSeries', () => {
  it('7日分の系列を生成し、欠損日は0埋めする', () => {
    const usageRecord: UsageRecord = {
      '2026-02-20': {
        cached_input_tokens: 10,
        uncached_input_tokens: 20,
        total_input_tokens: 30,
        output_tokens: 7,
        reasoning_tokens: 2,
        total_tokens: 37,
        by_model: [
          {
            provider: 'openai',
            model: 'gpt-5',
            cached_input_tokens: 10,
            uncached_input_tokens: 20,
            total_input_tokens: 30,
            output_tokens: 7,
            reasoning_tokens: 2,
            total_tokens: 37,
          },
        ],
      },
      '2026-02-22': {
        cached_input_tokens: 5,
        uncached_input_tokens: 15,
        total_input_tokens: 20,
        output_tokens: 4,
        reasoning_tokens: 1,
        total_tokens: 24,
        by_model: [
          {
            provider: 'openai',
            model: 'gpt-5-mini',
            cached_input_tokens: 5,
            uncached_input_tokens: 15,
            total_input_tokens: 20,
            output_tokens: 4,
            reasoning_tokens: 1,
            total_tokens: 24,
          },
        ],
      },
    };

    const series = buildUsageStackedSeries(
      usageRecord,
      7,
      new Date('2026-02-22T12:00:00+09:00')
    );

    expect(series).toHaveLength(7);
    expect(series[0]?.dateKey).toBe('2026-02-16');
    expect(series[6]?.dateKey).toBe('2026-02-22');

    expect(series[0]).toMatchObject({
      cachedInputTokens: 0,
      uncachedInputTokens: 0,
      normalOutputTokens: 0,
      reasoningOutputTokens: 0,
      totalInputTokens: 0,
      totalOutputTokens: 0,
      totalTokens: 0,
      estimatedCostUsd: 0,
      estimatedCostIsPartial: false,
    });
    expect(series[4]).toMatchObject({
      cachedInputTokens: 10,
      uncachedInputTokens: 20,
      normalOutputTokens: 5,
      reasoningOutputTokens: 2,
      totalInputTokens: 30,
      totalOutputTokens: 7,
      totalTokens: 37,
      estimatedCostUsd: (10 * 0.125 + 20 * 1.25 + 7 * 10) / 1_000_000,
      estimatedCostIsPartial: false,
    });
    expect(series[6]).toMatchObject({
      cachedInputTokens: 5,
      uncachedInputTokens: 15,
      normalOutputTokens: 3,
      reasoningOutputTokens: 1,
      totalInputTokens: 20,
      totalOutputTokens: 4,
      totalTokens: 24,
      estimatedCostUsd: (5 * 0.025 + 15 * 0.25 + 4 * 2) / 1_000_000,
      estimatedCostIsPartial: false,
    });
  });

  it('host timezone が UTC でも Asia/Tokyo の usage 日付キーを生成する', () => {
    const previousTimezone = process.env.TZ;
    process.env.TZ = 'UTC';

    try {
      const usageRecord: UsageRecord = {
        '2026-02-22': {
          cached_input_tokens: 10,
          uncached_input_tokens: 20,
          total_input_tokens: 30,
          output_tokens: 7,
          reasoning_tokens: 2,
          total_tokens: 37,
          by_model: [
            {
              provider: 'openai',
              model: 'gpt-5',
              cached_input_tokens: 10,
              uncached_input_tokens: 20,
              total_input_tokens: 30,
              output_tokens: 7,
              reasoning_tokens: 2,
              total_tokens: 37,
            },
          ],
        },
      };

      const series = buildUsageStackedSeries(
        usageRecord,
        7,
        new Date('2026-02-22T12:00:00+09:00')
      );

      expect(series[0]?.dateKey).toBe('2026-02-16');
      expect(series[6]?.dateKey).toBe('2026-02-22');
      expect(series[6]?.totalTokens).toBe(37);
    } finally {
      if (previousTimezone === undefined) {
        delete process.env.TZ;
      } else {
        process.env.TZ = previousTimezone;
      }
    }
  });

  it('reasoning_tokens が output_tokens を超えるとエラーにする', () => {
    expect(() => {
      buildUsageStackedSeries(
        {
          '2026-02-22': {
            cached_input_tokens: 10,
            uncached_input_tokens: 10,
            total_input_tokens: 20,
            output_tokens: 5,
            reasoning_tokens: 6,
            total_tokens: 25,
            by_model: [
              {
                provider: 'openai',
                model: 'gpt-5',
                cached_input_tokens: 10,
                uncached_input_tokens: 10,
                total_input_tokens: 20,
                output_tokens: 5,
                reasoning_tokens: 6,
                total_tokens: 25,
              },
            ],
          },
        },
        7,
        new Date('2026-02-22T12:00:00+09:00')
      );
    }).toThrow(/reasoning_tokens exceeds output_tokens/);
  });

  it('by_model の合計が日次合計と一致しない場合はエラーにする', () => {
    expect(() => {
      buildUsageStackedSeries(
        {
          '2026-02-22': {
            cached_input_tokens: 10,
            uncached_input_tokens: 10,
            total_input_tokens: 20,
            output_tokens: 5,
            reasoning_tokens: 1,
            total_tokens: 25,
            by_model: [
              {
                provider: 'openai',
                model: 'gpt-5',
                cached_input_tokens: 1,
                uncached_input_tokens: 1,
                total_input_tokens: 2,
                output_tokens: 1,
                reasoning_tokens: 0,
                total_tokens: 3,
              },
            ],
          },
        },
        7,
        new Date('2026-02-22T12:00:00+09:00')
      );
    }).toThrow(/by_model totals are inconsistent/);
  });
});

describe('sumUsageBreakdown', () => {
  it('区分別トークンと合計値を返す', () => {
    const series = buildUsageStackedSeries(
      {
        '2026-02-21': {
          cached_input_tokens: 10,
          uncached_input_tokens: 10,
          total_input_tokens: 20,
          output_tokens: 5,
          reasoning_tokens: 1,
          total_tokens: 25,
          by_model: [
            {
              provider: 'openai',
              model: 'gpt-5',
              cached_input_tokens: 10,
              uncached_input_tokens: 10,
              total_input_tokens: 20,
              output_tokens: 5,
              reasoning_tokens: 1,
              total_tokens: 25,
            },
          ],
        },
        '2026-02-22': {
          cached_input_tokens: 2,
          uncached_input_tokens: 3,
          total_input_tokens: 5,
          output_tokens: 4,
          reasoning_tokens: 0,
          total_tokens: 9,
          by_model: [
            {
              provider: 'openai',
              model: 'gpt-5-mini',
              cached_input_tokens: 2,
              uncached_input_tokens: 3,
              total_input_tokens: 5,
              output_tokens: 4,
              reasoning_tokens: 0,
              total_tokens: 9,
            },
          ],
        },
      },
      7,
      new Date('2026-02-22T12:00:00+09:00')
    );

    const totals = sumUsageBreakdown(series);

    expect(totals).toEqual({
      cachedInputTokens: 12,
      uncachedInputTokens: 13,
      normalOutputTokens: 8,
      reasoningOutputTokens: 1,
      totalInputTokens: 25,
      totalOutputTokens: 9,
      totalTokens: 34,
      estimatedCostUsd:
        (10 * 0.125 + 10 * 1.25 + 5 * 10) / 1_000_000 +
        (2 * 0.025 + 3 * 0.25 + 4 * 2) / 1_000_000,
      estimatedCostIsPartial: false,
    });
  });

  it('未対応 provider だけの場合は推定コストを null にして partial 扱いにする', () => {
    const series = buildUsageStackedSeries(
      {
        '2026-02-22': {
          cached_input_tokens: 10,
          uncached_input_tokens: 10,
          total_input_tokens: 20,
          output_tokens: 5,
          reasoning_tokens: 1,
          total_tokens: 25,
          by_model: [
            {
              provider: 'lmstudio',
              model: 'local-model',
              cached_input_tokens: 10,
              uncached_input_tokens: 10,
              total_input_tokens: 20,
              output_tokens: 5,
              reasoning_tokens: 1,
              total_tokens: 25,
            },
          ],
        },
      },
      7,
      new Date('2026-02-22T12:00:00+09:00')
    );

    expect(series[6]?.estimatedCostUsd).toBeNull();
    expect(series[6]?.estimatedCostIsPartial).toBe(true);
    expect(sumUsageBreakdown(series).estimatedCostUsd).toBeNull();
    expect(sumUsageBreakdown(series).estimatedCostIsPartial).toBe(true);
  });

  it('未対応 provider が混ざっても計算できる分の推定コストを返す', () => {
    const knownCostUsd = (5 * 0.125 + 15 * 1.25 + 4 * 10) / 1_000_000;
    const series = buildUsageStackedSeries(
      {
        '2026-02-22': {
          cached_input_tokens: 15,
          uncached_input_tokens: 25,
          total_input_tokens: 40,
          output_tokens: 9,
          reasoning_tokens: 1,
          total_tokens: 49,
          by_model: [
            {
              provider: 'openai',
              model: 'gpt-5',
              cached_input_tokens: 5,
              uncached_input_tokens: 15,
              total_input_tokens: 20,
              output_tokens: 4,
              reasoning_tokens: 1,
              total_tokens: 24,
            },
            {
              provider: 'lmstudio',
              model: 'local-model',
              cached_input_tokens: 10,
              uncached_input_tokens: 10,
              total_input_tokens: 20,
              output_tokens: 5,
              reasoning_tokens: 0,
              total_tokens: 25,
            },
          ],
        },
      },
      7,
      new Date('2026-02-22T12:00:00+09:00')
    );
    const totals = sumUsageBreakdown(series);

    expect(series[6]?.estimatedCostUsd).toBe(knownCostUsd);
    expect(series[6]?.estimatedCostIsPartial).toBe(true);
    expect(totals.estimatedCostUsd).toBe(knownCostUsd);
    expect(totals.estimatedCostIsPartial).toBe(true);
  });
});

describe('buildUsageRatioMetrics', () => {
  it('cache/uncached と input/output の比率を返す', () => {
    const ratios = buildUsageRatioMetrics({
      cachedInputTokens: 25,
      uncachedInputTokens: 75,
      normalOutputTokens: 30,
      reasoningOutputTokens: 10,
      totalInputTokens: 100,
      totalOutputTokens: 40,
      totalTokens: 140,
      estimatedCostUsd: 0.01,
      estimatedCostIsPartial: false,
    });

    expect(ratios.cacheRateInInput).toBeCloseTo(0.25, 10);
    expect(ratios.uncachedRateInInput).toBeCloseTo(0.75, 10);
    expect(ratios.inputRateInTotal).toBeCloseTo(100 / 140, 10);
    expect(ratios.outputRateInTotal).toBeCloseTo(40 / 140, 10);
  });

  it('分母が0のときは0を返す', () => {
    const ratios = buildUsageRatioMetrics({
      cachedInputTokens: 0,
      uncachedInputTokens: 0,
      normalOutputTokens: 0,
      reasoningOutputTokens: 0,
      totalInputTokens: 0,
      totalOutputTokens: 0,
      totalTokens: 0,
      estimatedCostUsd: 0,
      estimatedCostIsPartial: false,
    });

    expect(ratios).toEqual({
      cacheRateInInput: 0,
      uncachedRateInInput: 0,
      inputRateInTotal: 0,
      outputRateInTotal: 0,
    });
  });
});

describe('filterNotes', () => {
  const notes: Note[] = [
    {
      id: 'note-1',
      title: 'Alpha Topic',
      content: 'First content',
      createdAt: '2026-02-20T00:00:00.000Z',
      updatedAt: '2026-02-20T00:00:00.000Z',
    },
    {
      id: 'note-2',
      title: 'Meeting',
      content: 'Discuss beta release',
      createdAt: '2026-02-21T00:00:00.000Z',
      updatedAt: '2026-02-21T00:00:00.000Z',
    },
  ];

  it('空クエリ時は全件を返す', () => {
    expect(filterNotes(notes, '  ')).toEqual(notes);
  });

  it('タイトル・本文の部分一致でフィルタする（大文字小文字を無視）', () => {
    expect(filterNotes(notes, 'alpha')).toEqual([notes[0]]);
    expect(filterNotes(notes, 'BETA')).toEqual([notes[1]]);
  });
});
