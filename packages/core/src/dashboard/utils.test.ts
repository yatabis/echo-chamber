import { describe, expect, it } from 'vitest';

import { buildUsageSeries, filterNotes, sumUsage } from './utils';

import type { Note, UsageRecord } from '../echo/types';

describe('buildUsageSeries', () => {
  it('7日分の系列を生成し、欠損日はnullを入れる', () => {
    const usageRecord: UsageRecord = {
      '2026-02-20': {
        cached_input_tokens: 10,
        uncached_input_tokens: 20,
        total_input_tokens: 30,
        output_tokens: 5,
        reasoning_tokens: 2,
        total_tokens: 35,
        total_cost: 0.001,
      },
      '2026-02-22': {
        cached_input_tokens: 5,
        uncached_input_tokens: 15,
        total_input_tokens: 20,
        output_tokens: 4,
        reasoning_tokens: 1,
        total_tokens: 24,
        total_cost: 0.0008,
      },
    };

    const series = buildUsageSeries(
      usageRecord,
      7,
      new Date('2026-02-22T12:00:00+09:00')
    );

    expect(series).toHaveLength(7);
    expect(series[0]?.dateKey).toBe('2026-02-16');
    expect(series[6]?.dateKey).toBe('2026-02-22');
    expect(series[0]?.usage).toBeNull();
    expect(series[4]?.usage?.total_tokens).toBe(35);
    expect(series[6]?.usage?.total_tokens).toBe(24);
  });

  it('30日分の系列を生成する', () => {
    const series = buildUsageSeries(
      {},
      30,
      new Date('2026-02-22T12:00:00+09:00')
    );

    expect(series).toHaveLength(30);
    expect(series[0]?.dateKey).toBe('2026-01-24');
    expect(series[29]?.dateKey).toBe('2026-02-22');
  });
});

describe('sumUsage', () => {
  it('トークンとコストの合計を返す', () => {
    const series = buildUsageSeries(
      {
        '2026-02-21': {
          cached_input_tokens: 10,
          uncached_input_tokens: 10,
          total_input_tokens: 20,
          output_tokens: 5,
          reasoning_tokens: 1,
          total_tokens: 25,
          total_cost: 0.001,
        },
        '2026-02-22': {
          cached_input_tokens: 2,
          uncached_input_tokens: 3,
          total_input_tokens: 5,
          output_tokens: 4,
          reasoning_tokens: 0,
          total_tokens: 9,
          total_cost: 0.0004,
        },
      },
      7,
      new Date('2026-02-22T12:00:00+09:00')
    );

    const total = sumUsage(series);

    expect(total.totalTokens).toBe(34);
    expect(total.totalCost).toBeCloseTo(0.0014, 10);
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
