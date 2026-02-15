import { describe, it, expect } from 'vitest';

import { formatDatetime, formatDate, formatElapsedTime } from './datetime';

describe('formatDatetime', () => {
  it('基本的な日時フォーマット（Asia/Tokyo）', () => {
    const date = new Date('2025-09-06T02:30:45Z');

    const result = formatDatetime(date);

    expect(result).toBe('2025/09/06 11:30:45');
  });

  it('深夜の時間を正しく表示', () => {
    const date = new Date('2025-09-06T18:00:00Z');

    const result = formatDatetime(date);

    expect(result).toBe('2025/09/07 03:00:00');
  });

  it('24時間表記で表示', () => {
    const date = new Date('2025-09-06T03:15:30Z');

    const result = formatDatetime(date);

    expect(result).toBe('2025/09/06 12:15:30');
  });

  it('タイムゾーン UTC を指定', () => {
    const date = new Date('2025-09-06T15:30:45Z');

    const result = formatDatetime(date, 'UTC');

    expect(result).toBe('2025/09/06 15:30:45');
  });

  it('タイムゾーン America/New_York を指定', () => {
    const date = new Date('2025-09-06T15:30:45Z');

    const result = formatDatetime(date, 'America/New_York');

    // New York は UTC-4 または UTC-5 (夏時間による)
    // 9月なので夏時間適用でUTC-4
    expect(result).toBe('2025/09/06 11:30:45');
  });

  it('月跨ぎの時間変換', () => {
    const date = new Date('2025-02-28T18:00:00Z');

    const result = formatDatetime(date);

    expect(result).toBe('2025/03/01 03:00:00');
  });

  it('年跨ぎの時間変換', () => {
    const date = new Date('2024-12-31T18:00:00Z');

    const result = formatDatetime(date);

    expect(result).toBe('2025/01/01 03:00:00');
  });
});

describe('formatDate', () => {
  it('基本的な日付フォーマット（Asia/Tokyo）', () => {
    const date = new Date('2025-09-06T15:30:45Z');

    const result = formatDate(date);

    expect(result).toBe('2025-09-07');
  });

  it('タイムゾーン UTC を指定', () => {
    const date = new Date('2025-09-06T15:30:45Z');

    const result = formatDate(date, 'UTC');

    expect(result).toBe('2025-09-06');
  });

  it('タイムゾーン America/New_York を指定', () => {
    const date = new Date('2025-09-06T15:30:45Z');

    const result = formatDate(date, 'America/New_York');

    // New York は UTC-4 (夏時間) なので同日
    expect(result).toBe('2025-09-06');
  });

  it('月跨ぎの日付変換', () => {
    const date = new Date('2025-02-28T18:00:00Z');

    const result = formatDate(date);

    expect(result).toBe('2025-03-01');
  });

  it('年跨ぎの日付変換', () => {
    const date = new Date('2024-12-31T18:00:00Z');

    const result = formatDate(date);

    expect(result).toBe('2025-01-01');
  });

  it('うるう年の2月29日', () => {
    const date = new Date('2024-02-29T12:00:00Z');

    const result = formatDate(date);

    expect(result).toBe('2024-02-29');
  });

  it('1桁の月と日が0埋めされる', () => {
    const date = new Date('2025-01-05T12:00:00Z');

    const result = formatDate(date);

    expect(result).toBe('2025-01-05');
  });
});

describe('formatElapsedTime', () => {
  it('0分経過の場合は "0分前"', () => {
    const now = new Date('2025-07-29T12:00:00Z');
    const since = new Date('2025-07-29T12:00:00Z');

    const result = formatElapsedTime(since, now);

    expect(result).toBe('0分前');
  });

  it('30分経過の場合は "30分前"', () => {
    const now = new Date('2025-07-29T12:30:00Z');
    const since = new Date('2025-07-29T12:00:00Z');

    const result = formatElapsedTime(since, now);

    expect(result).toBe('30分前');
  });

  it('59分経過の場合は "59分前"', () => {
    const now = new Date('2025-07-29T12:59:00Z');
    const since = new Date('2025-07-29T12:00:00Z');

    const result = formatElapsedTime(since, now);

    expect(result).toBe('59分前');
  });

  it('1時間経過の場合は "1時間前"', () => {
    const now = new Date('2025-07-29T13:00:00Z');
    const since = new Date('2025-07-29T12:00:00Z');

    const result = formatElapsedTime(since, now);

    expect(result).toBe('1時間前');
  });

  it('1時間30分経過の場合は "1時間前" に丸められる', () => {
    const now = new Date('2025-07-29T13:30:00Z');
    const since = new Date('2025-07-29T12:00:00Z');

    const result = formatElapsedTime(since, now);

    expect(result).toBe('1時間前');
  });

  it('23時間59分経過の場合は "23時間前"', () => {
    const now = new Date('2025-07-30T11:59:00Z');
    const since = new Date('2025-07-29T12:00:00Z');

    const result = formatElapsedTime(since, now);

    expect(result).toBe('23時間前');
  });

  it('24時間経過の場合は "1日前"', () => {
    const now = new Date('2025-07-30T12:00:00Z');
    const since = new Date('2025-07-29T12:00:00Z');

    const result = formatElapsedTime(since, now);

    expect(result).toBe('1日前');
  });

  it('2日3時間15分経過の場合は "2日前" に丸められる', () => {
    const now = new Date('2025-07-31T15:15:00Z');
    const since = new Date('2025-07-29T12:00:00Z');

    const result = formatElapsedTime(since, now);

    expect(result).toBe('2日前');
  });

  it('文字列形式の日時を受け付ける', () => {
    const now = new Date('2025-07-29T14:00:00Z');
    const since = '2025-07-29T12:00:00Z';

    const result = formatElapsedTime(since, now);

    expect(result).toBe('2時間前');
  });

  it('負の差分（未来の日時）の場合は "0分前"', () => {
    const now = new Date('2025-07-29T11:00:00Z');
    const since = new Date('2025-07-29T12:00:00Z');

    const result = formatElapsedTime(since, now);

    expect(result).toBe('0分前');
  });
});
