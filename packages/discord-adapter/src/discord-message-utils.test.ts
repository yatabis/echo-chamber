import { describe, expect, it } from 'vitest';

import { truncateForDiscord } from './discord-message-utils';

describe('truncateForDiscord', () => {
  it('上限以内ならそのまま返す', () => {
    expect(truncateForDiscord('hello')).toBe('hello');
  });

  it('ちょうど 2000 文字でもそのまま返す', () => {
    const content = 'a'.repeat(2000);

    expect(truncateForDiscord(content)).toBe(content);
  });

  it('2000 文字を超える場合は suffix 付きで切り詰める', () => {
    const content = 'a'.repeat(2100);
    const result = truncateForDiscord(content);

    expect(result).toBe(`${content.substring(0, 1986)}...(truncated)`);
    expect(result).toHaveLength(2000);
  });
});
