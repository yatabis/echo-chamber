import { vi } from 'vitest';

// Discord 関連の依存関係をグローバルにモック
vi.mock('@echo-chamber/core/discord', async (importOriginal) => {
  const actual =
    // eslint-disable-next-line @typescript-eslint/consistent-type-imports
    await importOriginal<typeof import('@echo-chamber/core/discord')>();

  return {
    ...actual,
    addReactionToMessage: vi.fn(),
    getChannelMessages: vi.fn(),
    getCurrentUser: vi.fn(),
    getNotificationDetails: vi.fn(),
    sendChannelMessage: vi.fn(),
  };
});

// Logger をグローバルにモック
vi.mock('../src/utils/logger', () => {
  const mockLogger = {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
    discordNotifyLevel: 'debug',
  };

  return {
    Logger: vi.fn(() => mockLogger),
    createLogger: vi.fn(() => mockLogger),
  };
});
