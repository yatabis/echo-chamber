import { vi } from 'vitest';

// Discord 関連の依存関係をグローバルにモック
const mockChatPort = {
  readMessages: vi.fn().mockResolvedValue([]),
  sendMessage: vi.fn().mockResolvedValue(undefined),
  addReaction: vi.fn().mockResolvedValue(undefined),
};

const mockNotificationPort = {
  getNotificationSummary: vi.fn().mockResolvedValue({
    unreadCount: 0,
    latestMessagePreview: null,
  }),
};

const mockThoughtLog = {
  send: vi.fn().mockResolvedValue(undefined),
};

vi.mock('../src/discord/client', () => {
  return {
    createDiscordChatPort: vi.fn(() => mockChatPort),
    createDiscordNotificationPort: vi.fn(() => mockNotificationPort),
    DiscordThoughtLog: vi.fn(() => mockThoughtLog),
    getUnreadMessageCount: vi.fn(),
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
