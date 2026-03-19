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

vi.mock('@echo-chamber/discord-adapter/chat-port', () => {
  return {
    createDiscordChatPort: vi.fn(() => mockChatPort),
  };
});

vi.mock('@echo-chamber/discord-adapter/notification-port', () => {
  return {
    createDiscordNotificationPort: vi.fn(() => mockNotificationPort),
  };
});

vi.mock('@echo-chamber/discord-adapter/discord-thought-log', () => {
  return {
    DiscordThoughtLog: vi.fn(() => mockThoughtLog),
  };
});

vi.mock('@echo-chamber/discord-adapter/notification-utils', () => {
  return {
    getUnreadMessageCount: vi.fn(),
  };
});

vi.mock('@echo-chamber/discord-adapter/api', () => {
  return {
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
