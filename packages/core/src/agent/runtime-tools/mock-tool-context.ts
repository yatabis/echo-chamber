import { vi } from 'vitest';

import type { ToolContext } from './tool';

export const mockToolContext: ToolContext = {
  chat: {
    readMessages: vi.fn().mockResolvedValue([]),
    sendMessage: vi.fn().mockResolvedValue(undefined),
    addReaction: vi.fn().mockResolvedValue(undefined),
  },
  notifications: {
    getNotificationSummary: vi.fn().mockResolvedValue([]),
  },
  memory: {
    store: vi.fn().mockResolvedValue(undefined),
    search: vi.fn().mockResolvedValue([]),
  },
  notes: {
    list: vi.fn().mockResolvedValue([]),
    get: vi.fn().mockResolvedValue(null),
    search: vi.fn().mockResolvedValue([]),
    create: vi.fn(),
    update: vi.fn().mockResolvedValue(null),
    delete: vi.fn().mockResolvedValue(false),
  },
  webPageReader: {
    readPage: vi.fn(),
  },
  zenn: {
    listTrendingArticles: vi.fn().mockResolvedValue([]),
    getArticleBySlug: vi.fn(),
  },
};
