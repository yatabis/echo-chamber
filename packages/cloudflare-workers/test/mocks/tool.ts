/* eslint-disable @typescript-eslint/no-unsafe-assignment */
import { env } from 'cloudflare:test';
import { vi } from 'vitest';

import type { EchoInstanceConfig } from '@echo-chamber/core';

import { createLogger } from '../../src/utils/logger';

import type { MemorySystem } from '../../src/echo/memory-system';
import type { NoteSystem } from '../../src/echo/note-system';
import type { ToolContext } from '../../src/llm/openai/functions';

export const mockStorage = {
  get: vi.fn(),
  put: vi.fn(),
  delete: vi.fn(),
  list: vi.fn(),
  deleteAll: vi.fn(),
  getAlarm: vi.fn(),
  setAlarm: vi.fn(),
  deleteAlarm: vi.fn(),
  sync: vi.fn(),
  transaction: vi.fn(),
  sql: {
    exec: vi.fn(),
    databaseSize: 0,
    Cursor: vi.fn(),
    Statement: vi.fn(),
  },
  transactionSync: vi.fn(),
  getCurrentBookmark: vi.fn(),
  getBookmarkForTime: vi.fn(),
  onNextSessionRestoreBookmark: vi.fn(),
};

const mockLogger = createLogger(env);
vi.spyOn(mockLogger, 'debug').mockImplementation(vi.fn());
vi.spyOn(mockLogger, 'info').mockImplementation(vi.fn());
vi.spyOn(mockLogger, 'warn').mockImplementation(vi.fn());
vi.spyOn(mockLogger, 'error').mockImplementation(vi.fn());

const mockMemorySystem: MemorySystem = {
  storeMemory: vi.fn(),
  searchMemory: vi.fn().mockResolvedValue([]),
} as unknown as MemorySystem;

const mockNoteSystem: NoteSystem = {
  createNote: vi.fn(),
  listNotes: vi.fn().mockResolvedValue([]),
  getNote: vi.fn().mockResolvedValue(null),
  searchNotes: vi.fn().mockResolvedValue([]),
  updateNote: vi.fn().mockResolvedValue(null),
  deleteNote: vi.fn().mockResolvedValue(false),
} as unknown as NoteSystem;

const mockDurableObjectStorage = mockStorage as DurableObjectStorage;

export const mockInstanceConfig: EchoInstanceConfig = {
  id: 'rin',
  name: 'テスト用リン',
  systemPrompt: 'Test system prompt',
  discordBotToken: 'mock-discord-token',
  chatChannelId: 'mock-chat-channel-id',
  thinkingChannelId: 'mock-thinking-channel-id',
};

export const mockToolContext: ToolContext = {
  instanceConfig: mockInstanceConfig,
  storage: mockDurableObjectStorage,
  memorySystem: mockMemorySystem,
  noteSystem: mockNoteSystem,
  logger: mockLogger,
};
