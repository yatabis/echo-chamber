import { beforeEach, describe, expect, it, vi } from 'vitest';

import { MemorySystem } from '@echo-chamber/cloudflare-runtime/memory-system';
import { canonicalRuntimeTools } from '@echo-chamber/core/agent/runtime-tools/catalog';
import { bindRuntimeTools } from '@echo-chamber/core/agent/runtime-tools/tool';
import { getEchoInstanceDefinition } from '@echo-chamber/core/echo/instance-definitions';

import { resolveEchoRuntimeBindings } from '../config/echo-runtime-bindings';
import { createEmbeddingService } from '../embedding/create-embedding-service';
import { createLogger } from '../utils/logger';

import { createToolExecutionContext } from './tool-context';

import { Echo } from './index';

const {
  mockEmbeddingService,
  mockExecutableTools,
  mockInstanceDefinition,
  mockLogger,
  mockMemorySystem,
  mockNoteSystem,
  mockRuntimeBindings,
  mockToolContext,
} = vi.hoisted(() => ({
  mockEmbeddingService: { provider: 'test-embedding' },
  mockExecutableTools: [{ name: 'tool-1' }],
  mockInstanceDefinition: {
    id: 'rin',
    name: 'リン',
    systemPrompt: '<persona>Rin</persona>',
  },
  mockLogger: {
    debug: vi.fn(async () => Promise.resolve()),
    info: vi.fn(async () => Promise.resolve()),
    warn: vi.fn(async () => Promise.resolve()),
    error: vi.fn(async () => Promise.resolve()),
  },
  mockMemorySystem: {
    reEmbedStaleMemories: vi.fn(async () => Promise.resolve()),
  },
  mockNoteSystem: {},
  mockRuntimeBindings: {
    discordBotToken: 'discord-token',
    chatChannelId: 'chat-channel',
    thinkingChannelId: 'thinking-channel',
    embeddingConfig: {
      provider: 'workersai' as const,
      model: '@cf/pfnet/plamo-embedding-1b',
    },
  },
  mockToolContext: { context: 'tool-context' },
}));

vi.mock('@echo-chamber/cloudflare-runtime/memory-system', () => ({
  MemorySystem: vi.fn(() => mockMemorySystem),
}));

vi.mock('@echo-chamber/cloudflare-runtime/note-system', () => ({
  NoteSystem: vi.fn(() => mockNoteSystem),
}));

vi.mock('cloudflare:workers', () => ({
  DurableObject: class {
    protected readonly ctx: DurableObjectState;
    protected readonly env: Env;

    constructor(ctx: DurableObjectState, env: Env) {
      this.ctx = ctx;
      this.env = env;
    }
  },
}));

vi.mock('@echo-chamber/core/agent/runtime-tools/catalog', () => ({
  canonicalRuntimeTools: ['runtime-tool-1', 'runtime-tool-2'],
}));

vi.mock('@echo-chamber/core/agent/runtime-tools/tool', () => ({
  bindRuntimeTools: vi.fn(() => mockExecutableTools),
}));

vi.mock('@echo-chamber/core/echo/instance-definitions', () => ({
  getEchoInstanceDefinition: vi.fn(() => mockInstanceDefinition),
}));

vi.mock('../config/echo-runtime-bindings', () => ({
  resolveEchoRuntimeBindings: vi.fn(async () =>
    Promise.resolve(mockRuntimeBindings)
  ),
}));

vi.mock('../embedding/create-embedding-service', () => ({
  createEmbeddingService: vi.fn(() => mockEmbeddingService),
}));

vi.mock('../utils/logger', () => ({
  createLogger: vi.fn(() => mockLogger),
}));

vi.mock('./tool-context', () => ({
  createToolExecutionContext: vi.fn(() => mockToolContext),
}));

function createMockStorage(): {
  storage: DurableObjectStorage;
  deleteFn: ReturnType<typeof vi.fn>;
  putFn: ReturnType<typeof vi.fn>;
} {
  const deleteFn = vi.fn(async () => Promise.resolve(false));
  const putFn = vi.fn(async () => Promise.resolve());

  return {
    storage: {
      delete: deleteFn,
      put: putFn,
      sql: { exec: vi.fn() },
    } as unknown as DurableObjectStorage,
    deleteFn,
    putFn,
  };
}

function createMockState(storage: DurableObjectStorage): DurableObjectState {
  return {
    storage,
    blockConcurrencyWhile: vi.fn(),
    waitUntil: vi.fn(),
  } as unknown as DurableObjectState;
}

function createMockEnv(): Env {
  return {
    ECHO_KV: {
      get: vi.fn(),
    },
    ENVIRONMENT: 'test',
  } as unknown as Env;
}

async function ensureInitialized(
  echo: Echo,
  id: 'rin' | 'marie'
): Promise<void> {
  await (
    echo as unknown as {
      ensureInitialized(instanceId: 'rin' | 'marie'): Promise<void>;
    }
  ).ensureInitialized(id);
}

describe('Echo.ensureInitialized', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('legacy key を削除せずに初期化する', async () => {
    const env = createMockEnv();
    const { storage, deleteFn, putFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);

    await ensureInitialized(echo, 'rin');

    expect(deleteFn).not.toHaveBeenCalled();
    expect(createLogger).toHaveBeenCalledWith(env);
    expect(getEchoInstanceDefinition).toHaveBeenCalledWith('rin');
    expect(resolveEchoRuntimeBindings).toHaveBeenCalledWith(
      env,
      env.ECHO_KV,
      'rin'
    );
    expect(createEmbeddingService).toHaveBeenCalledWith(
      env,
      mockRuntimeBindings.embeddingConfig
    );
    expect(MemorySystem).toHaveBeenCalledWith({
      sql: storage.sql,
      embeddingService: mockEmbeddingService,
      logger: mockLogger,
    });
    expect(createToolExecutionContext).toHaveBeenCalledWith({
      chatBindings: mockRuntimeBindings,
      memorySystem: mockMemorySystem,
      noteSystem: mockNoteSystem,
      logger: mockLogger,
    });
    expect(bindRuntimeTools).toHaveBeenCalledWith(
      canonicalRuntimeTools,
      mockToolContext
    );
    expect(mockMemorySystem.reEmbedStaleMemories).toHaveBeenCalledTimes(1);
    expect(putFn).toHaveBeenNthCalledWith(1, 'id', 'rin');
    expect(putFn).toHaveBeenNthCalledWith(2, 'name', 'リン');
  });

  it('同じ id で再初期化しても storage cleanup や再構築をしない', async () => {
    const env = createMockEnv();
    const { storage, deleteFn, putFn } = createMockStorage();
    const echo = new Echo(createMockState(storage), env);

    await ensureInitialized(echo, 'rin');
    vi.clearAllMocks();

    await ensureInitialized(echo, 'rin');

    expect(deleteFn).not.toHaveBeenCalled();
    expect(putFn).not.toHaveBeenCalled();
    expect(resolveEchoRuntimeBindings).not.toHaveBeenCalled();
    expect(createEmbeddingService).not.toHaveBeenCalled();
    expect(MemorySystem).not.toHaveBeenCalled();
    expect(createToolExecutionContext).not.toHaveBeenCalled();
    expect(bindRuntimeTools).not.toHaveBeenCalled();
    expect(mockMemorySystem.reEmbedStaleMemories).not.toHaveBeenCalled();
  });
});
