/* eslint-disable @typescript-eslint/require-await -- Stateful fixture ports implement asynchronous production contracts without external I/O. */

import {
  runAgentSession,
  ZERO_MODEL_USAGE,
} from '@echo-chamber/core/agent/session';
import type { ToolExecutionContext } from '@echo-chamber/core/agent/tool-context';
import type { Note } from '@echo-chamber/core/echo/types';
import type { ChatMessage } from '@echo-chamber/core/ports/chat';
import type {
  MemoryRecord,
  MemorySearchResult,
} from '@echo-chamber/core/ports/memory';
import type { ModelUsage } from '@echo-chamber/core/ports/model';
import type { ChannelNotificationSummary } from '@echo-chamber/core/ports/notification';
import type {
  ZennArticle,
  ZennTrendingArticleSummary,
} from '@echo-chamber/core/ports/zenn';

import {
  bindQwenEvaluationRuntimeTools,
  createRuntimeInitialInput,
  EvaluationTrace,
} from './runtime-harness';
import { summarizeChecks } from './scoring';

import type { RuntimeHarnessOptions } from './runtime-harness';
import type {
  RuntimeInjectedFault,
  RuntimeWorkflowFixture,
  RuntimeWorkflowSessionFixture,
} from './runtime-workflows';
import type {
  RuntimeContextSnapshot,
  RuntimeSessionTrace,
  RuntimeWorkflowResult,
  TraceCall,
} from './types';

interface SessionBoundary {
  callIndex: number;
  memoryCount: number;
  contextBefore: RuntimeContextSnapshot | null;
}

function cloneContext(
  context: RuntimeContextSnapshot | null
): RuntimeContextSnapshot | null {
  return context === null
    ? null
    : {
        ...context,
        emotion: { ...context.emotion, labels: [...context.emotion.labels] },
      };
}

function cloneMemories(memories: readonly MemoryRecord[]): MemoryRecord[] {
  return memories.map((memory) => ({
    ...memory,
    emotion: { ...memory.emotion, labels: [...memory.emotion.labels] },
  }));
}

function cloneNotes(notes: readonly Note[]): Note[] {
  return notes.map((note) => ({ ...note }));
}

class StatefulRuntimeWorld {
  readonly calls: TraceCall[] = [];
  readonly memories: MemoryRecord[];
  readonly notes: Note[];
  private readonly chatMessages = new Map<string, ChatMessage[]>();
  private readonly faultAttempts = new Map<string, number>();
  private activeSession: RuntimeWorkflowSessionFixture | null = null;
  private context: RuntimeContextSnapshot | null;

  constructor(
    private readonly fixture: RuntimeWorkflowFixture,
    private readonly startedAt: number
  ) {
    this.context = cloneContext(fixture.initialContext);
    this.memories = cloneMemories(fixture.initialMemories);
    this.notes = cloneNotes(fixture.initialNotes);
  }

  beginSession(session: RuntimeWorkflowSessionFixture): SessionBoundary {
    this.activeSession = session;
    if (session.clearContextBefore === true) {
      this.context = null;
    }
    for (const channelKey of session.clearChatHistoryBefore ?? []) {
      this.chatMessages.delete(channelKey);
    }
    for (const [channelKey, incoming] of Object.entries(
      session.incomingMessages
    )) {
      const current = this.chatMessages.get(channelKey) ?? [];
      this.chatMessages.set(channelKey, [...current, ...incoming]);
    }
    return {
      callIndex: this.calls.length,
      memoryCount: this.memories.length,
      contextBefore: cloneContext(this.context),
    };
  }

  callsSince(boundary: SessionBoundary): TraceCall[] {
    return this.calls.slice(boundary.callIndex);
  }

  loadContext(): RuntimeContextSnapshot | null {
    return cloneContext(this.context);
  }

  saveSessionContext(
    record: Pick<RuntimeContextSnapshot, 'content' | 'emotion'>
  ): void {
    const session = this.requireActiveSession();
    this.context = {
      content: record.content,
      emotion: {
        ...record.emotion,
        labels: [...record.emotion.labels],
      },
      createdAt: session.currentDatetime.toISOString(),
    };
  }

  loadRelatedMemories(
    context: RuntimeContextSnapshot | null
  ): MemorySearchResult[] {
    if (context === null) {
      return [];
    }
    const results = this.toSearchResults(this.memories);
    this.record('startup_memory_search', { query: context.content }, results);
    return results;
  }

  createToolContext(): ToolExecutionContext {
    return {
      notifications: {
        getNotificationSummary: async (): Promise<
          ChannelNotificationSummary[]
        > => {
          const result = this.requireActiveSession().notifications;
          this.record('notification_summary', {}, result);
          return result;
        },
      },
      chat: {
        readMessages: async (channelKey, limit): Promise<ChatMessage[]> => {
          const result = (this.chatMessages.get(channelKey) ?? []).slice(
            -limit
          );
          this.record('read_chat', { channelKey, limit }, result);
          return result;
        },
        sendMessage: async (channelKey, message): Promise<void> => {
          this.record('send_chat', { channelKey, message });
          const current = this.chatMessages.get(channelKey) ?? [];
          current.push({
            messageId: `fixture-agent-${this.calls.length}`,
            user: 'rin',
            message,
            createdAt:
              this.requireActiveSession().currentDatetime.toISOString(),
            reactions: [],
            images: [],
          });
          this.chatMessages.set(channelKey, current);
        },
        addReaction: async (channelKey, messageId, reaction): Promise<void> => {
          this.record('add_reaction', { channelKey, messageId, reaction });
        },
      },
      memory: {
        store: async (content, type): Promise<void> => {
          const now = this.requireActiveSession().currentDatetime.toISOString();
          const emotion = this.context?.emotion ?? {
            valence: 0,
            arousal: 0,
            labels: [],
          };
          const memory: MemoryRecord = {
            content,
            emotion: { ...emotion, labels: [...emotion.labels] },
            type,
            createdAt: now,
            updatedAt: now,
          };
          this.memories.push(memory);
          this.record('store_memory', { content, type }, memory);
        },
        search: async (query, type): Promise<MemorySearchResult[]> => {
          const matching =
            type === undefined
              ? this.memories
              : this.memories.filter((memory) => memory.type === type);
          const result = this.toSearchResults(matching);
          this.record('search_memory', { query, type }, result);
          return result;
        },
      },
      notes: this.createNotePort(),
      webPageReader: {
        readPage: async () => ({
          success: false,
          code: 'internal_error',
          error:
            'Public Web access is disabled in deterministic evaluation fixtures.',
          retryable: false,
        }),
      },
      zenn: {
        listTrendingArticles: async (): Promise<
          readonly ZennTrendingArticleSummary[]
        > => {
          this.record('list_zenn', {}, []);
          return [];
        },
        getArticleBySlug: async (slug): Promise<ZennArticle> => {
          const article = {
            slug,
            url: `https://zenn.dev/example/articles/${slug}`,
            title: 'Synthetic evaluation article',
            author: { username: 'fixture', name: 'Fixture' },
            topics: ['evaluation'],
            tableOfContents: [],
            content: 'Synthetic fixture content.',
          };
          this.record('get_zenn', { slug }, article);
          return article;
        },
      },
    };
  }

  private createNotePort(): ToolExecutionContext['notes'] {
    return {
      list: async (): Promise<Note[]> => {
        const result = cloneNotes(this.notes);
        this.record('list_notes', {}, result);
        return result;
      },
      get: async (id): Promise<Note | null> => {
        const note =
          this.notes.find((candidate) => candidate.id === id) ?? null;
        const result = note === null ? null : { ...note };
        this.record('get_note', { id }, result);
        return result;
      },
      search: async (query): Promise<Note[]> => {
        const normalized = query.toLocaleLowerCase();
        const result = this.notes
          .filter(
            (note) =>
              note.title.toLocaleLowerCase().includes(normalized) ||
              note.content.toLocaleLowerCase().includes(normalized)
          )
          .map((note) => ({ ...note }));
        this.record('search_notes', { query }, result);
        return result;
      },
      create: async (input): Promise<Note> => {
        const now = this.requireActiveSession().currentDatetime.toISOString();
        const note: Note = {
          id: `created-${this.notes.length + 1}`,
          title: input.title,
          content: input.content,
          createdAt: now,
          updatedAt: now,
        };
        this.notes.push(note);
        this.record('create_note', input, note);
        return { ...note };
      },
      update: async (id, patch): Promise<Note | null> => {
        const input = { id, ...patch };
        if (this.shouldInjectUpdateNoteFault()) {
          this.record('update_note', input, { injectedFailure: true });
          throw new Error(this.getUpdateNoteFault()?.message);
        }
        const index = this.notes.findIndex((note) => note.id === id);
        if (index < 0) {
          this.record('update_note', input, null);
          return null;
        }
        const current = this.notes[index];
        if (current === undefined) {
          throw new Error(`Missing note at resolved index ${index}`);
        }
        const updated: Note = {
          ...current,
          ...(patch.title === undefined ? {} : { title: patch.title }),
          ...(patch.content === undefined ? {} : { content: patch.content }),
          updatedAt: this.requireActiveSession().currentDatetime.toISOString(),
        };
        this.notes[index] = updated;
        this.record('update_note', input, updated);
        return { ...updated };
      },
      delete: async (id): Promise<boolean> => {
        const index = this.notes.findIndex((note) => note.id === id);
        const deleted = index >= 0;
        if (deleted) {
          this.notes.splice(index, 1);
        }
        this.record('delete_note', { id }, deleted);
        return deleted;
      },
    };
  }

  private toSearchResults(
    memories: readonly MemoryRecord[]
  ): MemorySearchResult[] {
    return [...memories]
      .reverse()
      .slice(0, 8)
      .map((memory, index) => ({
        ...memory,
        emotion: { ...memory.emotion, labels: [...memory.emotion.labels] },
        similarity: Math.max(0.5, 0.98 - index * 0.03),
      }));
  }

  private requireActiveSession(): RuntimeWorkflowSessionFixture {
    if (this.activeSession === null) {
      throw new Error('Workflow world has no active session');
    }
    return this.activeSession;
  }

  private getUpdateNoteFault(): RuntimeInjectedFault | undefined {
    const sessionId = this.requireActiveSession().id;
    return this.fixture.injectedFaults.find(
      (fault) => fault.sessionId === sessionId
    );
  }

  private shouldInjectUpdateNoteFault(): boolean {
    const fault = this.getUpdateNoteFault();
    if (fault === undefined) {
      return false;
    }
    const key = `${fault.sessionId}:update_note`;
    const attempts = this.faultAttempts.get(key) ?? 0;
    this.faultAttempts.set(key, attempts + 1);
    return attempts < fault.failures;
  }

  private record(
    kind: TraceCall['kind'],
    input: Record<string, unknown>,
    output?: unknown
  ): void {
    this.calls.push({
      kind,
      elapsedMs: Math.round(performance.now() - this.startedAt),
      input,
      ...(output === undefined ? {} : { output }),
    });
  }
}

interface SessionRunOutcome {
  usage: ModelUsage;
  terminationReason: RuntimeSessionTrace['terminationReason'];
  error?: string;
}

async function runWorkflowSession(input: {
  fixture: RuntimeWorkflowFixture;
  session: RuntimeWorkflowSessionFixture;
  world: StatefulRuntimeWorld;
  options: RuntimeHarnessOptions;
  workflowStartedAt: number;
}): Promise<RuntimeSessionTrace> {
  const sessionStartedAt = performance.now();
  const boundary = input.world.beginSession(input.session);
  const trace = new EvaluationTrace(input.workflowStartedAt);
  const tools = bindQwenEvaluationRuntimeTools(
    input.world.createToolContext(),
    (record): void => {
      input.world.saveSessionContext(record);
    }
  );
  const model = input.options.createModel({
    events: trace,
    generationProfile: input.options.generationProfile,
    ...(input.options.sessionId === undefined
      ? {}
      : { sessionId: input.options.sessionId }),
  });
  let outcome: SessionRunOutcome = {
    usage: ZERO_MODEL_USAGE,
    terminationReason: 'error',
  };

  try {
    const context = input.world.loadContext();
    const result = await runAgentSession({
      model,
      tools,
      initialInput: await createRuntimeInitialInput(
        {
          systemPrompt: input.options.systemPrompt,
          currentDatetime: input.session.currentDatetime,
          latestContext: context,
          relatedMemories: input.world.loadRelatedMemories(context),
        },
        tools
      ),
      events: trace,
      maxTurns: input.options.maxTurns,
    });
    outcome = {
      usage: result.usage,
      terminationReason: result.terminationReason,
    };
  } catch (cause) {
    outcome = {
      usage: ZERO_MODEL_USAGE,
      terminationReason: 'error',
      error:
        cause instanceof Error ? (cause.stack ?? cause.message) : String(cause),
    };
  }

  return {
    sessionId: input.session.id,
    title: input.session.title,
    elapsedMs: Math.round(performance.now() - sessionStartedAt),
    terminationReason: outcome.terminationReason,
    ...(outcome.error === undefined ? {} : { error: outcome.error }),
    usage: outcome.usage,
    calls: input.world.callsSince(boundary),
    events: trace.events,
    contextBefore: boundary.contextBefore,
    contextAfter: input.world.loadContext(),
    memoryCountBefore: boundary.memoryCount,
    memoryCountAfter: input.world.memories.length,
  };
}

/** 複数の実セッション境界と状態変化を一つの採点対象として実行する。 */
export async function runRuntimeWorkflow(
  fixture: RuntimeWorkflowFixture,
  options: RuntimeHarnessOptions
): Promise<RuntimeWorkflowResult> {
  const startedAt = performance.now();
  const world = new StatefulRuntimeWorld(fixture, startedAt);
  const sessions: RuntimeSessionTrace[] = [];

  for (const session of fixture.sessions) {
    // Workflow sessions intentionally depend on prior persisted state.
    // eslint-disable-next-line no-await-in-loop
    const sessionResult = await runWorkflowSession({
      fixture,
      session,
      world,
      options,
      workflowStartedAt: startedAt,
    });
    sessions.push(sessionResult);
  }

  const finalMemories = cloneMemories(world.memories);
  const finalNotes = cloneNotes(world.notes);
  const checks = fixture.evaluate({ sessions, finalMemories, finalNotes });
  return {
    workflowId: fixture.id,
    title: fixture.title,
    instructionMode: fixture.instructionMode,
    generationProfile: options.generationProfile,
    repetition: options.repetition,
    elapsedMs: Math.round(performance.now() - startedAt),
    sessions,
    finalMemories,
    finalNotes,
    checks,
    score: summarizeChecks(checks),
  };
}
