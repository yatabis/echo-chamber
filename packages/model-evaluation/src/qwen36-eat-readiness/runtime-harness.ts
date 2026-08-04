/* eslint-disable @typescript-eslint/require-await -- In-memory fixture ports intentionally implement asynchronous production contracts without external I/O. */

import { buildAgentPromptMessages } from '@echo-chamber/core/agent/prompt-builder';
import type {
  PromptContextSnapshot,
  PromptRelatedMemorySnapshot,
} from '@echo-chamber/core/agent/prompt-builder';
import { canonicalRuntimeTools } from '@echo-chamber/core/agent/runtime-tools/catalog';
import { bindRuntimeTools } from '@echo-chamber/core/agent/runtime-tools/tool';
import {
  runAgentSession,
  sanitizeToolOutputForModel,
  ZERO_MODEL_USAGE,
} from '@echo-chamber/core/agent/session';
import type { AgentSessionTool } from '@echo-chamber/core/agent/session';
import type { ToolExecutionContext } from '@echo-chamber/core/agent/tool-context';
import type { Note } from '@echo-chamber/core/echo/types';
import type { ChatMessage } from '@echo-chamber/core/ports/chat';
import type {
  EchoEvent,
  EchoEventPort,
} from '@echo-chamber/core/ports/echo-event';
import type { MemorySearchResult } from '@echo-chamber/core/ports/memory';
import type {
  ModelInputItem,
  ModelPort,
  ModelUsage,
} from '@echo-chamber/core/ports/model';
import type { ChannelNotificationSummary } from '@echo-chamber/core/ports/notification';
import type {
  ZennArticle,
  ZennTrendingArticleSummary,
} from '@echo-chamber/core/ports/zenn';

import { EVALUATION_DATETIME } from './runtime-scenarios';
import { summarizeChecks } from './scoring';

import type {
  RuntimeScenarioFixture,
  RuntimeScenarioObservation,
} from './runtime-scenarios';
import type {
  RuntimeGenerationProfile,
  RuntimeScenarioResult,
  TraceCall,
  TraceEvent,
} from './types';

export interface RuntimeHarnessOptions {
  createModel: RuntimeModelFactory;
  maxTurns: number;
  systemPrompt: string;
  generationProfile: RuntimeGenerationProfile;
  repetition: number;
  sessionId?: string;
}

/**
 * 一つの評価セッション用モデルを生成するときにランナーへ渡す情報。
 *
 * `sessionId` の利用方法はランナーが決める。不要なランナーは無視でき、
 * 専用ランタイムはprefix cacheの識別子などへ変換できる。
 */
export interface RuntimeModelFactoryInput {
  events: EchoEventPort;
  generationProfile: RuntimeGenerationProfile;
  sessionId?: string;
}

/** E.C.H.O.評価ハーネスへprovider実装を注入するモデル生成関数。 */
export type RuntimeModelFactory = (
  input: RuntimeModelFactoryInput
) => ModelPort;

interface RuntimeFixtureEnvironment {
  context: ToolExecutionContext;
  calls: TraceCall[];
}

export class EvaluationTrace implements EchoEventPort {
  readonly events: TraceEvent[] = [];

  constructor(private readonly startedAt: number) {}

  async emit(event: EchoEvent): Promise<void> {
    this.events.push({
      type: event.type,
      severity: event.severity,
      elapsedMs: Math.round(performance.now() - this.startedAt),
      summary: event.summary,
      ...(event.payload === undefined
        ? {}
        : { payload: sanitizeEventPayload(event) }),
    });
  }
}

function sanitizeEventPayload(event: EchoEvent): Record<string, unknown> {
  const payload = event.payload ?? {};
  if (event.type !== 'model.exchange.recorded') {
    return payload;
  }

  // OpenAI-compatible providers attach the full wire response, which must be
  // reduced before it is retained in evaluation artifacts. Native inference
  // already emits a bounded metrics-only payload, so preserve it verbatim.
  if (typeof payload.response !== 'object' || payload.response === null) {
    return payload;
  }

  const response = payload.response as Record<string, unknown>;
  const choices = Array.isArray(response.choices)
    ? response.choices.map((choice) => {
        if (typeof choice !== 'object' || choice === null) {
          return { invalidChoice: String(choice) };
        }
        const record = choice as Record<string, unknown>;
        return {
          index: record.index,
          finish_reason: record.finish_reason,
          message: record.message,
        };
      })
    : [];

  return {
    provider: payload.provider,
    model: payload.model,
    turnIndex: payload.turnIndex,
    response: {
      id: response.id,
      model: response.model,
      usage: response.usage,
      choices,
    },
  };
}

function cloneNotes(notes: readonly Note[]): Note[] {
  return notes.map((note) => ({ ...note }));
}

// Keeping every fake port in one boundary makes all permitted side effects auditable.
// eslint-disable-next-line max-lines-per-function
function createRuntimeEnvironment(
  fixture: RuntimeScenarioFixture,
  startedAt: number
): RuntimeFixtureEnvironment {
  const calls: TraceCall[] = [];
  const notes = cloneNotes(fixture.notes);

  const record = (
    kind: TraceCall['kind'],
    input: Record<string, unknown>,
    output?: unknown
  ): void => {
    calls.push({
      kind,
      elapsedMs: Math.round(performance.now() - startedAt),
      input,
      ...(output === undefined ? {} : { output }),
    });
  };

  return {
    calls,
    context: {
      notifications: {
        async getNotificationSummary(): Promise<ChannelNotificationSummary[]> {
          record('notification_summary', {}, fixture.notifications);
          return fixture.notifications;
        },
      },
      chat: {
        async readMessages(channelKey, limit): Promise<ChatMessage[]> {
          const messages = (fixture.chatMessages[channelKey] ?? []).slice(
            -limit
          );
          record('read_chat', { channelKey, limit }, messages);
          return messages;
        },
        async sendMessage(channelKey, message): Promise<void> {
          record('send_chat', { channelKey, message });
        },
        async addReaction(channelKey, messageId, reaction): Promise<void> {
          record('add_reaction', { channelKey, messageId, reaction });
        },
      },
      memory: {
        async store(content, emotion, type): Promise<void> {
          record('store_memory', { content, emotion, type });
        },
        async search(query, type): Promise<MemorySearchResult[]> {
          record('search_memory', { query, type }, fixture.memorySearchResults);
          return fixture.memorySearchResults;
        },
      },
      notes: {
        async list(): Promise<Note[]> {
          const result = cloneNotes(notes);
          record('list_notes', {}, result);
          return result;
        },
        async get(id): Promise<Note | null> {
          const note = notes.find((candidate) => candidate.id === id) ?? null;
          const result = note === null ? null : { ...note };
          record('get_note', { id }, result);
          return result;
        },
        async search(query): Promise<Note[]> {
          const normalized = query.toLocaleLowerCase();
          const result = notes
            .filter(
              (note) =>
                note.title.toLocaleLowerCase().includes(normalized) ||
                note.content.toLocaleLowerCase().includes(normalized)
            )
            .map((note) => ({ ...note }));
          record('search_notes', { query }, result);
          return result;
        },
        async create(input): Promise<Note> {
          const now = EVALUATION_DATETIME.toISOString();
          const note = {
            id: `created-${notes.length + 1}`,
            title: input.title,
            content: input.content,
            createdAt: now,
            updatedAt: now,
          };
          notes.push(note);
          record('create_note', input, note);
          return { ...note };
        },
        async update(id, patch): Promise<Note | null> {
          const index = notes.findIndex((note) => note.id === id);
          if (index < 0) {
            record('update_note', { id, ...patch }, null);
            return null;
          }

          const current = notes[index];
          if (current === undefined) {
            throw new Error(`Missing note at resolved index ${index}`);
          }
          const updated: Note = {
            ...current,
            ...(patch.title === undefined ? {} : { title: patch.title }),
            ...(patch.content === undefined ? {} : { content: patch.content }),
            updatedAt: EVALUATION_DATETIME.toISOString(),
          };
          notes[index] = updated;
          record('update_note', { id, ...patch }, updated);
          return { ...updated };
        },
        async delete(id): Promise<boolean> {
          const index = notes.findIndex((note) => note.id === id);
          const deleted = index >= 0;
          if (deleted) {
            notes.splice(index, 1);
          }
          record('delete_note', { id }, deleted);
          return deleted;
        },
      },
      zenn: {
        async listTrendingArticles(): Promise<
          readonly ZennTrendingArticleSummary[]
        > {
          record('list_zenn', {}, []);
          return [];
        },
        async getArticleBySlug(slug): Promise<ZennArticle> {
          const article = {
            slug,
            url: `https://zenn.dev/example/articles/${slug}`,
            title: 'Synthetic evaluation article',
            author: { username: 'fixture', name: 'Fixture' },
            topics: ['evaluation'],
            tableOfContents: [],
            content:
              'This is a synthetic fixture. No network request occurred.',
          };
          record('get_zenn', { slug }, article);
          return article;
        },
      },
    },
  };
}

export interface RuntimeInitialInputOptions {
  systemPrompt: string;
  currentDatetime: Date;
  latestContext: PromptContextSnapshot | null;
  relatedMemories: readonly PromptRelatedMemorySnapshot[];
}

export async function createRuntimeInitialInput(
  input: RuntimeInitialInputOptions,
  tools: readonly AgentSessionTool[]
): Promise<ModelInputItem[]> {
  const promptMessages = buildAgentPromptMessages({
    systemPrompt: input.systemPrompt,
    currentDatetime: input.currentDatetime,
    latestContext: input.latestContext,
    relatedMemories: input.relatedMemories,
  });
  const startupTool = tools.find((tool) => tool.name === 'check_notifications');
  if (startupTool === undefined) {
    throw new Error('check_notifications tool is required');
  }

  const callId = 'check_notifications';
  return [
    ...promptMessages.map<ModelInputItem>((message) => ({
      role: message.role,
      content: message.content,
    })),
    {
      type: 'tool_call',
      callId,
      toolName: startupTool.name,
      input: '{}',
    },
    {
      type: 'tool_result',
      callId,
      output: sanitizeToolOutputForModel(await startupTool.execute('{}')),
    },
  ];
}

/**
 * 一つのE.C.H.O.固定シナリオを本物のprompt/session/tool実装で実行する。
 */
export async function runRuntimeScenario(
  fixture: RuntimeScenarioFixture,
  options: RuntimeHarnessOptions
): Promise<RuntimeScenarioResult> {
  const startedAt = performance.now();
  const trace = new EvaluationTrace(startedAt);
  const environment = createRuntimeEnvironment(fixture, startedAt);
  const tools = bindRuntimeTools(canonicalRuntimeTools, environment.context);
  const model = options.createModel({
    events: trace,
    generationProfile: options.generationProfile,
    ...(options.sessionId === undefined
      ? {}
      : { sessionId: options.sessionId }),
  });
  let usage: ModelUsage = ZERO_MODEL_USAGE;
  let terminationReason: RuntimeScenarioResult['terminationReason'] = 'error';
  let error: string | undefined;

  try {
    const session = await runAgentSession({
      model,
      tools,
      initialInput: await createRuntimeInitialInput(
        {
          systemPrompt: options.systemPrompt,
          currentDatetime: EVALUATION_DATETIME,
          latestContext: fixture.latestContext,
          relatedMemories: fixture.relatedMemories,
        },
        tools
      ),
      events: trace,
      maxTurns: options.maxTurns,
    });
    usage = session.usage;
    terminationReason = session.terminationReason;
  } catch (cause) {
    error =
      cause instanceof Error ? (cause.stack ?? cause.message) : String(cause);
  }

  const observation: RuntimeScenarioObservation = {
    calls: environment.calls,
    events: trace.events,
    terminationReason,
  };
  const checks = fixture.evaluate(observation);

  return {
    scenarioId: fixture.id,
    title: fixture.title,
    instructionMode: fixture.instructionMode,
    generationProfile: options.generationProfile,
    repetition: options.repetition,
    elapsedMs: Math.round(performance.now() - startedAt),
    terminationReason,
    ...(error === undefined ? {} : { error }),
    usage,
    calls: environment.calls,
    events: trace.events,
    checks,
    score: summarizeChecks(checks),
  };
}
