import { DurableObject } from 'cloudflare:workers';
import { Hono } from 'hono';

import { MemorySystem } from '@echo-chamber/cloudflare-runtime/memory-system';
import { NoteSystem } from '@echo-chamber/cloudflare-runtime/note-system';
import {
  parseDashboardInstanceSummary,
  parseEchoStatus,
} from '@echo-chamber/contracts/dashboard/schemas';
import type {
  DashboardInstanceSummary,
  DashboardRuntimeConfig,
  EchoStatus,
} from '@echo-chamber/contracts/dashboard/types';
import {
  buildUsageStackedSeries,
  sumUsageBreakdown,
} from '@echo-chamber/contracts/dashboard/utils';
import { canonicalRuntimeTools } from '@echo-chamber/core/agent/runtime-tools/catalog';
import { bindRuntimeTools } from '@echo-chamber/core/agent/runtime-tools/tool';
import type { AgentSessionTool } from '@echo-chamber/core/agent/session';
import { ThinkingEngine as AgentThinkingEngine } from '@echo-chamber/core/agent/thinking-engine';
import {
  ALARM_CONFIG,
  SCHEDULING_CONFIG,
} from '@echo-chamber/core/echo/constants';
import {
  getEchoInstanceDefinition,
  type EchoInstanceDefinition,
} from '@echo-chamber/core/echo/instance-definitions';
import type {
  EchoState,
  Note,
  Usage,
  UsageRecord,
} from '@echo-chamber/core/echo/types';
import {
  addUsage,
  calculateDynamicTokenLimit,
  convertUsage,
  findNextTokenLimitRecoveryTime,
  getTodayUsageKey,
  normalizeUsageRecord,
} from '@echo-chamber/core/echo/usage';
import type { ContextSnapshot } from '@echo-chamber/core/ports/context';
import { emitEchoEvent } from '@echo-chamber/core/ports/echo-event';
import type { EchoEventPort } from '@echo-chamber/core/ports/echo-event';
import type { ModelPort } from '@echo-chamber/core/ports/model';
import type { EchoInstanceId } from '@echo-chamber/core/types/echo-config';
import { isValidInstanceId } from '@echo-chamber/core/types/echo-config';
import { formatDatetime } from '@echo-chamber/core/utils/datetime';
import { getErrorMessage } from '@echo-chamber/core/utils/error';
import { getUnreadMessageCount } from '@echo-chamber/discord-adapter/notification-utils';
import { OpenAIChatCompletionsModel } from '@echo-chamber/openai-adapter/openai-chat-completions-model';
import { OpenAIResponsesModel } from '@echo-chamber/openai-adapter/openai-responses-model';

import {
  resolveEchoRuntimeBindings,
  type EchoChatChannelBinding,
  type EchoRuntimeBindings,
} from '../config/echo-runtime-bindings';
import { resolveMainLLMConfig } from '../config/main-llm-config';
import {
  resolveTokenLimitConfig,
  type TokenLimitConfig,
} from '../config/token-limit-config';
import { createEmbeddingService } from '../embedding/create-embedding-service';
import { createRerankingService } from '../reranking/create-reranking-service';
import { createCloudflareEchoEventPort } from '../utils/echo-event';

import { createToolExecutionContext } from './tool-context';

async function fetchUnreadMessageCounts(
  token: string,
  chatChannels: readonly EchoChatChannelBinding[]
): Promise<{ channel: EchoChatChannelBinding; unreadCount: number }[]> {
  return await Promise.all(
    chatChannels.map(async (channel) => ({
      channel,
      unreadCount: await getUnreadMessageCount(token, channel.discordChannelId),
    }))
  );
}

function findLatestUpdatedAt(
  rows: readonly { updatedAt: string }[]
): string | null {
  return rows.reduce<string | null>((latest, row) => {
    if (latest === null) {
      return row.updatedAt;
    }

    const latestTime = new Date(latest).getTime();
    const rowTime = new Date(row.updatedAt).getTime();
    if (Number.isNaN(rowTime)) {
      return latest;
    }
    if (Number.isNaN(latestTime) || rowTime > latestTime) {
      return row.updatedAt;
    }

    return latest;
  }, null);
}

interface RunDecision {
  shouldRun: boolean;
  unreadCheckMs: number;
}

interface RunExecutionResult {
  unreadCheckMs: number;
  thinkMs: number;
}

export class Echo extends DurableObject<Env> {
  private readonly store: KVNamespace;
  private readonly storage: DurableObjectStorage;
  private readonly router: Hono;
  private readonly events: EchoEventPort;
  private readonly _env: Env;
  private readonly noteSystem: NoteSystem;
  private currentSessionId: string | null = null;

  // 遅延初期化されるプロパティ（ensureInitializedで設定されるためreadonlyではない）
  private executableTools: readonly AgentSessionTool[] | null = null;
  private instanceDefinition: EchoInstanceDefinition | null = null;
  private runtimeBindings: EchoRuntimeBindings | null = null;
  private memorySystem: MemorySystem | null = null;
  private lastUnreadMessageDetails: {
    totalUnreadCount: number;
    channels: {
      key: string;
      displayName: string;
      unreadCount: number;
    }[];
  } | null = null;

  /**
   * The constructor is invoked once upon creation of the Durable Object, i.e. the first call to
   * 	`DurableObjectStub::get` for a given identifier (no-op constructors can be omitted)
   *
   * @param ctx - The interface for interacting with Durable Object state
   * @param env - The interface to reference bindings declared in wrangler.jsonc
   */
  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.store = env.ECHO_KV;
    this.storage = ctx.storage;
    this.events = createCloudflareEchoEventPort({
      source: 'cloudflare-workers',
      getInstanceId: (): string | null => this.instanceDefinition?.id ?? null,
      getSessionId: (): string | null => this.currentSessionId,
      getDiscordConfig: (): { token: string; channelId: string } | null => {
        if (this.runtimeBindings === null) {
          return null;
        }

        return {
          token: this.runtimeBindings.discordBotToken,
          channelId: this.runtimeBindings.thinkingChannelId,
        };
      },
    });
    this._env = env;
    this.noteSystem = new NoteSystem({
      storage: this.storage,
    });
    this.router = new Hono()
      .basePath('/:id')
      // 全リクエストで遅延初期化を実行するミドルウェア
      .use('*', async (c, next) => {
        const id = c.req.param('id');
        if (!isValidInstanceId(id)) {
          return c.text(`Invalid instance ID: ${id}`, 400);
        }
        await this.ensureInitialized(id);
        await next();
      })
      .get('/', async (c) => {
        return c.json(await this.getStatus());
      })
      .get('/summary', async (c) => {
        return c.json(await this.getSummary());
      })
      .post('/wake', async (c) => {
        await this.wake(true);
        return c.text('OK.');
      })
      .post('/sleep', async (c) => {
        await this.sleep(true);
        return c.text('OK.');
      })
      .post('/run', async (c) => {
        if (env.ENVIRONMENT !== 'local') {
          return c.notFound();
        }
        await this.run();
        return c.text('OK.');
      });
  }

  async fetch(request: Request): Promise<Response> {
    return this.router.fetch(request);
  }

  /**
   * インスタンスの遅延初期化
   * 最初のリクエスト時に呼び出され、definition と runtime bindings を設定する
   */
  private async ensureInitialized(id: EchoInstanceId): Promise<void> {
    // 既に同じIDで初期化済みの場合はスキップ
    if (this.instanceDefinition?.id === id) {
      return;
    }

    this.instanceDefinition = getEchoInstanceDefinition(id);
    this.runtimeBindings = await resolveEchoRuntimeBindings(
      this._env,
      this.store,
      id
    );
    const embeddingService = createEmbeddingService(
      this._env,
      this.runtimeBindings.embeddingConfig,
      this.events
    );
    const rerankingService = createRerankingService(this._env);
    this.memorySystem = new MemorySystem({
      sql: this.ctx.storage.sql,
      embeddingService,
      rerankingService,
      events: this.events,
    });
    const toolContext = createToolExecutionContext({
      chatBindings: this.getRuntimeBindingsOrThrow(),
      memorySystem: this.memorySystem,
      noteSystem: this.noteSystem,
    });
    this.executableTools = bindRuntimeTools(canonicalRuntimeTools, toolContext);
    // embedding モデル変更時の自動再 embedding
    await this.memorySystem.reEmbedStaleMemories();

    // ストレージにID/名前を保存（alarmから参照するため）
    await this.storage.put('id', id);
    await this.storage.put('name', this.instanceDefinition.name);
  }

  /**
   * instance definition を取得（初期化されていない場合はエラー）
   */
  private getInstanceDefinitionOrThrow(): EchoInstanceDefinition {
    if (!this.instanceDefinition) {
      throw new Error('Echo instance definition not initialized');
    }
    return this.instanceDefinition;
  }

  /**
   * runtime bindings を取得（初期化されていない場合はエラー）
   */
  private getRuntimeBindingsOrThrow(): EchoRuntimeBindings {
    if (!this.runtimeBindings) {
      throw new Error('Echo runtime bindings not initialized');
    }
    return this.runtimeBindings;
  }

  /**
   * executableToolsを取得（初期化されていない場合はエラー）
   */
  private getExecutableToolsOrThrow(): readonly AgentSessionTool[] {
    if (!this.executableTools) {
      throw new Error('Thinking tools not initialized');
    }
    return this.executableTools;
  }

  /**
   * memorySystemを取得（初期化されていない場合はエラー）
   */
  private getMemorySystemOrThrow(): MemorySystem {
    if (!this.memorySystem) {
      throw new Error('MemorySystem not initialized');
    }
    return this.memorySystem;
  }

  async alarm(alarmInfo?: AlarmInvocationInfo): Promise<void> {
    const alarmStartedAt = Date.now();
    let runResult: RunExecutionResult = {
      unreadCheckMs: 0,
      thinkMs: 0,
    };
    let alarmStatus: 'completed' | 'failed' | 'sleep_scheduled' = 'completed';
    let alarmReason: string | undefined;

    try {
      await emitEchoEvent(this.events, {
        type: 'system.schedule.alarm_triggered',
        severity: 'debug',
        summary: 'alarm triggered',
        payload: {
          alarmInfo,
        },
      });

      // ストレージからIDを読み取り初期化
      const storedId = await this.storage.get<string>('id');
      if (storedId == null || !isValidInstanceId(storedId)) {
        alarmStatus = 'failed';
        alarmReason = 'invalid_instance_id';
        await this.emitRunPreconditionFailed({
          reason: 'invalid_instance_id',
          storedId: storedId ?? null,
        });
        await this.sleep(true);
        return;
      }

      await this.ensureInitialized(storedId);

      const now = new Date();
      const state = await this.getState();
      if (now.getHours() === 18 && state === 'Idling') {
        await this.sleep();
        const nextAlarm = new Date();
        nextAlarm.setHours(22, 0, 0, 0);
        await this.setNextAlarm(nextAlarm, 'daily_sleep_wake');
        alarmStatus = 'sleep_scheduled';
        alarmReason = 'daily_sleep_window';
        return;
      }
      if (now.getHours() === 22 && state === 'Sleeping') {
        await this.wake(true);
      }
      runResult = await this.run();
      await this.setNextAlarm();
    } finally {
      await emitEchoEvent(this.events, {
        type: 'system.schedule.alarm_completed',
        severity: alarmStatus === 'failed' ? 'error' : 'debug',
        summary: `alarm ${alarmStatus}`,
        payload: {
          status: alarmStatus,
          reason: alarmReason,
          alarmTotalMs: Date.now() - alarmStartedAt,
          unreadCheckMs: runResult.unreadCheckMs,
          thinkMs: runResult.thinkMs,
        },
      });
    }
  }

  async getNextAlarm(): Promise<string | null> {
    const nextAlarm = await this.storage.getAlarm();
    if (nextAlarm == null) {
      return null;
    }
    return formatDatetime(new Date(nextAlarm));
  }

  async setNextAlarm(
    nextAlarm?: Date,
    reason = 'regular_interval'
  ): Promise<void> {
    if (!nextAlarm) {
      nextAlarm = new Date();
      nextAlarm.setMinutes(
        nextAlarm.getMinutes() + ALARM_CONFIG.INTERVAL_MINUTES,
        0,
        0
      );
    }
    await this.storage.setAlarm(nextAlarm);
    await emitEchoEvent(this.events, {
      type: 'system.schedule.alarm_scheduled',
      severity: 'debug',
      summary: `alarm scheduled: ${nextAlarm.toISOString()}`,
      payload: {
        scheduledAt: nextAlarm.toISOString(),
        reason,
      },
    });
  }

  async getId(): Promise<string> {
    const id = await this.storage.get<string>('id');
    return id ?? 'Echo';
  }

  async getName(): Promise<string> {
    const name = await this.storage.get<string>('name');
    return name ?? 'NO_NAME';
  }

  async getState(): Promise<EchoState> {
    const state = await this.storage.get<EchoState>('state');
    return state ?? 'Idling';
  }

  async setState(newState: EchoState, reason = 'direct'): Promise<void> {
    const previousState = await this.getState();
    await this.storage.put('state', newState);
    if (previousState === newState) {
      return;
    }

    await emitEchoEvent(this.events, {
      type: 'system.echo_state.changed',
      severity: 'info',
      summary: `echo state changed: ${previousState} -> ${newState}`,
      payload: {
        previousState,
        nextState: newState,
        reason,
      },
    });
  }

  /**
   * Dashboard 詳細画面向けの状態スナップショットを返す。
   *
   * `EchoStatus` はインスタンスの表示名・状態・次回実行時刻に加え、
   * ノート/メモリ/usage の表示に必要な情報を 1 レスポンスで返す DTO。
   */
  async getStatus(): Promise<EchoStatus> {
    const definition = this.getInstanceDefinitionOrThrow();
    const state = await this.getState();
    const nextAlarm = await this.getNextAlarm();
    const usage = await this.getAllUsage();

    const memories = this.getMemorySystemOrThrow()
      .getAllMemories()
      .map((row) => ({
        content: row.content,
        type: row.type,
        emotion: {
          valence: row.emotion_valence,
          arousal: row.emotion_arousal,
          labels: JSON.parse(row.emotion_labels) as string[],
        },
        embedding_model: row.embedding_model,
        createdAt: row.created_at,
        updatedAt: row.updated_at,
      }));
    const notes = await this.getNotes();

    return parseEchoStatus({
      id: definition.id,
      name: definition.name,
      state,
      nextAlarm,
      runtime: this.getDashboardRuntimeConfig(),
      memories,
      notes,
      usage,
    });
  }

  /**
   * Dashboard 一覧画面向けの軽量サマリーを返す。
   *
   * 一覧では name/state/nextAlarm のみ使うため、詳細 DTO より小さい形で返す。
   */
  async getSummary(): Promise<DashboardInstanceSummary> {
    const definition = this.getInstanceDefinitionOrThrow();
    const notes = await this.getNotes();
    const memories = this.getMemorySystemOrThrow()
      .getAllMemories()
      .map((row) => ({
        updatedAt: row.updated_at,
      }));
    const usage = await this.getAllUsage();
    const todayUsageTokens = (await this.getTodayUsage())?.total_tokens ?? 0;
    const sevenDayUsageTokens = sumUsageBreakdown(
      buildUsageStackedSeries(usage, 7)
    ).totalTokens;
    const thirtyDayUsageTokens = sumUsageBreakdown(
      buildUsageStackedSeries(usage, 30)
    ).totalTokens;

    return parseDashboardInstanceSummary({
      id: definition.id,
      name: definition.name,
      state: await this.getState(),
      nextAlarm: await this.getNextAlarm(),
      runtime: this.getDashboardRuntimeConfig(),
      noteCount: notes.length,
      memoryCount: memories.length,
      todayUsageTokens,
      sevenDayUsageTokens,
      thirtyDayUsageTokens,
      latestNoteUpdatedAt: findLatestUpdatedAt(notes),
      latestMemoryUpdatedAt: findLatestUpdatedAt(memories),
    });
  }

  async getNotes(query = ''): Promise<Note[]> {
    const normalizedQuery = query.trim();
    if (normalizedQuery.length > 0) {
      return await this.noteSystem.searchNotes(normalizedQuery);
    }
    return await this.noteSystem.listNotes();
  }

  /**
   * 全期間のUsage履歴を取得
   */
  async getAllUsage(): Promise<UsageRecord> {
    const usage = await this.storage.get('usage');
    return normalizeUsageRecord(usage);
  }

  /**
   * 今日のUsage情報を取得
   */
  async getTodayUsage(): Promise<Usage | null> {
    const usageRecord = await this.getAllUsage();
    return usageRecord[getTodayUsageKey()] ?? null;
  }

  async wake(force = false): Promise<void> {
    const state = await this.getState();

    if (!force && state === 'Sleeping') {
      await this.emitEchoStateChangeRejected({
        currentState: state,
        requestedState: 'Idling',
        reason: 'cannot_wake_while_sleeping',
        severity: 'warn',
      });
      return;
    }

    await this.setNextAlarm(undefined, 'wake');
    await this.setState('Idling', 'wake');
  }

  async sleep(force = false): Promise<void> {
    const state = await this.getState();

    if (state === 'Sleeping') {
      await this.emitEchoStateChangeRejected({
        currentState: state,
        requestedState: 'Sleeping',
        reason: 'already_sleeping',
        severity: 'info',
      });
      return;
    }

    if (!force && state === 'Running') {
      await this.emitEchoStateChangeRejected({
        currentState: state,
        requestedState: 'Sleeping',
        reason: 'cannot_sleep_while_running',
        severity: 'warn',
      });
      return;
    }

    try {
      await this.setState('Sleeping', 'sleep');
      await this.storage.deleteAlarm();
      // sleep 処理
    } catch (error) {
      await emitEchoEvent(this.events, {
        type: 'system.echo_state.change_failed',
        severity: 'error',
        summary: `echo state change failed: ${getErrorMessage(error)}`,
        payload: {
          currentState: state,
          requestedState: 'Sleeping',
          reason: 'sleep_failed',
          error: getErrorMessage(error),
        },
      });
    } finally {
      // await this.setNextAlarm();
      // await this.setState('Idling');
    }
  }

  async run(): Promise<RunExecutionResult> {
    const runDecision = await this.resolveRunDecision();
    if (!runDecision.shouldRun) {
      return {
        unreadCheckMs: runDecision.unreadCheckMs,
        thinkMs: 0,
      };
    }

    await this.setState('Running', 'run_started');
    this.currentSessionId = crypto.randomUUID();
    let thinkMs = 0;
    let thinkStartedAt = 0;

    try {
      thinkStartedAt = Date.now();
      const { context, nextWakeAt, usage } =
        await this.createThinkingEngine().think();
      thinkMs = Date.now() - thinkStartedAt;
      if (context != null) {
        await this.saveContext(context);
      }
      if (nextWakeAt == null) {
        await this.clearNextWakeAt('finish_thinking');
      } else {
        await this.saveNextWakeAt(nextWakeAt, 'finish_thinking');
      }
      const totalUsage = await this.updateUsage(
        convertUsage(usage, this.getMainLLMUsageIdentity())
      );
      await emitEchoEvent(this.events, {
        type: 'usage.recorded',
        severity: 'info',
        summary: `usage recorded: ${usage.totalTokens} tokens`,
        payload: {
          usage,
          totalUsage,
        },
      });
    } catch (error) {
      if (thinkStartedAt !== 0) {
        thinkMs = Date.now() - thinkStartedAt;
      }
      await emitEchoEvent(this.events, {
        type: 'system.run.failed',
        severity: 'error',
        summary: `run failed: ${getErrorMessage(error)}`,
        payload: {
          error: getErrorMessage(error),
          thinkMs,
        },
      });
    } finally {
      this.currentSessionId = null;
      await this.setState('Idling', 'run_completed');
    }

    return {
      unreadCheckMs: runDecision.unreadCheckMs,
      thinkMs,
    };
  }

  /**
   * 今回の alarm / run が実行されるべきかと、補助メトリクスを返す。
   *
   * @returns 実行可否と未読確認にかかった時間
   */
  private async resolveRunDecision(): Promise<RunDecision> {
    // Stateチェック
    if (!(await this.validateEchoState())) {
      await this.emitRunDecisionEvaluated({
        shouldRun: false,
        reason: 'invalid_state',
        unreadCheckMs: 0,
      });
      return {
        shouldRun: false,
        unreadCheckMs: 0,
      };
    }

    // 未読メッセージがあれば実行
    const unreadCheckStartedAt = Date.now();
    const hasUnreadMessages = await this.validateChatMessage();
    const unreadCheckMs = Date.now() - unreadCheckStartedAt;
    if (hasUnreadMessages) {
      await this.emitRunDecisionEvaluated({
        shouldRun: true,
        reason: 'unread_messages',
        unreadCheckMs,
        unreadTotalCount: this.lastUnreadMessageDetails?.totalUnreadCount,
        unreadChannels: this.lastUnreadMessageDetails?.channels,
      });
      return {
        shouldRun: true,
        unreadCheckMs,
      };
    }

    const todayUsage = await this.getTodayUsage();
    const totalTokens = todayUsage?.total_tokens ?? 0;
    const { nextWakeAt, hasReachedNextWakeAt } =
      await this.resolveNextWakeAtStatus();

    return await this.resolveScheduledRunDecision({
      unreadCheckMs,
      totalTokens,
      nextWakeAt,
      hasReachedNextWakeAt,
    });
  }

  /**
   * 未読以外の通常起動条件を token limit と next_wake_at から判定する。
   */
  private async resolveScheduledRunDecision(input: {
    unreadCheckMs: number;
    totalTokens: number;
    nextWakeAt: Date | null;
    hasReachedNextWakeAt: boolean;
  }): Promise<RunDecision> {
    if (
      !(await this.validateHardTokenLimit(
        input.totalTokens,
        input.nextWakeAt,
        input.hasReachedNextWakeAt
      ))
    ) {
      await this.emitRunDecisionEvaluated({
        shouldRun: false,
        reason: 'hard_token_limit',
        unreadCheckMs: input.unreadCheckMs,
        totalTokens: input.totalTokens,
        nextWakeAt: input.nextWakeAt?.toISOString() ?? null,
      });
      return {
        shouldRun: false,
        unreadCheckMs: input.unreadCheckMs,
      };
    }

    if (input.hasReachedNextWakeAt && input.nextWakeAt !== null) {
      await this.emitRunDecisionEvaluated({
        shouldRun: true,
        reason: 'next_wake_at_reached',
        unreadCheckMs: input.unreadCheckMs,
        totalTokens: input.totalTokens,
        nextWakeAt: input.nextWakeAt.toISOString(),
      });
      return {
        shouldRun: true,
        unreadCheckMs: input.unreadCheckMs,
      };
    }

    const softLimitRun = this.evaluateSoftLimitRun(
      input.totalTokens,
      input.nextWakeAt
    );
    await this.emitRunDecisionEvaluated({
      shouldRun: softLimitRun.shouldRun,
      reason: softLimitRun.reason,
      unreadCheckMs: input.unreadCheckMs,
      totalTokens: input.totalTokens,
      softLimit: softLimitRun.softLimit,
      nextWakeAt: input.nextWakeAt?.toISOString() ?? null,
    });

    return {
      shouldRun: softLimitRun.shouldRun,
      unreadCheckMs: input.unreadCheckMs,
    };
  }

  /**
   * run / alarm の起動判定イベントを送る。
   */
  private async emitRunDecisionEvaluated(input: {
    shouldRun: boolean;
    reason: string;
    unreadCheckMs: number;
    totalTokens?: number;
    softLimit?: number;
    nextWakeAt?: string | null;
    unreadTotalCount?: number;
    unreadChannels?: {
      key: string;
      displayName: string;
      unreadCount: number;
    }[];
  }): Promise<void> {
    await emitEchoEvent(this.events, {
      type: 'system.run_decision.evaluated',
      severity: input.shouldRun ? 'info' : 'debug',
      summary: `run decision: ${input.shouldRun ? 'run' : 'skip'} (${input.reason})`,
      payload: input,
    });
  }

  private async emitRunPreconditionFailed(
    payload: Record<string, unknown>
  ): Promise<void> {
    await emitEchoEvent(this.events, {
      type: 'system.run.precondition_failed',
      severity: 'warn',
      summary: `run precondition failed: ${payload.reason as string}`,
      payload,
    });
  }

  private async emitEchoStateChangeRejected(input: {
    currentState: EchoState;
    requestedState: EchoState;
    reason: string;
    severity: 'info' | 'warn';
  }): Promise<void> {
    await emitEchoEvent(this.events, {
      type: 'system.echo_state.change_rejected',
      severity: input.severity,
      summary: `echo state change rejected: ${input.reason}`,
      payload: input,
    });
  }

  /**
   * 保存済みの `next_wake_at` を読み込み、比較可能な状態へ正規化する。
   *
   * @returns 比較可能な next_wake_at と、現在時刻までに到達済みかどうか
   */
  private async resolveNextWakeAtStatus(): Promise<{
    nextWakeAt: Date | null;
    hasReachedNextWakeAt: boolean;
  }> {
    const storedNextWakeAt = await this.loadNextWakeAt();
    const nextWakeAt =
      storedNextWakeAt === null
        ? null
        : await this.parseNextWakeAt(storedNextWakeAt);

    return {
      nextWakeAt,
      hasReachedNextWakeAt:
        nextWakeAt !== null && this.hasNextWakeAtReached(nextWakeAt),
    };
  }

  /**
   * soft limit の範囲で通常起動できるかを判定する。
   * next_wake_at が直近にある場合は、soft limit 未満でも次回起動時刻を優先して待機する。
   *
   * @param totalTokens 今日すでに消費した総トークン数
   * @param nextWakeAt 比較可能な next_wake_at。未設定または不正なら `null`
   * @returns soft limit 起動を許可する場合は `true`
   */
  private evaluateSoftLimitRun(
    totalTokens: number,
    nextWakeAt: Date | null
  ): {
    shouldRun: boolean;
    reason:
      | 'soft_limit_allows_run'
      | 'soft_token_limit'
      | 'next_wake_at_suppression';
    softLimit: number;
  } {
    // soft limit 未満なら通常起動。ただし直近の next_wake_at があるときは待機する。
    const tokenLimits = this.getTokenLimitConfig();
    const softLimit = calculateDynamicTokenLimit(tokenLimits.dailySoftLimit);
    if (totalTokens >= softLimit) {
      return {
        shouldRun: false,
        reason: 'soft_token_limit',
        softLimit,
      };
    }

    if (
      nextWakeAt !== null &&
      this.isNextWakeAtWithinSoftLimitWindow(nextWakeAt)
    ) {
      return {
        shouldRun: false,
        reason: 'next_wake_at_suppression',
        softLimit,
      };
    }

    return {
      shouldRun: true,
      reason: 'soft_limit_allows_run',
      softLimit,
    };
  }

  /**
   * 未読メッセージ以外の通常起動で使えるトークン量が残っているかを検証する。
   * hard limit を超えた場合は next_wake_at に到達していても起動しない。
   *
   * @param totalTokens 今日すでに消費した総トークン数
   * @param nextWakeAt 正規化済みの next_wake_at。未設定なら `null`
   * @param shouldWarnOnHardLimit hard limit 到達時に warn と fallback を行うべきなら `true`
   * @returns hard limit 未満なら `true`
   */
  private async validateHardTokenLimit(
    totalTokens: number,
    nextWakeAt: Date | null,
    shouldWarnOnHardLimit: boolean
  ): Promise<boolean> {
    const tokenLimits = this.getTokenLimitConfig();
    const hardLimit = calculateDynamicTokenLimit(
      tokenLimits.dailyHardLimit,
      tokenLimits.hardLimitBufferFactor
    );
    if (totalTokens < hardLimit) {
      return true;
    }

    if (shouldWarnOnHardLimit && nextWakeAt !== null) {
      const fallbackNextWakeAt = findNextTokenLimitRecoveryTime(
        totalTokens,
        tokenLimits.dailyHardLimit,
        tokenLimits.hardLimitBufferFactor
      );
      await this.saveNextWakeAt(
        fallbackNextWakeAt.toISOString(),
        'hard_token_limit_defer',
        {
          totalTokens,
          hardLimit,
        },
        nextWakeAt.toISOString()
      );
    }
    return false;
  }

  /**
   * Echoの状態を検証
   */
  private async validateEchoState(): Promise<boolean> {
    const id = await this.getId();
    const name = await this.getName();

    // IDが未登録の場合は実行できない
    if (id === 'Echo') {
      await this.emitRunPreconditionFailed({
        reason: 'missing_echo_id',
        id,
      });
      return false;
    }

    const state = await this.getState();

    // 睡眠中は実行できない
    if (state === 'Sleeping') {
      await this.emitRunPreconditionFailed({
        reason: 'sleeping',
        state,
        name,
      });
      return false;
    }

    // 既に実行中の場合は何もしない
    if (state === 'Running') {
      await this.emitRunPreconditionFailed({
        reason: 'running',
        state,
        name,
      });
      return false;
    }

    return true;
  }

  /**
   * 未読メッセージがあるか検証
   */
  private async validateChatMessage(): Promise<boolean> {
    const name = await this.getName();
    const runtimeBindings = this.getRuntimeBindingsOrThrow();
    this.lastUnreadMessageDetails = null;

    if (runtimeBindings.chatChannels.length === 0) {
      await this.emitRunPreconditionFailed({
        reason: 'missing_chat_channels',
        name,
      });
      return false;
    }

    const unreadCounts = await fetchUnreadMessageCounts(
      runtimeBindings.discordBotToken,
      runtimeBindings.chatChannels
    );

    const unreadChannels = unreadCounts.filter(
      ({ unreadCount }) => unreadCount > 0
    );
    this.lastUnreadMessageDetails =
      unreadChannels.length === 0
        ? null
        : {
            totalUnreadCount: unreadChannels.reduce(
              (total, { unreadCount }) => total + unreadCount,
              0
            ),
            channels: unreadChannels.map(({ channel, unreadCount }) => ({
              key: channel.key,
              displayName: channel.displayName,
              unreadCount,
            })),
          };
    return unreadChannels.length > 0;
  }

  /**
   * 保存済みの `next_wake_at` を比較可能な時刻へ正規化する。
   * 不正値は warn を出して storage から破棄し、未設定扱いにする。
   *
   * @param nextWakeAt 保存済みの next_wake_at
   * @returns 比較可能な Date。未設定または不正なら `null`
   */
  private async parseNextWakeAt(nextWakeAt: string): Promise<Date | null> {
    const parsed = new Date(nextWakeAt);
    if (Number.isNaN(parsed.getTime())) {
      await this.invalidateNextWakeAt(nextWakeAt, 'invalid_date');
      return null;
    }

    return parsed;
  }

  /**
   * @param nextWakeAt 正規化済みの next_wake_at
   * @returns `next_wake_at` が現在時刻以前なら `true`
   */
  private hasNextWakeAtReached(nextWakeAt: Date): boolean {
    return Date.now() >= nextWakeAt.getTime();
  }

  /**
   * soft limit 起動を抑制すべきほど近い `next_wake_at` かを判定する。
   *
   * @param nextWakeAt 正規化済みの next_wake_at
   * @returns 未来の `next_wake_at` が suppression window 以内なら `true`
   */
  private isNextWakeAtWithinSoftLimitWindow(nextWakeAt: Date): boolean {
    const msUntilNextWakeAt = nextWakeAt.getTime() - Date.now();
    const suppressionWindowMs =
      SCHEDULING_CONFIG.SOFT_LIMIT_NEXT_WAKE_AT_WINDOW_MINUTES * 60 * 1000;

    return msUntilNextWakeAt > 0 && msUntilNextWakeAt <= suppressionWindowMs;
  }

  /**
   * Usage情報を日別に累積保存
   */
  async updateUsage(usage: Usage): Promise<Usage> {
    const dateKey = getTodayUsageKey();
    const usageRecord = await this.getAllUsage();
    const updatedUsageRecord = addUsage(usageRecord, dateKey, usage);
    const totalUsage = updatedUsageRecord[dateKey];
    if (totalUsage === undefined) {
      throw new Error(`Usage was not accumulated for ${dateKey}`);
    }

    await this.storage.put('usage', updatedUsageRecord);
    return totalUsage;
  }

  /**
   * usage 内訳へ保存するメイン LLM の provider / model 名を返す。
   */
  private getMainLLMUsageIdentity(): { provider: string; model: string } {
    const config = resolveMainLLMConfig(
      this._env,
      this.getInstanceDefinitionOrThrow()
    );

    return {
      provider: config.provider,
      model: config.model ?? 'unknown',
    };
  }

  /**
   * Dashboard に表示する実効 runtime 設定を返す。
   *
   * API key は secret なので含めず、provider/model と token limit だけを公開する。
   *
   * @returns Dashboard DTO に載せる runtime 設定
   */
  private getDashboardRuntimeConfig(): DashboardRuntimeConfig {
    const mainLlm = resolveMainLLMConfig(
      this._env,
      this.getInstanceDefinitionOrThrow()
    );
    const tokenLimits = this.getTokenLimitConfig();

    return {
      mainLlm: {
        provider: mainLlm.provider,
        model: mainLlm.model ?? 'unknown',
      },
      tokenLimits,
    };
  }

  /**
   * 現在の instance に対応する token limit 設定を返す。
   *
   * @returns 実行判定で使う token limit 設定
   */
  private getTokenLimitConfig(): TokenLimitConfig {
    return resolveTokenLimitConfig(
      this._env,
      this.getInstanceDefinitionOrThrow()
    );
  }

  /**
   * 前回 `finish_thinking` が残した context を DO storage から読み出す。
   *
   * @returns 保存済み context。未保存なら `null`
   */
  private async loadContext(): Promise<ContextSnapshot | null> {
    return (await this.storage.get<ContextSnapshot>('context')) ?? null;
  }

  /**
   * 次回起動時に注入する最新 context を DO storage へ保存する。
   *
   * @param context 今回セッションの終了時に確定した context snapshot
   */
  private async saveContext(context: ContextSnapshot): Promise<void> {
    await this.storage.put('context', context);
  }

  /**
   * 前回 `finish_thinking` が残した next_wake_at を DO storage から読み出す。
   *
   * @returns 保存済み next_wake_at。未保存なら `null`
   */
  private async loadNextWakeAt(): Promise<string | null> {
    return (await this.storage.get<string>('next_wake_at')) ?? null;
  }

  /**
   * 次回起動の目安時刻を DO storage へ保存する。
   *
   * @param nextWakeAt 今回の終了時に確定した次回起動時刻
   */
  private async saveNextWakeAt(
    nextWakeAt: string,
    reason: string,
    metadata: Record<string, unknown> = {},
    previousValueOverride?: string | null
  ): Promise<void> {
    const previousValue =
      previousValueOverride === undefined
        ? await this.loadNextWakeAt()
        : previousValueOverride;
    await this.storage.put('next_wake_at', nextWakeAt);
    await emitEchoEvent(this.events, {
      type: 'system.schedule.next_wake_at_updated',
      severity: 'info',
      summary: `next_wake_at updated: ${nextWakeAt}`,
      payload: {
        previousValue,
        nextValue: nextWakeAt,
        reason,
        ...metadata,
      },
    });
  }

  /**
   * 保存済みの次回起動時刻を破棄する。
   * 今回の session が next_wake_at を指定しなかった場合のリセットに使う。
   */
  private async clearNextWakeAt(reason: string): Promise<void> {
    const previousValue = await this.loadNextWakeAt();
    await this.storage.delete('next_wake_at');
    if (previousValue === null) {
      return;
    }

    await emitEchoEvent(this.events, {
      type: 'system.schedule.next_wake_at_cleared',
      severity: 'info',
      summary: 'next_wake_at cleared',
      payload: {
        previousValue,
        reason,
      },
    });
  }

  private async invalidateNextWakeAt(
    storedValue: string,
    reason: string
  ): Promise<void> {
    await this.storage.delete('next_wake_at');
    await emitEchoEvent(this.events, {
      type: 'system.schedule.next_wake_at_invalidated',
      severity: 'warn',
      summary: `next_wake_at invalidated: ${reason}`,
      payload: {
        storedValue,
        reason,
      },
    });
  }

  /**
   * @returns メイン LLM 用の OpenAI-compatible model adapter
   */
  private createMainLLMClient(): ModelPort {
    const config = resolveMainLLMConfig(
      this._env,
      this.getInstanceDefinitionOrThrow()
    );

    if (config.api === 'chat_completions') {
      if (config.model === undefined) {
        throw new Error('Chat Completions model is not configured');
      }

      return new OpenAIChatCompletionsModel({
        apiKey: config.apiKey,
        model: config.model,
        baseURL: config.baseURL,
        events: this.events,
        maxTokens: config.maxTokens,
        temperature: config.temperature,
        topP: config.topP,
        presencePenalty: config.presencePenalty,
        extraBody: config.extraBody,
      });
    }

    return new OpenAIResponsesModel({
      apiKey: config.apiKey,
      model: config.model,
      events: this.events,
      reasoningEffort: config.reasoningEffort,
    });
  }

  /**
   * @returns provider/runtime 非依存の core ThinkingEngine
   */
  private createThinkingEngine(): AgentThinkingEngine {
    const definition = this.getInstanceDefinitionOrThrow();
    const memorySystem = this.getMemorySystemOrThrow();

    return new AgentThinkingEngine({
      model: this.createMainLLMClient(),
      events: this.events,
      context: {
        load: async (): Promise<ContextSnapshot | null> =>
          await this.loadContext(),
      },
      memory: {
        search: async (query) => await memorySystem.searchMemory(query),
      },
      tools: this.getExecutableToolsOrThrow(),
      systemPrompt: definition.systemPrompt,
    });
  }
}
