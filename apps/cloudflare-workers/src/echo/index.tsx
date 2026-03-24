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
  EchoStatus,
} from '@echo-chamber/contracts/dashboard/types';
import { canonicalRuntimeTools } from '@echo-chamber/core/agent/runtime-tools/catalog';
import { bindRuntimeTools } from '@echo-chamber/core/agent/runtime-tools/tool';
import type { AgentSessionTool } from '@echo-chamber/core/agent/session';
import { ThinkingEngine as AgentThinkingEngine } from '@echo-chamber/core/agent/thinking-engine';
import {
  ALARM_CONFIG,
  SCHEDULING_CONFIG,
  TOKEN_LIMITS,
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
  getTodayUsageKey,
} from '@echo-chamber/core/echo/usage';
import type { ContextSnapshot } from '@echo-chamber/core/ports/context';
import type { EchoInstanceId } from '@echo-chamber/core/types/echo-config';
import { isValidInstanceId } from '@echo-chamber/core/types/echo-config';
import { formatDatetime } from '@echo-chamber/core/utils/datetime';
import { getErrorMessage } from '@echo-chamber/core/utils/error';
import { DiscordThoughtLog } from '@echo-chamber/discord-adapter/discord-thought-log';
import { getUnreadMessageCount } from '@echo-chamber/discord-adapter/notification-utils';
import { OpenAIResponsesModel } from '@echo-chamber/openai-adapter/openai-responses-model';

import {
  resolveEchoRuntimeBindings,
  type EchoChatChannelBinding,
  type EchoRuntimeBindings,
} from '../config/echo-runtime-bindings';
import { createEmbeddingService } from '../embedding/create-embedding-service';
import { createRerankingService } from '../reranking/create-reranking-service';
import { createLogger } from '../utils/logger';

import { createToolExecutionContext } from './tool-context';

import type { Logger } from '../utils/logger';

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
  private readonly logger: Logger;
  private readonly _env: Env;
  private readonly noteSystem: NoteSystem;

  // 遅延初期化されるプロパティ（ensureInitializedで設定されるためreadonlyではない）
  private executableTools: readonly AgentSessionTool[] | null = null;
  private instanceDefinition: EchoInstanceDefinition | null = null;
  private runtimeBindings: EchoRuntimeBindings | null = null;
  private memorySystem: MemorySystem | null = null;

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
    this.logger = createLogger(env);
    this._env = env;
    this.noteSystem = new NoteSystem({
      storage: this.storage,
      logger: this.logger,
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
    console.log('Echo Durable Object created');
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
      this.runtimeBindings.embeddingConfig
    );
    const rerankingService = createRerankingService(this._env);
    this.memorySystem = new MemorySystem({
      sql: this.ctx.storage.sql,
      embeddingService,
      rerankingService,
      logger: this.logger,
    });
    const toolContext = createToolExecutionContext({
      chatBindings: this.getRuntimeBindingsOrThrow(),
      memorySystem: this.memorySystem,
      noteSystem: this.noteSystem,
      logger: this.logger,
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

    try {
      await this.logger.debug(
        `Alarm triggered with info: ${JSON.stringify(alarmInfo)}`
      );

      // ストレージからIDを読み取り初期化
      const storedId = await this.storage.get<string>('id');
      if (storedId == null || !isValidInstanceId(storedId)) {
        await this.logger.error(
          'No valid instance ID in storage. Cannot run alarm.\nEcho going to sleep.'
        );
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
        await this.setNextAlarm(nextAlarm);
        await this.logger.info(
          `Echo is going to sleep and will wake at ${formatDatetime(nextAlarm)}.`
        );
        return;
      }
      if (now.getHours() === 22 && state === 'Sleeping') {
        await this.wake(true);
      }
      runResult = await this.run();
      await this.setNextAlarm();
    } finally {
      await this.logger.debug('Echo alarm metrics', {
        alarm_total_ms: Date.now() - alarmStartedAt,
        unread_check_ms: runResult.unreadCheckMs,
        think_ms: runResult.thinkMs,
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

  async setNextAlarm(nextAlarm?: Date): Promise<void> {
    if (!nextAlarm) {
      nextAlarm = new Date();
      nextAlarm.setMinutes(
        nextAlarm.getMinutes() + ALARM_CONFIG.INTERVAL_MINUTES,
        0,
        0
      );
    }
    await this.storage.setAlarm(nextAlarm);
    await this.logger.debug(`Next alarm set for ${nextAlarm.toISOString()}`);
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

  async setState(newState: EchoState): Promise<void> {
    await this.storage.put('state', newState);
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

    return parseDashboardInstanceSummary({
      id: definition.id,
      name: definition.name,
      state: await this.getState(),
      nextAlarm: await this.getNextAlarm(),
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
    const usage = await this.storage.get<UsageRecord>('usage');
    return usage ?? {};
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
      await this.logger.warn(
        'Echo is currently sleeping! Cannot wake while sleeping.'
      );
      return;
    }

    await this.setNextAlarm();
    await this.setState('Idling');
  }

  async sleep(force = false): Promise<void> {
    const state = await this.getState();

    if (state === 'Sleeping') {
      await this.logger.info('Echo is already sleeping.');
      return;
    }

    if (!force && state === 'Running') {
      await this.logger.warn(
        'Echo is currently running! Cannot sleep while running.'
      );
      return;
    }

    try {
      await this.setState('Sleeping');
      await this.storage.deleteAlarm();
      // sleep 処理
    } catch (error) {
      await this.logger.error(
        `Echo encountered an error during sleep: ${getErrorMessage(error)}`
      );
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

    await this.setState('Running');
    const name = await this.getName();
    await this.logger.info(`${name}が思考を開始しました。`);
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
        await this.clearNextWakeAt();
      } else {
        await this.saveNextWakeAt(nextWakeAt);
      }
      await this.logger.info(`usage: ${usage.totalTokens}`);
      const totalUsage = await this.updateUsage(convertUsage(usage));
      await this.createThoughtLog().send(
        `Usage: ${usage.totalTokens} tokens (Total: ${totalUsage.total_tokens} tokens)`
      );
      await this.logger.info(`${name}が思考を正常に完了しました。`);
    } catch (error) {
      if (thinkStartedAt !== 0) {
        thinkMs = Date.now() - thinkStartedAt;
      }
      await this.logger.error(
        `${name}の思考中にエラーが発生しました: ${getErrorMessage(error)}`
      );
    } finally {
      await this.setState('Idling');
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
      return {
        shouldRun: true,
        unreadCheckMs,
      };
    }

    const todayUsage = await this.getTodayUsage();
    const totalTokens = todayUsage?.total_tokens ?? 0;
    const { nextWakeAt, hasReachedNextWakeAt } =
      await this.resolveNextWakeAtStatus();
    if (
      !(await this.validateHardTokenLimit(totalTokens, hasReachedNextWakeAt))
    ) {
      return {
        shouldRun: false,
        unreadCheckMs,
      };
    }

    if (hasReachedNextWakeAt && nextWakeAt !== null) {
      await this.logger.info(
        `next_wake_at reached: ${nextWakeAt.toISOString()}`
      );
      return {
        shouldRun: true,
        unreadCheckMs,
      };
    }

    const shouldRunForSoftLimit = await this.validateSoftLimitRun(
      totalTokens,
      nextWakeAt
    );

    return {
      shouldRun: shouldRunForSoftLimit,
      unreadCheckMs,
    };
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
  private async validateSoftLimitRun(
    totalTokens: number,
    nextWakeAt: Date | null
  ): Promise<boolean> {
    // soft limit 未満なら通常起動。ただし直近の next_wake_at があるときは待機する。
    const softLimit = calculateDynamicTokenLimit(TOKEN_LIMITS.DAILY_SOFT_LIMIT);
    if (totalTokens >= softLimit) {
      return false;
    }

    if (
      nextWakeAt !== null &&
      this.isNextWakeAtWithinSoftLimitWindow(nextWakeAt)
    ) {
      await this.logger.info(
        `Skipping soft-limit run because next_wake_at is within ${SCHEDULING_CONFIG.SOFT_LIMIT_NEXT_WAKE_AT_WINDOW_MINUTES} minutes: ${nextWakeAt.toISOString()}`
      );
      return false;
    }

    await this.logger.info(
      `Usage: ${totalTokens}  (Soft limit: ${Math.floor(softLimit)})`
    );
    return true;
  }

  /**
   * 未読メッセージ以外の通常起動で使えるトークン量が残っているかを検証する。
   * hard limit を超えた場合は next_wake_at に到達していても起動しない。
   *
   * @param totalTokens 今日すでに消費した総トークン数
   * @param shouldWarnOnHardLimit hard limit 到達時に warn を出すべきなら `true`
   * @returns hard limit 未満なら `true`
   */
  private async validateHardTokenLimit(
    totalTokens: number,
    shouldWarnOnHardLimit: boolean
  ): Promise<boolean> {
    const hardLimit = calculateDynamicTokenLimit(
      TOKEN_LIMITS.DAILY_HARD_LIMIT,
      TOKEN_LIMITS.HARD_LIMIT_BUFFER_FACTOR
    );
    if (totalTokens < hardLimit) {
      return true;
    }

    if (shouldWarnOnHardLimit) {
      await this.logger.warn(
        `Usage hard limit reached: ${totalTokens}  (Hard limit: ${Math.floor(hardLimit)})`
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
      await this.logger.error('Echo ID is not set. Cannot validate state.');
      return false;
    }

    const state = await this.getState();

    // 睡眠中は実行できない
    if (state === 'Sleeping') {
      await this.logger.warn(
        `${name} is currently sleeping! Cannot run while sleeping.`
      );
      return false;
    }

    // 既に実行中の場合は何もしない
    if (state === 'Running') {
      await this.logger.warn(`${name} is already running.`);
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

    if (runtimeBindings.chatChannels.length === 0) {
      await this.logger.error(
        `${name}のチャットチャンネルが設定されていません。`
      );
      return false;
    }

    const unreadCounts = await fetchUnreadMessageCounts(
      runtimeBindings.discordBotToken,
      runtimeBindings.chatChannels
    );

    const unreadChannels = unreadCounts.filter(
      ({ unreadCount }) => unreadCount > 0
    );
    if (unreadChannels.length > 0) {
      const totalUnreadCount = unreadChannels.reduce(
        (total, { unreadCount }) => total + unreadCount,
        0
      );
      await this.logger.info(
        `${name}の未読メッセージ数: ${totalUnreadCount} (${unreadChannels
          .map(
            ({ channel, unreadCount }) =>
              `${channel.displayName}(${channel.key}): ${unreadCount}`
          )
          .join(', ')})`
      );
    } else {
      await this.logger.debug(`${name}の未読メッセージはありません。`);
    }

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
      await this.logger.warn(
        `Stored next_wake_at is invalid and will be ignored: ${nextWakeAt}`
      );
      await this.clearNextWakeAt();
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
    await this.logger.debug(
      `Usage accumulated for ${dateKey}: ${JSON.stringify(totalUsage, null, 2)}`
    );
    return totalUsage;
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
  private async saveNextWakeAt(nextWakeAt: string): Promise<void> {
    await this.storage.put('next_wake_at', nextWakeAt);
  }

  /**
   * 保存済みの次回起動時刻を破棄する。
   * 今回の session が next_wake_at を指定しなかった場合のリセットに使う。
   */
  private async clearNextWakeAt(): Promise<void> {
    await this.storage.delete('next_wake_at');
  }

  /**
   * @returns 実行ごとの thought log adapter
   */
  private createThoughtLog(): DiscordThoughtLog {
    const runtimeBindings = this.getRuntimeBindingsOrThrow();

    return new DiscordThoughtLog({
      token: runtimeBindings.discordBotToken,
      channelId: runtimeBindings.thinkingChannelId,
    });
  }

  /**
   * @param thoughtLog 実行ごとの thought log adapter
   * @returns OpenAI Responses API 用 model adapter
   */
  private createOpenAIClient(
    thoughtLog: DiscordThoughtLog
  ): OpenAIResponsesModel {
    return new OpenAIResponsesModel({
      apiKey: this._env.OPENAI_API_KEY,
      logger: this.logger,
      thoughtLog,
    });
  }

  /**
   * @returns provider/runtime 非依存の core ThinkingEngine
   */
  private createThinkingEngine(): AgentThinkingEngine {
    const definition = this.getInstanceDefinitionOrThrow();
    const memorySystem = this.getMemorySystemOrThrow();
    const thoughtLog = this.createThoughtLog();

    return new AgentThinkingEngine({
      model: this.createOpenAIClient(thoughtLog),
      thoughtLog,
      logger: this.logger,
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
