import { DurableObject } from 'cloudflare:workers';
import { Hono } from 'hono';

import {
  ALARM_CONFIG,
  TOKEN_LIMITS,
  addUsage,
  calculateDynamicTokenLimit,
  convertUsage,
  formatDatetime,
  getTodayUsageKey,
  getErrorMessage,
  isValidInstanceId,
} from '@echo-chamber/core';
import type {
  DashboardInstanceSummary,
  EchoStatus,
  EchoInstanceConfig,
  EchoInstanceId,
  EchoState,
  Note,
  Usage,
  UsageRecord,
} from '@echo-chamber/core';

import { getInstanceConfig } from '../config/echo-registry';
import { getUnreadMessageCount } from '../discord/client';
import { createEmbeddingService } from '../llm/embedding-factory';
import { MemorySystem } from '../runtime/memory-system';
import { NoteSystem } from '../runtime/note-system';
import { createLogger } from '../utils/logger';

import { ThinkingEngine } from './thinking-engine';

import type { Logger } from '../utils/logger';

async function fetchUnreadMessageCount(
  token: string,
  channelId: string
): Promise<number> {
  return getUnreadMessageCount(token, channelId);
}

export class Echo extends DurableObject<Env> {
  private readonly store: KVNamespace;
  private readonly storage: DurableObjectStorage;
  private readonly router: Hono;
  private readonly logger: Logger;
  private readonly _env: Env;
  private readonly noteSystem: NoteSystem;

  // 遅延初期化されるプロパティ（ensureInitializedで設定されるためreadonlyではない）
  private instanceConfig: EchoInstanceConfig | null = null;
  private thinkingEngine: ThinkingEngine | null = null;

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
    void this.logger.info('Echo Durable Object created');
  }

  async fetch(request: Request): Promise<Response> {
    return this.router.fetch(request);
  }

  /**
   * インスタンスの遅延初期化
   * 最初のリクエスト時に呼び出され、instanceConfigとThinkingEngineを設定する
   */
  private async ensureInitialized(id: EchoInstanceId): Promise<void> {
    await this.storage.delete('context');
    await this.storage.delete('tasks');
    await this.storage.delete('knowledge');

    // 既に同じIDで初期化済みの場合はスキップ
    if (this.instanceConfig?.id === id) {
      return;
    }

    this.instanceConfig = await getInstanceConfig(this._env, this.store, id);
    this.thinkingEngine = new ThinkingEngine({
      env: this._env,
      storage: this.storage,
      sql: this.ctx.storage.sql,
      logger: this.logger,
      instanceConfig: this.instanceConfig,
    });
    // embedding モデル変更時の自動再 embedding
    await this.thinkingEngine.initialize();

    // ストレージにID/名前を保存（alarmから参照するため）
    await this.storage.put('id', id);
    await this.storage.put('name', this.instanceConfig.name);
  }

  /**
   * instanceConfigを取得（初期化されていない場合はエラー）
   */
  private getInstanceConfigOrThrow(): EchoInstanceConfig {
    if (!this.instanceConfig) {
      throw new Error('Echo instance not initialized');
    }
    return this.instanceConfig;
  }

  /**
   * ThinkingEngineを取得（初期化されていない場合はエラー）
   */
  private getThinkingEngineOrThrow(): ThinkingEngine {
    if (!this.thinkingEngine) {
      throw new Error('ThinkingEngine not initialized');
    }
    return this.thinkingEngine;
  }

  async alarm(alarmInfo?: AlarmInvocationInfo): Promise<void> {
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
    await this.run();
    await this.setNextAlarm();
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
    const instanceConfig = this.getInstanceConfigOrThrow();
    const state = await this.getState();
    const nextAlarm = await this.getNextAlarm();
    const usage = await this.getAllUsage();

    const embeddingService = createEmbeddingService(
      this._env,
      instanceConfig.embeddingConfig
    );
    const memorySystem = new MemorySystem({
      sql: this.ctx.storage.sql,
      embeddingService,
      logger: this.logger,
    });
    const memories = memorySystem.getAllMemories().map((row) => ({
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

    return {
      id: instanceConfig.id,
      name: instanceConfig.name,
      state,
      nextAlarm,
      memories,
      notes,
      usage,
    };
  }

  /**
   * Dashboard 一覧画面向けの軽量サマリーを返す。
   *
   * 一覧では name/state/nextAlarm のみ使うため、詳細 DTO より小さい形で返す。
   */
  async getSummary(): Promise<DashboardInstanceSummary> {
    const config = this.getInstanceConfigOrThrow();

    return {
      id: config.id,
      name: config.name,
      state: await this.getState(),
      nextAlarm: await this.getNextAlarm(),
    };
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

  async run(): Promise<void> {
    if (!(await this.validateRunPreconditions())) {
      return;
    }

    await this.setState('Running');
    const name = await this.getName();
    await this.logger.info(`${name}が思考を開始しました。`);

    try {
      const usage = await this.getThinkingEngineOrThrow().think();
      await this.logger.info(`usage: ${usage.totalTokens}`);
      await this.updateUsage(convertUsage(usage));
      await this.logger.info(`${name}が思考を正常に完了しました。`);
    } catch (error) {
      await this.logger.error(
        `${name}の思考中にエラーが発生しました: ${getErrorMessage(error)}`
      );
    } finally {
      await this.setState('Idling');
    }
  }

  /**
   * 実行前の前提条件を検証
   */
  private async validateRunPreconditions(): Promise<boolean> {
    // Stateチェック
    if (!(await this.validateEchoState())) {
      return false;
    }

    // 未読メッセージがあれば実行
    if (await this.validateChatMessage()) {
      return true;
    }

    // tokenが余っていれば実行
    const softLimit = calculateDynamicTokenLimit(TOKEN_LIMITS.DAILY_SOFT_LIMIT);
    const todayUsage = await this.getTodayUsage();
    const totalTokens = todayUsage?.total_tokens ?? 0;
    if (totalTokens < softLimit) {
      await this.logger.info(
        `Usage: ${totalTokens}  (Soft limit: ${Math.floor(softLimit)})`
      );
      return true;
    }

    // どの条件も満たしていない場合は実行しない
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
    const instanceConfig = this.getInstanceConfigOrThrow();

    if (instanceConfig.chatChannelId === '') {
      await this.logger.error(`${name}のチャンネルIDが設定されていません。`);
      return false;
    }

    const unreadCount = await fetchUnreadMessageCount(
      instanceConfig.discordBotToken,
      instanceConfig.chatChannelId
    );
    if (unreadCount > 0) {
      await this.logger.info(`${name}の未読メッセージ数: ${unreadCount}`);
    } else {
      await this.logger.debug(`${name}の未読メッセージはありません。`);
    }

    return unreadCount > 0;
  }

  /**
   * Usage情報を日別に累積保存
   */
  async updateUsage(usage: Usage): Promise<void> {
    const dateKey = getTodayUsageKey();
    const usageRecord = await this.getAllUsage();
    await this.storage.put('usage', addUsage(usageRecord, dateKey, usage));
    await this.logger.debug(
      `Usage accumulated for ${dateKey}: ${JSON.stringify(usageRecord[dateKey], null, 2)}`
    );
  }
}
