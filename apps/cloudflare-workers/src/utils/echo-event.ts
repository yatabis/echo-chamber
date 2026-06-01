import type {
  EchoEvent,
  EchoEventPort,
} from '@echo-chamber/core/ports/echo-event';
import { sendChannelMessage } from '@echo-chamber/discord-adapter/api';

export interface ConsoleEchoEventPortOptions {
  source: string;
  getInstanceId(): string | null;
  getSessionId(): string | null;
}

export interface DiscordEchoEventConfig {
  token: string;
  channelId: string;
}

export interface DiscordEchoEventPortOptions
  extends ConsoleEchoEventPortOptions {
  getDiscordConfig(): DiscordEchoEventConfig | null;
}

export interface EchoEventArchive {
  recordEvent(
    event: EchoEvent,
    context: {
      sessionId: string | null;
    }
  ): Promise<void>;
}

export interface ArchiveEchoEventPortOptions
  extends ConsoleEchoEventPortOptions {
  eventArchive: EchoEventArchive;
}

export interface CloudflareEchoEventPortOptions
  extends DiscordEchoEventPortOptions,
    ArchiveEchoEventPortOptions {}

const DISCORD_MESSAGE_MAX_LENGTH = 2000;

/**
 * EchoEventPort を Cloudflare Workers の console log へ構造化出力する。
 *
 * Discord 通知とは独立した観察・分析用イベントとして扱う。
 */
export class ConsoleEchoEventPort implements EchoEventPort {
  private readonly options: ConsoleEchoEventPortOptions;

  /**
   * @param options 出力元名と実行時 context の取得 callback
   */
  constructor(options: ConsoleEchoEventPortOptions) {
    this.options = options;
  }

  /**
   * Echo event を Cloudflare observability に載る JSON log として出力する。
   *
   * @param event 出力する Echo event
   * @returns 出力完了
   */
  async emit(event: EchoEvent): Promise<void> {
    const logString = JSON.stringify({
      timestamp: new Date().toISOString(),
      kind: 'echo_event',
      source: this.options.source,
      instanceId: this.options.getInstanceId(),
      sessionId: this.options.getSessionId(),
      ...event,
    });

    switch (event.severity) {
      case 'debug':
        console.debug(logString);
        break;
      case 'info':
        console.info(logString);
        break;
      case 'warn':
        console.warn(logString);
        break;
      case 'error':
        console.error(logString);
        break;
    }
    await Promise.resolve();
  }
}

/**
 * Echo event を Discord 通知として送る port。
 *
 * Discord は即時確認用なので、session lifecycle と system stream の warn/error に絞る。
 */
export class DiscordEchoEventPort implements EchoEventPort {
  private readonly options: DiscordEchoEventPortOptions;

  /**
   * @param options Discord 送信先と実行時 context の取得 callback
   */
  constructor(options: DiscordEchoEventPortOptions) {
    this.options = options;
  }

  /**
   * Discord 通知対象の event だけを送信する。
   *
   * @param event 出力する Echo event
   * @returns 出力完了
   */
  async emit(event: EchoEvent): Promise<void> {
    if (!shouldNotifyDiscord(event)) {
      return;
    }

    const config = this.options.getDiscordConfig();
    if (config === null) {
      return;
    }

    await sendChannelMessage(config.token, config.channelId, {
      content: truncateDiscordMessage(
        formatDiscordEventMessage(event, {
          source: this.options.source,
          instanceId: this.options.getInstanceId(),
          sessionId: this.options.getSessionId(),
        })
      ),
    });
  }
}

/**
 * Echo event を dashboard / R2 archive 用の保存層へ送る port。
 */
export class ArchiveEchoEventPort implements EchoEventPort {
  private readonly options: ArchiveEchoEventPortOptions;

  /**
   * @param options archive 保存先と実行時 context の取得 callback
   */
  constructor(options: ArchiveEchoEventPortOptions) {
    this.options = options;
  }

  /**
   * Echo event を archive 保存層へ記録する。
   *
   * @param event 保存する Echo event
   */
  async emit(event: EchoEvent): Promise<void> {
    await this.options.eventArchive.recordEvent(event, {
      sessionId: this.options.getSessionId(),
    });
  }
}

/**
 * 複数の EchoEventPort へ同じ event を配送する。
 */
export class CompositeEchoEventPort implements EchoEventPort {
  private readonly ports: readonly EchoEventPort[];

  /**
   * @param ports 配送先 port
   */
  constructor(ports: readonly EchoEventPort[]) {
    this.ports = ports;
  }

  /**
   * 登録済み port へ順に event を送る。
   *
   * @param event 出力する Echo event
   * @returns 出力完了
   */
  async emit(event: EchoEvent): Promise<void> {
    let firstError: Error | undefined;

    for (const port of this.ports) {
      try {
        // Observability sinks should preserve order for the same event.
        // eslint-disable-next-line no-await-in-loop
        await port.emit(event);
      } catch (error) {
        firstError ??=
          error instanceof Error ? error : new Error('Echo event port failed');
      }
    }

    if (firstError !== undefined) {
      throw firstError;
    }
  }
}

/**
 * Discord に通知する event かを判定する。
 *
 * @param event 判定対象の Echo event
 * @returns Discord に送るなら `true`
 */
export function shouldNotifyDiscord(event: EchoEvent): boolean {
  if (event.category === 'session') {
    return true;
  }
  if (event.severity === 'error') {
    return true;
  }
  if (event.severity === 'warn' && event.streams.includes('system')) {
    return true;
  }

  return false;
}

/**
 * Discord に送る event message を組み立てる。
 *
 * @param event Echo event
 * @param context 実行時 context
 * @returns Discord 投稿本文
 */
function formatDiscordEventMessage(
  event: EchoEvent,
  context: {
    source: string;
    instanceId: string | null;
    sessionId: string | null;
  }
): string {
  const lines = [
    `**[${event.severity.toUpperCase()}] ${event.type}**`,
    event.summary,
    `source: ${context.source}`,
    context.instanceId === null ? undefined : `instance: ${context.instanceId}`,
    context.sessionId === null ? undefined : `session: ${context.sessionId}`,
  ].filter((line) => line !== undefined);
  const payload =
    event.payload === undefined
      ? ''
      : `\n\n\`\`\`json\n${stringifyPayload(event.payload)}\n\`\`\``;

  return `${lines.join('\n')}${payload}`;
}

/**
 * payload を Discord 表示用 JSON にする。
 *
 * @param payload event payload
 * @returns JSON 文字列
 */
function stringifyPayload(payload: Record<string, unknown>): string {
  try {
    return JSON.stringify(payload, null, 2);
  } catch {
    return JSON.stringify({ error: 'payload serialization failed' });
  }
}

/**
 * Discord の投稿上限に合わせて本文を切り詰める。
 *
 * @param message 投稿本文
 * @returns Discord に渡せる長さの本文
 */
function truncateDiscordMessage(message: string): string {
  if (message.length <= DISCORD_MESSAGE_MAX_LENGTH) {
    return message;
  }

  return `${message.slice(0, DISCORD_MESSAGE_MAX_LENGTH - 3)}...`;
}

/**
 * Cloudflare console 出力用の EchoEventPort を作る。
 *
 * @param options 出力元名と実行時 context の取得 callback
 * @returns EchoEventPort 実装
 */
export function createConsoleEchoEventPort(
  options: ConsoleEchoEventPortOptions
): EchoEventPort {
  return new ConsoleEchoEventPort(options);
}

/**
 * Cloudflare runtime 用の EchoEventPort を作る。
 *
 * @param options 出力元名、実行時 context、archive 保存先、Discord 送信先の取得 callback
 * @returns console、archive、Discord へ配送する EchoEventPort
 */
export function createCloudflareEchoEventPort(
  options: CloudflareEchoEventPortOptions
): EchoEventPort {
  const ports: EchoEventPort[] = [
    new ConsoleEchoEventPort(options),
    new ArchiveEchoEventPort(options),
    new DiscordEchoEventPort(options),
  ];

  return new CompositeEchoEventPort(ports);
}
