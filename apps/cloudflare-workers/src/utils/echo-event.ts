import type {
  EchoEvent,
  EchoEventPort,
} from '@echo-chamber/core/ports/echo-event';

export interface ConsoleEchoEventPortOptions {
  source: string;
  getInstanceId(): string | null;
  getSessionId(): string | null;
}

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
