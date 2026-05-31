export type EchoEventSeverity = 'debug' | 'info' | 'warn' | 'error';

export type EchoEventCategory =
  | 'session'
  | 'model'
  | 'tool'
  | 'memory'
  | 'control'
  | 'usage';

export type EchoEventStream = 'thought' | 'system' | 'analysis';

export type EchoEventType =
  | 'session.started'
  | 'session.completed'
  | 'session.failed'
  | 'model.turn.started'
  | 'model.turn.completed'
  | 'tool.called'
  | 'tool.completed'
  | 'tool.failed'
  | 'memory.search.started'
  | 'memory.search.completed'
  | 'run_decision.evaluated'
  | 'usage.recorded';

const ECHO_EVENT_CATEGORIES: Record<EchoEventType, EchoEventCategory> = {
  'session.started': 'session',
  'session.completed': 'session',
  'session.failed': 'session',
  'model.turn.started': 'model',
  'model.turn.completed': 'model',
  'tool.called': 'tool',
  'tool.completed': 'tool',
  'tool.failed': 'tool',
  'memory.search.started': 'memory',
  'memory.search.completed': 'memory',
  'run_decision.evaluated': 'control',
  'usage.recorded': 'usage',
};

const ECHO_EVENT_STREAMS: Record<EchoEventType, EchoEventStream[]> = {
  'session.started': ['thought', 'system', 'analysis'],
  'session.completed': ['thought', 'system', 'analysis'],
  'session.failed': ['thought', 'system', 'analysis'],
  'model.turn.started': ['analysis'],
  'model.turn.completed': ['analysis'],
  'tool.called': ['thought', 'analysis'],
  'tool.completed': ['system', 'analysis'],
  'tool.failed': ['thought', 'system', 'analysis'],
  'memory.search.started': ['system', 'analysis'],
  'memory.search.completed': ['system', 'analysis'],
  'run_decision.evaluated': ['system', 'analysis'],
  'usage.recorded': ['system', 'analysis'],
};

export interface EchoEventInput {
  type: EchoEventType;
  severity: EchoEventSeverity;
  summary: string;
  payload?: Record<string, unknown>;
}

export interface EchoEvent extends EchoEventInput {
  category: EchoEventCategory;
  streams: EchoEventStream[];
}

/**
 * Echo の運用・分析イベントを受け取る port。
 *
 * 実装側は Discord 通知、console 出力、D1/R2 archive などへ自由に配送できる。
 */
export interface EchoEventPort {
  emit(event: EchoEvent): Promise<void>;
}

/**
 * event type から意味カテゴリを決める。
 *
 * @param type Echo event type
 * @returns event の意味カテゴリ
 */
export function getEchoEventCategory(type: EchoEventType): EchoEventCategory {
  return ECHO_EVENT_CATEGORIES[type];
}

/**
 * event type から読み取り用途の stream を決める。
 *
 * @param type Echo event type
 * @returns dashboard / archive 上の論理 stream
 */
export function getEchoEventStreams(type: EchoEventType): EchoEventStream[] {
  return ECHO_EVENT_STREAMS[type];
}

/**
 * 発火元から渡された event input を、統一 event 形状に正規化する。
 *
 * @param event 発火元が生成した最小 event
 * @returns category / streams を補った Echo event
 */
export function createEchoEvent(event: EchoEventInput): EchoEvent {
  return {
    ...event,
    category: getEchoEventCategory(event.type),
    streams: getEchoEventStreams(event.type),
  };
}

/**
 * イベント配送失敗で agent 本体を落とさないための best-effort helper。
 *
 * @param events イベント配送 port。未指定なら何もしない
 * @param event 送信するイベント
 */
export async function emitEchoEvent(
  events: EchoEventPort | undefined,
  event: EchoEventInput
): Promise<void> {
  try {
    await events?.emit(createEchoEvent(event));
  } catch {
    // Observability must not affect agent execution.
  }
}
