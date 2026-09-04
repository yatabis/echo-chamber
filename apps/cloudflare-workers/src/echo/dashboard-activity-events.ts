import type { EchoEventType } from '@echo-chamber/core/ports/echo-event';

/** Dashboard session log が Activity の材料として受け取る event type。 */
export const DASHBOARD_ACTIVITY_EVENT_TYPES = [
  'session.started',
  'session.completed',
  'session.failed',
  'model.turn.completed',
  'model.output.emitted',
  'model.provider.warning',
  'tool.called',
  'tool.completed',
  'tool.failed',
  'memory.search.completed',
  'memory.search.failed',
  'cognitive.phase.committed',
  'cognitive.phase.failed',
] as const satisfies readonly EchoEventType[];

const DASHBOARD_ACTIVITY_EVENT_TYPE_SET = new Set<string>(
  DASHBOARD_ACTIVITY_EVENT_TYPES
);

/**
 * event type が Dashboard session log の Activity 候補かを返す。
 *
 * @param type 判定対象の event type
 * @returns Activity 候補なら `true`
 */
export function isDashboardActivityEventType(type: string): boolean {
  return DASHBOARD_ACTIVITY_EVENT_TYPE_SET.has(type);
}
