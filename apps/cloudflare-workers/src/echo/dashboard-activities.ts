import type {
  DashboardActivity,
  DashboardEchoEvent,
  DashboardSessionLog,
  DashboardSessionLogsResponse,
} from '@echo-chamber/contracts/dashboard/types';

import type { EchoEventArchiveDay } from './event-archive';

type DashboardSessionEchoEvent = Omit<DashboardEchoEvent, 'sessionId'> & {
  sessionId: string;
};

interface ActivityBuildContext {
  completedToolCallIds: Set<string>;
  toolCallsById: Map<string, DashboardSessionEchoEvent>;
}

interface MutableSessionLog {
  id: string;
  activities: DashboardActivity[];
  latestActivityAt: string;
  sessionId: string;
  startedAt: string;
  title: string;
}

/**
 * raw event archive day を dashboard がそのまま描画できる activity response へ射影する。
 *
 * @param input raw event archive day
 * @returns dashboard activity response
 */
export function buildDashboardSessionLogsResponse(
  input: EchoEventArchiveDay
): DashboardSessionLogsResponse {
  return {
    archiveDay: input.archiveDay,
    sessionLogs: buildSessionLogs(input.events),
  };
}

/**
 * raw event 列を session log へ射影する。
 */
function buildSessionLogs(
  events: readonly DashboardEchoEvent[]
): DashboardSessionLog[] {
  const eventsBySession = groupEventsBySession(events);

  return Array.from(eventsBySession.entries())
    .flatMap(([sessionId, sessionEvents]) => {
      const sessionLog = buildSessionLog(sessionId, sessionEvents);

      return sessionLog === null ? [] : [sessionLog];
    })
    .sort(compareSessionLogsByLatest);
}

function groupEventsBySession(
  events: readonly DashboardEchoEvent[]
): Map<string, DashboardSessionEchoEvent[]> {
  const eventsBySession = new Map<string, DashboardSessionEchoEvent[]>();

  for (const event of [...events].sort(compareEventsByCreatedAt)) {
    if (!hasSessionId(event)) {
      continue;
    }

    const sessionEvents = eventsBySession.get(event.sessionId) ?? [];
    sessionEvents.push(event);
    eventsBySession.set(event.sessionId, sessionEvents);
  }

  return eventsBySession;
}

/**
 * 1 session 分の raw event 列を、dashboard で読むための activity log にする。
 */
function buildSessionLog(
  sessionId: string,
  events: readonly DashboardSessionEchoEvent[]
): DashboardSessionLog | null {
  const activities = buildActivityEntries(events);
  if (activities.length === 0) {
    return null;
  }
  const firstActivity = activities[0];
  const lastActivity = activities[activities.length - 1];
  if (firstActivity === undefined || lastActivity === undefined) {
    return null;
  }

  const mutableSessionLog: MutableSessionLog = {
    id: getSessionLogId(sessionId),
    activities,
    latestActivityAt: lastActivity.createdAt,
    sessionId,
    startedAt: firstActivity.createdAt,
    title: getSessionLogTitle(sessionId),
  };

  return toDashboardSessionLog(mutableSessionLog);
}

function buildActivityEntries(
  events: readonly DashboardSessionEchoEvent[]
): DashboardActivity[] {
  const context = buildActivityContext(events);

  return events.flatMap((event) => {
    const entry = createSessionLogActivity(event, context);

    return entry === null ? [] : [entry];
  });
}

function createSessionLogActivity(
  event: DashboardSessionEchoEvent,
  context: ActivityBuildContext
): DashboardActivity | null {
  switch (event.category) {
    case 'session':
      return createSessionLifecycleActivity(event);
    case 'model':
      return createModelActivity(event);
    case 'tool':
      return createToolActivity(event, context);
    case 'memory':
      return createMemoryActivity(event);
    default:
      return null;
  }
}

function createSessionLifecycleActivity(
  event: DashboardSessionEchoEvent
): DashboardActivity | null {
  switch (event.type) {
    case 'session.started':
      return createSessionStartedActivity(event);
    case 'session.completed':
      return createSessionCompletedActivity(event);
    case 'session.failed':
      return createSessionFailedActivity(event);
    default:
      return null;
  }
}

function createModelActivity(
  event: DashboardSessionEchoEvent
): DashboardActivity | null {
  switch (event.type) {
    case 'model.output.emitted':
      return createModelOutputActivity(event);
    case 'model.provider.warning':
      return createModelProviderWarningActivity(event);
    case 'model.turn.completed':
      return createNoToolCallsActivity(event);
    default:
      return null;
  }
}

function createToolActivity(
  event: DashboardSessionEchoEvent,
  context: ActivityBuildContext
): DashboardActivity | null {
  switch (event.type) {
    case 'tool.called':
      return createToolCalledActivity(event, context);
    case 'tool.completed':
    case 'tool.failed':
      return createToolFinishedActivity(event, context);
    default:
      return null;
  }
}

function createMemoryActivity(
  event: DashboardSessionEchoEvent
): DashboardActivity | null {
  switch (event.type) {
    case 'memory.search.completed':
      return createMemorySearchCompletedActivity(event);
    case 'memory.search.failed':
      return createMemorySearchFailedActivity(event);
    case 'cognitive.phase.committed':
      return createCognitivePhaseCommittedActivity(event);
    case 'cognitive.phase.failed':
      return createCognitivePhaseFailedActivity(event);
    default:
      return null;
  }
}

/** Cognitive phase commit を検索・記銘・Emotion更新の確定 activity にする。 */
function createCognitivePhaseCommittedActivity(
  event: DashboardSessionEchoEvent
): DashboardActivity {
  const payload = getEventPayload(event);
  const phase = getPayloadString(payload, 'phase');
  const committedVersion = getPayloadNumber(payload, 'committedVersion');
  return {
    id: event.id,
    body: `Cognitive state was committed for ${phase ?? 'a cognitive phase'}.`,
    createdAt: event.createdAt,
    details: compactDetails({
      boundaryId: getPayloadString(payload, 'boundaryId'),
      committedVersion,
      memoryUpdates: getPayloadNumber(payload, 'memoryUpdates'),
      phase,
    }),
    kind: 'knowledge',
    meta: createActivityMeta(event, [
      phase,
      committedVersion === undefined
        ? undefined
        : `domain v${committedVersion}`,
    ]),
    tone: 'positive',
    title: 'Cognitive phase committed',
  };
}

/** Cognitive phase failure を generic Memory search failure と混同せず表示する。 */
function createCognitivePhaseFailedActivity(
  event: DashboardSessionEchoEvent
): DashboardActivity {
  const payload = getEventPayload(event);
  return {
    id: event.id,
    body: event.summary,
    createdAt: event.createdAt,
    details: compactDetails({
      boundaryId: getPayloadString(payload, 'boundaryId'),
      commitError: getPayloadString(payload, 'commitError'),
      emotion: payload.emotion,
      memory: payload.memory,
      phase: getPayloadString(payload, 'phase'),
    }),
    kind: 'issue',
    meta: createActivityMeta(event, []),
    tone: 'critical',
    title: 'Cognitive phase failed',
  };
}

function toDashboardSessionLog(
  sessionLog: MutableSessionLog
): DashboardSessionLog {
  const warningCount = sessionLog.activities.filter(
    (activity) => activity.tone === 'warning' || activity.tone === 'critical'
  ).length;

  return {
    id: sessionLog.id,
    activities: sessionLog.activities,
    activityCount: sessionLog.activities.length,
    latestActivityAt: sessionLog.latestActivityAt,
    meta: buildSessionLogMeta({
      activityCount: sessionLog.activities.length,
      latestActivityAt: sessionLog.latestActivityAt,
      sessionId: sessionLog.sessionId,
      startedAt: sessionLog.startedAt,
      warningCount,
    }),
    sessionId: sessionLog.sessionId,
    startedAt: sessionLog.startedAt,
    title: sessionLog.title,
    warningCount,
  };
}

function buildSessionLogMeta(input: {
  activityCount: number;
  latestActivityAt: string;
  sessionId: string;
  startedAt: string;
  warningCount: number;
}): string[] {
  const timeRange =
    input.startedAt === input.latestActivityAt
      ? formatDateTime(input.startedAt)
      : `${formatDateTime(input.startedAt)} - ${formatDateTime(input.latestActivityAt)}`;

  return [
    timeRange,
    `session ${input.sessionId}`,
    formatCountLabel(input.activityCount, 'activity', 'activities'),
    input.warningCount === 0
      ? undefined
      : formatCountLabel(input.warningCount, 'warning', 'warnings'),
  ].filter((item): item is string => item !== undefined);
}

function compareSessionLogsByLatest(
  left: DashboardSessionLog,
  right: DashboardSessionLog
): number {
  const latestCompare = compareDateTimeString(
    right.latestActivityAt,
    left.latestActivityAt
  );
  if (latestCompare !== 0) {
    return latestCompare;
  }

  return left.id.localeCompare(right.id);
}

function getSessionLogId(sessionId: string): string {
  return `session:${sessionId}`;
}

function getSessionLogTitle(sessionId: string): string {
  return `Session ${formatSessionLabel(sessionId)}`;
}

function formatSessionLabel(sessionId: string): string {
  return sessionId.length <= 12 ? sessionId : sessionId.slice(0, 8);
}

/**
 * activity log の構築に必要な tool call index を作る。
 */
function buildActivityContext(
  events: readonly DashboardSessionEchoEvent[]
): ActivityBuildContext {
  const completedToolCallIds = new Set<string>();
  const toolCallsById = new Map<string, DashboardSessionEchoEvent>();

  for (const event of events) {
    const callId = getToolCallId(event);
    if (callId === undefined) {
      continue;
    }
    if (event.type === 'tool.called') {
      toolCallsById.set(callId, event);
    }
    if (event.type === 'tool.completed' || event.type === 'tool.failed') {
      completedToolCallIds.add(callId);
    }
  }

  return {
    completedToolCallIds,
    toolCallsById,
  };
}

function createSessionStartedActivity(
  event: DashboardSessionEchoEvent
): DashboardActivity {
  return {
    id: event.id,
    body: 'A thinking session started.',
    createdAt: event.createdAt,
    details: null,
    kind: 'session',
    meta: createActivityMeta(event, []),
    tone: 'positive',
    title: 'Started thinking',
  };
}

function createSessionCompletedActivity(
  event: DashboardSessionEchoEvent
): DashboardActivity {
  const payload = getEventPayload(event);
  const totalTokens = getPayloadNumber(payload, 'totalTokens');
  const nextWakeAt = getPayloadString(payload, 'nextWakeAt');
  const terminationReason = getPayloadString(payload, 'terminationReason');

  return {
    id: event.id,
    body: [
      `Finished with ${formatCodeLabel(terminationReason)}.`,
      totalTokens === undefined
        ? undefined
        : `${formatNumber(totalTokens)} tokens used.`,
      nextWakeAt === undefined
        ? undefined
        : `Next wake at ${formatDateTime(nextWakeAt)}.`,
    ]
      .filter((line): line is string => line !== undefined)
      .join(' '),
    createdAt: event.createdAt,
    details: compactDetails({
      committedCognitivePhases: getPayloadNumber(
        payload,
        'committedCognitivePhases'
      ),
      nextWakeAt,
      terminationReason,
      totalTokens,
    }),
    kind: 'session',
    meta: createActivityMeta(event, []),
    tone: event.severity === 'warn' ? 'warning' : 'positive',
    title: 'Finished thinking',
  };
}

function createSessionFailedActivity(
  event: DashboardSessionEchoEvent
): DashboardActivity {
  return {
    id: event.id,
    body: event.summary,
    createdAt: event.createdAt,
    details: compactDetails({
      error: getPayloadString(getEventPayload(event), 'error'),
    }),
    kind: 'issue',
    meta: createActivityMeta(event, []),
    tone: 'critical',
    title: 'Thinking failed',
  };
}

function createModelOutputActivity(
  event: DashboardSessionEchoEvent
): DashboardActivity | null {
  const payload = getEventPayload(event);
  const content = getPayloadString(payload, 'content');
  if (content === undefined || content.trim() === '') {
    return null;
  }

  const turnIndex = getPayloadNumber(payload, 'turnIndex');
  const cognitiveModule = getPayloadString(payload, 'cognitiveModule');
  const cognitiveModuleLabel = getCognitiveModuleLabel(cognitiveModule);
  return {
    id: event.id,
    body: content,
    createdAt: event.createdAt,
    details: compactDetails({
      cognitiveModule,
      model: getPayloadString(payload, 'model'),
      provider: getPayloadString(payload, 'provider'),
      turnIndex,
    }),
    kind: 'thought',
    meta: createActivityMeta(event, [
      turnIndex === undefined ? undefined : `turn ${turnIndex}`,
      cognitiveModuleLabel,
    ]),
    tone: 'neutral',
    title: cognitiveModuleLabel ?? 'Echo',
  };
}

function createToolCalledActivity(
  event: DashboardSessionEchoEvent,
  context: ActivityBuildContext
): DashboardActivity | null {
  const callId = getToolCallId(event);
  if (callId !== undefined && context.completedToolCallIds.has(callId)) {
    return null;
  }

  const payload = getEventPayload(event);
  const toolName = getPayloadString(payload, 'toolName') ?? 'tool';
  const turnIndex = getPayloadNumber(payload, 'turnIndex');
  return {
    id: event.id,
    body: 'Started and has not recorded a completion yet.',
    createdAt: event.createdAt,
    details: compactDetails({
      callId,
      input: parseMaybeJson(payload.input),
      turnIndex,
    }),
    kind: 'action',
    meta: createActivityMeta(
      event,
      turnIndex === undefined ? [] : [`turn ${turnIndex}`]
    ),
    tone: 'neutral',
    title: `Called ${toolName}`,
  };
}

function createToolFinishedActivity(
  event: DashboardSessionEchoEvent,
  context: ActivityBuildContext
): DashboardActivity {
  const payload = getEventPayload(event);
  const callId = getToolCallId(event);
  const startedEvent =
    callId === undefined ? undefined : context.toolCallsById.get(callId);
  const startedPayload =
    startedEvent === undefined ? {} : getEventPayload(startedEvent);
  const toolName =
    getPayloadString(payload, 'toolName') ??
    getPayloadString(startedPayload, 'toolName') ??
    'tool';
  const turnIndex = getPayloadNumber(payload, 'turnIndex');
  const durationMs = getPayloadNumber(payload, 'durationMs');
  const operation = getPayloadString(payload, 'operation');
  const entityId = getPayloadString(payload, 'entityId');
  const success = event.type === 'tool.completed';

  return {
    id: event.id,
    body: formatToolActivityBody({
      entityId,
      error: payload.error,
      operation,
      success,
    }),
    createdAt: event.createdAt,
    details: compactDetails({
      callId,
      durationMs,
      entityId,
      entityType: getPayloadString(payload, 'entityType'),
      error: payload.error,
      input: parseMaybeJson(startedPayload.input),
      operation,
      outputLength: getPayloadNumber(payload, 'outputLength'),
      success: getPayloadBoolean(payload, 'success'),
      turnIndex,
    }),
    kind: success ? 'action' : 'issue',
    meta: createActivityMeta(event, [
      turnIndex === undefined ? undefined : `turn ${turnIndex}`,
      durationMs === undefined ? undefined : `${durationMs}ms`,
    ]),
    tone: success ? 'positive' : 'warning',
    title: success ? `Used ${toolName}` : `Tool failed: ${toolName}`,
  };
}

function createNoToolCallsActivity(
  event: DashboardSessionEchoEvent
): DashboardActivity | null {
  const payload = getEventPayload(event);
  const warnings = getPayloadStringArray(payload, 'warnings');
  if (!warnings.includes('no_tool_calls')) {
    return null;
  }

  const turnIndex = getPayloadNumber(payload, 'turnIndex');
  return {
    id: event.id,
    body: 'The model returned no tool calls in this turn, so the session continued.',
    createdAt: event.createdAt,
    details: compactDetails({
      durationMs: getPayloadNumber(payload, 'durationMs'),
      outputItemCount: getPayloadNumber(payload, 'outputItemCount'),
      toolCallCount: getPayloadNumber(payload, 'toolCallCount'),
      turnIndex,
      warnings,
    }),
    kind: 'decision',
    meta: createActivityMeta(
      event,
      turnIndex === undefined ? [] : [`turn ${turnIndex}`]
    ),
    tone: 'warning',
    title: 'No tool calls returned',
  };
}

function createModelProviderWarningActivity(
  event: DashboardSessionEchoEvent
): DashboardActivity {
  const payload = getEventPayload(event);
  const cognitiveModule = getPayloadString(payload, 'cognitiveModule');
  const cognitiveModuleLabel = getCognitiveModuleLabel(cognitiveModule);
  return {
    id: event.id,
    body: event.summary,
    createdAt: event.createdAt,
    details: compactDetails({
      code: getPayloadString(payload, 'code'),
      cognitiveModule,
      model: getPayloadString(payload, 'model'),
      provider: getPayloadString(payload, 'provider'),
      turnIndex: getPayloadNumber(payload, 'turnIndex'),
    }),
    kind: 'issue',
    meta: createActivityMeta(event, [cognitiveModuleLabel]),
    tone: 'warning',
    title: 'Model warning',
  };
}

/** Cognitive model event の attribution を Dashboard 表示名へ正規化する。 */
function getCognitiveModuleLabel(
  cognitiveModule: string | undefined
): 'Memory Module' | 'Emotion Module' | undefined {
  if (cognitiveModule === 'memory') {
    return 'Memory Module';
  }
  if (cognitiveModule === 'emotion') {
    return 'Emotion Module';
  }
  return undefined;
}

function createMemorySearchFailedActivity(
  event: DashboardSessionEchoEvent
): DashboardActivity {
  const payload = getEventPayload(event);
  return {
    id: event.id,
    body: event.summary,
    createdAt: event.createdAt,
    details: compactDetails({
      error: getPayloadString(payload, 'error'),
      query: getPayloadString(payload, 'query'),
      source: getPayloadString(payload, 'source'),
    }),
    kind: 'issue',
    meta: createActivityMeta(event, []),
    tone: 'warning',
    title: 'Memory search failed',
  };
}

function createMemorySearchCompletedActivity(
  event: DashboardSessionEchoEvent
): DashboardActivity {
  const payload = getEventPayload(event);
  const query = getPayloadString(payload, 'query');
  const finalResultCount = getPayloadNumber(payload, 'finalResultCount') ?? 0;

  return {
    id: event.id,
    body:
      query === undefined
        ? `${formatNumber(finalResultCount)} related memories found.`
        : `${formatNumber(finalResultCount)} related memories found for "${query}".`,
    createdAt: event.createdAt,
    details: compactDetails({
      durationMs: getPayloadNumber(payload, 'durationMs'),
      finalResultCount,
      query,
      sourceCount: getPayloadNumber(payload, 'sourceCount'),
      type: getPayloadString(payload, 'type'),
      vectorCandidateCount: getPayloadNumber(payload, 'vectorCandidateCount'),
    }),
    kind: 'knowledge',
    meta: createActivityMeta(event, []),
    tone: 'neutral',
    title: 'Checked memory',
  };
}

function formatToolActivityBody(input: {
  entityId: string | undefined;
  error: unknown;
  operation: string | undefined;
  success: boolean;
}): string {
  if (!input.success) {
    return input.error === undefined
      ? 'The tool returned a failure.'
      : `The tool returned a failure: ${formatUnknownInline(input.error)}`;
  }
  if (input.operation !== undefined && input.entityId !== undefined) {
    return `${formatCodeLabel(input.operation)} completed for ${input.entityId}.`;
  }
  if (input.operation !== undefined) {
    return `${formatCodeLabel(input.operation)} completed.`;
  }

  return 'Completed successfully.';
}

function getEventPayload(event: DashboardEchoEvent): Record<string, unknown> {
  return event.payload ?? {};
}

function getPayloadString(
  payload: Record<string, unknown>,
  key: string
): string | undefined {
  const value = payload[key];
  return typeof value === 'string' ? value : undefined;
}

function getPayloadNumber(
  payload: Record<string, unknown>,
  key: string
): number | undefined {
  const value = payload[key];
  return typeof value === 'number' ? value : undefined;
}

function getPayloadBoolean(
  payload: Record<string, unknown>,
  key: string
): boolean | undefined {
  const value = payload[key];
  return typeof value === 'boolean' ? value : undefined;
}

function getPayloadStringArray(
  payload: Record<string, unknown>,
  key: string
): string[] {
  const value = payload[key];
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter((item): item is string => typeof item === 'string');
}

function getToolCallId(event: DashboardEchoEvent): string | undefined {
  return getPayloadString(getEventPayload(event), 'callId');
}

function hasSessionId(
  event: DashboardEchoEvent
): event is DashboardSessionEchoEvent {
  return event.sessionId !== null;
}

function createActivityMeta(
  event: DashboardEchoEvent,
  extra: readonly (string | undefined)[]
): string[] {
  return [formatDateTime(event.createdAt), ...extra].filter(
    (item): item is string => item !== undefined
  );
}

function compareEventsByCreatedAt(
  left: DashboardEchoEvent,
  right: DashboardEchoEvent
): number {
  return compareDateTimeString(left.createdAt, right.createdAt);
}

function compareDateTimeString(left: string, right: string): number {
  const leftTime = new Date(left).getTime();
  const rightTime = new Date(right).getTime();
  if (!Number.isNaN(leftTime) && !Number.isNaN(rightTime)) {
    return leftTime - rightTime;
  }

  return left.localeCompare(right);
}

function compactDetails(
  details: Record<string, unknown>
): Record<string, unknown> | null {
  const compacted = Object.fromEntries(
    Object.entries(details).filter(([, value]) => value !== undefined)
  );

  return Object.keys(compacted).length === 0 ? null : compacted;
}

function parseMaybeJson(value: unknown): unknown {
  if (typeof value !== 'string') {
    return value;
  }

  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function formatUnknownInline(value: unknown): string {
  if (typeof value === 'string') {
    return value;
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }

  try {
    return JSON.stringify(value);
  } catch {
    return 'unserializable error';
  }
}

function formatCodeLabel(value: string | undefined): string {
  return value === undefined ? '-' : value.replaceAll('_', ' ');
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat('ja-JP').format(value);
}

function formatCountLabel(
  value: number,
  singularLabel: string,
  pluralLabel: string
): string {
  return `${formatNumber(value)} ${value === 1 ? singularLabel : pluralLabel}`;
}

function formatDateTime(value: string | null): string {
  if (value === null) {
    return '-';
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat('ja-JP', {
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    month: '2-digit',
    second: '2-digit',
    timeZone: 'Asia/Tokyo',
    year: 'numeric',
  }).format(date);
}
