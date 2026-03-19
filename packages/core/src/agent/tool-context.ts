import type { ChatPort } from '../ports/chat';
import type { LoggerPort } from '../ports/logger';
import type { MemoryPort } from '../ports/memory';
import type { NotePort } from '../ports/note';
import type { NotificationPort } from '../ports/notification';

export interface ToolExecutionContext {
  chat: ChatPort;
  notifications: NotificationPort;
  memory: Pick<MemoryPort, 'store' | 'search'>;
  notes: NotePort;
  logger: LoggerPort;
}
