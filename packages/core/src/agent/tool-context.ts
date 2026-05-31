import type { ChatPort } from '../ports/chat';
import type { MemoryPort } from '../ports/memory';
import type { NotePort } from '../ports/note';
import type { NotificationPort } from '../ports/notification';
import type { ZennPort } from '../ports/zenn';

export interface ToolExecutionContext {
  chat: ChatPort;
  notifications: NotificationPort;
  memory: Pick<MemoryPort, 'store' | 'search'>;
  notes: NotePort;
  zenn: ZennPort;
}
