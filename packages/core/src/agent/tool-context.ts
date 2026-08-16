import type { ChatPort } from '../ports/chat';
import type { MemoryPort } from '../ports/memory';
import type { NotePort } from '../ports/note';
import type { NotificationPort } from '../ports/notification';
import type { WebPageReaderPort } from '../ports/web-page-reader';
import type { ZennPort } from '../ports/zenn';

export interface ToolExecutionContext {
  chat: ChatPort;
  notifications: NotificationPort;
  memory: Pick<MemoryPort, 'store' | 'search'>;
  notes: NotePort;
  webPageReader: WebPageReaderPort;
  zenn: ZennPort;
}
