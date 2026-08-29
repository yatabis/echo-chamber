import type { MemoryType } from '../echo/types';
import type { ChatPort } from '../ports/chat';
import type { MemoryPort } from '../ports/memory';
import type { NotePort } from '../ports/note';
import type { NotificationPort } from '../ports/notification';
import type { WebPageReaderPort } from '../ports/web-page-reader';
import type { ZennPort } from '../ports/zenn';

/** Runtime memory toolが必要とする、現在感情を内部で関連付けるport。 */
export interface ToolMemoryPort {
  store(content: string, type: MemoryType): Promise<void>;
  search: MemoryPort['search'];
}

export interface ToolExecutionContext {
  chat: ChatPort;
  notifications: NotificationPort;
  memory: ToolMemoryPort;
  notes: NotePort;
  webPageReader: WebPageReaderPort;
  zenn: ZennPort;
}
