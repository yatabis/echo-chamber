import type { Note } from '@echo-chamber/core/echo/types';
import type { LoggerPort } from '@echo-chamber/core/ports/logger';

import {
  getMaxNoteSequence,
  getNoteStorageKey,
  isNoteRecord,
  NOTE_ITEM_PREFIX,
  sortByUpdatedAtDesc,
} from './note-storage';
import {
  validateContent,
  validateQuery,
  validateTitle,
} from './note-validation';

export const MAX_NOTE_COUNT = 200;

interface CreateNoteInput {
  title: string;
  content: string;
}

interface UpdateNoteInput {
  title?: string;
  content?: string;
}

/**
 * ノートシステム
 * DurableObjectStorage上でメモを管理する。
 */
export class NoteSystem {
  private readonly storage: DurableObjectStorage;
  private readonly logger: Pick<LoggerPort, 'info'>;

  /**
   * Durable Object storage 上で動く note runtime を構築する。
   *
   * @param options storage と logger
   */
  constructor(options: {
    storage: DurableObjectStorage;
    logger: Pick<LoggerPort, 'info'>;
  }) {
    this.storage = options.storage;
    this.logger = options.logger;
  }

  /**
   * すべてのノートを updatedAt 降順で返す。
   *
   * @returns 最新更新順に並んだノート一覧
   */
  async listNotes(): Promise<Note[]> {
    const notes = await this.readNotes();
    return sortByUpdatedAtDesc(notes);
  }

  /**
   * ID でノートを 1 件取得する。
   *
   * @param id note ID
   * @returns ノート。存在しなければ `null`
   */
  async getNote(id: string): Promise<Note | null> {
    const noteId = id.trim();
    if (noteId.length === 0) {
      throw new Error('Note ID is required');
    }

    const storedNote = await this.storage.get<Note>(getNoteStorageKey(noteId));
    if (!isNoteRecord(storedNote)) {
      return null;
    }

    return storedNote;
  }

  /**
   * title / content の部分一致でノートを検索する。
   *
   * @param query 検索クエリ
   * @returns 一致したノート一覧
   */
  async searchNotes(query: string): Promise<Note[]> {
    const normalizedQuery = validateQuery(query).toLowerCase();
    const notes = await this.listNotes();

    return notes.filter((note) => {
      return (
        note.title.toLowerCase().includes(normalizedQuery) ||
        note.content.toLowerCase().includes(normalizedQuery)
      );
    });
  }

  /**
   * 新しいノートを作成する。
   *
   * @param input title と content
   * @returns 作成されたノート
   */
  async createNote({ title, content }: CreateNoteInput): Promise<Note> {
    const normalizedTitle = validateTitle(title);
    const normalizedContent = validateContent(content);
    const notes = await this.readNotes();

    if (notes.length >= MAX_NOTE_COUNT) {
      throw new Error(`Note capacity reached (max ${MAX_NOTE_COUNT})`);
    }

    const nextId = `note-${getMaxNoteSequence(notes) + 1}`;
    const now = new Date().toISOString();
    const newNote: Note = {
      id: nextId,
      title: normalizedTitle,
      content: normalizedContent,
      createdAt: now,
      updatedAt: now,
    };

    await this.storage.put(getNoteStorageKey(newNote.id), newNote);
    await this.logger.info(`Note created: ${newNote.id}`);

    return newNote;
  }

  /**
   * 既存ノートを部分更新する。
   *
   * @param id note ID
   * @param patch title / content の更新内容
   * @returns 更新後ノート。存在しなければ `null`
   */
  async updateNote(id: string, patch: UpdateNoteInput): Promise<Note | null> {
    const noteId = id.trim();
    if (noteId.length === 0) {
      throw new Error('Note ID is required');
    }

    if (patch.title === undefined && patch.content === undefined) {
      throw new Error('Either title or content is required');
    }

    const storageKey = getNoteStorageKey(noteId);
    const storedNote = await this.storage.get<Note>(storageKey);
    if (!isNoteRecord(storedNote)) {
      return null;
    }

    const updatedNote: Note = {
      ...storedNote,
      title:
        patch.title !== undefined
          ? validateTitle(patch.title)
          : storedNote.title,
      content:
        patch.content !== undefined
          ? validateContent(patch.content)
          : storedNote.content,
      updatedAt: new Date().toISOString(),
    };

    await this.storage.put(storageKey, updatedNote);
    await this.logger.info(`Note updated: ${updatedNote.id}`);

    return updatedNote;
  }

  /**
   * ノートを削除する。
   *
   * @param id note ID
   * @returns 削除できた場合は `true`
   */
  async deleteNote(id: string): Promise<boolean> {
    const noteId = id.trim();
    if (noteId.length === 0) {
      throw new Error('Note ID is required');
    }

    const deleted = await this.storage.delete(getNoteStorageKey(noteId));
    if (deleted) {
      await this.logger.info(`Note deleted: ${noteId}`);
    }

    return deleted;
  }

  private async readNotes(): Promise<Note[]> {
    const entries = await this.storage.list({
      prefix: NOTE_ITEM_PREFIX,
    });
    return Array.from(entries.values()).filter(isNoteRecord);
  }
}
