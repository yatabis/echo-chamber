import type { Note } from '@echo-chamber/core/echo/types';

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

interface ListNotesInput {
  limit?: number;
}

export interface NoteDashboardSummary {
  count: number;
  latestUpdatedAt: string | null;
}

/**
 * ノートシステム
 * DurableObjectStorage上でメモを管理する。
 */
export class NoteSystem {
  private readonly storage: DurableObjectStorage;

  /**
   * Durable Object storage 上で動く note runtime を構築する。
   *
   * @param options storage
   */
  constructor(options: { storage: DurableObjectStorage }) {
    this.storage = options.storage;
  }

  /**
   * すべてのノートを updatedAt 降順で返す。
   *
   * @returns 最新更新順に並んだノート一覧
   */
  async listNotes(input: ListNotesInput = {}): Promise<Note[]> {
    const notes = await this.readNotes({
      limit: input.limit ?? MAX_NOTE_COUNT,
    });
    return sortByUpdatedAtDesc(notes);
  }

  /**
   * Dashboard summary 用に note 件数と最新更新時刻を返す。
   *
   * note は上限が 200 件なので、正確性を保つため bounded list で集計する。
   *
   * @returns note summary
   */
  async getDashboardNoteSummary(): Promise<NoteDashboardSummary> {
    const notes = await this.readNotes({
      limit: MAX_NOTE_COUNT,
    });
    const [latestNote] = sortByUpdatedAtDesc(notes);

    return {
      count: notes.length,
      latestUpdatedAt: latestNote?.updatedAt ?? null,
    };
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
    const notes = await this.listNotes({
      limit: MAX_NOTE_COUNT,
    });

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
    const notes = await this.readNotes({
      limit: MAX_NOTE_COUNT,
    });

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

    return await this.storage.delete(getNoteStorageKey(noteId));
  }

  private async readNotes(input: ListNotesInput = {}): Promise<Note[]> {
    const entries = await this.storage.list({
      prefix: NOTE_ITEM_PREFIX,
      reverse: true,
      limit: input.limit ?? MAX_NOTE_COUNT,
    });
    return Array.from(entries.values()).filter(isNoteRecord);
  }
}
