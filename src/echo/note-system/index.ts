import type { Logger } from '../../utils/logger';
import type { Note } from '../types';

const NOTE_ITEM_PREFIX = 'note:item:';
const NOTE_ID_PATTERN = /^note-(\d+)$/;

export const MAX_NOTE_COUNT = 200;
export const MAX_NOTE_TITLE_LENGTH = 120;
export const MAX_NOTE_CONTENT_LENGTH = 2000;
export const MAX_NOTE_QUERY_LENGTH = 200;

interface CreateNoteInput {
  title: string;
  content: string;
}

interface UpdateNoteInput {
  title?: string;
  content?: string;
}

function validateTitle(title: string): string {
  const normalizedTitle = title.trim();
  if (normalizedTitle.length === 0) {
    throw new Error('Title is required');
  }
  if (normalizedTitle.length > MAX_NOTE_TITLE_LENGTH) {
    throw new Error(
      `Title must be at most ${MAX_NOTE_TITLE_LENGTH} characters`
    );
  }
  return normalizedTitle;
}

function validateContent(content: string): string {
  const normalizedContent = content.trim();
  if (normalizedContent.length === 0) {
    throw new Error('Content is required');
  }
  if (normalizedContent.length > MAX_NOTE_CONTENT_LENGTH) {
    throw new Error(
      `Content must be at most ${MAX_NOTE_CONTENT_LENGTH} characters`
    );
  }
  return normalizedContent;
}

function validateQuery(query: string): string {
  const normalizedQuery = query.trim();
  if (normalizedQuery.length === 0) {
    throw new Error('Query is required');
  }
  if (normalizedQuery.length > MAX_NOTE_QUERY_LENGTH) {
    throw new Error(
      `Query must be at most ${MAX_NOTE_QUERY_LENGTH} characters`
    );
  }
  return normalizedQuery;
}

function sortByUpdatedAtDesc(notes: Note[]): Note[] {
  return [...notes].sort(
    (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
  );
}

function parseNoteSequence(id: string): number | null {
  const match = NOTE_ID_PATTERN.exec(id);
  if (!match) {
    return null;
  }
  const [, numericPart] = match;
  if (numericPart === undefined) {
    return null;
  }
  const parsed = Number.parseInt(numericPart, 10);
  if (Number.isNaN(parsed)) {
    return null;
  }
  return parsed;
}

function getMaxNoteSequence(notes: Note[]): number {
  return notes.reduce((max, note) => {
    const parsed = parseNoteSequence(note.id);
    if (parsed === null) {
      return max;
    }
    return Math.max(max, parsed);
  }, 0);
}

function getNoteStorageKey(id: string): string {
  return `${NOTE_ITEM_PREFIX}${id}`;
}

function isNoteRecord(value: unknown): value is Note {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const note = value as Record<string, unknown>;
  return (
    typeof note.id === 'string' &&
    typeof note.title === 'string' &&
    typeof note.content === 'string' &&
    typeof note.createdAt === 'string' &&
    typeof note.updatedAt === 'string'
  );
}

/**
 * ノートシステム
 * DurableObjectStorage上でメモを管理する。
 */
export class NoteSystem {
  private readonly storage: DurableObjectStorage;
  private readonly logger: Logger;

  constructor(options: { storage: DurableObjectStorage; logger: Logger }) {
    this.storage = options.storage;
    this.logger = options.logger;
  }

  async listNotes(): Promise<Note[]> {
    const notes = await this.readNotes();
    return sortByUpdatedAtDesc(notes);
  }

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
