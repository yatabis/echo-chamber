import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { Note } from '@echo-chamber/core/echo/types';

import {
  createNoteFunction,
  deleteNoteFunction,
  getNoteFunction,
  listNotesFunction,
  searchNotesFunction,
  updateNoteFunction,
} from './note';

import type { ToolContext } from './index';

const createNoteMock =
  vi.fn<(input: { title: string; content: string }) => Promise<Note>>();
const listNotesMock = vi.fn<() => Promise<Note[]>>();
const getNoteMock = vi.fn<(id: string) => Promise<Note | null>>();
const searchNotesMock = vi.fn<(query: string) => Promise<Note[]>>();
const updateNoteMock =
  vi.fn<
    (
      id: string,
      patch: { title?: string; content?: string }
    ) => Promise<Note | null>
  >();
const deleteNoteMock = vi.fn<(id: string) => Promise<boolean>>();

const mockToolContext: ToolContext = {
  chat: {
    readMessages: vi.fn().mockResolvedValue([]),
    sendMessage: vi.fn().mockResolvedValue(undefined),
    addReaction: vi.fn().mockResolvedValue(undefined),
  },
  notifications: {
    getNotificationSummary: vi.fn().mockResolvedValue({
      unreadCount: 0,
      latestMessagePreview: null,
    }),
  },
  memory: {
    store: vi.fn().mockResolvedValue(undefined),
    search: vi.fn().mockResolvedValue([]),
  },
  notes: {
    create: createNoteMock,
    list: listNotesMock,
    get: getNoteMock,
    search: searchNotesMock,
    update: updateNoteMock,
    delete: deleteNoteMock,
  },
  logger: {
    log: vi.fn().mockResolvedValue(undefined),
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  } as unknown as ToolContext['logger'],
};

beforeEach(() => {
  vi.resetAllMocks();
  listNotesMock.mockResolvedValue([]);
  getNoteMock.mockResolvedValue(null);
  searchNotesMock.mockResolvedValue([]);
  updateNoteMock.mockResolvedValue(null);
  deleteNoteMock.mockResolvedValue(false);
});

const createMockNote = (overrides?: Partial<Note>): Note => {
  return {
    id: 'note-1',
    title: 'Test Note',
    content: 'Test Content',
    createdAt: '2026-02-17T00:00:00.000Z',
    updatedAt: '2026-02-17T00:00:00.000Z',
    ...overrides,
  };
};

describe('Note Functions', () => {
  describe('createNoteFunction', () => {
    it('name', () => {
      expect(createNoteFunction.name).toBe('create_note');
    });

    it('handler creates note', async () => {
      const note = createMockNote();
      createNoteMock.mockResolvedValue(note);

      const result = await createNoteFunction.handler(
        {
          title: 'Meeting Notes',
          content: 'Discuss roadmap',
        },
        mockToolContext
      );

      expect(createNoteMock).toHaveBeenCalledWith({
        title: 'Meeting Notes',
        content: 'Discuss roadmap',
      });
      expect(result).toEqual({
        success: true,
        note,
      });
    });

    it('capacity reached errorをそのまま返す', async () => {
      createNoteMock.mockRejectedValue(
        new Error('Note capacity reached (max 200)')
      );

      const result = await createNoteFunction.handler(
        {
          title: 'New Note',
          content: 'Body',
        },
        mockToolContext
      );

      expect(result).toEqual({
        success: false,
        error: 'Note capacity reached (max 200)',
      });
    });
  });

  describe('listNotesFunction', () => {
    it('name', () => {
      expect(listNotesFunction.name).toBe('list_notes');
    });

    it('handler lists notes', async () => {
      const notes = [createMockNote(), createMockNote({ id: 'note-2' })];
      listNotesMock.mockResolvedValue(notes);

      const result = await listNotesFunction.handler({}, mockToolContext);

      expect(listNotesMock).toHaveBeenCalled();
      const noteSummaries = notes.map((note) => ({
        id: note.id,
        title: note.title,
        createdAt: note.createdAt,
        updatedAt: note.updatedAt,
      }));
      expect(result).toEqual({
        success: true,
        notes: noteSummaries,
      });
      if (result.success) {
        const summary = (result.notes as Record<string, unknown>[])[0];
        expect(summary).not.toHaveProperty('content');
      }
    });
  });

  describe('getNoteFunction', () => {
    it('name', () => {
      expect(getNoteFunction.name).toBe('get_note');
    });

    it('handler gets note', async () => {
      const note = createMockNote({ id: 'note-3' });
      getNoteMock.mockResolvedValue(note);

      const result = await getNoteFunction.handler(
        { id: 'note-3' },
        mockToolContext
      );

      expect(getNoteMock).toHaveBeenCalledWith('note-3');
      expect(result).toEqual({
        success: true,
        note,
      });
    });

    it('存在しないノートはnot found', async () => {
      getNoteMock.mockResolvedValue(null);

      const result = await getNoteFunction.handler(
        { id: 'missing-note' },
        mockToolContext
      );

      expect(result).toEqual({
        success: false,
        error: 'Note not found',
      });
    });
  });

  describe('searchNotesFunction', () => {
    it('name', () => {
      expect(searchNotesFunction.name).toBe('search_notes');
    });

    it('handler searches notes', async () => {
      const notes = [createMockNote({ title: 'Roadmap' })];
      searchNotesMock.mockResolvedValue(notes);

      const result = await searchNotesFunction.handler(
        { query: 'road' },
        mockToolContext
      );

      expect(searchNotesMock).toHaveBeenCalledWith('road');
      expect(result).toEqual({
        success: true,
        notes,
      });
    });
  });

  describe('updateNoteFunction', () => {
    it('name', () => {
      expect(updateNoteFunction.name).toBe('update_note');
    });

    it('handler updates note', async () => {
      const updated = createMockNote({
        title: 'Updated',
        updatedAt: '2026-02-17T00:10:00.000Z',
      });
      updateNoteMock.mockResolvedValue(updated);

      const result = await updateNoteFunction.handler(
        { id: 'note-1', title: 'Updated' },
        mockToolContext
      );

      expect(updateNoteMock).toHaveBeenCalledWith('note-1', {
        title: 'Updated',
        content: undefined,
      });
      expect(result).toEqual({
        success: true,
        note: updated,
      });
    });

    it('title/content未指定時はバリデーションエラー', async () => {
      const result = await updateNoteFunction.handler(
        { id: 'note-1', title: undefined, content: undefined },
        mockToolContext
      );

      expect(result).toEqual({
        success: false,
        error: 'Either title or content is required',
      });
    });

    it('更新対象がない場合はnot found', async () => {
      updateNoteMock.mockResolvedValue(null);

      const result = await updateNoteFunction.handler(
        { id: 'missing-note', content: 'Body' },
        mockToolContext
      );

      expect(result).toEqual({
        success: false,
        error: 'Note not found',
      });
    });
  });

  describe('deleteNoteFunction', () => {
    it('name', () => {
      expect(deleteNoteFunction.name).toBe('delete_note');
    });

    it('handler deletes note', async () => {
      deleteNoteMock.mockResolvedValue(true);

      const result = await deleteNoteFunction.handler(
        { id: 'note-1' },
        mockToolContext
      );

      expect(deleteNoteMock).toHaveBeenCalledWith('note-1');
      expect(result).toEqual({
        success: true,
      });
    });

    it('削除対象がない場合はnot found', async () => {
      deleteNoteMock.mockResolvedValue(false);

      const result = await deleteNoteFunction.handler(
        { id: 'missing-note' },
        mockToolContext
      );

      expect(result).toEqual({
        success: false,
        error: 'Note not found',
      });
    });
  });
});
