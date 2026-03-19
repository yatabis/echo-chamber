import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { LoggerPort } from '@echo-chamber/core/ports/logger';

import { MAX_NOTE_COUNT, NoteSystem } from './note-system';

type StorageMap = Map<string, unknown>;

interface MockStorage extends Partial<DurableObjectStorage> {
  _data: StorageMap;
  get: ReturnType<typeof vi.fn>;
  put: ReturnType<typeof vi.fn>;
  delete: ReturnType<typeof vi.fn>;
  list: ReturnType<typeof vi.fn>;
}

function createMockStorage(): MockStorage {
  const data: StorageMap = new Map<string, unknown>();
  return {
    _data: data,
    get: vi.fn(async (key: string) => Promise.resolve(data.get(key))),
    put: vi.fn(async (key: string, value: unknown) => {
      data.set(key, value);
      return Promise.resolve();
    }),
    delete: vi.fn(async (key: string) => {
      const deleted = data.delete(key);
      return Promise.resolve(deleted);
    }),
    list: vi.fn(
      async ({
        prefix,
      }: {
        prefix?: string;
      } = {}) => {
        const result = new Map<string, unknown>();
        for (const [key, value] of data.entries()) {
          if (prefix === undefined || key.startsWith(prefix)) {
            result.set(key, value);
          }
        }
        return Promise.resolve(result);
      }
    ),
  };
}

function createMockLogger(): Pick<LoggerPort, 'info'> {
  return {
    debug: vi.fn(async () => Promise.resolve()),
    info: vi.fn(async () => Promise.resolve()),
    warn: vi.fn(async () => Promise.resolve()),
    error: vi.fn(async () => Promise.resolve()),
    log: vi.fn(async () => Promise.resolve()),
  } as unknown as LoggerPort;
}

async function createManyNotes(
  noteSystem: NoteSystem,
  count: number,
  offset = 0
): Promise<void> {
  await Array.from({ length: count }).reduce(async (prev, _, i) => {
    await prev;
    await noteSystem.createNote({
      title: `Note ${i + offset}`,
      content: `Content ${i + offset}`,
    });
  }, Promise.resolve());
}

describe('NoteSystem', () => {
  let storage: MockStorage;
  let noteSystem: NoteSystem;

  beforeEach(() => {
    vi.resetAllMocks();
    storage = createMockStorage();
    noteSystem = new NoteSystem({
      storage: storage as DurableObjectStorage,
      logger: createMockLogger(),
    });
  });

  describe('createNote', () => {
    it('先頭のIDは note-1 になる', async () => {
      const note = await noteSystem.createNote({
        title: 'Meeting',
        content: 'Discuss timeline',
      });

      expect(note.id).toBe('note-1');
    });

    it('連続作成時にIDがインクリメントされる', async () => {
      await noteSystem.createNote({
        title: 'Meeting',
        content: 'Discuss timeline',
      });
      const second = await noteSystem.createNote({
        title: 'Roadmap',
        content: 'Plan next sprint',
      });

      expect(second.id).toBe('note-2');
    });

    it('作成したノートはlistNotesで取得できる', async () => {
      await noteSystem.createNote({
        title: 'Meeting',
        content: 'Discuss timeline',
      });

      const notes = await noteSystem.listNotes();
      expect(notes[0]?.title).toBe('Meeting');
    });
  });

  describe('listNotes', () => {
    it('updatedAt降順で先頭が最新ノートになる', async () => {
      vi.useFakeTimers();
      try {
        vi.setSystemTime(new Date('2026-02-17T00:00:00.000Z'));
        await noteSystem.createNote({
          title: 'Older',
          content: 'old content',
        });

        vi.setSystemTime(new Date('2026-02-17T00:01:00.000Z'));
        const newer = await noteSystem.createNote({
          title: 'Newer',
          content: 'new content',
        });

        const notes = await noteSystem.listNotes();
        expect(notes[0]?.id).toBe(newer.id);
      } finally {
        vi.useRealTimers();
      }
    });
  });

  describe('getNote', () => {
    it('id指定でノートを1件取得できる', async () => {
      const created = await noteSystem.createNote({
        title: 'Meeting',
        content: 'Discuss timeline',
      });

      const found = await noteSystem.getNote(created.id);
      expect(found).toEqual(created);
    });

    it('存在しないIDはnullを返す', async () => {
      const found = await noteSystem.getNote('missing-note');
      expect(found).toBeNull();
    });

    it('空白IDはバリデーションエラー', async () => {
      await expect(noteSystem.getNote('   ')).rejects.toThrowError(
        'Note ID is required'
      );
    });
  });

  describe('searchNotes', () => {
    it('titleの部分一致で1件ヒットする', async () => {
      await noteSystem.createNote({
        title: 'Project Alpha',
        content: 'Implement login feature',
      });
      await noteSystem.createNote({
        title: 'Daily Log',
        content: 'Reviewed API schema',
      });

      const matches = await noteSystem.searchNotes('alpha');
      expect(matches).toHaveLength(1);
    });

    it('contentの部分一致で対象ノートを返す', async () => {
      await noteSystem.createNote({
        title: 'Project Alpha',
        content: 'Implement login feature',
      });
      const target = await noteSystem.createNote({
        title: 'Daily Log',
        content: 'Reviewed API schema',
      });

      const matches = await noteSystem.searchNotes('api');
      expect(matches[0]?.id).toBe(target.id);
    });
  });

  describe('updateNote', () => {
    it('更新結果として新しいtitleを返す', async () => {
      const note = await noteSystem.createNote({
        title: 'Draft',
        content: 'Initial content',
      });

      const updated = await noteSystem.updateNote(note.id, {
        title: 'Final',
      });

      expect(updated?.title).toBe('Final');
    });

    it('update後のtitleはlistNotesにも反映される', async () => {
      const note = await noteSystem.createNote({
        title: 'Draft',
        content: 'Initial content',
      });
      await noteSystem.updateNote(note.id, { title: 'Final' });

      const notes = await noteSystem.listNotes();
      expect(notes.find((item) => item.id === note.id)?.title).toBe('Final');
    });

    it('update後のcontentはlistNotesにも反映される', async () => {
      const note = await noteSystem.createNote({
        title: 'Draft',
        content: 'Initial content',
      });
      await noteSystem.updateNote(note.id, { content: 'Updated content' });

      const notes = await noteSystem.listNotes();
      expect(notes.find((item) => item.id === note.id)?.content).toBe(
        'Updated content'
      );
    });

    it('存在しないノート更新はnullを返す', async () => {
      const result = await noteSystem.updateNote('missing-note', {
        title: 'updated',
      });

      expect(result).toBeNull();
    });
  });

  describe('deleteNote', () => {
    it('既存ノートの削除はtrueを返す', async () => {
      const note = await noteSystem.createNote({
        title: 'Temp',
        content: 'To be deleted',
      });

      const deleted = await noteSystem.deleteNote(note.id);
      expect(deleted).toBe(true);
    });

    it('削除したノートはlistNotesに残らない', async () => {
      const note = await noteSystem.createNote({
        title: 'Temp',
        content: 'To be deleted',
      });
      await noteSystem.deleteNote(note.id);

      const notes = await noteSystem.listNotes();
      expect(notes.find((item) => item.id === note.id)).toBeUndefined();
    });

    it('存在しないノート削除はfalseを返す', async () => {
      const deleted = await noteSystem.deleteNote('missing-note');
      expect(deleted).toBe(false);
    });
  });

  describe('capacity', () => {
    it('200件作成後の件数は200件になる', async () => {
      await createManyNotes(noteSystem, MAX_NOTE_COUNT);

      const notes = await noteSystem.listNotes();
      expect(notes).toHaveLength(MAX_NOTE_COUNT);
    });

    it('201件目作成時はエラーになる', async () => {
      await createManyNotes(noteSystem, MAX_NOTE_COUNT);

      await expect(
        noteSystem.createNote({
          title: 'Overflow note',
          content: 'This should fail',
        })
      ).rejects.toThrowError('Note capacity reached (max 200)');
    });

    it('201件目失敗後も件数は200件のまま', async () => {
      await createManyNotes(noteSystem, MAX_NOTE_COUNT);
      const overflow = noteSystem.createNote({
        title: 'Overflow note',
        content: 'This should fail',
      });
      await expect(overflow).rejects.toThrowError(
        'Note capacity reached (max 200)'
      );

      const notes = await noteSystem.listNotes();
      expect(notes).toHaveLength(MAX_NOTE_COUNT);
    });
  });

  describe('storage', () => {
    it('note:item の保存値が壊れている場合は無視する', async () => {
      storage._data.set('note:item:bad-1', 'invalid-value');
      storage._data.set('note:item:bad-2', { id: 'bad-2' });

      const notes = await noteSystem.listNotes();
      expect(notes).toEqual([]);
    });

    it('prefix外のキーは一覧対象に含めない', async () => {
      storage._data.set('random:key', {
        id: 'note-999',
        title: 'should not be listed',
      });

      const notes = await noteSystem.listNotes();
      expect(notes).toEqual([]);
    });
  });
});
