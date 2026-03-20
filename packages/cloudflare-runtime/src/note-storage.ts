import type { Note } from '@echo-chamber/core/echo/types';

export const NOTE_ITEM_PREFIX = 'note:item:';
const NOTE_ID_PATTERN = /^note-(\d+)$/;

/**
 * ノートを updatedAt の降順で並べ替える。
 *
 * @param notes 対象ノート一覧
 * @returns 最新更新順のノート一覧
 */
export function sortByUpdatedAtDesc(notes: Note[]): Note[] {
  return [...notes].sort(
    (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
  );
}

/**
 * `note-<number>` 形式の ID から連番部分を取り出す。
 *
 * @param id note ID
 * @returns 連番。形式不正なら `null`
 */
export function parseNoteSequence(id: string): number | null {
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

/**
 * ノート一覧から最大の連番を求める。
 *
 * 不正な ID は無視する。
 *
 * @param notes 対象ノート一覧
 * @returns 最大連番。該当がなければ `0`
 */
export function getMaxNoteSequence(notes: Pick<Note, 'id'>[]): number {
  return notes.reduce((max, note) => {
    const parsed = parseNoteSequence(note.id);
    if (parsed === null) {
      return max;
    }
    return Math.max(max, parsed);
  }, 0);
}

/**
 * note ID から Durable Object storage key を組み立てる。
 *
 * @param id note ID
 * @returns storage key
 */
export function getNoteStorageKey(id: string): string {
  return `${NOTE_ITEM_PREFIX}${id}`;
}

/**
 * storage から取得した unknown 値が Note 形状かを判定する。
 *
 * @param value 判定対象
 * @returns Note 形状なら `true`
 */
export function isNoteRecord(value: unknown): value is Note {
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
