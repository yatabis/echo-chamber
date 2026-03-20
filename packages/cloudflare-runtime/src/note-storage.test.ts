import { describe, expect, it } from 'vitest';

import type { Note } from '@echo-chamber/core/echo/types';

import {
  getMaxNoteSequence,
  getNoteStorageKey,
  isNoteRecord,
  parseNoteSequence,
  sortByUpdatedAtDesc,
} from './note-storage';

function createNote(id: string, updatedAt: string): Note {
  return {
    id,
    title: `Title ${id}`,
    content: `Content ${id}`,
    createdAt: '2026-01-01T00:00:00.000Z',
    updatedAt,
  };
}

describe('note-storage', () => {
  it('sortByUpdatedAtDescはupdatedAtの降順で並べる', () => {
    const sorted = sortByUpdatedAtDesc([
      createNote('note-1', '2026-01-01T00:00:00.000Z'),
      createNote('note-2', '2026-01-01T00:02:00.000Z'),
      createNote('note-3', '2026-01-01T00:01:00.000Z'),
    ]);

    expect(sorted.map((note) => note.id)).toEqual([
      'note-2',
      'note-3',
      'note-1',
    ]);
  });

  it('parseNoteSequenceは正しいIDから連番を取り出す', () => {
    expect(parseNoteSequence('note-42')).toBe(42);
  });

  it('parseNoteSequenceは不正なIDをnullにする', () => {
    expect(parseNoteSequence('memo-42')).toBeNull();
    expect(parseNoteSequence('note-abc')).toBeNull();
  });

  it('getMaxNoteSequenceは不正なIDを無視して最大値を返す', () => {
    expect(
      getMaxNoteSequence([
        { id: 'note-2' },
        { id: 'invalid' },
        { id: 'note-10' },
      ])
    ).toBe(10);
  });

  it('getNoteStorageKeyはprefix付きのkeyを返す', () => {
    expect(getNoteStorageKey('note-3')).toBe('note:item:note-3');
  });

  it('isNoteRecordは有効なNote形状をtrueにする', () => {
    expect(
      isNoteRecord({
        id: 'note-1',
        title: 'Title',
        content: 'Body',
        createdAt: '2026-01-01T00:00:00.000Z',
        updatedAt: '2026-01-01T00:00:00.000Z',
      })
    ).toBe(true);
  });

  it('isNoteRecordは不正な形状をfalseにする', () => {
    expect(isNoteRecord(null)).toBe(false);
    expect(
      isNoteRecord({
        id: 'note-1',
        title: 'Title',
        content: 'Body',
        createdAt: '2026-01-01T00:00:00.000Z',
      })
    ).toBe(false);
    expect(
      isNoteRecord({
        id: 'note-1',
        title: 'Title',
        content: 123,
        createdAt: '2026-01-01T00:00:00.000Z',
        updatedAt: '2026-01-01T00:00:00.000Z',
      })
    ).toBe(false);
  });
});
