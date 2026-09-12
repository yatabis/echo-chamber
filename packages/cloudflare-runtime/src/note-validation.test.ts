import { describe, expect, it } from 'vitest';

import {
  MAX_NOTE_CONTENT_LENGTH,
  MAX_NOTE_QUERY_LENGTH,
  MAX_NOTE_TITLE_LENGTH,
} from '@echo-chamber/core/echo/note-constraints';

import {
  validateContent,
  validateQuery,
  validateTitle,
} from './note-validation';

describe('note-validation', () => {
  it('validateTitleはtrimした値を返す', () => {
    expect(validateTitle('  Meeting  ')).toBe('Meeting');
  });

  it('validateTitleは空文字を拒否する', () => {
    expect(() => validateTitle('   ')).toThrow('Title is required');
  });

  it('validateTitleは長すぎる値を拒否する', () => {
    expect(() => validateTitle('a'.repeat(MAX_NOTE_TITLE_LENGTH + 1))).toThrow(
      `Title must be at most ${MAX_NOTE_TITLE_LENGTH} characters`
    );
  });

  it('validateContentはtrimした値を返す', () => {
    expect(validateContent('  Body  ')).toBe('Body');
  });

  it('validateContentは空文字を拒否する', () => {
    expect(() => validateContent('   ')).toThrow('Content is required');
  });

  it('validateContentは長すぎる値を拒否する', () => {
    expect(() =>
      validateContent('a'.repeat(MAX_NOTE_CONTENT_LENGTH + 1))
    ).toThrow(`Content must be at most ${MAX_NOTE_CONTENT_LENGTH} characters`);
  });

  it('validateQueryはtrimした値を返す', () => {
    expect(validateQuery('  search term  ')).toBe('search term');
  });

  it('validateQueryは空文字を拒否する', () => {
    expect(() => validateQuery('   ')).toThrow('Query is required');
  });

  it('validateQueryは長すぎる値を拒否する', () => {
    expect(() => validateQuery('a'.repeat(MAX_NOTE_QUERY_LENGTH + 1))).toThrow(
      `Query must be at most ${MAX_NOTE_QUERY_LENGTH} characters`
    );
  });
});
