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
    expect(() => validateTitle('   ')).toThrowError('Title is required');
  });

  it('validateTitleは長すぎる値を拒否する', () => {
    expect(() =>
      validateTitle('a'.repeat(MAX_NOTE_TITLE_LENGTH + 1))
    ).toThrowError(`Title must be at most ${MAX_NOTE_TITLE_LENGTH} characters`);
  });

  it('validateContentはtrimした値を返す', () => {
    expect(validateContent('  Body  ')).toBe('Body');
  });

  it('validateContentは空文字を拒否する', () => {
    expect(() => validateContent('   ')).toThrowError('Content is required');
  });

  it('validateContentは長すぎる値を拒否する', () => {
    expect(() =>
      validateContent('a'.repeat(MAX_NOTE_CONTENT_LENGTH + 1))
    ).toThrowError(
      `Content must be at most ${MAX_NOTE_CONTENT_LENGTH} characters`
    );
  });

  it('validateQueryはtrimした値を返す', () => {
    expect(validateQuery('  search term  ')).toBe('search term');
  });

  it('validateQueryは空文字を拒否する', () => {
    expect(() => validateQuery('   ')).toThrowError('Query is required');
  });

  it('validateQueryは長すぎる値を拒否する', () => {
    expect(() =>
      validateQuery('a'.repeat(MAX_NOTE_QUERY_LENGTH + 1))
    ).toThrowError(`Query must be at most ${MAX_NOTE_QUERY_LENGTH} characters`);
  });
});
