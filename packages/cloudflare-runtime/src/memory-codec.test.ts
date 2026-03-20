import { describe, expect, it } from 'vitest';

import { bufferToNumberArray, float32ArrayToBuffer } from './memory-codec';

describe('memory-codec', () => {
  it('number配列をArrayBufferへ変換できる', () => {
    const buffer = float32ArrayToBuffer([0.1, 0.2, 0.3]);

    expect(buffer).toBeInstanceOf(ArrayBuffer);
    expect(buffer.byteLength).toBe(Float32Array.BYTES_PER_ELEMENT * 3);
  });

  it('ArrayBufferをnumber配列へ戻せる', () => {
    const values = bufferToNumberArray(float32ArrayToBuffer([1, 2, 3]));

    expect(values).toEqual([1, 2, 3]);
  });

  it('空配列のround tripも壊れない', () => {
    const values = bufferToNumberArray(float32ArrayToBuffer([]));

    expect(values).toEqual([]);
  });
});
