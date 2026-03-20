/**
 * number 配列を SQLite BLOB 保存用の ArrayBuffer に変換する。
 *
 * @param values embedding ベクトル
 * @returns Float32Array ベースの ArrayBuffer
 */
export function float32ArrayToBuffer(values: number[]): ArrayBuffer {
  return new Float32Array(values).buffer;
}

/**
 * SQLite BLOB から復元した ArrayBuffer を number 配列へ戻す。
 *
 * @param buffer SQLite から読み出した embedding buffer
 * @returns number 配列化された embedding ベクトル
 */
export function bufferToNumberArray(buffer: ArrayBuffer): number[] {
  return Array.from(new Float32Array(buffer));
}
