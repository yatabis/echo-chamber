/**
 * 日時を指定されたタイムゾーンで 'YYYY/MM/DD HH:mm:ss' 形式にフォーマットする
 *
 * @param date - フォーマット対象の日時
 * @param timezone - 使用するタイムゾーン（デフォルト: 'Asia/Tokyo'）
 */
export function formatDatetime(date: Date, timezone = 'Asia/Tokyo'): string {
  const formatter = new Intl.DateTimeFormat('ja-JP', {
    timeZone: timezone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
  return formatter.format(date);
}

/**
 * 日付を指定されたタイムゾーンで 'YYYY-MM-DD' 形式にフォーマットする
 *
 * @param date - フォーマット対象の日時
 * @param timezone - 使用するタイムゾーン（デフォルト: 'Asia/Tokyo'）
 */
export function formatDate(date: Date, timezone = 'Asia/Tokyo'): string {
  const formatter = new Intl.DateTimeFormat('ja-JP', {
    timeZone: timezone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  });
  return formatter.format(date).replaceAll('/', '-');
}

/**
 * 経過時間を最も大きい単位に丸めて「〜前」形式でフォーマット
 * 例: 1時間30分 → "1時間前"、2日3時間 → "2日前"
 *
 * @param since - 開始日時（文字列またはDate）
 * @param until - 終了日時（デフォルト: 現在時刻）
 */
export function formatElapsedTime(
  since: string | Date,
  until: Date = new Date()
): string {
  const startTime = typeof since === 'string' ? new Date(since) : since;
  const diffMs = until.getTime() - startTime.getTime();

  if (diffMs < 0) return '0分前';

  const minutes = Math.floor(diffMs / (1000 * 60));
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days}日前`;
  if (hours > 0) return `${hours}時間前`;
  return `${minutes}分前`;
}
