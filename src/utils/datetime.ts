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

/**
 * 経過時間を「〜前」形式でフォーマット（エージェント向け）
 * - 0分以下: `たった今`
 * - 1-59分: `n分前`
 * - 1-23時間: `n時間前`
 * - 1日以上: `n日前`
 */
function formatElapsedForAgent(diffMs: number): string {
  if (diffMs <= 0) return 'たった今';

  const minutes = Math.floor(diffMs / (1000 * 60));
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days}日前`;
  if (hours > 0) return `${hours}時間前`;
  if (minutes > 0) return `${minutes}分前`;
  return 'たった今';
}

/**
 * 日時を 'YYYY年MM月DD日 HH:mm:ss' 形式でフォーマット
 */
export function formatJapaneseDatetime(
  date: Date,
  timezone = 'Asia/Tokyo'
): string {
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

  const parts = formatter.formatToParts(date);
  const year = parts.find((p) => p.type === 'year')?.value;
  const month = parts.find((p) => p.type === 'month')?.value;
  const day = parts.find((p) => p.type === 'day')?.value;
  const hour = parts.find((p) => p.type === 'hour')?.value;
  const minute = parts.find((p) => p.type === 'minute')?.value;
  const second = parts.find((p) => p.type === 'second')?.value;

  return `${year}年${month}月${day}日 ${hour}:${minute}:${second}`;
}

/**
 * エージェント（OpenAI API）に渡す日時フォーマット
 * 出力: `n日前 (YYYY年MM月DD日 HH:mm:ss)`
 *
 * @param date - フォーマット対象の日時
 * @param referenceDate - 基準日時（デフォルト: 現在時刻）
 * @param timezone - 使用するタイムゾーン（デフォルト: 'Asia/Tokyo'）
 */
export function formatDatetimeForAgent(
  date: Date,
  referenceDate: Date = new Date(),
  timezone = 'Asia/Tokyo'
): string {
  const diffMs = referenceDate.getTime() - date.getTime();
  const elapsedStr = formatElapsedForAgent(diffMs);
  const datetimeStr = formatJapaneseDatetime(date, timezone);

  return `${elapsedStr} (${datetimeStr})`;
}
