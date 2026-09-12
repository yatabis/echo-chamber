/**
 * 公開WebページReaderが返す安定error code。
 */
export const WEB_PAGE_READER_ERROR_CODES = [
  'invalid_url',
  'sensitive_url',
  'destination_not_public',
  'redirect_not_allowed',
  'redirect_limit',
  'timeout',
  'http_status',
  'unsupported_content_type',
  'unsupported_charset',
  'response_too_large',
  'invalid_text_encoding',
  'content_extraction_failed',
  'empty_content',
  'budget_exceeded',
  'internal_error',
] as const;

/**
 * 公開WebページReaderが返す安定error code。
 */
export type WebPageReaderErrorCode =
  (typeof WEB_PAGE_READER_ERROR_CODES)[number];

/**
 * 抽出済み本文を切り詰めた理由。
 */
export type WebPageTruncationReason = 'document_character_limit';

/**
 * 抽出済み本文に含める公開リンク。
 */
export interface WebPageLink {
  text: string;
  url: string;
}

/**
 * provider/runtimeに依存しない公開Webページの抽出結果。
 */
export interface WebPageDocument {
  requestedUrl: string;
  finalUrl: string;
  retrievedAt: string;
  httpStatus: number;
  redirectCount: number;
  contentType: 'text/html' | 'application/xhtml+xml' | 'text/plain';
  title: string | null;
  contentFormat: 'markdown';
  rendering: 'static';
  content: string;
  extractedCharacters: number;
  documentTruncated: boolean;
  truncationReasons: readonly WebPageTruncationReason[];
  links: readonly WebPageLink[];
}

/**
 * 公開Webページを正常に抽出した結果。
 */
export interface WebPageReadSuccess {
  success: true;
  document: WebPageDocument;
}

/**
 * 公開Webページを安全に取得できなかった結果。
 */
export interface WebPageReadFailure {
  success: false;
  code: WebPageReaderErrorCode;
  error: string;
  retryable: boolean;
}

/**
 * 公開WebページReaderのprovider-neutralな結果。
 */
export type WebPageReadResult = WebPageReadSuccess | WebPageReadFailure;

/**
 * 許可された公開HTTP(S)ページを静的に読み取るport。
 */
export interface WebPageReaderPort {
  /**
   * URLを取得し、制御Markdownへ正規化する。
   *
   * @param url モデルが指定した公開ページURL
   * @returns 抽出済みdocumentまたは安定error
   */
  readPage(url: string): Promise<WebPageReadResult>;
}
