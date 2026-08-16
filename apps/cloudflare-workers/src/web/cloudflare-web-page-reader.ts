import type {
  WebPageDocument,
  WebPageReadFailure,
  WebPageReaderErrorCode,
  WebPageReaderPort,
  WebPageReadResult,
} from '@echo-chamber/core/ports/web-page-reader';

import { extractHtmlToMarkdown } from './html-to-markdown';
import { admitPublicWebUrl, admitWebRedirect } from './url-policy';

import type { ExtractedWebPage } from './html-to-markdown';

export const MAX_WEB_PAGE_BODY_BYTES = 1_048_576;

const MAX_REDIRECTS = 3;
const REQUEST_TIMEOUT_MILLISECONDS = 10_000;
const DOCUMENT_CHARACTER_LIMIT = 64_000;
const REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308]);
const SUPPORTED_CONTENT_TYPES = new Set([
  'text/html',
  'application/xhtml+xml',
  'text/plain',
]);
const SUPPORTED_CHARSETS = new Set(['utf-8', 'utf8', 'us-ascii', 'ascii']);
const REQUEST_HEADERS = {
  Accept: 'text/html, application/xhtml+xml, text/plain;q=0.8',
  'User-Agent': 'ECHO-Chamber-Public-Web-Reader/1.0',
} as const;
const DISALLOWED_CONTROL_CHARACTERS =
  // eslint-disable-next-line no-control-regex -- These exact C0 controls are removed from untrusted plain text.
  /[\u0001-\u0008\u000b\u000c\u000e-\u001f\u007f]/g;

type Fetcher = (input: string, init: RequestInit) => Promise<Response>;
type SupportedContentType =
  | 'text/html'
  | 'application/xhtml+xml'
  | 'text/plain';

interface CloudflareWebPageReaderOptions {
  fetcher?: Fetcher;
  now?(): Date;
}

interface ParsedContentType {
  mimeType: SupportedContentType;
  charset: 'utf-8' | 'us-ascii';
}

interface FetchDependencies {
  fetcher: Fetcher;
  signal: AbortSignal;
}

interface FinalResponse {
  response: Response;
  finalUrl: URL;
  redirectCount: number;
}

interface DocumentMetadata extends FinalResponse {
  requestedUrl: string;
  retrievedAt: string;
  contentType: SupportedContentType;
}

function failure(
  code: WebPageReaderErrorCode,
  error: string,
  retryable = false
): WebPageReadFailure {
  return { success: false, code, error, retryable };
}

function isFailure(value: object): value is WebPageReadFailure {
  return 'success' in value && value.success === false;
}

function isSupportedContentType(value: string): value is SupportedContentType {
  return SUPPORTED_CONTENT_TYPES.has(value);
}

function getDeclaredCharset(parameters: readonly string[]): string | undefined {
  for (const parameter of parameters) {
    const match =
      /^\s*charset\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s]+))\s*$/i.exec(parameter);
    if (match !== null) {
      return (match[1] ?? match[2] ?? match[3] ?? '').toLowerCase();
    }
  }
  return undefined;
}

function normalizeCharset(charset: string): ParsedContentType['charset'] {
  return charset === 'us-ascii' || charset === 'ascii' ? 'us-ascii' : 'utf-8';
}

function parseContentType(
  response: Response
): ParsedContentType | WebPageReadFailure {
  const header = response.headers.get('content-type');
  if (header === null) {
    return failure(
      'unsupported_content_type',
      'The response content type is not supported.'
    );
  }

  const [rawMimeType = '', ...parameters] = header.split(';');
  const mimeType = rawMimeType.trim().toLowerCase();
  if (!isSupportedContentType(mimeType)) {
    return failure(
      'unsupported_content_type',
      'The response content type is not supported.'
    );
  }

  const charset = getDeclaredCharset(parameters) ?? 'utf-8';
  if (!SUPPORTED_CHARSETS.has(charset)) {
    return failure(
      'unsupported_charset',
      'The response character encoding is not supported.'
    );
  }

  return { mimeType, charset: normalizeCharset(charset) };
}

function hasOversizedContentLength(response: Response): boolean {
  const value = response.headers.get('content-length');
  if (value === null || !/^\d+$/.test(value.trim())) {
    return false;
  }
  return Number(value) > MAX_WEB_PAGE_BODY_BYTES;
}

async function cancelResponseBody(response: Response): Promise<void> {
  try {
    await response.body?.cancel();
  } catch {
    // The rejection reason is authoritative; cancellation is best effort.
  }
}

async function rejectOversizedStream(
  reader: ReadableStreamDefaultReader<Uint8Array>
): Promise<WebPageReadFailure> {
  try {
    await reader.cancel();
  } catch {
    // The byte limit result is authoritative; cancellation is best effort.
  }
  return failure(
    'response_too_large',
    'The response body exceeds the Web reader byte limit.'
  );
}

function joinChunks(
  chunks: readonly Uint8Array[],
  totalBytes: number
): Uint8Array {
  const bytes = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

async function readBoundedBody(
  response: Response
): Promise<Uint8Array | WebPageReadFailure> {
  if (hasOversizedContentLength(response)) {
    await cancelResponseBody(response);
    return failure(
      'response_too_large',
      'The response body exceeds the Web reader byte limit.'
    );
  }
  if (response.body === null) {
    return new Uint8Array();
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;
  let current = await reader.read();
  while (!current.done) {
    totalBytes += current.value.byteLength;
    if (totalBytes > MAX_WEB_PAGE_BODY_BYTES) {
      return rejectOversizedStream(reader);
    }
    chunks.push(current.value);
    // Stream chunks must be consumed sequentially from the same reader.
    // eslint-disable-next-line no-await-in-loop
    current = await reader.read();
  }

  return joinChunks(chunks, totalBytes);
}

function isBinaryLike(bytes: Uint8Array): boolean {
  let suspiciousControlBytes = 0;
  for (const byte of bytes) {
    if (byte === 0) {
      return true;
    }
    if (
      (byte < 0x20 && byte !== 0x09 && byte !== 0x0a && byte !== 0x0d) ||
      byte === 0x7f
    ) {
      suspiciousControlBytes += 1;
    }
  }
  return (
    suspiciousControlBytes >= 8 && suspiciousControlBytes * 10 > bytes.length
  );
}

function decodeText(
  bytes: Uint8Array,
  charset: ParsedContentType['charset']
): string | WebPageReadFailure {
  if (isBinaryLike(bytes)) {
    return failure(
      'unsupported_content_type',
      'The response body does not appear to be text.'
    );
  }
  if (charset === 'us-ascii' && bytes.some((byte) => byte > 0x7f)) {
    return failure(
      'invalid_text_encoding',
      'The response body is not valid for its declared character encoding.'
    );
  }

  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
  } catch {
    return failure(
      'invalid_text_encoding',
      'The response body is not valid for its declared character encoding.'
    );
  }
}

function escapeMarkdownText(value: string): string {
  return value
    .replaceAll('\\', '\\\\')
    .replace(/([`*_{}[\]<>~|])/g, '\\$1')
    .replace(/(^|\n)(\s*)(#{1,6}|>|[-+]|\d+\.)\s/g, '$1$2\\$3 ');
}

function sliceAtSafeUtf16Boundary(value: string, end: number): string {
  let safeEnd = Math.min(end, value.length);
  const finalCodeUnit = value.charCodeAt(safeEnd - 1);
  if (finalCodeUnit >= 0xd800 && finalCodeUnit <= 0xdbff) {
    safeEnd -= 1;
  }
  return value.slice(0, safeEnd);
}

function normalizePlainText(value: string): {
  content: string;
  extractedCharacters: number;
  documentTruncated: boolean;
} {
  const normalized = value
    .replace(/\r\n?/g, '\n')
    .replace(DISALLOWED_CONTROL_CHARACTERS, '')
    .replace(/[ \t]+$/gm, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  const markdown = escapeMarkdownText(normalized);
  return {
    content: sliceAtSafeUtf16Boundary(
      markdown,
      DOCUMENT_CHARACTER_LIMIT
    ).trimEnd(),
    extractedCharacters: markdown.length,
    documentTruncated: markdown.length > DOCUMENT_CHARACTER_LIMIT,
  };
}

function isRetryableHttpStatus(status: number): boolean {
  return status === 408 || status === 425 || status === 429 || status >= 500;
}

function isTimeout(error: unknown, signal: AbortSignal): boolean {
  return (
    signal.aborted ||
    (error instanceof DOMException &&
      (error.name === 'TimeoutError' || error.name === 'AbortError'))
  );
}

function createRequestInit(signal: AbortSignal): RequestInit {
  return {
    method: 'GET',
    redirect: 'manual',
    cache: 'no-store',
    credentials: 'omit',
    headers: REQUEST_HEADERS,
    signal,
  };
}

async function followRedirect(
  response: Response,
  currentUrl: URL,
  redirectCount: number,
  dependencies: FetchDependencies
): Promise<FinalResponse | WebPageReadFailure> {
  const location = response.headers.get('location');
  if (redirectCount >= MAX_REDIRECTS) {
    await cancelResponseBody(response);
    return failure(
      'redirect_limit',
      'The response exceeded the Web reader redirect limit.'
    );
  }
  if (location === null) {
    await cancelResponseBody(response);
    return failure(
      'redirect_not_allowed',
      'The redirect destination is not allowed.'
    );
  }

  const redirect = admitWebRedirect(currentUrl, location);
  await cancelResponseBody(response);
  if (!redirect.success) {
    return redirect;
  }
  return fetchFinalResponse(redirect.url, redirectCount + 1, dependencies);
}

async function fetchFinalResponse(
  currentUrl: URL,
  redirectCount: number,
  dependencies: FetchDependencies
): Promise<FinalResponse | WebPageReadFailure> {
  const response = await dependencies.fetcher(
    currentUrl.href,
    createRequestInit(dependencies.signal)
  );
  if (REDIRECT_STATUSES.has(response.status)) {
    return followRedirect(response, currentUrl, redirectCount, dependencies);
  }
  return { response, finalUrl: currentUrl, redirectCount };
}

async function validateFinalResponse(
  response: Response
): Promise<ParsedContentType | WebPageReadFailure> {
  if (response.status < 200 || response.status >= 300) {
    await cancelResponseBody(response);
    return failure(
      'http_status',
      'The Web page returned an unsuccessful HTTP status.',
      isRetryableHttpStatus(response.status)
    );
  }

  const contentType = parseContentType(response);
  if (isFailure(contentType)) {
    await cancelResponseBody(response);
  }
  return contentType;
}

function createDocumentBase(
  metadata: DocumentMetadata
): Pick<
  WebPageDocument,
  | 'requestedUrl'
  | 'finalUrl'
  | 'retrievedAt'
  | 'httpStatus'
  | 'redirectCount'
  | 'contentType'
  | 'contentFormat'
  | 'rendering'
> {
  return {
    requestedUrl: metadata.requestedUrl,
    finalUrl: metadata.finalUrl.href,
    retrievedAt: metadata.retrievedAt,
    httpStatus: metadata.response.status,
    redirectCount: metadata.redirectCount,
    contentType: metadata.contentType,
    contentFormat: 'markdown',
    rendering: 'static',
  };
}

function createPlainTextResult(
  decoded: string,
  metadata: DocumentMetadata
): WebPageReadResult {
  const extracted = normalizePlainText(decoded);
  if (extracted.content === '') {
    return failure(
      'empty_content',
      'The Web page did not contain readable text.'
    );
  }

  return {
    success: true,
    document: {
      ...createDocumentBase(metadata),
      title: null,
      content: extracted.content,
      extractedCharacters: extracted.extractedCharacters,
      documentTruncated: extracted.documentTruncated,
      truncationReasons: extracted.documentTruncated
        ? ['document_character_limit']
        : [],
      links: [],
    },
  };
}

function createHtmlDocument(
  extracted: ExtractedWebPage,
  metadata: DocumentMetadata
): WebPageReadResult {
  if (extracted.content === '') {
    return failure(
      'empty_content',
      'The Web page did not contain readable text.'
    );
  }

  return {
    success: true,
    document: {
      ...createDocumentBase(metadata),
      title: extracted.title,
      content: extracted.content,
      extractedCharacters: extracted.extractedCharacters,
      documentTruncated: extracted.documentTruncated,
      truncationReasons: extracted.truncationReasons,
      links: extracted.links,
    },
  };
}

async function createHtmlResult(
  decoded: string,
  metadata: DocumentMetadata
): Promise<WebPageReadResult> {
  try {
    const extracted = await extractHtmlToMarkdown(decoded, metadata.finalUrl);
    return createHtmlDocument(extracted, metadata);
  } catch {
    return failure(
      'content_extraction_failed',
      'The Web page content could not be extracted.'
    );
  }
}

async function extractFinalResponse(
  finalResponse: FinalResponse,
  requestedUrl: string,
  retrievedAt: string
): Promise<WebPageReadResult> {
  const contentType = await validateFinalResponse(finalResponse.response);
  if (isFailure(contentType)) {
    return contentType;
  }

  const body = await readBoundedBody(finalResponse.response);
  if (!(body instanceof Uint8Array)) {
    return body;
  }
  const decoded = decodeText(body, contentType.charset);
  if (typeof decoded !== 'string') {
    return decoded;
  }

  const metadata: DocumentMetadata = {
    ...finalResponse,
    requestedUrl,
    retrievedAt,
    contentType: contentType.mimeType,
  };
  return contentType.mimeType === 'text/plain'
    ? createPlainTextResult(decoded, metadata)
    : createHtmlResult(decoded, metadata);
}

async function readPublicWebPage(
  rawUrl: string,
  fetcher: Fetcher,
  now: () => Date
): Promise<WebPageReadResult> {
  const admitted = admitPublicWebUrl(rawUrl);
  if (!admitted.success) {
    return admitted;
  }

  const signal = AbortSignal.timeout(REQUEST_TIMEOUT_MILLISECONDS);
  try {
    const finalResponse = await fetchFinalResponse(admitted.url, 0, {
      fetcher,
      signal,
    });
    if (isFailure(finalResponse)) {
      return finalResponse;
    }
    return await extractFinalResponse(
      finalResponse,
      admitted.url.href,
      now().toISOString()
    );
  } catch (error) {
    return isTimeout(error, signal)
      ? failure('timeout', 'The Web page request timed out.', true)
      : failure(
          'internal_error',
          'The Web page could not be read because of an internal error.'
        );
  }
}

async function defaultFetcher(
  input: string,
  init: RequestInit
): Promise<Response> {
  return await fetch(input, init);
}

function currentDate(): Date {
  return new Date();
}

/**
 * Cloudflare Workerのfetch/HTMLRewriter境界で公開Webページを取得する。
 */
export function createCloudflareWebPageReader(
  options: CloudflareWebPageReaderOptions = {}
): WebPageReaderPort {
  const fetcher = options.fetcher ?? defaultFetcher;
  const now = (): Date => options.now?.() ?? currentDate();
  return {
    async readPage(rawUrl: string): Promise<WebPageReadResult> {
      return await readPublicWebPage(rawUrl, fetcher, now);
    },
  };
}
