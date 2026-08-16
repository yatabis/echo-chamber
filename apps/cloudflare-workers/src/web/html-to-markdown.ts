import type { WebPageLink } from '@echo-chamber/core/ports/web-page-reader';

import { admitPublicWebUrl } from './url-policy';

const DOCUMENT_CHARACTER_LIMIT = 64_000;
const TITLE_CHARACTER_LIMIT = 300;
const LINK_LIMIT = 20;
const LINK_TEXT_LIMIT = 160;
const LINK_URL_LIMIT = 4_096;
const MIN_MAIN_CANDIDATE_CHARACTERS = 200;
const DISALLOWED_CONTROL_CHARACTERS =
  // eslint-disable-next-line no-control-regex -- These exact C0 controls are removed from untrusted page text.
  /[\u0001-\u0008\u000b\u000c\u000e-\u001f\u007f]/g;

interface CandidateMarkers {
  start: string;
  end: string;
}

interface LinkOccurrence extends CandidateMarkers {
  url: string;
}

interface RewriterContext {
  rewriter: HTMLRewriter;
  registry: MarkerRegistry;
  chunks: string[];
}

interface ExtractionMarkers {
  mainCandidates: CandidateMarkers[];
  bodyCandidates: CandidateMarkers[];
  titleCandidates: CandidateMarkers[];
  linkOccurrences: LinkOccurrence[];
}

/**
 * HTMLから抽出した静的な制御Markdown。
 */
export interface ExtractedWebPage {
  title: string | null;
  content: string;
  extractedCharacters: number;
  documentTruncated: boolean;
  truncationReasons: readonly ['document_character_limit'] | readonly [];
  links: readonly WebPageLink[];
}

class MarkerRegistry {
  private readonly values = new Map<number, string>();
  private nextId = 0;
  readonly prefix: string;

  constructor(source: string) {
    let suffix = 0;
    while (source.includes(`\uE000ECHO${suffix}:`)) {
      suffix += 1;
    }
    this.prefix = `\uE000ECHO${suffix}:`;
  }

  create(value = ''): string {
    const id = this.nextId;
    this.nextId += 1;
    this.values.set(id, value);
    return `${this.prefix}${id}\uE001`;
  }

  set(marker: string, value: string): void {
    const id = this.getMarkerId(marker);
    if (id !== null) {
      this.values.set(id, value);
    }
  }

  render(source: string): string {
    const matcher = this.createMatcher();
    let rendered = '';
    let previousEnd = 0;

    for (const match of source.matchAll(matcher)) {
      const { index } = match;
      const idText = match[1];
      rendered += escapeMarkdownText(source.slice(previousEnd, index));
      rendered += this.values.get(Number(idText)) ?? '';
      previousEnd = index + match[0].length;
    }

    rendered += escapeMarkdownText(source.slice(previousEnd));
    return normalizeRenderedMarkdown(rendered);
  }

  remove(source: string): string {
    return source.replace(this.createMatcher(), '');
  }

  private createMatcher(): RegExp {
    return new RegExp(
      `${escapeRegularExpression(this.prefix)}(\\d+)\uE001`,
      'g'
    );
  }

  private getMarkerId(marker: string): number | null {
    if (!marker.startsWith(this.prefix) || !marker.endsWith('\uE001')) {
      return null;
    }
    const id = Number(marker.slice(this.prefix.length, -1));
    return Number.isInteger(id) ? id : null;
  }
}

function escapeRegularExpression(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function escapeMarkdownText(value: string): string {
  return value
    .replaceAll('\\', '\\\\')
    .replace(/([`*_{}[\]<>~|])/g, '\\$1')
    .replace(/(^|\n)(\s*)(#{1,6}|>|[-+]|\d+\.)\s/g, '$1$2\\$3 ');
}

function decodeHtmlEntities(value: string): string {
  const namedEntities: Record<string, string> = {
    amp: '&',
    apos: "'",
    copy: '©',
    gt: '>',
    hellip: '…',
    laquo: '«',
    lt: '<',
    mdash: '—',
    nbsp: ' ',
    ndash: '–',
    quot: '"',
    raquo: '»',
    reg: '®',
  };

  return value.replace(
    /&(#x[\da-f]+|#\d+|[a-z][\da-z]+);/gi,
    (entity, name: string): string => {
      if (name.startsWith('#x') || name.startsWith('#X')) {
        return decodeNumericEntity(entity, name.slice(2), 16);
      }
      if (name.startsWith('#')) {
        return decodeNumericEntity(entity, name.slice(1), 10);
      }
      return namedEntities[name.toLowerCase()] ?? entity;
    }
  );
}

function decodeNumericEntity(
  original: string,
  digits: string,
  radix: number
): string {
  const codePoint = Number.parseInt(digits, radix);
  if (
    !Number.isInteger(codePoint) ||
    codePoint < 0 ||
    codePoint > 0x10ffff ||
    (codePoint >= 0xd800 && codePoint <= 0xdfff)
  ) {
    return original;
  }

  return String.fromCodePoint(codePoint);
}

function normalizeSourceText(value: string): string {
  return decodeHtmlEntities(value)
    .replaceAll('\u0000', '')
    .replace(DISALLOWED_CONTROL_CHARACTERS, '')
    .replace(/[\s\u00a0]+/g, ' ')
    .trim();
}

function normalizeRenderedMarkdown(value: string): string {
  return value
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n[ \t]+/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function sliceAtSafeUtf16Boundary(value: string, end: number): string {
  let safeEnd = Math.min(end, value.length);
  const finalCodeUnit = value.charCodeAt(safeEnd - 1);
  if (finalCodeUnit >= 0xd800 && finalCodeUnit <= 0xdbff) {
    safeEnd -= 1;
  }
  return value.slice(0, safeEnd);
}

function truncateDocument(content: string): {
  content: string;
  truncated: boolean;
} {
  if (content.length <= DOCUMENT_CHARACTER_LIMIT) {
    return { content, truncated: false };
  }

  const blockBoundary = content.lastIndexOf('\n\n', DOCUMENT_CHARACTER_LIMIT);
  const end =
    blockBoundary >= Math.floor(DOCUMENT_CHARACTER_LIMIT * 0.7)
      ? blockBoundary
      : DOCUMENT_CHARACTER_LIMIT;
  return {
    content: sliceAtSafeUtf16Boundary(content, end).trimEnd(),
    truncated: true,
  };
}

function addBoundaryHandler(
  context: RewriterContext,
  selector: string,
  candidates: CandidateMarkers[]
): void {
  context.rewriter.on(selector, {
    element(element): void {
      if (element.removed) {
        return;
      }
      const candidate = {
        start: context.registry.create(),
        end: context.registry.create(),
      };
      candidates.push(candidate);
      context.chunks.push(candidate.start);
      element.onEndTag(() => {
        context.chunks.push(candidate.end);
      });
    },
  });
}

function addWrappingHandler(
  context: RewriterContext,
  selector: string,
  before: string,
  after: string
): void {
  context.rewriter.on(selector, {
    element(element): void {
      if (element.removed) {
        return;
      }
      context.chunks.push(context.registry.create(before));
      element.onEndTag(() => {
        context.chunks.push(context.registry.create(after));
      });
    },
  });
}

function addLeadingHandler(
  context: RewriterContext,
  selector: string,
  value: string
): void {
  context.rewriter.on(selector, {
    element(element): void {
      if (!element.removed) {
        context.chunks.push(context.registry.create(value));
      }
    },
  });
}

function findMarkedContent(
  source: string,
  markers: CandidateMarkers
): string | null {
  const start = source.indexOf(markers.start);
  if (start < 0) {
    return null;
  }
  const contentStart = start + markers.start.length;
  const end = source.indexOf(markers.end, contentStart);
  return end < 0 ? null : source.slice(contentStart, end);
}

function selectMainContent(
  source: string,
  candidates: readonly CandidateMarkers[],
  bodyCandidates: readonly CandidateMarkers[],
  registry: MarkerRegistry
): string {
  const substantiveCandidates = candidates
    .map((markers) => findMarkedContent(source, markers))
    .filter((candidate): candidate is string => candidate !== null)
    .map((candidate) => ({
      candidate,
      length: normalizeSourceText(registry.remove(candidate)).length,
    }))
    .filter(({ length }) => length >= MIN_MAIN_CANDIDATE_CHARACTERS)
    .sort((left, right) => right.length - left.length);
  const selected = substantiveCandidates[0]?.candidate;
  if (selected !== undefined) {
    return selected;
  }

  for (const markers of bodyCandidates) {
    const body = findMarkedContent(source, markers);
    if (body !== null) {
      return body;
    }
  }

  return source;
}

function resolveLink(href: string, baseUrl: URL): string | null {
  if (href.trim() === '' || href.trim().startsWith('#')) {
    return null;
  }

  let resolved: URL;
  try {
    resolved = new URL(href, baseUrl);
  } catch {
    return null;
  }
  if (resolved.href.length > LINK_URL_LIMIT) {
    return null;
  }

  const admitted = admitPublicWebUrl(resolved.href);
  return admitted.success ? admitted.url.href : null;
}

function configureSelectedLinks(
  selectedSource: string,
  occurrences: readonly LinkOccurrence[],
  registry: MarkerRegistry
): WebPageLink[] {
  const selectedUrls = new Set<string>();
  const links: WebPageLink[] = [];

  for (const occurrence of occurrences) {
    const start = selectedSource.indexOf(occurrence.start);
    if (start < 0) {
      continue;
    }
    const anchorStart = start + occurrence.start.length;
    const end = selectedSource.indexOf(occurrence.end, anchorStart);
    if (end < 0) {
      continue;
    }

    if (!selectedUrls.has(occurrence.url) && selectedUrls.size < LINK_LIMIT) {
      const anchor = normalizeSourceText(
        registry.remove(selectedSource.slice(anchorStart, end))
      );
      selectedUrls.add(occurrence.url);
      links.push({
        text: sliceAtSafeUtf16Boundary(anchor, LINK_TEXT_LIMIT),
        url: occurrence.url,
      });
    }

    if (selectedUrls.has(occurrence.url)) {
      registry.set(occurrence.start, '[');
      registry.set(occurrence.end, `](${occurrence.url})`);
    }
  }

  return links;
}

function configureDiscardedElements(context: RewriterContext): void {
  context.rewriter.on(
    'script, style, noscript, template, svg, canvas, iframe, object, embed, form, nav, footer, aside',
    {
      element(element): void {
        element.remove();
      },
      text(text): void {
        text.remove();
      },
    }
  );
}

function configureContentBoundaries(
  context: RewriterContext,
  markers: ExtractionMarkers
): void {
  addBoundaryHandler(context, 'body', markers.bodyCandidates);
  addBoundaryHandler(context, 'article', markers.mainCandidates);
  addBoundaryHandler(context, 'main', markers.mainCandidates);
  addBoundaryHandler(context, '[role="main"]', markers.mainCandidates);
  addBoundaryHandler(context, 'title', markers.titleCandidates);
}

function configureMarkdownStructure(context: RewriterContext): void {
  for (let level = 1; level <= 6; level += 1) {
    addWrappingHandler(
      context,
      `h${level}`,
      `\n\n${'#'.repeat(level)} `,
      '\n\n'
    );
  }
  addWrappingHandler(context, 'p', '\n\n', '\n\n');
  addWrappingHandler(context, 'blockquote', '\n\n> ', '\n\n');
  addWrappingHandler(context, 'pre', '\n\n~~~\n', '\n~~~\n\n');
  addWrappingHandler(context, 'table', '\n\n', '\n\n');
  addWrappingHandler(context, 'th, td', '| ', ' ');
  addWrappingHandler(context, 'tr', '\n', '|\n');
  addLeadingHandler(context, 'ul > li', '\n- ');
  addLeadingHandler(context, 'ol > li', '\n1. ');
  addLeadingHandler(context, 'br', '\n');
  addLeadingHandler(context, 'hr', '\n\n---\n\n');
  addLeadingHandler(context, 'div, section', '\n\n');
}

function configureLinks(
  context: RewriterContext,
  occurrences: LinkOccurrence[],
  baseUrl: URL
): void {
  context.rewriter.on('a', {
    element(element): void {
      if (element.removed) {
        return;
      }
      const href = element.getAttribute('href');
      if (href === null) {
        return;
      }
      const url = resolveLink(href, baseUrl);
      if (url === null) {
        return;
      }

      const occurrence = {
        start: context.registry.create(),
        end: context.registry.create(),
        url,
      };
      occurrences.push(occurrence);
      context.chunks.push(occurrence.start);
      element.onEndTag(() => {
        context.chunks.push(occurrence.end);
      });
    },
  });
}

function configureTextCollection(context: RewriterContext): void {
  context.rewriter.onDocument({
    text(text): void {
      if (!text.removed) {
        context.chunks.push(text.text);
      }
    },
  });
}

async function collectRewrittenSource(
  html: string,
  context: RewriterContext
): Promise<string> {
  const rewrittenResponse = context.rewriter.transform(
    new Response(html, {
      headers: { 'content-type': 'text/html; charset=utf-8' },
    })
  );
  await rewrittenResponse.text();
  return context.chunks.join('');
}

function createExtractedPage(
  decodedSource: string,
  context: RewriterContext,
  markers: ExtractionMarkers
): ExtractedWebPage {
  const selectedSource = selectMainContent(
    decodedSource,
    markers.mainCandidates,
    markers.bodyCandidates,
    context.registry
  );
  const links = configureSelectedLinks(
    selectedSource,
    markers.linkOccurrences,
    context.registry
  );
  const normalizedSource = normalizeSourceText(selectedSource);
  const markdown = context.registry.render(normalizedSource);
  const truncated = truncateDocument(markdown);

  const titleSource = markers.titleCandidates
    .map((candidate) => findMarkedContent(decodedSource, candidate))
    .find((candidate): candidate is string => candidate !== null);
  const title =
    titleSource === undefined
      ? null
      : sliceAtSafeUtf16Boundary(
          normalizeSourceText(context.registry.remove(titleSource)),
          TITLE_CHARACTER_LIMIT
        );

  return {
    title: title === '' ? null : title,
    content: truncated.content,
    extractedCharacters: markdown.length,
    documentTruncated: truncated.truncated,
    truncationReasons: truncated.truncated ? ['document_character_limit'] : [],
    links,
  };
}

/**
 * HTMLRewriterで実行可能要素を除去し、静的な制御Markdownへ変換する。
 */
export async function extractHtmlToMarkdown(
  html: string,
  baseUrl: URL
): Promise<ExtractedWebPage> {
  const context: RewriterContext = {
    rewriter: new HTMLRewriter(),
    registry: new MarkerRegistry(html),
    chunks: [],
  };
  const markers: ExtractionMarkers = {
    mainCandidates: [],
    bodyCandidates: [],
    titleCandidates: [],
    linkOccurrences: [],
  };

  configureDiscardedElements(context);
  configureContentBoundaries(context, markers);
  configureMarkdownStructure(context);
  configureLinks(context, markers.linkOccurrences, baseUrl);
  configureTextCollection(context);

  const decodedSource = await collectRewrittenSource(html, context);
  return createExtractedPage(decodedSource, context, markers);
}
