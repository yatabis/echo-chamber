import { z } from 'zod';

import {
  READ_WEB_PAGE_DEFAULT_MAX_CHARACTERS,
  READ_WEB_PAGE_INTERNAL_ERROR_MESSAGE,
  readWebPageResultSchema,
  readWebPageToolSpec,
} from '../tools/web';

import type { RuntimeTool, ToolContext } from './tool';
import type { ModelToolContract } from '../../ports/model';
import type { WebPageDocument } from '../../ports/web-page-reader';

type ReadWebPageResult = z.infer<typeof readWebPageResultSchema>;

const INTERNAL_ERROR_RESULT: ReadWebPageResult = {
  success: false,
  code: 'internal_error',
  error: READ_WEB_PAGE_INTERNAL_ERROR_MESSAGE,
  retryable: false,
};

const INVALID_ARGUMENTS_RESULT: ReadWebPageResult = {
  success: false,
  code: 'invalid_url',
  error: 'The read_web_page arguments are invalid.',
  retryable: false,
};

/**
 * read_web_pageの最終モデル向けresultだけを一度検証してJSON化する。
 */
function serializeResult(result: ReadWebPageResult): string {
  const parsed = readWebPageResultSchema.safeParse(result);
  return JSON.stringify(parsed.success ? parsed.data : INTERNAL_ERROR_RESULT);
}

/**
 * UTF-16 surrogate pairを分断しない位置まで文字列を切る。
 */
function sliceAtSafeUtf16Boundary(content: string, end: number): string {
  let safeEnd = Math.min(end, content.length);
  const finalCodeUnit = content.charCodeAt(safeEnd - 1);
  if (finalCodeUnit >= 0xd800 && finalCodeUnit <= 0xdbff) {
    safeEnd -= 1;
  }

  return content.slice(0, safeEnd);
}

/**
 * 本文を可能な限り意味ブロック境界で切り詰める。
 */
function truncateDocumentText(
  content: string,
  maxCharacters: number
): { text: string; truncated: boolean } {
  if (content.length <= maxCharacters) {
    return { text: content, truncated: false };
  }

  const minimumUsefulBoundary = Math.floor(maxCharacters * 0.7);
  const blockBoundary = content.lastIndexOf('\n\n', maxCharacters);
  const end =
    blockBoundary >= minimumUsefulBoundary ? blockBoundary : maxCharacters;

  return {
    text: sliceAtSafeUtf16Boundary(content, end).trimEnd(),
    truncated: true,
  };
}

/**
 * provider-neutral documentをread_web_pageのモデル向けshapeへ写像する。
 */
function createSuccessResult(
  document: WebPageDocument,
  maxCharacters: number
): ReadWebPageResult {
  const truncatedText = truncateDocumentText(document.content, maxCharacters);
  const truncationReasons = new Set<
    'output_character_limit' | 'document_character_limit'
  >(document.truncationReasons);
  if (truncatedText.truncated) {
    truncationReasons.add('output_character_limit');
  }

  return {
    success: true,
    source: {
      requestedUrl: document.requestedUrl,
      finalUrl: document.finalUrl,
      retrievedAt: document.retrievedAt,
      httpStatus: document.httpStatus,
      redirectCount: document.redirectCount,
      contentType: document.contentType,
      title: document.title,
    },
    document: {
      format: document.contentFormat,
      rendering: document.rendering,
      text: truncatedText.text,
      returnedCharacters: truncatedText.text.length,
      extractedCharacters: document.extractedCharacters,
      truncated: document.documentTruncated || truncatedText.truncated,
      truncationReasons: [...truncationReasons],
      links: [...document.links],
    },
    trust: 'untrusted_external_content',
  };
}

/**
 * 公開WebページReaderをモデル向けtool contractへ接続するruntime tool。
 * 最終result以外の共通tool実行基盤へoutput検証を追加しない。
 */
export const readWebPageTool: RuntimeTool = {
  name: readWebPageToolSpec.name,
  description: readWebPageToolSpec.description,
  parameters: readWebPageToolSpec.parameters,
  contract: {
    name: readWebPageToolSpec.name,
    description: readWebPageToolSpec.description,
    inputSchema: z.toJSONSchema(z.object(readWebPageToolSpec.parameters)),
    outputSchema: z.toJSONSchema(readWebPageToolSpec.outputSchema),
    strict: false,
  } satisfies ModelToolContract,

  async execute(args: string, ctx: ToolContext): Promise<string> {
    let parsedArgs: z.infer<z.ZodObject<typeof readWebPageToolSpec.parameters>>;
    try {
      parsedArgs = z.parse(
        z.object(readWebPageToolSpec.parameters),
        JSON.parse(args)
      );
    } catch {
      return serializeResult(INVALID_ARGUMENTS_RESULT);
    }

    try {
      const result = await ctx.webPageReader.readPage(parsedArgs.url);
      if (!result.success) {
        return serializeResult(result);
      }

      return serializeResult(
        createSuccessResult(
          result.document,
          parsedArgs.maxCharacters ?? READ_WEB_PAGE_DEFAULT_MAX_CHARACTERS
        )
      );
    } catch {
      return serializeResult(INTERNAL_ERROR_RESULT);
    }
  },
};
