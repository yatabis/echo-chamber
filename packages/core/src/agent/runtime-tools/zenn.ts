import { getErrorMessage } from '../../utils/error';
import {
  getZennArticleToolSpec,
  listTrendingZennArticlesToolSpec,
} from '../tools/zenn';

import { Tool } from './tool';

const DEFAULT_TRENDING_ZENN_ARTICLE_LIMIT = 20;

/**
 * 長い本文を段落または単語境界で切り詰める。
 *
 * @param content 元の本文
 * @param maxCharacters 最大文字数
 * @returns 切り詰め済み本文と truncated フラグ
 */
function truncateContent(
  content: string,
  maxCharacters: number
): { content: string; truncated: boolean } {
  if (content.length <= maxCharacters) {
    return {
      content,
      truncated: false,
    };
  }

  const paragraphBoundary = content.lastIndexOf('\n\n', maxCharacters);
  if (paragraphBoundary >= Math.floor(maxCharacters * 0.7)) {
    return {
      content: content.slice(0, paragraphBoundary).trimEnd(),
      truncated: true,
    };
  }

  const wordBoundary = content.lastIndexOf(' ', maxCharacters);
  if (wordBoundary >= Math.floor(maxCharacters * 0.7)) {
    return {
      content: content.slice(0, wordBoundary).trimEnd(),
      truncated: true,
    };
  }

  return {
    content: content.slice(0, maxCharacters).trimEnd(),
    truncated: true,
  };
}

export const listTrendingZennArticlesTool = new Tool(
  listTrendingZennArticlesToolSpec,
  async ({ articleType }, ctx) => {
    try {
      const articles = await ctx.zenn.listTrendingArticles();
      const filteredArticles =
        articleType == null
          ? articles
          : articles.filter((article) => article.articleType === articleType);

      return {
        success: true,
        articles: filteredArticles.slice(
          0,
          DEFAULT_TRENDING_ZENN_ARTICLE_LIMIT
        ),
      };
    } catch (error) {
      await ctx.logger.error(
        `Error listing Zenn trending articles: ${getErrorMessage(error)}`
      );
      return {
        success: false,
        error: 'Failed to list Zenn trending articles',
      };
    }
  }
);

export const getZennArticleTool = new Tool(
  getZennArticleToolSpec,
  async ({ slug, maxCharacters }, ctx) => {
    try {
      const zennArticle = await ctx.zenn.getArticleBySlug(slug);
      const truncatedContent =
        maxCharacters == null
          ? {
              content: zennArticle.content,
              truncated: false,
            }
          : truncateContent(zennArticle.content, maxCharacters);

      return {
        success: true,
        slug: zennArticle.slug,
        url: zennArticle.url,
        title: zennArticle.title,
        author: zennArticle.author,
        topics: [...zennArticle.topics],
        tableOfContents: [...zennArticle.tableOfContents],
        content: truncatedContent.content,
        truncated: truncatedContent.truncated,
      };
    } catch (error) {
      await ctx.logger.error(
        `Error fetching Zenn article: ${getErrorMessage(error)}`
      );
      return {
        success: false,
        error: 'Failed to fetch Zenn article',
      };
    }
  }
);
