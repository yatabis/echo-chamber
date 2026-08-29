import { z } from 'zod';

import {
  emotionSchema,
  memoryQuerySchema,
  memoryStoreInputSchema,
  memoryTypeSchema,
} from '../../echo/schemas';

import { createToolResultSchema, defineToolSpecification } from './shared';

const memorySearchResultSchema = z.object({
  content: z.string(),
  type: memoryTypeSchema,
  emotion: emotionSchema,
  createdAt: z.string(),
});

/** Main が必要に応じて長期記憶を保存するための tool contract。 */
export const storeMemoryToolSpec = defineToolSpecification({
  name: 'store_memory',
  description:
    'Main が将来のセマンティック検索に残す価値があると必要に応じて判断した場合に、記憶を保存できる。利用は任意である。現在の感情状態はシステムが自動的に関連付ける。システムはセマンティック検索のためにエンベディングを使用し、容量がいっぱいになると最も古い記憶を削除して自動的に管理する。',
  parameters: memoryStoreInputSchema.shape,
  outputSchema: createToolResultSchema({}),
});

/** Main が必要に応じて長期記憶を検索するための tool contract。 */
export const searchMemoryToolSpec = defineToolSpecification({
  name: 'search_memory',
  description:
    'Main が過去の経験や関連情報を参照する必要があると判断した場合に、セマンティック類似性を使用して記憶を検索できる。利用は任意である。セマンティック類似性でソートされた最大5件の最も関連性の高い記憶を返す。',
  parameters: {
    query: memoryQuerySchema,
    type: memoryTypeSchema
      .optional()
      .describe(
        '検索対象の記憶タイプ。指定しない場合は全タイプを検索する。semantic: 事実や一般的な知識。episode: 特定の体験や出来事。'
      ),
  },
  outputSchema: createToolResultSchema({
    results: z.array(memorySearchResultSchema),
  }),
});
