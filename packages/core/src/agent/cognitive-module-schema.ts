import { z } from 'zod';

import {
  emotionSchema,
  memoryQuerySchema,
  memoryStoreInputSchema,
} from '../echo/schemas';

import type { ModelStructuredOutputFormat } from '../ports/model';

/** 構造化出力の検証位置と違反規則。 */
export interface CognitiveModuleSchemaIssue {
  path: string;
  code: string;
}

/** Cognitive Moduleの構造化出力がcontractに一致しない。 */
export class CognitiveModuleSchemaValidationError extends Error {
  override readonly name = 'CognitiveModuleSchemaValidationError';
  readonly code = 'strict_schema' as const;

  /** @param issues event payloadへ載せられるbounded Zod issue */
  constructor(readonly issues: readonly CognitiveModuleSchemaIssue[] = []) {
    super('Cognitive module schema validation failed: strict_schema');
  }
}

const memoryRecallCognitiveModuleOutputSchema = z
  .object({
    query: memoryQuerySchema,
  })
  .strict();

/** Mainの各model turn前にMemory Moduleが生成するsearch_memory入力。 */
export type MemoryRecallCognitiveModuleOutput = z.infer<
  typeof memoryRecallCognitiveModuleOutputSchema
>;

/** 思考session終了時にMemory Moduleが生成するstore_memory入力。 */
export type MemoryStoreCognitiveModuleOutput = z.infer<
  typeof memoryStoreInputSchema
>;

/** Memory Moduleが実行phaseに応じて返す出力。 */
export type MemoryCognitiveModuleOutput =
  | MemoryRecallCognitiveModuleOutput
  | MemoryStoreCognitiveModuleOutput;

/** Emotion Moduleが毎回返す更新後の感情状態。 */
export type EmotionCognitiveModuleOutput = z.infer<typeof emotionSchema>;

/** JSON Schemaとruntime validationが共有するstrict output formatを作る。 */
function createOutputFormat(
  name: string,
  schema: z.ZodType
): ModelStructuredOutputFormat {
  return {
    type: 'json_schema',
    name,
    strict: true,
    schema: z.toJSONSchema(schema, {
      target: 'draft-7',
      io: 'output',
      reused: 'inline',
    }) as Record<string, unknown>,
  };
}

/** Zod issueをboundedな検証診断へ変換する。 */
function createSchemaIssues(
  error: z.ZodError
): readonly CognitiveModuleSchemaIssue[] {
  return error.issues.slice(0, 16).map((issue) => ({
    path: issue.path.map((segment) => String(segment)).join('.'),
    code: issue.code,
  }));
}

/** Schema不一致を共通のbounded errorへ変換する。 */
function parseOutput<T>(schema: z.ZodType<T>, value: unknown): T {
  const parsed = schema.safeParse(value);
  if (!parsed.success) {
    throw new CognitiveModuleSchemaValidationError(
      createSchemaIssues(parsed.error)
    );
  }
  return parsed.data;
}

/** Memory recall output用のprovider-neutral strict JSON Schema。 */
export function createMemoryRecallCognitiveModuleOutputFormat(): ModelStructuredOutputFormat {
  return createOutputFormat(
    'cognitive_memory_recall',
    memoryRecallCognitiveModuleOutputSchema
  );
}

/** Memory store output用のprovider-neutral strict JSON Schema。 */
export function createMemoryStoreCognitiveModuleOutputFormat(): ModelStructuredOutputFormat {
  return createOutputFormat('cognitive_memory_store', memoryStoreInputSchema);
}

/** Emotion output用のprovider-neutral strict JSON Schema。 */
export function createEmotionCognitiveModuleOutputFormat(): ModelStructuredOutputFormat {
  return createOutputFormat('cognitive_emotion_update', emotionSchema);
}

/** Memory recall modelのunknown outputを検証済みqueryへ変換する。 */
export function parseMemoryRecallCognitiveModuleOutput(
  value: unknown
): MemoryRecallCognitiveModuleOutput {
  return parseOutput(memoryRecallCognitiveModuleOutputSchema, value);
}

/** Memory store modelのunknown outputを検証済み入力へ変換する。 */
export function parseMemoryStoreCognitiveModuleOutput(
  value: unknown
): MemoryStoreCognitiveModuleOutput {
  return parseOutput(memoryStoreInputSchema, value);
}

/** Emotion modelのunknown outputを検証済み感情状態へ変換する。 */
export function parseEmotionCognitiveModuleOutput(
  value: unknown
): EmotionCognitiveModuleOutput {
  return parseOutput(emotionSchema, value);
}
