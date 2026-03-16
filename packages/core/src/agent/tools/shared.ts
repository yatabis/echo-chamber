import { z } from 'zod';

export interface ToolSpecification<
  Parameters extends z.ZodRawShape,
  OutputSchema extends z.ZodType,
> {
  name: string;
  description: string;
  parameters: Parameters;
  outputSchema: OutputSchema;
}

export function defineToolSpecification<
  Parameters extends z.ZodRawShape,
  OutputSchema extends z.ZodType,
>(
  spec: ToolSpecification<Parameters, OutputSchema>
): ToolSpecification<Parameters, OutputSchema> {
  return spec;
}

export const toolErrorResultSchema = z.object({
  success: z.literal(false),
  error: z.string(),
});

function createToolSuccessResultSchema<Shape extends z.ZodRawShape>(
  shape: Shape
): z.ZodObject<{ success: z.ZodLiteral<true> } & Shape> {
  return z.object({
    success: z.literal(true),
    ...shape,
  });
}

export function createToolResultSchema<Shape extends z.ZodRawShape>(
  shape: Shape
): z.ZodUnion<
  [
    z.ZodObject<{ success: z.ZodLiteral<true> } & Shape>,
    typeof toolErrorResultSchema,
  ]
> {
  return z.union([createToolSuccessResultSchema(shape), toolErrorResultSchema]);
}
