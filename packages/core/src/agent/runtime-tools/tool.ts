import { z } from 'zod';

import { getErrorMessage } from '../../utils/error';

import type { ModelToolContract } from '../../ports/model';
import type { AgentSessionTool } from '../session';
import type { ToolExecutionContext } from '../tool-context';
import type { ToolSpecification } from '../tools/shared';

export type ToolContext = ToolExecutionContext;

interface ToolResultSuccess {
  success: true;
  [key: string]: unknown;
}

interface ToolResultError {
  success: false;
  error: string;
}

export type ToolResult = ToolResultSuccess | ToolResultError;

export interface RuntimeTool {
  name: string;
  description: string;
  parameters: z.ZodRawShape;
  contract: ModelToolContract;
  execute(args: string, ctx: ToolContext): Promise<string>;
}

export class Tool<
  Parameters extends z.ZodRawShape,
  OutputSchema extends z.ZodType,
> implements RuntimeTool
{
  constructor(
    readonly specification: ToolSpecification<Parameters, OutputSchema>,
    readonly handler: (
      args: z.infer<z.ZodObject<Parameters>>,
      ctx: ToolContext
    ) => ToolResult | Promise<ToolResult>
  ) {}

  get name(): string {
    return this.specification.name;
  }

  get description(): string {
    return this.specification.description;
  }

  get parameters(): Parameters {
    return this.specification.parameters;
  }

  get contract(): ModelToolContract {
    const strict = Object.values(this.parameters).every((param) => {
      return !z.safeParse(param, undefined).success;
    });

    return {
      name: this.name,
      description: this.description,
      inputSchema: z.toJSONSchema(z.object(this.parameters)),
      strict,
    };
  }

  async execute(args: string, ctx: ToolContext): Promise<string> {
    try {
      const parsedArgs = z.parse(z.object(this.parameters), JSON.parse(args));
      const result = await this.handler(parsedArgs, ctx);
      return JSON.stringify(result);
    } catch (error) {
      // JSON.parse のエラー
      if (error instanceof SyntaxError) {
        return JSON.stringify({
          success: false,
          error: `arguments is not valid JSON`,
        });
      }

      // Zod のパースエラー
      if (error instanceof z.ZodError) {
        return JSON.stringify({ success: false, error: error.issues });
      }

      // handler のエラー
      return JSON.stringify({ success: false, error: getErrorMessage(error) });
    }
  }
}

export function bindRuntimeTool(
  tool: RuntimeTool,
  toolContext: ToolContext
): AgentSessionTool {
  return {
    name: tool.name,
    contract: tool.contract,
    execute: async (input: string): Promise<string> =>
      await tool.execute(input, toolContext),
  };
}

export function bindRuntimeTools(
  tools: readonly RuntimeTool[],
  toolContext: ToolContext
): readonly AgentSessionTool[] {
  return tools.map((tool) => bindRuntimeTool(tool, toolContext));
}
