import { z } from 'zod';

import { getErrorMessage } from '@echo-chamber/core';
import type { ToolExecutionContext } from '@echo-chamber/core/agent/tool-context';

export type ToolContext = ToolExecutionContext;

interface FunctionToolDefinition {
  type: 'function';
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  strict: boolean;
}

interface ToolResultSuccess {
  success: true;
  [key: string]: unknown;
}

interface ToolResultError {
  success: false;
  error: string;
}

export type ToolResult = ToolResultSuccess | ToolResultError;

export interface ITool {
  name: string;
  description: string;
  definition: FunctionToolDefinition;
  execute(args: string, ctx: ToolContext): Promise<string>;
}

export class Tool<Args extends z.ZodRawShape> implements ITool {
  constructor(
    readonly name: string,
    readonly description: string,
    readonly parameters: Args,
    readonly handler: (
      args: z.infer<z.ZodObject<Args>>,
      ctx: ToolContext
    ) => ToolResult | Promise<ToolResult>
  ) {}

  get definition(): FunctionToolDefinition {
    const strict = Object.values(this.parameters).every((param) => {
      return !z.safeParse(param, undefined).success;
    });

    return {
      type: 'function',
      name: this.name,
      description: this.description,
      parameters: z.toJSONSchema(z.object(this.parameters)) as Record<
        string,
        unknown
      >,
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
