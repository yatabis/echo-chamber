import { z } from 'zod';

import { getErrorMessage } from '@echo-chamber/core';
import type { ToolExecutionContext } from '@echo-chamber/core/agent/tool-context';
import type { ModelToolContract } from '@echo-chamber/core/ports/model';

export type ToolContext = ToolExecutionContext;

interface FunctionToolDefinition {
  type: 'function';
  name: string;
  description: string;
  parameters: unknown;
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
  contract: ModelToolContract;
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

  get definition(): FunctionToolDefinition {
    const { name, description, inputSchema, strict } = this.contract;
    return {
      type: 'function',
      name,
      description,
      parameters: inputSchema,
      strict: strict ?? false,
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
