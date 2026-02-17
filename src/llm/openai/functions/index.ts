import { z } from 'zod';

import { getErrorMessage } from '../../../utils/error';

import type { MemorySystem } from '../../../echo/memory-system';
import type { NoteSystem } from '../../../echo/note-system';
import type { EchoInstanceConfig } from '../../../types/echo-config';
import type { Logger } from '../../../utils/logger';
import type { FunctionTool } from 'openai/resources/responses/responses';

export interface ToolContext {
  instanceConfig: EchoInstanceConfig;
  storage: DurableObjectStorage;
  memorySystem: MemorySystem;
  noteSystem: NoteSystem;
  logger: Logger;
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
  definition: FunctionTool;
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

  get definition(): FunctionTool {
    const strict = Object.values(this.parameters).every((param) => {
      return !z.safeParse(param, undefined).success;
    });

    return {
      type: 'function',
      name: this.name,
      description: this.description,
      parameters: z.toJSONSchema(z.object(this.parameters)),
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
