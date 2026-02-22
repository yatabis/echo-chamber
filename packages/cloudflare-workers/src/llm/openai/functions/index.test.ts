import { beforeEach, describe, expect, it, vi } from 'vitest';
import { z } from 'zod';

import { mockToolContext } from '../../../../test/mocks/tool';

import { Tool, type ToolResult } from './index';

describe('Tool', () => {
  const mockName = 'test_tool';
  const mockDescription = 'A test tool for testing purposes';
  const mockParameters = {
    message: z.string().describe('The message to process'),
    count: z.number().min(1).describe('Number of times to process'),
  };
  const mockHandler = vi.fn();

  describe('definition', () => {
    it('正しいFunctionTool形式のオブジェクトを返す', () => {
      const tool = new Tool(
        mockName,
        mockDescription,
        mockParameters,
        mockHandler
      );
      const { type, name, description, parameters, strict } = tool.definition;

      expect(type).toBe('function');
      expect(name).toBe(mockName);
      expect(description).toBe(mockDescription);
      expect(parameters).toEqual(z.toJSONSchema(z.object(mockParameters)));
      expect(strict).toBe(true);
    });

    it('空のparametersでも正しく動作する', () => {
      const emptyParameters = {};
      const tool = new Tool(
        mockName,
        mockDescription,
        emptyParameters,
        mockHandler
      );
      const { type, name, description, parameters, strict } = tool.definition;

      expect(type).toBe('function');
      expect(name).toBe(mockName);
      expect(description).toBe(mockDescription);
      expect(parameters).toEqual(z.toJSONSchema(z.object(emptyParameters)));
      expect(strict).toBe(true);
    });
  });

  describe('execute', () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    describe('成功ケース', () => {
      it('正常なJSON文字列でhandlerが実行される', async () => {
        const successResult: ToolResult = { success: true };
        mockHandler.mockReturnValue(successResult);

        const tool = new Tool(
          mockName,
          mockDescription,
          mockParameters,
          mockHandler
        );
        const args = JSON.stringify({ message: 'test', count: 1 });
        const result = await tool.execute(args, mockToolContext);

        expect(mockHandler).toHaveBeenCalledWith(
          { message: 'test', count: 1 },
          mockToolContext
        );
        expect(result).toBe(JSON.stringify(successResult));
      });

      it('async handlerが正しく動作する', async () => {
        const successResult: ToolResult = { success: true };
        mockHandler.mockResolvedValue(successResult);

        const tool = new Tool(
          mockName,
          mockDescription,
          mockParameters,
          mockHandler
        );
        const args = JSON.stringify({ message: 'test', count: 1 });
        const result = await tool.execute(args, mockToolContext);

        expect(mockHandler).toHaveBeenCalledWith(
          { message: 'test', count: 1 },
          mockToolContext
        );
        expect(result).toBe(JSON.stringify(successResult));
      });

      it('handlerのエラー結果が正しく返される', async () => {
        const errorResult: ToolResult = {
          success: false,
          error: 'Handler error',
        };
        mockHandler.mockReturnValue(errorResult);

        const tool = new Tool(
          mockName,
          mockDescription,
          mockParameters,
          mockHandler
        );
        const args = JSON.stringify({ message: 'test', count: 1 });
        const result = await tool.execute(args, mockToolContext);

        expect(mockHandler).toHaveBeenCalledWith(
          { message: 'test', count: 1 },
          mockToolContext
        );
        expect(result).toBe(JSON.stringify(errorResult));
      });
    });

    describe('失敗ケース', () => {
      it('不正なJSON文字列でエラーレスポンスが返される', async () => {
        const tool = new Tool(
          mockName,
          mockDescription,
          mockParameters,
          mockHandler
        );
        const args = '{"message": "test", "count":}';
        const result = await tool.execute(args, mockToolContext);

        expect(result).toBe(
          JSON.stringify({
            success: false,
            error: 'arguments is not valid JSON',
          })
        );
        expect(mockHandler).not.toHaveBeenCalled();
      });

      it('パラメータバリデーション失敗でエラーレスポンスが返される', async () => {
        const tool = new Tool(
          mockName,
          mockDescription,
          mockParameters,
          mockHandler
        );
        const args = JSON.stringify({
          message: 'test',
          count: 'invalid',
        });
        const result = await tool.execute(args, mockToolContext);
        const expected = {
          success: false,
          error: [
            {
              expected: 'number',
              code: 'invalid_type',
              path: ['count'],
              message: 'Invalid input: expected number, received string',
            },
          ],
        };

        expect(result).toBe(JSON.stringify(expected));
        expect(mockHandler).not.toHaveBeenCalled();
      });

      it('必須パラメータが不足している場合エラーレスポンスが返される', async () => {
        const tool = new Tool(
          mockName,
          mockDescription,
          mockParameters,
          mockHandler
        );
        const args = JSON.stringify({ message: 'test' });
        const result = await tool.execute(args, mockToolContext);
        const expected = {
          success: false,
          error: [
            {
              expected: 'number',
              code: 'invalid_type',
              path: ['count'],
              message: 'Invalid input: expected number, received undefined',
            },
          ],
        };

        expect(result).toBe(JSON.stringify(expected));
        expect(mockHandler).not.toHaveBeenCalled();
      });

      it('handler内で例外が発生した場合エラーレスポンスが返される', async () => {
        const error = new Error('Handler exception');
        mockHandler.mockImplementation(() => {
          throw error;
        });

        const tool = new Tool(
          mockName,
          mockDescription,
          mockParameters,
          mockHandler
        );
        const args = JSON.stringify({ message: 'test', count: 1 });
        const result = await tool.execute(args, mockToolContext);

        expect(result).toBe(
          JSON.stringify({
            success: false,
            error: error.message,
          })
        );
      });

      it('async handler内で例外が発生した場合エラーレスポンスが返される', async () => {
        const error = new Error('Async handler exception');
        const asyncHandler = vi.fn().mockRejectedValue(error);

        const tool = new Tool(
          mockName,
          mockDescription,
          mockParameters,
          asyncHandler
        );
        const args = JSON.stringify({ message: 'test', count: 1 });
        const result = await tool.execute(args, mockToolContext);

        expect(result).toBe(
          JSON.stringify({
            success: false,
            error: error.message,
          })
        );
      });
    });

    describe('エッジケース', () => {
      it('空のオブジェクトパラメータで正しく動作する', async () => {
        const emptyParameters = {};
        const successResult: ToolResult = { success: true };
        const handler = vi.fn().mockReturnValue(successResult);

        const tool = new Tool(
          mockName,
          mockDescription,
          emptyParameters,
          handler
        );
        const args = JSON.stringify({});
        const result = await tool.execute(args, mockToolContext);

        expect(handler).toHaveBeenCalledWith({}, mockToolContext);
        expect(result).toBe(JSON.stringify(successResult));
      });

      it('オプショナルパラメータが省略された場合でも動作する', async () => {
        const parametersWithOptional = {
          message: z.string().describe('Required message'),
          count: z.number().optional().describe('Optional count'),
        };
        const successResult: ToolResult = { success: true };
        const handler = vi.fn().mockReturnValue(successResult);

        const tool = new Tool(
          mockName,
          mockDescription,
          parametersWithOptional,
          handler
        );
        const args = JSON.stringify({ message: 'test' });
        const result = await tool.execute(args, mockToolContext);

        expect(handler).toHaveBeenCalledWith(
          { message: 'test' },
          mockToolContext
        );
        expect(result).toBe(JSON.stringify(successResult));
      });

      it('デフォルト値が設定されたパラメータで動作する', async () => {
        const parametersWithDefault = {
          message: z.string().describe('Required message'),
          count: z
            .number()
            .optional()
            .default(1)
            .describe('Count with default value'),
        };
        const successResult: ToolResult = { success: true };
        const handler = vi.fn().mockReturnValue(successResult);

        const tool = new Tool(
          mockName,
          mockDescription,
          parametersWithDefault,
          handler
        );
        const args = JSON.stringify({ message: 'test' });
        const result = await tool.execute(args, mockToolContext);

        expect(handler).toHaveBeenCalledWith(
          { message: 'test', count: 1 },
          mockToolContext
        );
        expect(result).toBe(JSON.stringify(successResult));
      });
    });
  });
});
