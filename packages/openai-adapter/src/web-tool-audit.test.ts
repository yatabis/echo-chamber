import { describe, expect, it } from 'vitest';

import {
  projectChatCompletionsWebToolExchange,
  projectResponsesWebToolExchange,
} from './web-tool-audit';

const WEB_ARGUMENT_CANARY = 'https://public.example/path?token=WEB_ARG_CANARY';
const WEB_RESULT_CANARY = 'WEB_RESULT_CANARY';
const NON_WEB_CANARY = 'NON_WEB_CANARY';

const webResult = JSON.stringify({
  success: true,
  source: {
    requestedUrl: WEB_ARGUMENT_CANARY,
    finalUrl: 'https://public.example/final',
    title: 'private title',
    httpStatus: 200,
    contentType: 'text/html',
    redirectCount: 1,
  },
  document: {
    text: WEB_RESULT_CANARY,
    returnedCharacters: 17,
    extractedCharacters: 42,
    truncated: true,
    links: [
      {
        text: 'secret link',
        url: 'https://public.example/secret-link',
      },
    ],
  },
});

describe('projectResponsesWebToolExchange', () => {
  it('Web call IDを次exchangeへ引き継ぎ、Web部分だけmetadataへ置換する', () => {
    const firstRequest = {
      input: [{ role: 'user', content: 'start' }],
      model: 'gpt-test',
    };
    const firstResponse = {
      id: 'response-1',
      output: [
        {
          type: 'function_call',
          call_id: 'web-call',
          name: 'read_web_page',
          arguments: JSON.stringify({
            url: WEB_ARGUMENT_CANARY,
            maxCharacters: 9000,
          }),
        },
        {
          type: 'function_call',
          call_id: 'other-call',
          name: 'think_deeply',
          arguments: NON_WEB_CANARY,
        },
      ],
    };

    const first = projectResponsesWebToolExchange(
      firstRequest,
      firstResponse,
      new Set()
    );
    const projectedWebCall = first.response.output[0];
    if (projectedWebCall?.arguments === undefined) {
      throw new Error('Expected a projected Responses Web tool call');
    }
    expect(JSON.parse(projectedWebCall.arguments)).toEqual({
      redacted: true,
      urlLength: WEB_ARGUMENT_CANARY.length,
      hasQuery: true,
      maxCharacters: 9000,
    });
    expect(first.response.output[1]).toBe(firstResponse.output[1]);
    expect(first.response.output[1]?.arguments).toBe(NON_WEB_CANARY);
    expect(firstResponse.output[0]?.arguments).toContain(WEB_ARGUMENT_CANARY);

    const secondRequest = {
      input: [
        {
          type: 'function_call_output',
          call_id: 'web-call',
          output: webResult,
        },
        {
          type: 'function_call',
          call_id: 'other-call-2',
          name: 'think_deeply',
          arguments: NON_WEB_CANARY,
        },
        {
          type: 'function_call_output',
          call_id: 'other-call-2',
          output: NON_WEB_CANARY,
        },
      ],
      model: 'gpt-test',
    };
    const secondResponse = { id: 'response-2', output: [] };

    const second = projectResponsesWebToolExchange(
      secondRequest,
      secondResponse,
      first.webToolCallIds
    );
    const projectedWebResult = second.request.input[0];
    if (projectedWebResult?.output === undefined) {
      throw new Error('Expected a projected Responses Web tool result');
    }
    expect(JSON.parse(projectedWebResult.output)).toEqual({
      redacted: true,
      success: true,
      httpStatus: 200,
      contentType: 'text/html',
      redirectCount: 1,
      returnedCharacters: 17,
      extractedCharacters: 42,
      truncated: true,
      linkCount: 1,
    });
    expect(second.request.input[1]).toBe(secondRequest.input[1]);
    expect(second.request.input[2]).toBe(secondRequest.input[2]);
    expect(JSON.stringify(second)).not.toContain(WEB_RESULT_CANARY);
    expect(JSON.stringify(second)).toContain(NON_WEB_CANARY);
    expect(secondRequest.input[0]?.output).toBe(webResult);
  });
});

describe('projectChatCompletionsWebToolExchange', () => {
  it('保持historyとresponseのWeb call/resultだけをmetadataへ置換する', () => {
    const request = {
      model: 'chat-test',
      messages: [
        { role: 'user', content: 'start' },
        {
          role: 'assistant',
          content: null,
          tool_calls: [
            {
              id: 'web-call',
              type: 'function',
              function: {
                name: 'read_web_page',
                arguments: JSON.stringify({ url: WEB_ARGUMENT_CANARY }),
              },
            },
            {
              id: 'other-call',
              type: 'function',
              function: {
                name: 'think_deeply',
                arguments: NON_WEB_CANARY,
              },
            },
          ],
        },
        { role: 'tool', tool_call_id: 'web-call', content: webResult },
        {
          role: 'tool',
          tool_call_id: 'other-call',
          content: NON_WEB_CANARY,
        },
      ],
    };
    const response = {
      id: 'chat-1',
      choices: [
        {
          index: 0,
          message: {
            role: 'assistant',
            content: null,
            tool_calls: [
              {
                id: 'next-web-call',
                type: 'function',
                function: {
                  name: 'read_web_page',
                  arguments: JSON.stringify({ url: WEB_ARGUMENT_CANARY }),
                },
              },
            ],
          },
        },
      ],
    };

    const projected = projectChatCompletionsWebToolExchange(
      request,
      response,
      new Set()
    );
    const serialized = JSON.stringify(projected);

    expect(serialized).not.toContain('WEB_ARG_CANARY');
    expect(serialized).not.toContain(WEB_RESULT_CANARY);
    expect(serialized).not.toContain('private title');
    expect(serialized).not.toContain('secret-link');
    expect(serialized).toContain(NON_WEB_CANARY);
    expect(projected.request.messages[0]).toBe(request.messages[0]);
    expect(projected.request.messages[3]).toBe(request.messages[3]);
    expect(JSON.stringify(request)).toContain(WEB_RESULT_CANARY);
    expect(projected.webToolCallIds).toEqual(new Set(['next-web-call']));
  });

  it('解析不能なWeb値もrawへfallbackせず完全に伏せる', () => {
    const projected = projectChatCompletionsWebToolExchange(
      {
        messages: [
          {
            role: 'assistant',
            tool_calls: [
              {
                id: 'web-call',
                type: 'function',
                function: {
                  name: 'read_web_page',
                  arguments: 'MALFORMED_WEB_ARGUMENT',
                },
              },
            ],
          },
          {
            role: 'tool',
            tool_call_id: 'web-call',
            content: 'MALFORMED_WEB_RESULT',
          },
        ],
      },
      { choices: [] },
      new Set()
    );

    expect(JSON.stringify(projected)).not.toContain('MALFORMED_WEB');
  });
});
