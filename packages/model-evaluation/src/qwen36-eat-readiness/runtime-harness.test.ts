import { describe, expect, it } from 'vitest';

import { createEchoEvent } from '@echo-chamber/core/ports/echo-event';

import { EvaluationTrace } from './runtime-harness';

describe('EvaluationTrace', () => {
  it('retains bounded native inference metrics on exchange events', async () => {
    const trace = new EvaluationTrace(performance.now());
    const payload = {
      provider: 'echo.native_inference',
      instanceId: 'rin-workflow',
      turnIndex: 2,
      stateSequenceLength: 8_245,
      inputItemCount: 1,
      outputItemCount: 1,
      finishReason: 'stop_token',
      metrics: {
        cached_prefix_tokens: 8_192,
        input_tokens_processed: 53,
        generated_tokens: 24,
        input_model_execution_count: 1,
      },
    };

    await trace.emit(
      createEchoEvent({
        type: 'model.exchange.recorded',
        severity: 'debug',
        summary: 'native exchange',
        payload,
      })
    );

    expect(trace.events).toHaveLength(1);
    expect(trace.events[0]?.payload).toEqual(payload);
  });

  it('still removes unbounded fields from OpenAI-compatible responses', async () => {
    const trace = new EvaluationTrace(performance.now());

    await trace.emit(
      createEchoEvent({
        type: 'model.exchange.recorded',
        severity: 'debug',
        summary: 'OpenAI-compatible exchange',
        payload: {
          provider: 'openai-compatible',
          model: 'fixture',
          turnIndex: 0,
          response: {
            id: 'response-1',
            model: 'fixture',
            usage: { prompt_tokens: 12, completion_tokens: 3 },
            choices: [
              {
                index: 0,
                finish_reason: 'stop',
                message: { role: 'assistant', content: 'bounded' },
                logprobs: { large: 'discard me' },
              },
            ],
            providerSpecificLargeField: 'discard me',
          },
        },
      })
    );

    expect(trace.events[0]?.payload).toEqual({
      provider: 'openai-compatible',
      model: 'fixture',
      turnIndex: 0,
      response: {
        id: 'response-1',
        model: 'fixture',
        usage: { prompt_tokens: 12, completion_tokens: 3 },
        choices: [
          {
            index: 0,
            finish_reason: 'stop',
            message: { role: 'assistant', content: 'bounded' },
          },
        ],
      },
    });
  });
});
