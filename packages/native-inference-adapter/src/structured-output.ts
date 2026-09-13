import { Ajv } from 'ajv';

import { ModelGenerationError } from '@echo-chamber/core/ports/model';
import type {
  ModelStructuredOutputFormat,
  ModelOutputItem,
  ModelUsage,
} from '@echo-chamber/core/ports/model';

/** A completed Native response violated the engine or transport output invariant. */
export class NativeStructuredOutputError extends ModelGenerationError {
  /**
   * @param code Failed output rule
   * @param usage Completed generation usage
   * @param responseToken Committed continuation capability
   */
  constructor(
    readonly code: 'unexpected_output' | 'invalid_json' | 'schema_mismatch',
    usage: ModelUsage,
    readonly responseToken: string
  ) {
    super(`native structured output failed: ${code}`, usage);
    this.name = 'NativeStructuredOutputError';
  }
}

/** Request-owned execution metadata and a final protocol-invariant validator. */
export interface NativeStructuredOutputContract {
  format: ModelStructuredOutputFormat;
  validate(
    output: readonly ModelOutputItem[],
    usage: ModelUsage,
    responseToken: string
  ): void;
}

/** Compile synchronously before dispatch; unsupported schema rules are errors, never ignored. */
export function prepareStructuredOutput(
  format: unknown
): NativeStructuredOutputContract | undefined {
  if (format === undefined) return undefined;
  if (
    !isRecord(format) ||
    format.type !== 'json_schema' ||
    format.strict !== true ||
    typeof format.name !== 'string' ||
    format.name.trim() === ''
  ) {
    throw new Error(
      'native responseFormat requires a named strict json_schema contract'
    );
  }
  const serialized = JSON.stringify(format, (_key, value: unknown): unknown => {
    if (
      value === undefined ||
      typeof value === 'function' ||
      typeof value === 'symbol' ||
      typeof value === 'bigint' ||
      (typeof value === 'number' && !Number.isFinite(value))
    ) {
      throw new Error('native responseFormat must contain only JSON values');
    }
    return value;
  });
  const snapshot = JSON.parse(serialized) as Record<string, unknown>;
  if (!isRecord(snapshot.schema) || '$async' in snapshot.schema) {
    throw new Error(
      'native responseFormat requires a synchronous JSON Schema object'
    );
  }
  // A fresh compiler scopes refs and IDs to this request and avoids retaining
  // every caller-created schema in a process-wide cache. No remote ref loading,
  // value coercion, property removal, or default insertion is enabled.
  const validate = new Ajv({ strict: true }).compile(snapshot.schema);
  return {
    format: snapshot as unknown as ModelStructuredOutputFormat,
    validate(output, usage, responseToken): void {
      if (output.length !== 1 || output[0]?.type !== 'message') {
        throw new NativeStructuredOutputError(
          'unexpected_output',
          usage,
          responseToken
        );
      }
      let value: unknown;
      try {
        value = JSON.parse(output[0].content) as unknown;
      } catch {
        throw new NativeStructuredOutputError(
          'invalid_json',
          usage,
          responseToken
        );
      }
      if (!validate(value))
        throw new NativeStructuredOutputError(
          'schema_mismatch',
          usage,
          responseToken
        );
    },
  };
}

/** Require JSON objects at the schema admission boundary. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
