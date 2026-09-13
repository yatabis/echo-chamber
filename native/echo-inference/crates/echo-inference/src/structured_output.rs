use std::path::Path;

use echo_mlx::{Array, Gpu};
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{SimpleVob, TokTrie};
use llguidance::{Matcher, ParserFactory};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use toktrie_hf_tokenizers::{ByteTokenizer, ByteTokenizerEnv};

use super::EngineError;
use super::runtime::RuntimeError;

/// Output constraints transported independently of the conversation prompt.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredOutputFormat {
    /// Only strict JSON Schema generation is admitted.
    #[serde(rename = "type")]
    pub kind: String,
    /// Caller-owned schema identity; never rendered into the prompt.
    pub name: String,
    /// Must be true; approximate schema enforcement is not supported.
    pub strict: bool,
    /// JSON Schema compiled before any model execution for the request.
    pub schema: Value,
}

/// Shared immutable tokenizer infrastructure; matchers remain request-owned.
pub(crate) struct OutputGrammarCompiler {
    factory: ParserFactory,
}

impl OutputGrammarCompiler {
    pub(crate) fn load(
        directory: &Path,
        eos_token: u32,
        vocabulary_size: usize,
    ) -> Result<Self, RuntimeError> {
        // The bridge preserves byte fragments, including tokens ending inside
        // UTF-8 characters. Decoding each vocabulary token as a String would
        // corrupt the language recognized by the grammar.
        let tokenizer =
            ByteTokenizer::from_file(directory.join("tokenizer.json")).map_err(constraint_error)?;
        Self::from_tokenizer(tokenizer, eos_token, vocabulary_size)
    }

    fn from_tokenizer(
        mut tokenizer: ByteTokenizer,
        eos_token: u32,
        vocabulary_size: usize,
    ) -> Result<Self, RuntimeError> {
        // Forced JSON bytes are literal content. HF's default encoding turns
        // text such as "<tool_call>" into a control token outside the grammar.
        // This affects only the grammar tokenizer, never chat-template input.
        tokenizer.hf_tokenizer.set_encode_special_tokens(true);
        tokenizer.set_eos_token(eos_token);
        let mut environment =
            ByteTokenizerEnv::new(tokenizer, Some(vocabulary_size)).map_err(constraint_error)?;
        let mut bytes = (0..vocabulary_size)
            .map(|id| {
                let token = u32::try_from(id).map_err(constraint_error)?;
                Ok(environment.tok_trie.token(token).to_vec())
            })
            .collect::<Result<Vec<_>, RuntimeError>>()?;
        // The bridge guesses that every added <...> token is special. Qwen
        // explicitly marks tool delimiters as ordinary tokens. Honor the
        // tokenizer's declaration so forced bytes, masks and output decoding
        // agree on the very same vocabulary token.
        for (id, added) in environment
            .tokenizer
            .hf_tokenizer
            .get_added_tokens_decoder()
        {
            if !added.special {
                bytes[usize::try_from(id).map_err(constraint_error)?] = added.content.into_bytes();
            }
        }
        environment.tok_trie = TokTrie::from(environment.tok_trie.info(), &bytes);
        let mut factory =
            ParserFactory::new_simple(&environment.to_env()).map_err(constraint_error)?;
        factory.quiet();
        Ok(Self { factory })
    }

    pub(crate) fn compile(
        &self,
        format: &StructuredOutputFormat,
    ) -> Result<OutputConstraint, RuntimeError> {
        if format.kind != "json_schema" || !format.strict || format.name.trim().is_empty() {
            return Err(constraint_error(
                "expected a named strict json_schema format",
            ));
        }
        admit_schema(&format.schema)?;
        let parser = self
            .factory
            .create_parser(TopLevelGrammar::from_json_schema(format.schema.clone()))
            .map_err(constraint_error)?;
        let mut matcher = Matcher::new(Ok(parser));
        if let Some(error) = matcher.get_error() {
            return Err(constraint_error(error));
        }
        let warnings = matcher.grammar_warnings();
        if !warnings.is_empty() {
            return Err(constraint_error(warnings.join("; ")));
        }
        let mut constraint = OutputConstraint {
            matcher,
            eos_token: self.factory.tok_env().eos_token(),
        };
        // Empty languages and compiler limits fail before prefill, never by
        // switching the request back to unconstrained sampling.
        constraint.mask()?;
        Ok(constraint)
    }
}

/// Prefix recognizer discarded with its generation transaction on every exit.
pub(crate) struct OutputConstraint {
    matcher: Matcher,
    eos_token: u32,
}

impl OutputConstraint {
    fn mask(&mut self) -> Result<SimpleVob, RuntimeError> {
        if self.matcher.is_stopped()
            && (!self.matcher.stop_reason().is_ok()
                || !self.matcher.is_accepting().map_err(constraint_error)?)
        {
            return Err(constraint_error(
                "grammar stopped before a complete JSON value",
            ));
        }
        let mask = self
            .matcher
            .compute_mask_or_eos()
            .map_err(constraint_error)?;
        if mask.num_set() == 0 {
            return Err(constraint_error("grammar has no admissible next token"));
        }
        Ok(mask)
    }

    /// Advance only the prefix actually sampled and return whether EOS closed it.
    pub(crate) fn consume(&mut self, token: u32) -> Result<bool, RuntimeError> {
        if token == self.eos_token {
            if !self.matcher.is_accepting().map_err(constraint_error)? {
                return Err(constraint_error("EOS preceded a complete JSON value"));
            }
            return Ok(true);
        }
        self.matcher
            .consume_token(token)
            .map_err(constraint_error)?;
        Ok(false)
    }
}

/// Apply per-row masks before greedy selection or any probabilistic filtering.
pub(crate) fn mask_output_logits<'a>(
    gpu: &Gpu,
    logits: &Array,
    constraints: impl IntoIterator<Item = &'a mut Option<OutputConstraint>>,
    vocabulary_size: usize,
) -> Result<Array, RuntimeError> {
    let constraints = constraints.into_iter().collect::<Vec<_>>();
    if constraints.iter().all(|constraint| constraint.is_none()) {
        return logits
            .try_clone()
            .map_err(EngineError::Mlx)
            .map_err(Into::into);
    }
    let mut allowed = vec![true; constraints.len() * vocabulary_size];
    for (row, constraint) in constraints.into_iter().enumerate() {
        if let Some(constraint) = constraint {
            let mask = constraint.mask()?;
            let row_mask = &mut allowed[row * vocabulary_size..(row + 1) * vocabulary_size];
            row_mask.fill(false);
            mask.iter_set_entries(|token| {
                if token < vocabulary_size {
                    row_mask[token] = true;
                }
            });
        }
    }
    let mask = Array::from_bool_slice(
        &allowed,
        &[allowed.len() / vocabulary_size, 1, vocabulary_size],
    )
    .map_err(EngineError::Mlx)?;
    let excluded = Array::from_f32_slice(&[f32::NEG_INFINITY], &[]).map_err(EngineError::Mlx)?;
    // A constrained sibling must not change the dtype of an unconstrained row.
    let excluded = gpu
        .astype(&excluded, logits.dtype())
        .map_err(EngineError::Mlx)?;
    gpu.where_condition(&mask, logits, &excluded)
        .map_err(EngineError::Mlx)
        .map_err(Into::into)
}

/// Admit an explicit dialect so unknown rules or compiler extension switches
/// cannot silently weaken the caller's schema. The compiler checks rule values.
fn admit_schema(schema: &Value) -> Result<(), RuntimeError> {
    if schema.is_boolean() {
        return Ok(());
    }
    let object = schema
        .as_object()
        .ok_or_else(|| constraint_error("invalid schema node"))?;
    for (key, value) in object {
        match key.as_str() {
            "$schema" if value == "http://json-schema.org/draft-07/schema#" => {}
            "$ref"
                if value
                    .as_str()
                    .is_some_and(|reference| reference.starts_with('#')) => {}
            "properties" | "definitions" | "$defs" => {
                for child in value
                    .as_object()
                    .ok_or_else(|| constraint_error("invalid schema map"))?
                    .values()
                {
                    admit_schema(child)?;
                }
            }
            "items" | "additionalProperties" => admit_schema(value)?,
            "anyOf" | "allOf" => {
                for child in value
                    .as_array()
                    .ok_or_else(|| constraint_error("invalid schema alternatives"))?
                {
                    admit_schema(child)?;
                }
            }
            "type" | "required" | "enum" | "const" | "minimum" | "maximum" | "exclusiveMinimum"
            | "exclusiveMaximum" | "multipleOf" | "minLength" | "maxLength" | "pattern"
            | "minItems" | "maxItems" | "title" | "description" | "$comment" | "default"
            | "examples" => {}
            _ => {
                return Err(constraint_error(format!(
                    "unsupported schema keyword: {key}"
                )));
            }
        }
    }
    Ok(())
}

fn constraint_error(error: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::InvalidRequest {
        detail: format!("structured output constraint: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use llguidance::toktrie::ApproximateTokEnv;
    use serde_json::json;

    use super::*;

    fn compiler() -> OutputGrammarCompiler {
        OutputGrammarCompiler {
            factory: ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env()).unwrap(),
        }
    }

    fn format(schema: Value) -> StructuredOutputFormat {
        StructuredOutputFormat {
            kind: "json_schema".into(),
            name: "test".into(),
            strict: true,
            schema,
        }
    }

    fn consume_text(constraint: &mut OutputConstraint, text: &str) {
        for byte in text.bytes() {
            assert!(
                constraint.mask().unwrap().is_allowed(u32::from(byte)),
                "rejected byte {byte} in {text}"
            );
            assert!(!constraint.consume(u32::from(byte)).unwrap());
        }
    }

    #[test]
    fn masks_forbid_wrong_types_extra_keys_and_early_eos() {
        let mut constraint = compiler()
            .compile(&format(json!({
                "type":"object", "properties":{"answer":{"type":"integer","const":7}},
                "required":["answer"], "additionalProperties":false
            })))
            .unwrap();
        assert!(!constraint.mask().unwrap().is_allowed(constraint.eos_token));
        assert!(!constraint.mask().unwrap().is_allowed(u32::from(b'N')));
        consume_text(&mut constraint, "{\"answer\":");
        for forbidden in [b'\"', b'8', b'n'] {
            assert!(!constraint.mask().unwrap().is_allowed(u32::from(forbidden)));
        }
        consume_text(&mut constraint, "7");
        assert!(!constraint.mask().unwrap().is_allowed(u32::from(b',')));
        consume_text(&mut constraint, "}");
        assert!(constraint.mask().unwrap().is_allowed(constraint.eos_token));
        assert!(constraint.consume(constraint.eos_token).unwrap());
    }

    #[test]
    fn literal_added_tokens_and_control_token_spellings_match_the_decoded_bytes() {
        let mut vocab = (33_u8..=126)
            .enumerate()
            .map(|(id, byte)| (char::from(byte).to_string(), json!(id)))
            .collect::<serde_json::Map<_, _>>();
        vocab.insert("\u{120}".into(), json!(94));
        let tokenizer_json = json!({
            "version":"1.0", "truncation":null, "padding":null,
            "added_tokens":[
                {"id":95,"content":"<tool_call>","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},
                {"id":96,"content":"<|im_end|>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}
            ],
            "normalizer":null,"post_processor":null,
            "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true},
            "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true},
            "model":{"type":"BPE","vocab":vocab,"merges":[]}
        });
        let tokenizer =
            ByteTokenizer::from_json_bytes(&serde_json::to_vec(&tokenizer_json).unwrap()).unwrap();
        let compiler = OutputGrammarCompiler::from_tokenizer(tokenizer, 96, 97).unwrap();
        let text = "\"<tool_call><|im_end|>\"";
        let tokens = compiler.factory.tok_env().tokenize(text);
        assert!(tokens.contains(&95));
        assert!(
            !tokens.contains(&96),
            "quoted EOS spelling is literal content"
        );
        let mut constraint = compiler
            .compile(&format(json!({"const":"<tool_call><|im_end|>"})))
            .unwrap();
        for token in tokens {
            assert!(constraint.mask().unwrap().is_allowed(token));
            assert!(!constraint.consume(token).unwrap());
        }
        assert!(constraint.mask().unwrap().is_allowed(96));
        assert!(constraint.consume(96).unwrap());
    }

    #[test]
    fn unicode_lengths_and_numeric_bounds_are_generation_constraints() {
        let compiler = compiler();
        let mut text = compiler
            .compile(&format(
                json!({"type":"string","minLength":1,"maxLength":1}),
            ))
            .unwrap();
        consume_text(&mut text, "\"猫");
        assert!(!text.mask().unwrap().is_allowed(u32::from(b'a')));
        consume_text(&mut text, "\"");
        let mut number = compiler
            .compile(&format(json!({"type":"number","minimum":0,"maximum":1})))
            .unwrap();
        assert!(!number.mask().unwrap().is_allowed(u32::from(b'2')));
        consume_text(&mut number, "0.5");
        assert!(number.mask().unwrap().is_allowed(number.eos_token));
    }

    #[test]
    fn request_matchers_do_not_share_prefix_state() {
        let compiler = compiler();
        let format = format(json!({"type":"string","enum":["ab","xy"]}));
        let mut first = compiler.compile(&format).unwrap();
        let mut second = compiler.compile(&format).unwrap();
        consume_text(&mut first, "\"a");
        consume_text(&mut second, "\"x");
        assert!(!first.mask().unwrap().is_allowed(u32::from(b'y')));
        assert!(!second.mask().unwrap().is_allowed(u32::from(b'b')));
        consume_text(&mut first, "b\"");
        consume_text(&mut second, "y\"");
    }

    #[test]
    fn masks_precede_greedy_and_top_k_selection_in_mixed_batches() {
        use crate::sampling::{SamplingConfig, sample_token_rows};

        let compiler = compiler();
        let vocabulary_size = compiler.factory.tok_env().tok_trie().vocab_size();
        let gpu = Gpu::new();
        let mut values = vec![0.0; 3 * vocabulary_size];
        for row in 0..3 {
            for (byte, score) in [(b'9', 100.0), (b'7', 10.0), (b'8', 9.0)] {
                values[row * vocabulary_size + usize::from(byte)] = score;
            }
        }
        let logits = Array::from_f32_slice(&values, &[3, 1, vocabulary_size]).unwrap();
        for temperature in [0.0, 0.7] {
            let mut constraints = [
                Some(compiler.compile(&format(json!({"const":7}))).unwrap()),
                Some(compiler.compile(&format(json!({"const":8}))).unwrap()),
                None,
            ];
            let masked =
                mask_output_logits(&gpu, &logits, &mut constraints, vocabulary_size).unwrap();
            let sampling = SamplingConfig {
                temperature,
                top_k: 1,
                ..SamplingConfig::default()
            };
            let sampled = sample_token_rows(
                &gpu,
                &masked,
                &[vec![], vec![], vec![]],
                &[sampling; 3],
                vocabulary_size,
            )
            .unwrap();
            for (row, expected) in [b'7', b'8', b'9'].into_iter().enumerate() {
                let token = gpu
                    .slice(
                        &sampled,
                        &[i32::try_from(row).unwrap(), 0],
                        &[i32::try_from(row + 1).unwrap(), 1],
                        &[1, 1],
                    )
                    .unwrap();
                assert_eq!(
                    gpu.reshape(&token, &[]).unwrap().item_u32().unwrap(),
                    u32::from(expected)
                );
            }
        }
    }

    #[test]
    fn unsupported_or_relaxed_schemas_are_rejected_before_generation() {
        for schema in [
            json!({"type":"string","format":"email"}),
            json!({"type":"array","uniqueItems":true}),
            json!({"type":"string","x-guidance":{"lenient":true}}),
            json!({"oneOf":[{"type":"number"},{"type":"integer"}]}),
            json!({"$ref":"https://example.com/schema"}),
            json!({"properties":{"nested":{"unknownRule":true}}}),
            json!({"type":"integer","minimum":2,"maximum":1}),
        ] {
            assert!(
                compiler().compile(&format(schema.clone())).is_err(),
                "admitted {schema}"
            );
        }
    }
}
