use std::path::Path;
use std::time::Duration;

use echo_inference_state::InstanceId;
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::EngineError;
use super::chat::{
    EchoChatPrompt, EchoInputItem, EchoMessage, EchoMessageContent, EchoMessageRole,
    Qwen35ChatTokenizer,
};
use super::runtime::{
    InferenceRequest, RequestState, ResidentEngine, ResidentEngineConfig, RuntimeMetrics,
};
use super::sampling::SamplingConfig;

const PERFORMANCE_MODE_ENVIRONMENT: &str = "ECHO_MOE_PERFORMANCE_MODE";
const BENCHMARK_PROMPT: &str = concat!(
    "これは推論性能の測定です。ツールは使わず、次の形式だけを出力してください。\n",
    "1から400までの整数を昇順に、それぞれ独立した行へ書いてください。\n",
    "各行は「0001: native-rapid-performance」のように、4桁ゼロ埋めの番号、コロン、半角空白、固定文字列 native-rapid-performance の順にしてください。\n",
    "前置き、説明、後書き、省略記号は加えないでください。"
);

/// One fixed-length real-model attempt in the feature-gated `MoE` diagnostic.
#[derive(Clone, Debug, Serialize)]
pub struct MoePerformanceAttempt {
    pub phase: &'static str,
    pub index: usize,
    pub generated_tokens: usize,
    pub output_token_sha256: String,
    pub decode_nanos_per_token: f64,
    pub metrics: RuntimeMetrics,
}

/// Median timings for the measured attempts in one diagnostic process.
#[derive(Clone, Debug, Serialize)]
pub struct MoePerformanceSummary {
    pub count: usize,
    pub median_decode_nanos_per_token: f64,
    pub median_graph_construction_nanos_per_token: f64,
    pub median_schedule_nanos_per_token: f64,
    pub median_token_wait_nanos_per_token: f64,
    pub median_request_nanos: f64,
}

/// Feature-gated invalid-output profile used to locate current `MoE` costs.
#[derive(Clone, Debug, Serialize)]
pub struct MoePerformanceDiagnostic {
    pub schema_version: u32,
    pub mode: String,
    pub prompt_tokens: usize,
    pub max_new_tokens: usize,
    pub warmup_runs: usize,
    pub measured_runs: usize,
    pub attempts: Vec<MoePerformanceAttempt>,
    pub summary: MoePerformanceSummary,
}

/// Runs one resident process for a fixed number of full-length decode attempts.
///
/// The Cargo feature and execution mode make this an explicit invalid-output
/// component diagnostic. Ordinary builds contain neither its CLI command nor
/// its runtime `MoE` branches.
///
/// # Errors
///
/// Returns [`EngineError`] when the mode, tokenizer, model, request, or metric
/// denominator is invalid.
pub fn run_moe_performance_diagnostic(
    model_directory: &Path,
    warmup_runs: usize,
    measured_runs: usize,
    max_new_tokens: usize,
) -> Result<MoePerformanceDiagnostic, EngineError> {
    if measured_runs == 0 || max_new_tokens == 0 {
        return Err(EngineError::Unsupported(
            "MoE performance diagnostic requires measured runs and generated tokens".into(),
        ));
    }
    let mode = std::env::var(PERFORMANCE_MODE_ENVIRONMENT).unwrap_or_else(|_| "full".into());
    let tokenizer = Qwen35ChatTokenizer::load(model_directory)
        .map_err(|error| EngineError::Unsupported(format!("load diagnostic tokenizer: {error}")))?;
    let prompt = tokenizer
        .encode_prompt(&EchoChatPrompt {
            input: vec![EchoInputItem::Message(EchoMessage {
                role: EchoMessageRole::User,
                content: EchoMessageContent::Text(BENCHMARK_PROMPT.into()),
            })],
            tools: Vec::new(),
        })
        .map_err(|error| EngineError::Unsupported(format!("encode diagnostic prompt: {error}")))?;
    let mut engine = ResidentEngine::load(
        model_directory,
        ResidentEngineConfig {
            max_new_tokens_per_request: max_new_tokens,
            ..ResidentEngineConfig::default()
        },
    )
    .map_err(|error| EngineError::Unsupported(format!("load diagnostic engine: {error}")))?;
    let mut attempts = Vec::with_capacity(warmup_runs.saturating_add(measured_runs));
    for ordinal in 0..warmup_runs.saturating_add(measured_runs) {
        let measured = ordinal >= warmup_runs;
        let index = if measured {
            ordinal - warmup_runs + 1
        } else {
            ordinal + 1
        };
        let response = engine
            .execute(InferenceRequest {
                instance_id: InstanceId::new(format!(
                    "moe-diagnostic-{mode}-{}-{index}",
                    if measured { "measured" } else { "warmup" }
                ))
                .map_err(|error| {
                    EngineError::Unsupported(format!("create diagnostic instance: {error}"))
                })?,
                state_transition: RequestState::Initial,
                input_tokens: prompt.token_ids.clone(),
                max_new_tokens,
                length_eos_token: None,
                sampling: SamplingConfig {
                    temperature: 0.0,
                    top_p: 1.0,
                    top_k: 0,
                    min_p: 0.0,
                    repetition_penalty: 1.0,
                    presence_penalty: 0.0,
                    seed: 42,
                },
            })
            .map_err(|error| {
                EngineError::Unsupported(format!("execute diagnostic request: {error}"))
            })?;
        let generated_tokens = response.generated_tokens.len();
        let generated_tokens_u32 = u32::try_from(generated_tokens).map_err(|error| {
            EngineError::Unsupported(format!("diagnostic token denominator overflow: {error}"))
        })?;
        if generated_tokens_u32 == 0 {
            return Err(EngineError::Unsupported(
                "MoE performance diagnostic generated no tokens".into(),
            ));
        }
        let decode_nanos_per_token =
            nanos_as_f64(response.metrics.decode_execution_nanos) / f64::from(generated_tokens_u32);
        attempts.push(MoePerformanceAttempt {
            phase: if measured { "measured" } else { "warmup" },
            index,
            generated_tokens,
            output_token_sha256: token_digest(&response.generated_tokens),
            decode_nanos_per_token,
            metrics: response.metrics,
        });
    }
    let summary = summarize_attempts(&attempts)?;
    Ok(MoePerformanceDiagnostic {
        schema_version: 1,
        mode,
        prompt_tokens: prompt.token_ids.len(),
        max_new_tokens,
        warmup_runs,
        measured_runs,
        attempts,
        summary,
    })
}

fn summarize_attempts(
    attempts: &[MoePerformanceAttempt],
) -> Result<MoePerformanceSummary, EngineError> {
    let measured = attempts
        .iter()
        .filter(|attempt| attempt.phase == "measured")
        .collect::<Vec<_>>();
    let per_token = |value: fn(&RuntimeMetrics) -> u64| -> Result<Vec<f64>, EngineError> {
        measured
            .iter()
            .map(|attempt| {
                let tokens = u32::try_from(attempt.generated_tokens).map_err(|error| {
                    EngineError::Unsupported(format!(
                        "diagnostic summary token denominator overflow: {error}"
                    ))
                })?;
                Ok(nanos_as_f64(value(&attempt.metrics)) / f64::from(tokens))
            })
            .collect()
    };
    Ok(MoePerformanceSummary {
        count: measured.len(),
        median_decode_nanos_per_token: median(
            measured
                .iter()
                .map(|attempt| attempt.decode_nanos_per_token)
                .collect(),
        )?,
        median_graph_construction_nanos_per_token: median(per_token(|metrics| {
            metrics.decode_graph_construction_nanos
        })?)?,
        median_schedule_nanos_per_token: median(per_token(|metrics| {
            metrics.decode_schedule_nanos
        })?)?,
        median_token_wait_nanos_per_token: median(per_token(|metrics| {
            metrics.decode_token_wait_nanos
        })?)?,
        median_request_nanos: median(
            measured
                .iter()
                .map(|attempt| nanos_as_f64(attempt.metrics.request_nanos))
                .collect(),
        )?,
    })
}

fn nanos_as_f64(nanos: u64) -> f64 {
    Duration::from_nanos(nanos).as_secs_f64() * 1_000_000_000.0
}

fn token_digest(tokens: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn median(mut values: Vec<f64>) -> Result<f64, EngineError> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::Unsupported(
            "MoE performance diagnostic median requires finite samples".into(),
        ));
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        Ok(values[middle - 1].midpoint(values[middle]))
    } else {
        Ok(values[middle])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)]
    fn median_handles_odd_and_even_sample_counts() {
        assert_eq!(median(vec![3.0, 1.0, 2.0]).expect("odd median"), 2.0);
        assert_eq!(median(vec![4.0, 1.0, 3.0, 2.0]).expect("even median"), 2.5);
    }
}
