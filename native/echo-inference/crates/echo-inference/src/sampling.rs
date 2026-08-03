use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use echo_mlx::{Array, Gpu};
use serde::{Deserialize, Serialize};

use super::EngineError;

// Rapid-MLX applies OpenAI-style presence penalty to generated output only and
// extends the MLX-LM default window to cover ordinary chat responses.
const PRESENCE_CONTEXT_SIZE: usize = 4_096;
const PRESENCE_HISTORY_SCOPE: &str = "generated_output";

#[derive(Debug, Deserialize)]
struct SamplingParityFixture {
    schema_version: u32,
    history_scope: String,
    presence_context_size: usize,
    logits: Vec<f32>,
    history_tokens: Vec<u32>,
    config: SamplingConfig,
    cases: Vec<SamplingParityFixtureCase>,
}

#[derive(Debug, Deserialize)]
struct SamplingParityFixtureCase {
    seed: u64,
    expected_token: u32,
}

/// One deterministic functional-key sample comparison.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SamplingParityCase {
    /// Functional MLX key seed.
    pub seed: u64,
    /// Token produced by the official Python/MLX oracle.
    pub expected_token: u32,
    /// Token produced through MLX C from Rust.
    pub actual_token: u32,
    /// Exact token equality.
    pub exact: bool,
}

/// Aggregate official Python/MLX versus native Rust production-sampler result.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SamplingParity {
    /// Fixture schema.
    pub schema_version: u32,
    /// Admitted generation profile.
    pub config: SamplingConfig,
    /// Rapid-MLX-compatible token history admitted by the fixture.
    pub history_scope: String,
    /// Rapid-MLX OpenAI-style presence window.
    pub presence_context_size: usize,
    /// Vocabulary width of the deterministic logits.
    pub vocabulary_size: usize,
    /// Number of previous tokens supplied to presence penalty.
    pub history_token_count: usize,
    /// Seed-by-seed comparisons.
    pub cases: Vec<SamplingParityCase>,
    /// `true` only when every sampled token matches.
    pub exact: bool,
}

/// Request-owned generation profile.
///
/// The native engine deliberately implements only the controls used by
/// E.C.H.O.'s production profile. `min_p` and repetition penalty remain on the
/// wire for auditability, but non-neutral values fail admission until they are
/// required and parity-tested.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct SamplingConfig {
    /// Zero selects greedy argmax. Positive values enable categorical sampling.
    pub temperature: f32,
    /// Nucleus threshold. Zero or one disables nucleus filtering.
    pub top_p: f32,
    /// Maximum number of candidate tokens. Zero disables top-k filtering.
    pub top_k: usize,
    /// Minimum-probability filter. Only the production-neutral zero is admitted.
    pub min_p: f32,
    /// Sign-aware repetition factor. Only the production-neutral one is admitted.
    pub repetition_penalty: f32,
    /// Additive penalty for tokens present in the generated-output context.
    pub presence_penalty: f32,
    /// Functional MLX random-key seed owned by this request.
    pub seed: u64,
}

impl SamplingConfig {
    /// Returns E.C.H.O.'s current Qwen non-thinking production profile.
    #[must_use]
    pub const fn echo_production(seed: u64) -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 1.5,
            seed,
        }
    }

    /// Validates this profile against the admitted model vocabulary.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] for non-finite or unsupported settings.
    pub fn validate(self, vocabulary_size: usize) -> Result<(), EngineError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(invalid_sampling(
                "temperature must be finite and greater than or equal to zero",
            ));
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            return Err(invalid_sampling("top_p must be finite and within 0..=1"));
        }
        if self.top_k >= vocabulary_size && self.top_k != 0 {
            return Err(invalid_sampling(&format!(
                "top_k must be zero or less than vocabulary size {vocabulary_size}"
            )));
        }
        if self.min_p.to_bits() != 0.0_f32.to_bits() {
            return Err(invalid_sampling(
                "non-zero min_p is outside the admitted E.C.H.O. production profile",
            ));
        }
        if self.repetition_penalty.to_bits() != 1.0_f32.to_bits() {
            return Err(invalid_sampling(
                "repetition_penalty other than 1 is outside the admitted E.C.H.O. production profile",
            ));
        }
        if !self.presence_penalty.is_finite() {
            return Err(invalid_sampling("presence_penalty must be finite"));
        }
        Ok(())
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 0.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            seed: 0,
        }
    }
}

/// Runs a deterministic production-sampler fixture produced by Python/MLX.
///
/// # Errors
///
/// Returns [`EngineError`] when the fixture is unreadable or invalid, or when
/// an MLX operation cannot be evaluated.
pub fn run_sampling_parity(fixture_path: &Path) -> Result<SamplingParity, EngineError> {
    let bytes = fs::read(fixture_path).map_err(|error| {
        EngineError::Unsupported(format!(
            "sampling fixture {}: {error}",
            fixture_path.display()
        ))
    })?;
    let fixture: SamplingParityFixture = serde_json::from_slice(&bytes).map_err(|error| {
        EngineError::Unsupported(format!(
            "sampling fixture {}: {error}",
            fixture_path.display()
        ))
    })?;
    if fixture.schema_version != 1 {
        return Err(EngineError::Unsupported(format!(
            "sampling fixture schema must be 1, observed {}",
            fixture.schema_version
        )));
    }
    if fixture.history_scope != PRESENCE_HISTORY_SCOPE
        || fixture.presence_context_size != PRESENCE_CONTEXT_SIZE
    {
        return Err(EngineError::Unsupported(format!(
            "sampling fixture must use {PRESENCE_HISTORY_SCOPE} history with a {PRESENCE_CONTEXT_SIZE}-token presence window"
        )));
    }
    if fixture.logits.is_empty() {
        return Err(EngineError::Unsupported(
            "sampling fixture logits must not be empty".into(),
        ));
    }
    let vocabulary_size = fixture.logits.len();
    fixture.config.validate(vocabulary_size)?;
    if fixture
        .history_tokens
        .iter()
        .any(|token| usize::try_from(*token).map_or(true, |token| token >= vocabulary_size))
    {
        return Err(EngineError::Unsupported(
            "sampling fixture history contains an out-of-vocabulary token".into(),
        ));
    }

    let gpu = Gpu::new();
    let logits = Array::from_f32_slice(&fixture.logits, &[1, 1, vocabulary_size])
        .map_err(EngineError::Mlx)?;
    let cases = fixture
        .cases
        .into_iter()
        .map(|case| {
            let token = sample_token(
                &gpu,
                &logits,
                &fixture.history_tokens,
                0,
                SamplingConfig {
                    seed: case.seed,
                    ..fixture.config
                },
                vocabulary_size,
            )?;
            gpu.eval(&[&token]).map_err(EngineError::Mlx)?;
            let actual_token = gpu
                .reshape(&token, &[])
                .and_then(|token| token.item_u32())
                .map_err(EngineError::Mlx)?;
            Ok(SamplingParityCase {
                seed: case.seed,
                expected_token: case.expected_token,
                actual_token,
                exact: actual_token == case.expected_token,
            })
        })
        .collect::<Result<Vec<_>, EngineError>>()?;
    let exact = cases.iter().all(|case| case.exact);
    Ok(SamplingParity {
        schema_version: fixture.schema_version,
        config: fixture.config,
        history_scope: fixture.history_scope,
        presence_context_size: fixture.presence_context_size,
        vocabulary_size,
        history_token_count: fixture.history_tokens.len(),
        cases,
        exact,
    })
}

/// Samples one token with the admitted MLX-LM operation order.
///
/// Presence penalty is applied to raw logits using generated output only, then
/// log probabilities are formed, followed by top-p, top-k, temperature
/// scaling, and categorical sampling. A request-owned functional key prevents
/// one instance from perturbing another instance's random sequence.
pub(crate) fn sample_token(
    gpu: &Gpu,
    logits: &Array,
    history_tokens: &[u32],
    generated_index: usize,
    config: SamplingConfig,
    vocabulary_size: usize,
) -> Result<Array, EngineError> {
    config.validate(vocabulary_size)?;
    let logits = last_logits(gpu, logits, vocabulary_size)?;
    let logits = apply_presence_penalty(gpu, &logits, history_tokens, config.presence_penalty)?;
    if config.temperature == 0.0 {
        return gpu
            .argmax_axis(&logits, -1, true)
            .and_then(|token| gpu.reshape(&token, &[1, 1]))
            .map_err(EngineError::Mlx);
    }

    let normalizer = gpu
        .logsumexp_axis(&logits, -1, true)
        .map_err(EngineError::Mlx)?;
    let mut logprobs = gpu
        .subtract(&logits, &normalizer)
        .map_err(EngineError::Mlx)?;
    if config.top_p > 0.0 && config.top_p < 1.0 {
        logprobs = apply_top_p(gpu, &logprobs, config.top_p)?;
    }
    if config.top_k > 0 {
        logprobs = apply_top_k(gpu, &logprobs, config.top_k)?;
    }
    let inverse_temperature = gpu
        .scalar_like(1.0 / config.temperature, logprobs.dtype())
        .map_err(EngineError::Mlx)?;
    let sampling_logits = gpu
        .multiply(&logprobs, &inverse_temperature)
        .map_err(EngineError::Mlx)?;
    let step = u64::try_from(generated_index)
        .map_err(|error| invalid_sampling(&format!("generated token index overflow: {error}")))?;
    gpu.categorical_with_seed(&sampling_logits, -1, config.seed.wrapping_add(step))
        .and_then(|token| gpu.reshape(&token, &[1, 1]))
        .map_err(EngineError::Mlx)
}

fn last_logits(
    gpu: &Gpu,
    logits: &Array,
    admitted_vocabulary_size: usize,
) -> Result<Array, EngineError> {
    let shape = logits.shape();
    let [batch_size, sequence_length, vocabulary_size] = <[usize; 3]>::try_from(shape.clone())
        .map_err(|shape| {
            EngineError::Unsupported(format!(
                "generation logits must be rank 3, observed {shape:?}"
            ))
        })?;
    if batch_size != 1 || sequence_length == 0 || vocabulary_size != admitted_vocabulary_size {
        return Err(EngineError::Unsupported(format!(
            "generation logits shape drift: observed {shape:?}"
        )));
    }
    gpu.slice(
        logits,
        &[0, dimension(sequence_length - 1, "last logit position")?, 0],
        &[
            1,
            dimension(sequence_length, "generation sequence length")?,
            dimension(admitted_vocabulary_size, "generation vocabulary size")?,
        ],
        &[1, 1, 1],
    )
    .and_then(|value| gpu.reshape(&value, &[1, admitted_vocabulary_size]))
    .map_err(EngineError::Mlx)
}

fn apply_presence_penalty(
    gpu: &Gpu,
    logits: &Array,
    history_tokens: &[u32],
    penalty: f32,
) -> Result<Array, EngineError> {
    if penalty == 0.0 || history_tokens.is_empty() {
        return logits.try_clone().map_err(EngineError::Mlx);
    }
    let start = history_tokens.len().saturating_sub(PRESENCE_CONTEXT_SIZE);
    let tokens = history_tokens[start..]
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|token| {
            i32::try_from(token)
                .map_err(|error| invalid_sampling(&format!("token ID {token}: {error}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let flat_indices = Array::from_i32_slice(&tokens, &[tokens.len()]).map_err(EngineError::Mlx)?;
    let indices = gpu
        .reshape(&flat_indices, &[1, tokens.len()])
        .map_err(EngineError::Mlx)?;
    let selected = gpu
        .take_axis(logits, &flat_indices, 1)
        .map_err(EngineError::Mlx)?;
    let penalty = gpu
        .scalar_like(penalty, selected.dtype())
        .map_err(EngineError::Mlx)?;
    let adjusted = gpu
        .subtract(&selected, &penalty)
        .map_err(EngineError::Mlx)?;
    gpu.put_along_axis(logits, &indices, &adjusted, -1)
        .map_err(EngineError::Mlx)
}

fn apply_top_p(gpu: &Gpu, logprobs: &Array, top_p: f32) -> Result<Array, EngineError> {
    let vocabulary_size = logprobs.shape().last().copied().ok_or_else(|| {
        EngineError::Unsupported("top-p log probabilities must have a vocabulary axis".into())
    })?;
    let probs = gpu.exp(logprobs).map_err(EngineError::Mlx)?;
    let sorted_indices = gpu.argsort_axis(logprobs, -1).map_err(EngineError::Mlx)?;
    let sorted_probs = gpu
        .take_along_axis(&probs, &sorted_indices, -1)
        .map_err(EngineError::Mlx)?;
    let sorted_cumulative = gpu.cumsum(&sorted_probs, -1).map_err(EngineError::Mlx)?;
    let inverse_seed = gpu.zeros_like(&sorted_indices).map_err(EngineError::Mlx)?;
    let positions = gpu
        .arange(vocabulary_size, sorted_indices.dtype())
        .and_then(|positions| gpu.reshape(&positions, &[1, vocabulary_size]))
        .map_err(EngineError::Mlx)?;
    let inverse_indices = gpu
        .put_along_axis(&inverse_seed, &sorted_indices, &positions, -1)
        .map_err(EngineError::Mlx)?;
    let cumulative = gpu
        .take_along_axis(&sorted_cumulative, &inverse_indices, -1)
        .map_err(EngineError::Mlx)?;
    let threshold = gpu
        .scalar_like(1.0 - top_p, cumulative.dtype())
        .map_err(EngineError::Mlx)?;
    let keep = gpu
        .greater(&cumulative, &threshold)
        .map_err(EngineError::Mlx)?;
    let negative_infinity = gpu
        .scalar_like(f32::NEG_INFINITY, logprobs.dtype())
        .map_err(EngineError::Mlx)?;
    gpu.where_condition(&keep, logprobs, &negative_infinity)
        .map_err(EngineError::Mlx)
}

fn apply_top_k(gpu: &Gpu, logprobs: &Array, top_k: usize) -> Result<Array, EngineError> {
    let vocabulary_size = logprobs.shape().last().copied().ok_or_else(|| {
        EngineError::Unsupported("top-k log probabilities must have a vocabulary axis".into())
    })?;
    let negative = gpu.negative(logprobs).map_err(EngineError::Mlx)?;
    let partition = gpu
        .argpartition_axis(&negative, dimension(top_k - 1, "top-k partition")?, -1)
        .map_err(EngineError::Mlx)?;
    let masked_indices = gpu
        .slice(
            &partition,
            &[0, dimension(top_k, "top-k start")?],
            &[1, dimension(vocabulary_size, "top-k vocabulary")?],
            &[1, 1],
        )
        .map_err(EngineError::Mlx)?;
    let negative_infinity = gpu
        .scalar_like(f32::NEG_INFINITY, logprobs.dtype())
        .map_err(EngineError::Mlx)?;
    gpu.put_along_axis(logprobs, &masked_indices, &negative_infinity, -1)
        .map_err(EngineError::Mlx)
}

fn dimension(value: usize, label: &str) -> Result<i32, EngineError> {
    i32::try_from(value)
        .map_err(|error| EngineError::Unsupported(format!("{label} does not fit int32: {error}")))
}

fn invalid_sampling(detail: &str) -> EngineError {
    EngineError::Unsupported(format!("sampling profile: {detail}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_profile_is_explicit_and_admitted() {
        let profile = SamplingConfig::echo_production(42);
        profile.validate(248_320).expect("production profile");
        assert_eq!(profile.temperature.to_bits(), 0.7_f32.to_bits());
        assert_eq!(profile.top_p.to_bits(), 0.8_f32.to_bits());
        assert_eq!(profile.top_k, 20);
        assert_eq!(profile.presence_penalty.to_bits(), 1.5_f32.to_bits());
    }

    #[test]
    fn top_p_keeps_a_dominant_token_when_it_alone_exceeds_the_cutoff() {
        let gpu = Gpu::new();
        let logits = Array::from_f32_slice(&[-10.0, -10.0, -10.0, 10.0], &[1, 1, 4])
            .expect("dominant-token logits");
        let token = sample_token(
            &gpu,
            &logits,
            &[],
            0,
            SamplingConfig {
                temperature: 0.7,
                top_p: 0.8,
                top_k: 0,
                seed: 0,
                ..SamplingConfig::default()
            },
            4,
        )
        .expect("top-p sample");
        gpu.eval(&[&token]).expect("evaluate sampled token");
        let actual = gpu
            .reshape(&token, &[])
            .and_then(|value| value.item_u32())
            .expect("sampled token ID");

        assert_eq!(actual, 3);
    }

    #[test]
    fn unsupported_sampling_controls_fail_closed() {
        assert!(
            SamplingConfig {
                min_p: 0.1,
                ..SamplingConfig::default()
            }
            .validate(100)
            .is_err()
        );
        assert!(
            SamplingConfig {
                repetition_penalty: 1.1,
                ..SamplingConfig::default()
            }
            .validate(100)
            .is_err()
        );
    }
}
