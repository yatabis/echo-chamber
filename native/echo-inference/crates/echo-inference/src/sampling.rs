use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use echo_mlx::{Array, Gpu};
use serde::{Deserialize, Serialize};

use super::{EngineError, MAX_ACTIVE_BATCH_SIZE};

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
    let sampling_logits = prepare_sampling_logits(gpu, &logits, config)?;
    let step = u64::try_from(generated_index)
        .map_err(|error| invalid_sampling(&format!("generated token index overflow: {error}")))?;
    gpu.categorical_with_seed(&sampling_logits, -1, config.seed.wrapping_add(step))
        .and_then(|token| gpu.reshape(&token, &[1, 1]))
        .map_err(EngineError::Mlx)
}

/// Samples one token per active batch row without sharing request-owned
/// history or random-key state between rows.
///
/// Compatible production rows share the deterministic full-vocabulary
/// filtering graph. Presence history and functional random keys remain
/// request-owned. Width one, greedy generation, and rows with different
/// controls retain the admitted single-request path.
pub(crate) fn sample_token_rows(
    gpu: &Gpu,
    logits: &Array,
    history_rows: &[Vec<u32>],
    configs: &[SamplingConfig],
    vocabulary_size: usize,
) -> Result<Array, EngineError> {
    let shape = logits.shape();
    let [batch_size, sequence_length, observed_vocabulary_size] =
        <[usize; 3]>::try_from(shape.clone()).map_err(|shape| {
            EngineError::Unsupported(format!(
                "batched generation logits must be rank 3, observed {shape:?}"
            ))
        })?;
    if batch_size == 0
        || sequence_length == 0
        || observed_vocabulary_size != vocabulary_size
        || history_rows.len() != batch_size
        || configs.len() != batch_size
    {
        return Err(EngineError::Unsupported(format!(
            "batched sampling shape or request rows drifted: logits={shape:?}, histories={}, configs={}, vocabulary={vocabulary_size}",
            history_rows.len(),
            configs.len()
        )));
    }
    if can_batch_sampling_filters(configs) {
        for config in configs {
            config.validate(vocabulary_size)?;
        }
        return sample_token_rows_with_batched_filters(
            gpu,
            logits,
            history_rows,
            configs,
            vocabulary_size,
        );
    }

    sample_token_rows_individually(
        gpu,
        logits,
        history_rows,
        configs,
        batch_size,
        sequence_length,
        vocabulary_size,
    )
}

fn sample_token_rows_individually(
    gpu: &Gpu,
    logits: &Array,
    history_rows: &[Vec<u32>],
    configs: &[SamplingConfig],
    batch_size: usize,
    sequence_length: usize,
    vocabulary_size: usize,
) -> Result<Array, EngineError> {
    let mut tokens = Vec::with_capacity(batch_size);
    for row in 0..batch_size {
        let row_logits = gpu
            .slice(
                logits,
                &[dimension(row, "sampling batch row")?, 0, 0],
                &[
                    dimension(row + 1, "sampling batch row stop")?,
                    dimension(sequence_length, "sampling sequence length")?,
                    dimension(vocabulary_size, "sampling vocabulary size")?,
                ],
                &[1, 1, 1],
            )
            .map_err(EngineError::Mlx)?;
        tokens.push(sample_token(
            gpu,
            &row_logits,
            &history_rows[row],
            history_rows[row].len(),
            configs[row],
            vocabulary_size,
        )?);
    }
    let references = tokens.iter().collect::<Vec<_>>();
    gpu.concatenate(&references, 0).map_err(EngineError::Mlx)
}

fn sample_token_rows_with_batched_filters(
    gpu: &Gpu,
    logits: &Array,
    history_rows: &[Vec<u32>],
    configs: &[SamplingConfig],
    vocabulary_size: usize,
) -> Result<Array, EngineError> {
    let sampling_logits = prepare_sampling_rows_with_batched_filters(
        gpu,
        logits,
        history_rows,
        configs,
        vocabulary_size,
    )?;

    let mut tokens = Vec::with_capacity(configs.len());
    for row in 0..configs.len() {
        let row_logits = matrix_row(gpu, &sampling_logits, row, vocabulary_size)?;
        let step = u64::try_from(history_rows[row].len()).map_err(|error| {
            invalid_sampling(&format!("generated token index overflow: {error}"))
        })?;
        tokens.push(
            gpu.categorical_with_seed(&row_logits, -1, configs[row].seed.wrapping_add(step))
                .and_then(|token| gpu.reshape(&token, &[1, 1]))
                .map_err(EngineError::Mlx)?,
        );
    }
    let token_references = tokens.iter().collect::<Vec<_>>();
    gpu.concatenate(&token_references, 0)
        .map_err(EngineError::Mlx)
}

fn prepare_sampling_rows_with_batched_filters(
    gpu: &Gpu,
    logits: &Array,
    history_rows: &[Vec<u32>],
    configs: &[SamplingConfig],
    vocabulary_size: usize,
) -> Result<Array, EngineError> {
    let (last_rows, _) = last_logit_rows(gpu, logits, vocabulary_size)?;
    let mut adjusted_rows = Vec::with_capacity(configs.len());
    for row in 0..configs.len() {
        let row_logits = matrix_row(gpu, &last_rows, row, vocabulary_size)?;
        adjusted_rows.push(apply_presence_penalty(
            gpu,
            &row_logits,
            &history_rows[row],
            configs[row].presence_penalty,
        )?);
    }
    let adjusted_references = adjusted_rows.iter().collect::<Vec<_>>();
    let adjusted_logits = gpu
        .concatenate(&adjusted_references, 0)
        .map_err(EngineError::Mlx)?;
    prepare_sampling_logits(gpu, &adjusted_logits, configs[0])
}

fn can_batch_sampling_filters(configs: &[SamplingConfig]) -> bool {
    let Some(first) = configs.first() else {
        return false;
    };
    (2..=MAX_ACTIVE_BATCH_SIZE).contains(&configs.len())
        && first.temperature > 0.0
        && configs
            .iter()
            .all(|config| sampling_controls_equal(*first, *config))
}

fn sampling_controls_equal(left: SamplingConfig, right: SamplingConfig) -> bool {
    left.temperature.to_bits() == right.temperature.to_bits()
        && left.top_p.to_bits() == right.top_p.to_bits()
        && left.top_k == right.top_k
        && left.min_p.to_bits() == right.min_p.to_bits()
        && left.repetition_penalty.to_bits() == right.repetition_penalty.to_bits()
        && left.presence_penalty.to_bits() == right.presence_penalty.to_bits()
}

fn prepare_sampling_logits(
    gpu: &Gpu,
    logits: &Array,
    config: SamplingConfig,
) -> Result<Array, EngineError> {
    let normalizer = gpu
        .logsumexp_axis(logits, -1, true)
        .map_err(EngineError::Mlx)?;
    let mut logprobs = gpu
        .subtract(logits, &normalizer)
        .map_err(EngineError::Mlx)?;
    if config.top_p > 0.0 && config.top_p < 1.0 && config.top_k > 0 {
        logprobs = apply_top_k_then_top_p(gpu, &logprobs, config.top_k, config.top_p)?;
    } else if config.top_p > 0.0 && config.top_p < 1.0 {
        logprobs = apply_top_p(gpu, &logprobs, config.top_p)?;
    } else if config.top_k > 0 {
        logprobs = apply_top_k(gpu, &logprobs, config.top_k)?;
    }
    scale_sampling_logits(gpu, &logprobs, config.temperature)
}

#[cfg(test)]
fn prepare_sampling_logits_with_full_sort(
    gpu: &Gpu,
    logits: &Array,
    config: SamplingConfig,
) -> Result<Array, EngineError> {
    let normalizer = gpu
        .logsumexp_axis(logits, -1, true)
        .map_err(EngineError::Mlx)?;
    let mut logprobs = gpu
        .subtract(logits, &normalizer)
        .map_err(EngineError::Mlx)?;
    if config.top_p > 0.0 && config.top_p < 1.0 {
        logprobs = apply_top_p(gpu, &logprobs, config.top_p)?;
    }
    if config.top_k > 0 {
        logprobs = apply_top_k(gpu, &logprobs, config.top_k)?;
    }
    scale_sampling_logits(gpu, &logprobs, config.temperature)
}

fn scale_sampling_logits(
    gpu: &Gpu,
    logprobs: &Array,
    temperature: f32,
) -> Result<Array, EngineError> {
    let inverse_temperature = gpu
        .scalar_like(1.0 / temperature, logprobs.dtype())
        .map_err(EngineError::Mlx)?;
    gpu.multiply(logprobs, &inverse_temperature)
        .map_err(EngineError::Mlx)
}

fn last_logits(
    gpu: &Gpu,
    logits: &Array,
    admitted_vocabulary_size: usize,
) -> Result<Array, EngineError> {
    let (rows, batch_size) = last_logit_rows(gpu, logits, admitted_vocabulary_size)?;
    if batch_size != 1 {
        return Err(EngineError::Unsupported(format!(
            "generation logits shape drift: observed {:?}",
            logits.shape()
        )));
    }
    Ok(rows)
}

fn last_logit_rows(
    gpu: &Gpu,
    logits: &Array,
    admitted_vocabulary_size: usize,
) -> Result<(Array, usize), EngineError> {
    let shape = logits.shape();
    let [batch_size, sequence_length, vocabulary_size] = <[usize; 3]>::try_from(shape.clone())
        .map_err(|shape| {
            EngineError::Unsupported(format!(
                "generation logits must be rank 3, observed {shape:?}"
            ))
        })?;
    if batch_size == 0 || sequence_length == 0 || vocabulary_size != admitted_vocabulary_size {
        return Err(EngineError::Unsupported(format!(
            "generation logits shape drift: observed {shape:?}"
        )));
    }
    let rows = gpu
        .slice(
            logits,
            &[0, dimension(sequence_length - 1, "last logit position")?, 0],
            &[
                dimension(batch_size, "generation batch size")?,
                dimension(sequence_length, "generation sequence length")?,
                dimension(admitted_vocabulary_size, "generation vocabulary size")?,
            ],
            &[1, 1, 1],
        )
        .and_then(|value| gpu.reshape(&value, &[batch_size, admitted_vocabulary_size]))
        .map_err(EngineError::Mlx)?;
    Ok((rows, batch_size))
}

fn matrix_row(
    gpu: &Gpu,
    values: &Array,
    row: usize,
    vocabulary_size: usize,
) -> Result<Array, EngineError> {
    gpu.slice(
        values,
        &[dimension(row, "sampling matrix row")?, 0],
        &[
            dimension(row + 1, "sampling matrix row stop")?,
            dimension(vocabulary_size, "sampling matrix vocabulary")?,
        ],
        &[1, 1],
    )
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
    let [batch_size, vocabulary_size] = rank_two_shape(logprobs, "top-p log probabilities")?;
    let probs = gpu.exp(logprobs).map_err(EngineError::Mlx)?;
    let sorted_indices = gpu.argsort_axis(logprobs, -1).map_err(EngineError::Mlx)?;
    let sorted_probs = gpu
        .take_along_axis(&probs, &sorted_indices, -1)
        .map_err(EngineError::Mlx)?;
    let sorted_cumulative = gpu.cumsum(&sorted_probs, -1).map_err(EngineError::Mlx)?;
    let inverse_seed = gpu.zeros_like(&sorted_indices).map_err(EngineError::Mlx)?;
    let position_row = gpu
        .arange(vocabulary_size, sorted_indices.dtype())
        .and_then(|positions| gpu.reshape(&positions, &[1, vocabulary_size]))
        .map_err(EngineError::Mlx)?;
    let positions = if batch_size == 1 {
        position_row
    } else {
        let position_rows = (0..batch_size).map(|_| &position_row).collect::<Vec<_>>();
        gpu.concatenate(&position_rows, 0)
            .map_err(EngineError::Mlx)?
    };
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

fn apply_top_k_then_top_p(
    gpu: &Gpu,
    logprobs: &Array,
    top_k: usize,
    top_p: f32,
) -> Result<Array, EngineError> {
    let [batch_size, _] = rank_two_shape(logprobs, "top-k-first log probabilities")?;
    let negative = gpu.negative(logprobs).map_err(EngineError::Mlx)?;
    let partition = gpu
        .argpartition_axis(
            &negative,
            dimension(top_k - 1, "top-k-first partition")?,
            -1,
        )
        .map_err(EngineError::Mlx)?;
    let selected_indices = gpu
        .slice(
            &partition,
            &[0, 0],
            &[
                dimension(batch_size, "top-k-first batch size")?,
                dimension(top_k, "top-k-first candidate count")?,
            ],
            &[1, 1],
        )
        .map_err(EngineError::Mlx)?;
    let selected_logprobs = gpu
        .take_along_axis(logprobs, &selected_indices, -1)
        .map_err(EngineError::Mlx)?;
    let selected_order = gpu
        .argsort_axis(&selected_logprobs, -1)
        .map_err(EngineError::Mlx)?;
    let sorted_indices = gpu
        .take_along_axis(&selected_indices, &selected_order, -1)
        .map_err(EngineError::Mlx)?;
    let sorted_logprobs = gpu
        .take_along_axis(&selected_logprobs, &selected_order, -1)
        .map_err(EngineError::Mlx)?;

    let all_probabilities = gpu.exp(logprobs).map_err(EngineError::Mlx)?;
    let total_probability = gpu
        .sum_axis(&all_probabilities, -1, true)
        .map_err(EngineError::Mlx)?;
    let selected_probabilities = gpu.exp(&sorted_logprobs).map_err(EngineError::Mlx)?;
    let selected_probability = gpu
        .sum_axis(&selected_probabilities, -1, true)
        .map_err(EngineError::Mlx)?;
    let outside_probability = gpu
        .subtract(&total_probability, &selected_probability)
        .map_err(EngineError::Mlx)?;
    let selected_cumulative = gpu
        .cumsum(&selected_probabilities, -1)
        .map_err(EngineError::Mlx)?;
    let cumulative = gpu
        .add(&selected_cumulative, &outside_probability)
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
    let selected_logits = gpu
        .where_condition(&keep, &sorted_logprobs, &negative_infinity)
        .map_err(EngineError::Mlx)?;
    let empty = gpu.zeros_like(logprobs).map_err(EngineError::Mlx)?;
    let empty = gpu
        .add(&empty, &negative_infinity)
        .map_err(EngineError::Mlx)?;
    gpu.put_along_axis(&empty, &sorted_indices, &selected_logits, -1)
        .map_err(EngineError::Mlx)
}

fn apply_top_k(gpu: &Gpu, logprobs: &Array, top_k: usize) -> Result<Array, EngineError> {
    let [batch_size, vocabulary_size] = rank_two_shape(logprobs, "top-k log probabilities")?;
    let negative = gpu.negative(logprobs).map_err(EngineError::Mlx)?;
    let partition = gpu
        .argpartition_axis(&negative, dimension(top_k - 1, "top-k partition")?, -1)
        .map_err(EngineError::Mlx)?;
    let masked_indices = gpu
        .slice(
            &partition,
            &[0, dimension(top_k, "top-k start")?],
            &[
                dimension(batch_size, "top-k batch size")?,
                dimension(vocabulary_size, "top-k vocabulary")?,
            ],
            &[1, 1],
        )
        .map_err(EngineError::Mlx)?;
    let negative_infinity = gpu
        .scalar_like(f32::NEG_INFINITY, logprobs.dtype())
        .map_err(EngineError::Mlx)?;
    gpu.put_along_axis(logprobs, &masked_indices, &negative_infinity, -1)
        .map_err(EngineError::Mlx)
}

fn rank_two_shape(values: &Array, label: &str) -> Result<[usize; 2], EngineError> {
    let shape = values.shape();
    match shape.as_slice() {
        [rows, columns] => Ok([*rows, *columns]),
        _ => Err(EngineError::Unsupported(format!(
            "{label} must be rank 2, observed {shape:?}"
        ))),
    }
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
    use echo_mlx::DType;

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

    #[test]
    fn batched_sampling_keeps_each_request_row_separate() {
        let gpu = Gpu::new();
        let logits = Array::from_f32_slice(&[0.0, 3.0, 1.0, -1.0, 4.0, 0.0], &[2, 1, 3])
            .expect("two-row logits");
        let tokens = sample_token_rows(
            &gpu,
            &logits,
            &[Vec::new(), Vec::new()],
            &[SamplingConfig::default(), SamplingConfig::default()],
            3,
        )
        .expect("sample rows");
        gpu.eval(&[&tokens]).expect("evaluate sampled rows");

        assert_eq!(tokens.shape(), vec![2, 1]);
        let first = gpu
            .slice(&tokens, &[0, 0], &[1, 1], &[1, 1])
            .and_then(|value| gpu.reshape(&value, &[]))
            .and_then(|value| value.item_u32())
            .expect("first sampled row");
        let second = gpu
            .slice(&tokens, &[1, 0], &[2, 1], &[1, 1])
            .and_then(|value| gpu.reshape(&value, &[]))
            .and_then(|value| value.item_u32())
            .expect("second sampled row");
        assert_eq!([first, second], [1, 1]);
    }

    #[test]
    fn production_filter_batch_is_exact_for_widths_two_through_six() {
        const VOCABULARY_SIZE: usize = 257;
        let gpu = Gpu::new();

        for batch_size in 2..=MAX_ACTIVE_BATCH_SIZE {
            let histories = sampling_histories(batch_size, VOCABULARY_SIZE);
            let configs = sampling_configs(batch_size);
            let logits = sampling_logits(batch_size, 2, VOCABULARY_SIZE);
            for dtype in [DType::Float32, DType::BFloat16] {
                let logits = gpu.astype(&logits, dtype).expect("typed sampling logits");
                let reference = prepare_sampling_rows_individually_for_test(
                    &gpu,
                    &logits,
                    &histories,
                    &configs,
                    VOCABULARY_SIZE,
                );
                let batched = prepare_sampling_rows_with_batched_filters(
                    &gpu,
                    &logits,
                    &histories,
                    &configs,
                    VOCABULARY_SIZE,
                )
                .expect("batched sampling filters");

                assert_filtered_rows_exact(&gpu, &reference, &batched);
            }
        }
    }

    #[test]
    fn top_k_first_filter_is_exact_at_the_production_vocabulary_width() {
        const VOCABULARY_SIZE: usize = 248_320;
        const BATCH_SIZE: usize = 6;
        let gpu = Gpu::new();
        let mut values = vec![-12.0_f32; BATCH_SIZE * VOCABULARY_SIZE];
        for row in 0..BATCH_SIZE {
            for rank in 0..40 {
                let token = (row * 65_537 + rank * 7_919) % VOCABULARY_SIZE;
                let rank = f32::from(u16::try_from(rank).expect("candidate rank"));
                values[row * VOCABULARY_SIZE + token] = 8.0 - rank * 0.25;
            }
        }
        let logits = Array::from_f32_slice(&values, &[BATCH_SIZE, VOCABULARY_SIZE])
            .and_then(|logits| gpu.astype(&logits, DType::BFloat16))
            .expect("production-width BF16 logits");
        let config = SamplingConfig::echo_production(42);
        let reference = prepare_sampling_logits_with_full_sort(&gpu, &logits, config)
            .expect("production-width full-sort filters");
        let candidate = prepare_sampling_logits(&gpu, &logits, config)
            .expect("production-width top-k-first filters");

        assert_filtered_rows_exact(&gpu, &reference, &candidate);
    }

    #[test]
    fn production_sampling_tokens_are_exact_for_widths_two_through_six() {
        const VOCABULARY_SIZE: usize = 257;
        let gpu = Gpu::new();

        for batch_size in 2..=MAX_ACTIVE_BATCH_SIZE {
            let histories = sampling_histories(batch_size, VOCABULARY_SIZE);
            let configs = sampling_configs(batch_size);
            let logits = sampling_logits(batch_size, 2, VOCABULARY_SIZE);
            let reference = sample_token_rows_individually(
                &gpu,
                &logits,
                &histories,
                &configs,
                batch_size,
                2,
                VOCABULARY_SIZE,
            )
            .expect("individual sampling");
            let batched = sample_token_rows(&gpu, &logits, &histories, &configs, VOCABULARY_SIZE)
                .expect("batched sampling");

            assert_eq!(
                token_rows(&gpu, &batched),
                token_rows(&gpu, &reference),
                "batch width {batch_size}"
            );
        }
    }

    #[test]
    fn production_sampling_is_invariant_to_row_order_and_cotenant_logits() {
        const VOCABULARY_SIZE: usize = 257;
        const BATCH_SIZE: usize = 3;
        let gpu = Gpu::new();
        let histories = sampling_histories(BATCH_SIZE, VOCABULARY_SIZE);
        let configs = sampling_configs(BATCH_SIZE);
        let logits_data = sampling_logits_data(BATCH_SIZE, 2, VOCABULARY_SIZE);
        let logits = Array::from_f32_slice(&logits_data, &[BATCH_SIZE, 2, VOCABULARY_SIZE])
            .expect("sampling logits");
        let original = sample_token_rows(&gpu, &logits, &histories, &configs, VOCABULARY_SIZE)
            .expect("original sampling rows");
        let original_tokens = token_rows(&gpu, &original);

        let reversed_data = reorder_logits_rows(&logits_data, &[2, 1, 0], 2, VOCABULARY_SIZE);
        let reversed_logits =
            Array::from_f32_slice(&reversed_data, &[BATCH_SIZE, 2, VOCABULARY_SIZE])
                .expect("reversed logits");
        let reversed_histories = histories.iter().rev().cloned().collect::<Vec<_>>();
        let reversed_configs = configs.iter().rev().copied().collect::<Vec<_>>();
        let reversed = sample_token_rows(
            &gpu,
            &reversed_logits,
            &reversed_histories,
            &reversed_configs,
            VOCABULARY_SIZE,
        )
        .expect("reversed sampling rows");
        let mut expected_reversed = original_tokens.clone();
        expected_reversed.reverse();
        assert_eq!(token_rows(&gpu, &reversed), expected_reversed);

        let mut changed_data = logits_data;
        let changed_row_start = 2 * VOCABULARY_SIZE;
        let changed_row_stop = changed_row_start + 2 * VOCABULARY_SIZE;
        for value in &mut changed_data[changed_row_start..changed_row_stop] {
            *value = -*value + 0.125;
        }
        let changed_logits =
            Array::from_f32_slice(&changed_data, &[BATCH_SIZE, 2, VOCABULARY_SIZE])
                .expect("changed cotenant logits");
        let changed =
            sample_token_rows(&gpu, &changed_logits, &histories, &configs, VOCABULARY_SIZE)
                .expect("changed cotenant sampling rows");
        assert_eq!(token_rows(&gpu, &changed)[0], original_tokens[0]);
    }

    #[test]
    fn batched_filters_require_matching_non_seed_controls() {
        let mut configs = sampling_configs(2);
        assert!(can_batch_sampling_filters(&configs));
        configs[1].top_p = 0.9;
        assert!(!can_batch_sampling_filters(&configs));
        configs[1].top_p = configs[0].top_p;
        configs[1].temperature = 0.0;
        assert!(!can_batch_sampling_filters(&configs));
        assert!(!can_batch_sampling_filters(&configs[..1]));
    }

    fn prepare_sampling_rows_individually_for_test(
        gpu: &Gpu,
        logits: &Array,
        history_rows: &[Vec<u32>],
        configs: &[SamplingConfig],
        vocabulary_size: usize,
    ) -> Array {
        let [batch_size, sequence_length, _] =
            <[usize; 3]>::try_from(logits.shape()).expect("rank-three sampling logits");
        let mut rows = Vec::with_capacity(batch_size);
        for row in 0..batch_size {
            let row_logits = gpu
                .slice(
                    logits,
                    &[i32::try_from(row).expect("row"), 0, 0],
                    &[
                        i32::try_from(row + 1).expect("row stop"),
                        i32::try_from(sequence_length).expect("sequence length"),
                        i32::try_from(vocabulary_size).expect("vocabulary size"),
                    ],
                    &[1, 1, 1],
                )
                .expect("sampling row");
            let row_logits = last_logits(gpu, &row_logits, vocabulary_size).expect("last logits");
            let row_logits = apply_presence_penalty(
                gpu,
                &row_logits,
                &history_rows[row],
                configs[row].presence_penalty,
            )
            .expect("presence penalty");
            rows.push(
                prepare_sampling_logits_with_full_sort(gpu, &row_logits, configs[row])
                    .expect("full-sort sampling filters"),
            );
        }
        let references = rows.iter().collect::<Vec<_>>();
        gpu.concatenate(&references, 0)
            .expect("concatenated sampling rows")
    }

    fn assert_filtered_rows_exact(gpu: &Gpu, reference: &Array, batched: &Array) {
        let negative_infinity = gpu
            .scalar_like(f32::NEG_INFINITY, reference.dtype())
            .expect("negative infinity");
        let reference_keep = gpu
            .greater(reference, &negative_infinity)
            .expect("reference finite mask");
        let batched_keep = gpu
            .greater(batched, &negative_infinity)
            .expect("batched finite mask");
        let reference_mask = gpu
            .astype(&reference_keep, DType::Float32)
            .expect("reference float mask");
        let batched_mask = gpu
            .astype(&batched_keep, DType::Float32)
            .expect("batched float mask");
        let mask_difference = gpu
            .max_abs_difference(&reference_mask, &batched_mask)
            .expect("filter mask difference");
        assert_eq!(mask_difference.to_bits(), 0.0_f32.to_bits());

        let zero = gpu
            .scalar_like(0.0, reference.dtype())
            .expect("filter comparison zero");
        let reference_values = gpu
            .where_condition(&reference_keep, reference, &zero)
            .expect("finite reference values");
        let batched_values = gpu
            .where_condition(&batched_keep, batched, &zero)
            .expect("finite batched values");
        let value_difference = gpu
            .max_abs_difference(&reference_values, &batched_values)
            .expect("finite value difference");
        assert_eq!(value_difference.to_bits(), 0.0_f32.to_bits());
    }

    fn sampling_configs(batch_size: usize) -> Vec<SamplingConfig> {
        (0..batch_size)
            .map(|row| SamplingConfig::echo_production(10_000 + row as u64 * 97))
            .collect()
    }

    fn sampling_histories(batch_size: usize, vocabulary_size: usize) -> Vec<Vec<u32>> {
        (0..batch_size)
            .map(|row| {
                (0..=row)
                    .map(|offset| {
                        u32::try_from((row * 31 + offset * 17) % vocabulary_size)
                            .expect("test token ID")
                    })
                    .collect()
            })
            .collect()
    }

    fn sampling_logits(batch_size: usize, sequence_length: usize, vocabulary_size: usize) -> Array {
        let values = sampling_logits_data(batch_size, sequence_length, vocabulary_size);
        Array::from_f32_slice(&values, &[batch_size, sequence_length, vocabulary_size])
            .expect("sampling logits")
    }

    fn sampling_logits_data(
        batch_size: usize,
        sequence_length: usize,
        vocabulary_size: usize,
    ) -> Vec<f32> {
        (0..batch_size * sequence_length * vocabulary_size)
            .map(|index| {
                let token = index % vocabulary_size;
                let sequence = index / vocabulary_size % sequence_length;
                let row = index / (vocabulary_size * sequence_length);
                let mixed = (token * 37 + sequence * 19 + row * 53) % 1_009;
                (f32::from(u16::try_from(mixed).expect("mixed test logit")) - 504.0) / 5_040.0
            })
            .collect()
    }

    fn reorder_logits_rows(
        logits: &[f32],
        order: &[usize],
        sequence_length: usize,
        vocabulary_size: usize,
    ) -> Vec<f32> {
        let row_size = sequence_length * vocabulary_size;
        let mut reordered = Vec::with_capacity(logits.len());
        for row in order {
            let start = row * row_size;
            reordered.extend_from_slice(&logits[start..start + row_size]);
        }
        reordered
    }

    fn token_rows(gpu: &Gpu, tokens: &Array) -> Vec<u32> {
        gpu.eval(&[tokens]).expect("evaluate sampled rows");
        (0..tokens.shape()[0])
            .map(|row| {
                gpu.slice(
                    tokens,
                    &[i32::try_from(row).expect("token row"), 0],
                    &[i32::try_from(row + 1).expect("token row stop"), 1],
                    &[1, 1],
                )
                .and_then(|value| gpu.reshape(&value, &[]))
                .and_then(|value| value.item_u32())
                .expect("sampled token")
            })
            .collect()
    }
}
