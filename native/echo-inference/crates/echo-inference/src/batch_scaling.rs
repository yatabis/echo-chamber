use std::collections::HashSet;
use std::path::Path;
use std::time::{Duration, Instant};

use echo_mlx::{Array, Gpu, MetalMemoryStats, metal_memory_stats, reset_peak_memory};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::chat::{
    EchoChatPrompt, EchoInputItem, EchoMessage, EchoMessageContent, EchoMessageRole,
    Qwen35ChatTokenizer,
};
use super::decoder::MoeKernel;
use super::full_model::{
    RuntimeModelExecution, compact_runtime_state, evaluate_runtime_execution,
    execute_runtime_model, greedy_token, prepare_merged_runtime_state, prepare_runtime_state,
    schedule_runtime_execution, split_runtime_state,
};
use super::gdn::GdnKernel;
use super::model_state::{LayerState, MlxInferenceState};
use super::runtime::{
    ResidentEngine, ResidentEngineConfig, ResidentEngineInfo, RuntimeMemoryStats,
};
use super::sampling::{SamplingConfig, sample_token_rows};
use super::weights::BoundModelWeights;
use super::{EngineError, ModelPlan};

const CONTEXT_TARGETS: [usize; 3] = [4_096, 16_384, 32_768];
const PREFILL_CHUNK_SIZE: usize = 2_048;
const MAX_SUPPORTED_BATCH_SIZE: usize = 6;
const ROW_LABELS: [char; MAX_SUPPORTED_BATCH_SIZE] = ['A', 'B', 'C', 'D', 'E', 'F'];
const BENCHMARK_PROMPT_PREFIX: &str = concat!(
    "これは可変batch幅の推論性能測定です。ツールは使わず、",
    "指定された固定文字列だけを独立した行へ繰り返してください。\n",
    "各行には前置き、番号、説明、後書き、省略記号を加えないでください。\n",
    "固定文字列: native-batch-width-row-"
);

/// Process-wide Metal allocator observation around one fixed-width attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct BatchWidthMemoryObservation {
    pub start: RuntimeMemoryStats,
    pub end: RuntimeMemoryStats,
    pub active_delta_nbytes: i64,
    pub peak_above_start_active_nbytes: u64,
}

/// One fixed-width decode measurement from an already resident context.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BatchWidthScalingAttempt {
    pub target_context_tokens: usize,
    pub batch_size: usize,
    pub phase: &'static str,
    pub round: usize,
    pub order_position: usize,
    pub batch_total_nanos: u64,
    pub batch_decode_nanos: u64,
    pub ttft_nanos: u64,
    pub aggregate_decode_tokens_per_second: f64,
    pub per_request_decode_tokens_per_second: f64,
    pub output_token_sha256: Vec<String>,
    pub state_sequence_lengths: Vec<usize>,
    pub state_lengths_exact: bool,
    pub memory: BatchWidthMemoryObservation,
}

/// Median behavior for one resident context and fixed batch width.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BatchWidthScalingSummary {
    pub target_context_tokens: usize,
    pub batch_size: usize,
    pub count: usize,
    pub median_batch_total_nanos: u64,
    pub median_batch_decode_nanos: u64,
    pub median_ttft_nanos: u64,
    pub median_aggregate_decode_tokens_per_second: f64,
    pub median_per_request_decode_tokens_per_second: f64,
    pub aggregate_speedup_over_batch_one: f64,
    pub ideal_scaling_efficiency: f64,
    pub median_active_delta_nbytes: i64,
    pub median_peak_above_start_active_nbytes: u64,
    pub exact_state_attempts: usize,
}

/// Same-shape row isolation at the largest requested batch width.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MaximumBatchIsolationObservation {
    pub batch_size: usize,
    pub generated_tokens_per_row: usize,
    pub changed_co_tenants_row_zero_output_exact: bool,
    pub changed_co_tenants_row_zero_state_max_absolute_difference: f32,
    pub reversed_row_output_exact: Vec<bool>,
    pub reversed_row_state_max_absolute_difference: Vec<f32>,
    pub baseline_state_rows_pairwise_distinct: bool,
    pub exact: bool,
}

/// One stage while active rows shrink from the requested maximum to one.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BatchWidthShrinkStep {
    pub batch_size: usize,
    pub logical_row_ids: Vec<usize>,
    pub emitted_tokens_per_row: Vec<Vec<u32>>,
    pub state_sequence_lengths: Vec<usize>,
    pub expected_state_sequence_length: usize,
    pub state_lengths_exact: bool,
    pub state_rows_pairwise_distinct: bool,
}

/// State ownership and token accounting while a batch shrinks one row at a time.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BatchWidthShrinkObservation {
    pub initial_batch_size: usize,
    pub generated_tokens_per_stage: usize,
    pub visited_batch_sizes: Vec<usize>,
    pub steps: Vec<BatchWidthShrinkStep>,
    pub exact: bool,
}

/// Width-one through width-six scaling across E.C.H.O.'s context tiers.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BatchWidthScalingDiagnostic {
    pub schema_version: u32,
    pub engine: ResidentEngineInfo,
    pub context_targets: [usize; 3],
    pub prefill_chunk_size_tokens: usize,
    pub max_batch_size: usize,
    pub row_input_tokens: Vec<u32>,
    pub generated_tokens_per_row: usize,
    pub warmup_rounds: usize,
    pub measured_rounds: usize,
    pub base_prefill_nanos: [u64; 3],
    pub base_state_logical_nbytes: [usize; 3],
    pub attempts: Vec<BatchWidthScalingAttempt>,
    pub summaries: Vec<BatchWidthScalingSummary>,
    pub maximum_batch_isolation: MaximumBatchIsolationObservation,
    pub shrinking_membership: BatchWidthShrinkObservation,
    pub all_state_checks_passed: bool,
    pub final_metal_memory: RuntimeMemoryStats,
}

/// Production-sampling width scaling at the latency-sensitive 4K tier.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProductionBatchWidthScalingDiagnostic {
    pub schema_version: u32,
    pub engine: ResidentEngineInfo,
    pub target_context_tokens: usize,
    pub prefill_chunk_size_tokens: usize,
    pub max_batch_size: usize,
    pub row_input_tokens: Vec<u32>,
    pub request_sampling_configs: Vec<SamplingConfig>,
    pub generated_tokens_per_row: usize,
    pub warmup_rounds: usize,
    pub measured_rounds: usize,
    pub base_prefill_nanos: u64,
    pub base_state_logical_nbytes: usize,
    pub attempts: Vec<BatchWidthScalingAttempt>,
    pub summaries: Vec<BatchWidthScalingSummary>,
    pub maximum_batch_isolation: MaximumBatchIsolationObservation,
    pub all_state_checks_passed: bool,
    pub final_metal_memory: RuntimeMemoryStats,
}

struct EngineParts<'a> {
    gpu: &'a Gpu,
    plan: &'a ModelPlan,
    weights: &'a BoundModelWeights,
    gdn_kernel: &'a GdnKernel,
    moe_kernel: &'a MoeKernel,
}

struct WidthExecution {
    output: Vec<Vec<u32>>,
    states: Vec<MlxInferenceState>,
    total_nanos: u64,
    decode_nanos: u64,
    ttft_nanos: u64,
    memory: BatchWidthMemoryObservation,
}

struct WidthCursor {
    logical_row_id: usize,
    state: MlxInferenceState,
    pending_token: u32,
}

/// Measures fixed batch widths from one through `max_batch_size` without
/// changing the production scheduler.
///
/// # Errors
///
/// Returns [`EngineError`] when model admission, context preparation,
/// fixed-width execution, row isolation, shrinking membership, timing, or
/// state accounting is invalid.
#[allow(clippy::too_many_lines)]
pub fn run_batch_width_scaling_diagnostic(
    model_directory: &Path,
    max_batch_size: usize,
    warmup_rounds: usize,
    measured_rounds: usize,
    generated_tokens_per_row: usize,
) -> Result<BatchWidthScalingDiagnostic, EngineError> {
    if !(2..=MAX_SUPPORTED_BATCH_SIZE).contains(&max_batch_size)
        || measured_rounds == 0
        || generated_tokens_per_row == 0
    {
        return Err(EngineError::Unsupported(format!(
            "batch width diagnostic requires max batch 2..={MAX_SUPPORTED_BATCH_SIZE}, measured rounds, and generated tokens"
        )));
    }
    let tokenizer = Qwen35ChatTokenizer::load(model_directory).map_err(|error| {
        EngineError::Unsupported(format!("load batch width tokenizer: {error}"))
    })?;
    let prompt_rows = ROW_LABELS[..max_batch_size]
        .iter()
        .map(|label| encode_benchmark_prompt(&tokenizer, *label))
        .collect::<Result<Vec<_>, _>>()?;
    let row_input_tokens = distinct_row_tokens(&prompt_rows)?;
    let engine = ResidentEngine::load(
        model_directory,
        ResidentEngineConfig {
            max_new_tokens_per_request: generated_tokens_per_row,
            ..ResidentEngineConfig::default()
        },
    )
    .map_err(|error| EngineError::Unsupported(format!("load batch width engine: {error}")))?;
    let parts = engine_parts(&engine);
    let mut base_prefill_nanos = [0_u64; 3];
    let mut base_state_logical_nbytes = [0_usize; 3];
    let mut attempts = Vec::new();
    let mut bases = Vec::with_capacity(CONTEXT_TARGETS.len());

    for (target_index, target_context_tokens) in CONTEXT_TARGETS.into_iter().enumerate() {
        let context_tokens = repeated_token_context(&prompt_rows[0], target_context_tokens)?;
        let prefill_started = Instant::now();
        let base = prefill_context_state(
            &parts,
            &context_tokens,
            PREFILL_CHUNK_SIZE,
            generated_tokens_per_row.saturating_add(1),
        )?;
        base_prefill_nanos[target_index] = duration_nanos(prefill_started.elapsed());
        base_state_logical_nbytes[target_index] = base.logical_nbytes()?;
        if base.sequence_length()? != target_context_tokens {
            return Err(EngineError::Unsupported(format!(
                "batch width prefill produced length {}, expected {target_context_tokens}",
                base.sequence_length()?
            )));
        }

        for ordinal in 0..warmup_rounds.saturating_add(measured_rounds) {
            let measured = ordinal >= warmup_rounds;
            let phase = if measured { "measured" } else { "warmup" };
            let round = if measured {
                ordinal - warmup_rounds + 1
            } else {
                ordinal + 1
            };
            let order = rotated_batch_widths(max_batch_size, target_index + ordinal);
            for (order_position, batch_size) in order.into_iter().enumerate() {
                let execution = run_fixed_width_context(
                    &parts,
                    &base,
                    &row_input_tokens[..batch_size],
                    generated_tokens_per_row,
                )?;
                attempts.push(to_attempt(
                    target_context_tokens,
                    batch_size,
                    phase,
                    round,
                    order_position + 1,
                    &execution,
                    generated_tokens_per_row,
                )?);
            }
        }
        bases.push(base);
    }

    let mut summaries = Vec::with_capacity(CONTEXT_TARGETS.len() * max_batch_size);
    for target in CONTEXT_TARGETS {
        let batch_one = summarize_raw(&attempts, target, 1)?;
        let batch_one_throughput = batch_one.median_aggregate_decode_tokens_per_second;
        for batch_size in 1..=max_batch_size {
            let mut summary = summarize_raw(&attempts, target, batch_size)?;
            summary.aggregate_speedup_over_batch_one =
                summary.median_aggregate_decode_tokens_per_second / batch_one_throughput;
            summary.ideal_scaling_efficiency = summary.aggregate_speedup_over_batch_one
                / f64::from(u32::try_from(batch_size).map_err(|error| {
                    EngineError::Unsupported(format!("batch size does not fit u32: {error}"))
                })?);
            summaries.push(summary);
        }
    }

    let four_k_base = bases.first().ok_or_else(|| {
        EngineError::Unsupported("batch width diagnostic lost its 4K base state".into())
    })?;
    let maximum_batch_isolation = run_maximum_batch_isolation(
        &parts,
        four_k_base,
        &row_input_tokens,
        None,
        generated_tokens_per_row.min(32),
    )?;
    let shrinking_membership = run_shrinking_membership(&parts, four_k_base, &row_input_tokens)?;
    let all_state_checks_passed = attempts.iter().all(|attempt| attempt.state_lengths_exact)
        && maximum_batch_isolation.exact
        && shrinking_membership.exact;
    let final_metal_memory = metal_memory_stats().map_err(EngineError::Mlx)?.into();
    Ok(BatchWidthScalingDiagnostic {
        schema_version: 1,
        engine: engine.info().clone(),
        context_targets: CONTEXT_TARGETS,
        prefill_chunk_size_tokens: PREFILL_CHUNK_SIZE,
        max_batch_size,
        row_input_tokens,
        generated_tokens_per_row,
        warmup_rounds,
        measured_rounds,
        base_prefill_nanos,
        base_state_logical_nbytes,
        attempts,
        summaries,
        maximum_batch_isolation,
        shrinking_membership,
        all_state_checks_passed,
        final_metal_memory,
    })
}

/// Measures production sampling at each fixed batch width from one through
/// `max_batch_size` using one resident 4K state.
///
/// # Errors
///
/// Returns [`EngineError`] when model admission, context preparation,
/// request-owned sampling, row isolation, timing, or state accounting fails.
#[allow(clippy::too_many_lines)]
pub fn run_production_batch_width_scaling_diagnostic(
    model_directory: &Path,
    max_batch_size: usize,
    warmup_rounds: usize,
    measured_rounds: usize,
    generated_tokens_per_row: usize,
) -> Result<ProductionBatchWidthScalingDiagnostic, EngineError> {
    if !(2..=MAX_SUPPORTED_BATCH_SIZE).contains(&max_batch_size)
        || measured_rounds == 0
        || generated_tokens_per_row == 0
    {
        return Err(EngineError::Unsupported(format!(
            "production batch width diagnostic requires max batch 2..={MAX_SUPPORTED_BATCH_SIZE}, measured rounds, and generated tokens"
        )));
    }
    let tokenizer = Qwen35ChatTokenizer::load(model_directory).map_err(|error| {
        EngineError::Unsupported(format!("load production width tokenizer: {error}"))
    })?;
    let prompt_rows = ROW_LABELS[..max_batch_size]
        .iter()
        .map(|label| encode_benchmark_prompt(&tokenizer, *label))
        .collect::<Result<Vec<_>, _>>()?;
    let row_input_tokens = distinct_row_tokens(&prompt_rows)?;
    let request_sampling_configs = [10_001, 20_003, 30_007, 40_009, 50_021, 60_013]
        [..max_batch_size]
        .iter()
        .copied()
        .map(SamplingConfig::echo_production)
        .collect::<Vec<_>>();
    let engine = ResidentEngine::load(
        model_directory,
        ResidentEngineConfig {
            max_new_tokens_per_request: generated_tokens_per_row,
            ..ResidentEngineConfig::default()
        },
    )
    .map_err(|error| EngineError::Unsupported(format!("load production width engine: {error}")))?;
    let parts = engine_parts(&engine);
    for config in &request_sampling_configs {
        config.validate(parts.plan.vocabulary_size)?;
    }
    let context_tokens = repeated_token_context(&prompt_rows[0], CONTEXT_TARGETS[0])?;
    let prefill_started = Instant::now();
    let base = prefill_context_state(
        &parts,
        &context_tokens,
        PREFILL_CHUNK_SIZE,
        generated_tokens_per_row.saturating_add(1),
    )?;
    let base_prefill_nanos = duration_nanos(prefill_started.elapsed());
    let base_state_logical_nbytes = base.logical_nbytes()?;
    if base.sequence_length()? != CONTEXT_TARGETS[0] {
        return Err(EngineError::Unsupported(format!(
            "production width prefill produced length {}, expected {}",
            base.sequence_length()?,
            CONTEXT_TARGETS[0]
        )));
    }

    let mut attempts = Vec::new();
    for ordinal in 0..warmup_rounds.saturating_add(measured_rounds) {
        let measured = ordinal >= warmup_rounds;
        let phase = if measured { "measured" } else { "warmup" };
        let round = if measured {
            ordinal - warmup_rounds + 1
        } else {
            ordinal + 1
        };
        for (order_position, batch_size) in rotated_batch_widths(max_batch_size, ordinal)
            .into_iter()
            .enumerate()
        {
            let execution = run_fixed_width_context_with_configs(
                &parts,
                &base,
                &row_input_tokens[..batch_size],
                Some(&request_sampling_configs[..batch_size]),
                generated_tokens_per_row,
            )?;
            attempts.push(to_attempt(
                CONTEXT_TARGETS[0],
                batch_size,
                phase,
                round,
                order_position + 1,
                &execution,
                generated_tokens_per_row,
            )?);
        }
    }

    let batch_one = summarize_raw(&attempts, CONTEXT_TARGETS[0], 1)?;
    let batch_one_throughput = batch_one.median_aggregate_decode_tokens_per_second;
    let mut summaries = Vec::with_capacity(max_batch_size);
    for batch_size in 1..=max_batch_size {
        let mut summary = summarize_raw(&attempts, CONTEXT_TARGETS[0], batch_size)?;
        summary.aggregate_speedup_over_batch_one =
            summary.median_aggregate_decode_tokens_per_second / batch_one_throughput;
        summary.ideal_scaling_efficiency = summary.aggregate_speedup_over_batch_one
            / f64::from(u32::try_from(batch_size).map_err(|error| {
                EngineError::Unsupported(format!("batch size does not fit u32: {error}"))
            })?);
        summaries.push(summary);
    }
    let maximum_batch_isolation = run_maximum_batch_isolation(
        &parts,
        &base,
        &row_input_tokens,
        Some(&request_sampling_configs),
        generated_tokens_per_row.min(32),
    )?;
    let all_state_checks_passed =
        attempts.iter().all(|attempt| attempt.state_lengths_exact) && maximum_batch_isolation.exact;
    let final_metal_memory = metal_memory_stats().map_err(EngineError::Mlx)?.into();
    Ok(ProductionBatchWidthScalingDiagnostic {
        schema_version: 1,
        engine: engine.info().clone(),
        target_context_tokens: CONTEXT_TARGETS[0],
        prefill_chunk_size_tokens: PREFILL_CHUNK_SIZE,
        max_batch_size,
        row_input_tokens,
        request_sampling_configs,
        generated_tokens_per_row,
        warmup_rounds,
        measured_rounds,
        base_prefill_nanos,
        base_state_logical_nbytes,
        attempts,
        summaries,
        maximum_batch_isolation,
        all_state_checks_passed,
        final_metal_memory,
    })
}

fn run_fixed_width_context(
    parts: &EngineParts<'_>,
    base: &MlxInferenceState,
    input_tokens: &[u32],
    generated_tokens: usize,
) -> Result<WidthExecution, EngineError> {
    run_fixed_width_context_with_configs(parts, base, input_tokens, None, generated_tokens)
}

fn run_fixed_width_context_with_configs(
    parts: &EngineParts<'_>,
    base: &MlxInferenceState,
    input_tokens: &[u32],
    sampling_configs: Option<&[SamplingConfig]>,
    generated_tokens: usize,
) -> Result<WidthExecution, EngineError> {
    if input_tokens.is_empty() || generated_tokens == 0 {
        return Err(EngineError::Unsupported(
            "fixed width execution requires rows and generated tokens".into(),
        ));
    }
    if sampling_configs.is_some_and(|configs| configs.len() != input_tokens.len()) {
        return Err(EngineError::Unsupported(
            "fixed width sampling configs differ from active rows".into(),
        ));
    }
    reset_peak_memory().map_err(EngineError::Mlx)?;
    let start_memory = metal_memory_stats().map_err(EngineError::Mlx)?;
    let started = Instant::now();
    let input = batched_token_array(input_tokens)?;
    let base_rows = vec![base; input_tokens.len()];
    let runtime_state = prepare_merged_runtime_state(
        parts.gpu,
        &base_rows,
        generated_tokens.saturating_add(1),
        parts.plan,
    )?;
    let mut execution = execute_runtime_model(
        parts.gpu,
        &input,
        runtime_state,
        parts.weights,
        parts.plan,
        parts.gdn_kernel,
        parts.moe_kernel,
    )?;
    evaluate_runtime_execution(parts.gpu, &execution)?;
    let decode_started = Instant::now();
    let mut output = (0..input_tokens.len())
        .map(|_| Vec::with_capacity(generated_tokens))
        .collect::<Vec<_>>();
    let mut ttft_nanos = None;
    for _ in 0..generated_tokens {
        let RuntimeModelExecution { logits, state } = execution;
        let tokens = if let Some(configs) = sampling_configs {
            sample_token_rows(
                parts.gpu,
                &logits,
                &output,
                configs,
                parts.plan.vocabulary_size,
            )?
        } else {
            greedy_token(parts.gpu, &logits, input_tokens.len(), parts.plan)?
        };
        let next = execute_runtime_model(
            parts.gpu,
            &tokens,
            state,
            parts.weights,
            parts.plan,
            parts.gdn_kernel,
            parts.moe_kernel,
        )?;
        schedule_runtime_execution(parts.gpu, &tokens, &next)?;
        for (row, token) in batched_token_values(parts.gpu, &tokens, input_tokens.len())?
            .into_iter()
            .enumerate()
        {
            output[row].push(token);
        }
        ttft_nanos.get_or_insert_with(|| duration_nanos(started.elapsed()));
        execution = next;
    }
    evaluate_runtime_execution(parts.gpu, &execution)?;
    let decode_nanos = duration_nanos(decode_started.elapsed());
    let states = split_runtime_state(parts.gpu, execution.state, parts.plan)?;
    let total_nanos = duration_nanos(started.elapsed());
    let end_memory = metal_memory_stats().map_err(EngineError::Mlx)?;
    Ok(WidthExecution {
        output,
        states,
        total_nanos,
        decode_nanos,
        ttft_nanos: ttft_nanos.ok_or_else(|| {
            EngineError::Unsupported("fixed width execution produced no first token".into())
        })?,
        memory: memory_observation(start_memory, end_memory)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn to_attempt(
    target_context_tokens: usize,
    batch_size: usize,
    phase: &'static str,
    round: usize,
    order_position: usize,
    execution: &WidthExecution,
    generated_tokens: usize,
) -> Result<BatchWidthScalingAttempt, EngineError> {
    if execution.output.len() != batch_size
        || execution.states.len() != batch_size
        || execution
            .output
            .iter()
            .any(|tokens| tokens.len() != generated_tokens)
    {
        return Err(EngineError::Unsupported(
            "fixed width execution produced the wrong row or token count".into(),
        ));
    }
    let total_tokens = generated_tokens
        .checked_mul(batch_size)
        .ok_or_else(|| EngineError::Unsupported("batch width token denominator overflow".into()))?;
    let state_sequence_lengths = execution
        .states
        .iter()
        .map(MlxInferenceState::sequence_length)
        .collect::<Result<Vec<_>, _>>()?;
    let expected_state_length = target_context_tokens
        .checked_add(generated_tokens)
        .and_then(|length| length.checked_add(1))
        .ok_or_else(|| EngineError::Unsupported("batch width state length overflow".into()))?;
    Ok(BatchWidthScalingAttempt {
        target_context_tokens,
        batch_size,
        phase,
        round,
        order_position,
        batch_total_nanos: execution.total_nanos,
        batch_decode_nanos: execution.decode_nanos,
        ttft_nanos: execution.ttft_nanos,
        aggregate_decode_tokens_per_second: tokens_per_second(
            total_tokens,
            execution.decode_nanos,
        )?,
        per_request_decode_tokens_per_second: tokens_per_second(
            generated_tokens,
            execution.decode_nanos,
        )?,
        output_token_sha256: execution
            .output
            .iter()
            .map(|tokens| token_digest(tokens))
            .collect(),
        state_lengths_exact: state_sequence_lengths
            .iter()
            .all(|length| *length == expected_state_length),
        state_sequence_lengths,
        memory: execution.memory,
    })
}

fn summarize_raw(
    attempts: &[BatchWidthScalingAttempt],
    target_context_tokens: usize,
    batch_size: usize,
) -> Result<BatchWidthScalingSummary, EngineError> {
    let measured = attempts
        .iter()
        .filter(|attempt| {
            attempt.phase == "measured"
                && attempt.target_context_tokens == target_context_tokens
                && attempt.batch_size == batch_size
        })
        .collect::<Vec<_>>();
    let integers = |extract: fn(&BatchWidthScalingAttempt) -> u64| {
        measured
            .iter()
            .map(|attempt| extract(attempt))
            .collect::<Vec<_>>()
    };
    let signed = |extract: fn(&BatchWidthScalingAttempt) -> i64| {
        measured
            .iter()
            .map(|attempt| extract(attempt))
            .collect::<Vec<_>>()
    };
    let floats = |extract: fn(&BatchWidthScalingAttempt) -> f64| {
        measured
            .iter()
            .map(|attempt| extract(attempt))
            .collect::<Vec<_>>()
    };
    Ok(BatchWidthScalingSummary {
        target_context_tokens,
        batch_size,
        count: measured.len(),
        median_batch_total_nanos: median_u64(integers(|attempt| attempt.batch_total_nanos))?,
        median_batch_decode_nanos: median_u64(integers(|attempt| attempt.batch_decode_nanos))?,
        median_ttft_nanos: median_u64(integers(|attempt| attempt.ttft_nanos))?,
        median_aggregate_decode_tokens_per_second: median_f64(floats(|attempt| {
            attempt.aggregate_decode_tokens_per_second
        }))?,
        median_per_request_decode_tokens_per_second: median_f64(floats(|attempt| {
            attempt.per_request_decode_tokens_per_second
        }))?,
        aggregate_speedup_over_batch_one: 0.0,
        ideal_scaling_efficiency: 0.0,
        median_active_delta_nbytes: median_i64(signed(|attempt| {
            attempt.memory.active_delta_nbytes
        }))?,
        median_peak_above_start_active_nbytes: median_u64(integers(|attempt| {
            attempt.memory.peak_above_start_active_nbytes
        }))?,
        exact_state_attempts: measured
            .iter()
            .filter(|attempt| attempt.state_lengths_exact)
            .count(),
    })
}

fn run_maximum_batch_isolation(
    parts: &EngineParts<'_>,
    base: &MlxInferenceState,
    input_tokens: &[u32],
    sampling_configs: Option<&[SamplingConfig]>,
    generated_tokens: usize,
) -> Result<MaximumBatchIsolationObservation, EngineError> {
    let batch_size = input_tokens.len();
    let baseline = run_fixed_width_context_with_configs(
        parts,
        base,
        input_tokens,
        sampling_configs,
        generated_tokens,
    )?;
    let mut changed_tokens = input_tokens.to_vec();
    changed_tokens[1..].rotate_left(1);
    let mut changed_configs = sampling_configs.map(<[SamplingConfig]>::to_vec);
    if let Some(configs) = &mut changed_configs {
        configs[1..].rotate_left(1);
    }
    let changed = run_fixed_width_context_with_configs(
        parts,
        base,
        &changed_tokens,
        changed_configs.as_deref(),
        generated_tokens,
    )?;
    let reversed_tokens = input_tokens.iter().rev().copied().collect::<Vec<_>>();
    let reversed_configs = sampling_configs.map(|configs| {
        configs
            .iter()
            .rev()
            .copied()
            .collect::<Vec<SamplingConfig>>()
    });
    let reversed = run_fixed_width_context_with_configs(
        parts,
        base,
        &reversed_tokens,
        reversed_configs.as_deref(),
        generated_tokens,
    )?;

    let changed_co_tenants_row_zero_output_exact = baseline.output[0] == changed.output[0];
    let changed_co_tenants_row_zero_state_max_absolute_difference =
        state_max_absolute_difference(parts, &baseline.states[0], &changed.states[0])?;
    let mut reversed_row_output_exact = Vec::with_capacity(batch_size);
    let mut reversed_row_state_max_absolute_difference = Vec::with_capacity(batch_size);
    for row in 0..batch_size {
        let reversed_row = batch_size - row - 1;
        reversed_row_output_exact.push(baseline.output[row] == reversed.output[reversed_row]);
        reversed_row_state_max_absolute_difference.push(state_max_absolute_difference(
            parts,
            &baseline.states[row],
            &reversed.states[reversed_row],
        )?);
    }
    let baseline_state_rows_pairwise_distinct =
        state_rows_pairwise_distinct(parts, &baseline.states)?;
    let exact = changed_co_tenants_row_zero_output_exact
        && changed_co_tenants_row_zero_state_max_absolute_difference == 0.0
        && reversed_row_output_exact.iter().all(|value| *value)
        && reversed_row_state_max_absolute_difference
            .iter()
            .all(|difference| *difference == 0.0)
        && baseline_state_rows_pairwise_distinct;
    Ok(MaximumBatchIsolationObservation {
        batch_size,
        generated_tokens_per_row: generated_tokens,
        changed_co_tenants_row_zero_output_exact,
        changed_co_tenants_row_zero_state_max_absolute_difference,
        reversed_row_output_exact,
        reversed_row_state_max_absolute_difference,
        baseline_state_rows_pairwise_distinct,
        exact,
    })
}

fn run_shrinking_membership(
    parts: &EngineParts<'_>,
    base: &MlxInferenceState,
    input_tokens: &[u32],
) -> Result<BatchWidthShrinkObservation, EngineError> {
    const GENERATED_TOKENS_PER_STAGE: usize = 2;
    let base_length = base.sequence_length()?;
    let mut cursors = input_tokens
        .iter()
        .copied()
        .enumerate()
        .map(|(logical_row_id, token)| prepare_cursor(parts, base, logical_row_id, token))
        .collect::<Result<Vec<_>, _>>()?;
    let mut steps = Vec::with_capacity(cursors.len());
    let mut visited_batch_sizes = Vec::with_capacity(cursors.len());
    let mut completed_stages = 0_usize;
    while !cursors.is_empty() {
        let batch_size = cursors.len();
        let (emitted_tokens_per_row, advanced) =
            advance_cursors(parts, cursors, GENERATED_TOKENS_PER_STAGE)?;
        cursors = advanced;
        completed_stages += 1;
        let expected_state_sequence_length = base_length
            .checked_add(1)
            .and_then(|length| length.checked_add(completed_stages * GENERATED_TOKENS_PER_STAGE))
            .ok_or_else(|| EngineError::Unsupported("shrink state length overflow".into()))?;
        let state_sequence_lengths = cursors
            .iter()
            .map(|cursor| cursor.state.sequence_length())
            .collect::<Result<Vec<_>, _>>()?;
        let state_lengths_exact = state_sequence_lengths
            .iter()
            .all(|length| *length == expected_state_sequence_length);
        let states = cursors
            .iter()
            .map(|cursor| &cursor.state)
            .collect::<Vec<_>>();
        let state_rows_pairwise_distinct = state_refs_pairwise_distinct(parts, &states)?;
        visited_batch_sizes.push(batch_size);
        steps.push(BatchWidthShrinkStep {
            batch_size,
            logical_row_ids: cursors.iter().map(|cursor| cursor.logical_row_id).collect(),
            emitted_tokens_per_row,
            state_sequence_lengths,
            expected_state_sequence_length,
            state_lengths_exact,
            state_rows_pairwise_distinct,
        });
        cursors.pop();
    }
    let expected_widths = (1..=input_tokens.len()).rev().collect::<Vec<_>>();
    let exact = visited_batch_sizes == expected_widths
        && steps.iter().all(|step| {
            step.state_lengths_exact
                && step.state_rows_pairwise_distinct
                && step
                    .emitted_tokens_per_row
                    .iter()
                    .all(|tokens| tokens.len() == GENERATED_TOKENS_PER_STAGE)
        });
    Ok(BatchWidthShrinkObservation {
        initial_batch_size: input_tokens.len(),
        generated_tokens_per_stage: GENERATED_TOKENS_PER_STAGE,
        visited_batch_sizes,
        steps,
        exact,
    })
}

fn prepare_cursor(
    parts: &EngineParts<'_>,
    base: &MlxInferenceState,
    logical_row_id: usize,
    input_token: u32,
) -> Result<WidthCursor, EngineError> {
    let input = batched_token_array(&[input_token])?;
    let runtime_state = prepare_runtime_state(parts.gpu, base, 1, 1, parts.plan)?;
    let execution = execute_runtime_model(
        parts.gpu,
        &input,
        runtime_state,
        parts.weights,
        parts.plan,
        parts.gdn_kernel,
        parts.moe_kernel,
    )?;
    evaluate_runtime_execution(parts.gpu, &execution)?;
    let pending = greedy_token(parts.gpu, &execution.logits, 1, parts.plan)?;
    let pending_token = batched_token_values(parts.gpu, &pending, 1)?[0];
    Ok(WidthCursor {
        logical_row_id,
        state: compact_runtime_state(parts.gpu, execution.state, parts.plan)?,
        pending_token,
    })
}

fn advance_cursors(
    parts: &EngineParts<'_>,
    cursors: Vec<WidthCursor>,
    generated_tokens: usize,
) -> Result<(Vec<Vec<u32>>, Vec<WidthCursor>), EngineError> {
    if cursors.is_empty() || generated_tokens == 0 {
        return Err(EngineError::Unsupported(
            "cursor advance requires active rows and generated tokens".into(),
        ));
    }
    let batch_size = cursors.len();
    let state_refs = cursors
        .iter()
        .map(|cursor| &cursor.state)
        .collect::<Vec<_>>();
    let mut runtime_state =
        prepare_merged_runtime_state(parts.gpu, &state_refs, generated_tokens, parts.plan)?;
    let mut pending_tokens = cursors
        .iter()
        .map(|cursor| cursor.pending_token)
        .collect::<Vec<_>>();
    let mut outputs = (0..batch_size)
        .map(|_| Vec::with_capacity(generated_tokens))
        .collect::<Vec<_>>();
    for _ in 0..generated_tokens {
        let input = batched_token_array(&pending_tokens)?;
        let execution = execute_runtime_model(
            parts.gpu,
            &input,
            runtime_state,
            parts.weights,
            parts.plan,
            parts.gdn_kernel,
            parts.moe_kernel,
        )?;
        evaluate_runtime_execution(parts.gpu, &execution)?;
        for (row, token) in pending_tokens.iter().copied().enumerate() {
            outputs[row].push(token);
        }
        let pending = greedy_token(parts.gpu, &execution.logits, batch_size, parts.plan)?;
        pending_tokens = batched_token_values(parts.gpu, &pending, batch_size)?;
        runtime_state = execution.state;
    }
    let states = split_runtime_state(parts.gpu, runtime_state, parts.plan)?;
    let advanced = cursors
        .into_iter()
        .zip(states)
        .zip(pending_tokens)
        .map(|((cursor, state), pending_token)| WidthCursor {
            logical_row_id: cursor.logical_row_id,
            state,
            pending_token,
        })
        .collect();
    Ok((outputs, advanced))
}

fn state_rows_pairwise_distinct(
    parts: &EngineParts<'_>,
    states: &[MlxInferenceState],
) -> Result<bool, EngineError> {
    let refs = states.iter().collect::<Vec<_>>();
    state_refs_pairwise_distinct(parts, &refs)
}

fn state_refs_pairwise_distinct(
    parts: &EngineParts<'_>,
    states: &[&MlxInferenceState],
) -> Result<bool, EngineError> {
    for left in 0..states.len() {
        for right in left + 1..states.len() {
            if state_max_absolute_difference(parts, states[left], states[right])? == 0.0 {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn state_max_absolute_difference(
    parts: &EngineParts<'_>,
    left: &MlxInferenceState,
    right: &MlxInferenceState,
) -> Result<f32, EngineError> {
    left.validate(parts.plan, 1)?;
    right.validate(parts.plan, 1)?;
    if left.layer_count() != right.layer_count() {
        return Err(EngineError::Unsupported(
            "batch width state comparison layer counts differ".into(),
        ));
    }
    let mut maximum = 0.0_f32;
    for (index, (left_layer, right_layer)) in left.layers().iter().zip(right.layers()).enumerate() {
        let pairs = match (left_layer, right_layer) {
            (
                LayerState::Gdn {
                    convolution: left_first,
                    recurrent: left_second,
                },
                LayerState::Gdn {
                    convolution: right_first,
                    recurrent: right_second,
                },
            )
            | (
                LayerState::Attention {
                    keys: left_first,
                    values: left_second,
                },
                LayerState::Attention {
                    keys: right_first,
                    values: right_second,
                },
            ) => [(left_first, right_first), (left_second, right_second)],
            _ => {
                return Err(EngineError::Unsupported(format!(
                    "batch width state comparison layer {index} kinds differ"
                )));
            }
        };
        for (left_array, right_array) in pairs {
            if left_array.shape() != right_array.shape()
                || left_array.dtype() != right_array.dtype()
            {
                return Err(EngineError::Unsupported(format!(
                    "batch width state comparison layer {index} tensor layout differs"
                )));
            }
            maximum = maximum.max(
                parts
                    .gpu
                    .max_abs_difference(left_array, right_array)
                    .map_err(EngineError::Mlx)?,
            );
        }
    }
    Ok(maximum)
}

fn encode_benchmark_prompt(
    tokenizer: &Qwen35ChatTokenizer,
    row_label: char,
) -> Result<Vec<u32>, EngineError> {
    tokenizer
        .encode_prompt(&EchoChatPrompt {
            input: vec![EchoInputItem::Message(EchoMessage {
                role: EchoMessageRole::User,
                content: EchoMessageContent::Text(format!("{BENCHMARK_PROMPT_PREFIX}{row_label}")),
            })],
            tools: Vec::new(),
        })
        .map(|encoded| encoded.token_ids)
        .map_err(|error| EngineError::Unsupported(format!("encode batch width prompt: {error}")))
}

fn distinct_row_tokens(prompt_rows: &[Vec<u32>]) -> Result<Vec<u32>, EngineError> {
    let first = prompt_rows
        .first()
        .ok_or_else(|| EngineError::Unsupported("batch width prompt rows are empty".into()))?;
    if first.is_empty() || prompt_rows.iter().any(|row| row.len() != first.len()) {
        return Err(EngineError::Unsupported(format!(
            "batch width prompts must have one equal non-zero length, observed {:?}",
            prompt_rows.iter().map(Vec::len).collect::<Vec<_>>()
        )));
    }
    for position in 0..first.len() {
        let values = prompt_rows
            .iter()
            .map(|row| row[position])
            .collect::<Vec<_>>();
        if values.iter().copied().collect::<HashSet<_>>().len() == prompt_rows.len() {
            return Ok(values);
        }
    }
    Err(EngineError::Unsupported(
        "batch width prompts do not expose one distinct token per row".into(),
    ))
}

fn repeated_token_context(seed: &[u32], target: usize) -> Result<Vec<u32>, EngineError> {
    if seed.is_empty() || target == 0 {
        return Err(EngineError::Unsupported(
            "batch width context seed and target must be non-empty".into(),
        ));
    }
    Ok(seed.iter().copied().cycle().take(target).collect())
}

fn prefill_context_state(
    parts: &EngineParts<'_>,
    tokens: &[u32],
    chunk_size: usize,
    additional_tokens: usize,
) -> Result<MlxInferenceState, EngineError> {
    if tokens.is_empty() || chunk_size == 0 {
        return Err(EngineError::Unsupported(
            "batch width prefill requires tokens and a positive chunk size".into(),
        ));
    }
    let initial = MlxInferenceState::empty(parts.gpu, 1, parts.plan)?;
    let capacity = tokens
        .len()
        .checked_add(additional_tokens)
        .ok_or_else(|| EngineError::Unsupported("batch width capacity overflow".into()))?;
    let mut state = prepare_runtime_state(parts.gpu, &initial, 1, capacity, parts.plan)?;
    for chunk in tokens.chunks(chunk_size) {
        let input = sequence_token_array(chunk)?;
        let execution = execute_runtime_model(
            parts.gpu,
            &input,
            state,
            parts.weights,
            parts.plan,
            parts.gdn_kernel,
            parts.moe_kernel,
        )?;
        evaluate_runtime_execution(parts.gpu, &execution)?;
        state = execution.state;
    }
    compact_runtime_state(parts.gpu, state, parts.plan)
}

fn sequence_token_array(tokens: &[u32]) -> Result<Array, EngineError> {
    if tokens.is_empty() {
        return Err(EngineError::Unsupported(
            "sequence token array requires at least one token".into(),
        ));
    }
    let values = token_i32_values(tokens)?;
    Array::from_i32_slice(&values, &[1, values.len()]).map_err(EngineError::Mlx)
}

fn batched_token_array(tokens: &[u32]) -> Result<Array, EngineError> {
    if tokens.is_empty() {
        return Err(EngineError::Unsupported(
            "batched token array requires at least one row".into(),
        ));
    }
    let values = token_i32_values(tokens)?;
    Array::from_i32_slice(&values, &[values.len(), 1]).map_err(EngineError::Mlx)
}

fn token_i32_values(tokens: &[u32]) -> Result<Vec<i32>, EngineError> {
    tokens
        .iter()
        .map(|token| {
            i32::try_from(*token).map_err(|error| {
                EngineError::Unsupported(format!("token ID {token} does not fit int32: {error}"))
            })
        })
        .collect()
}

fn batched_token_values(
    gpu: &Gpu,
    tokens: &Array,
    batch_size: usize,
) -> Result<Vec<u32>, EngineError> {
    if tokens.shape() != [batch_size, 1] {
        return Err(EngineError::Unsupported(format!(
            "sampled tokens must have shape [{batch_size}, 1], observed {:?}",
            tokens.shape()
        )));
    }
    (0..batch_size)
        .map(|row| {
            let row = i32::try_from(row).map_err(|error| {
                EngineError::Unsupported(format!("batch row does not fit int32: {error}"))
            })?;
            gpu.slice(tokens, &[row, 0], &[row + 1, 1], &[1, 1])
                .and_then(|value| gpu.reshape(&value, &[]))
                .and_then(|value| value.item_u32())
                .map_err(EngineError::Mlx)
        })
        .collect()
}

fn engine_parts(engine: &ResidentEngine) -> EngineParts<'_> {
    EngineParts {
        gpu: engine.gpu(),
        plan: engine.diagnostic_plan(),
        weights: engine.diagnostic_weights(),
        gdn_kernel: engine.diagnostic_gdn_kernel(),
        moe_kernel: engine.diagnostic_moe_kernel(),
    }
}

fn rotated_batch_widths(max_batch_size: usize, ordinal: usize) -> Vec<usize> {
    let mut widths = (1..=max_batch_size).collect::<Vec<_>>();
    if !widths.is_empty() {
        let count = widths.len();
        widths.rotate_left(ordinal % count);
    }
    widths
}

fn memory_observation(
    start: MetalMemoryStats,
    end: MetalMemoryStats,
) -> Result<BatchWidthMemoryObservation, EngineError> {
    let start_active = i128::try_from(start.active_nbytes).map_err(|error| {
        EngineError::Unsupported(format!("start active memory does not fit i128: {error}"))
    })?;
    let end_active = i128::try_from(end.active_nbytes).map_err(|error| {
        EngineError::Unsupported(format!("end active memory does not fit i128: {error}"))
    })?;
    let active_delta_nbytes = i64::try_from(end_active - start_active).map_err(|error| {
        EngineError::Unsupported(format!("active memory delta does not fit i64: {error}"))
    })?;
    Ok(BatchWidthMemoryObservation {
        start: start.into(),
        end: end.into(),
        active_delta_nbytes,
        peak_above_start_active_nbytes: u64::try_from(
            end.peak_nbytes.saturating_sub(start.active_nbytes),
        )
        .map_err(|error| {
            EngineError::Unsupported(format!("peak memory delta does not fit u64: {error}"))
        })?,
    })
}

fn token_digest(tokens: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn tokens_per_second(tokens: usize, nanos: u64) -> Result<f64, EngineError> {
    if tokens == 0 || nanos == 0 {
        return Err(EngineError::Unsupported(
            "batch width throughput requires non-zero tokens and duration".into(),
        ));
    }
    let tokens = u32::try_from(tokens).map_err(|error| {
        EngineError::Unsupported(format!("batch width token count does not fit u32: {error}"))
    })?;
    Ok(f64::from(tokens) / Duration::from_nanos(nanos).as_secs_f64())
}

fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn median_u64(mut values: Vec<u64>) -> Result<u64, EngineError> {
    if values.is_empty() {
        return Err(EngineError::Unsupported(
            "batch width median requires samples".into(),
        ));
    }
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        let lower = values[middle - 1];
        Ok(lower.saturating_add((values[middle] - lower) / 2))
    } else {
        Ok(values[middle])
    }
}

fn median_i64(mut values: Vec<i64>) -> Result<i64, EngineError> {
    if values.is_empty() {
        return Err(EngineError::Unsupported(
            "batch width median requires samples".into(),
        ));
    }
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        let lower = i128::from(values[middle - 1]);
        let upper = i128::from(values[middle]);
        i64::try_from(lower + (upper - lower) / 2).map_err(|error| {
            EngineError::Unsupported(format!("batch width signed median overflow: {error}"))
        })
    } else {
        Ok(values[middle])
    }
}

fn median_f64(mut values: Vec<f64>) -> Result<f64, EngineError> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::Unsupported(
            "batch width median requires finite samples".into(),
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
    fn rotates_every_batch_width_through_the_first_position() {
        assert_eq!(rotated_batch_widths(6, 0), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(rotated_batch_widths(6, 1), vec![2, 3, 4, 5, 6, 1]);
        assert_eq!(rotated_batch_widths(6, 5), vec![6, 1, 2, 3, 4, 5]);
        assert_eq!(rotated_batch_widths(6, 6), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn signed_median_handles_negative_allocator_deltas() {
        assert_eq!(median_i64(vec![-8, 4, -2]).expect("odd median"), -2);
        assert_eq!(median_i64(vec![-8, 4]).expect("even median"), -2);
    }
}
