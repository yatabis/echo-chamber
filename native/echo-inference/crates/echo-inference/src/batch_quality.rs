use std::collections::BTreeSet;
use std::path::Path;
use std::time::{Duration, Instant};

use echo_mlx::{Array, Gpu, metal_memory_stats};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::chat::{
    EchoChatPrompt, EchoInputItem, EchoMessage, EchoMessageContent, EchoMessageRole,
    EchoToolContract, EchoToolResult, Qwen35ChatTokenizer, ToolResultKind,
};
use super::decoder::MoeKernel;
use super::full_model::{
    RuntimeInferenceState, RuntimeModelExecution, compact_runtime_state,
    evaluate_runtime_execution, execute_runtime_model, prepare_merged_runtime_state,
    prepare_runtime_state, schedule_runtime_execution, split_runtime_state,
};
use super::gdn::GdnKernel;
use super::model_state::{LayerState, MlxInferenceState};
use super::runtime::{
    GenerationFinishReason, ResidentEngine, ResidentEngineConfig, ResidentEngineInfo,
    RuntimeMemoryStats,
};
use super::sampling::{SamplingConfig, sample_token, sample_token_rows};
use super::tool_output::{EchoOutputItem, parse_qwen_output};
use super::weights::BoundModelWeights;
use super::{EngineError, ModelPlan};

const PREFILL_CHUNK_SIZE: usize = 2_048;
const TOOL_NAME: &str = "record_runtime_probe";

/// Production-sampling execution strategy used by the adoption diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProductionBatchMode {
    /// Complete one request before decoding the second.
    FifoSerial,
    /// Decode both active requests as a two-row model batch.
    FixedBatchTwo,
}

/// One measured production-sampling throughput attempt.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProductionBatchPerformanceAttempt {
    pub phase: &'static str,
    pub round: usize,
    pub order_position: usize,
    pub mode: ProductionBatchMode,
    pub pair_decode_nanos: u64,
    pub request_decode_nanos: [u64; 2],
    pub aggregate_tokens_per_second: f64,
    pub request_tokens_per_second: [f64; 2],
    pub output_token_sha256: [String; 2],
    pub state_sequence_lengths: [usize; 2],
    pub state_lengths_exact: bool,
}

/// Median production-sampling behavior for one execution strategy.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProductionBatchPerformanceSummary {
    pub mode: ProductionBatchMode,
    pub count: usize,
    pub median_pair_decode_nanos: u64,
    pub median_request_decode_nanos: [u64; 2],
    pub median_aggregate_tokens_per_second: f64,
    pub median_request_tokens_per_second: [f64; 2],
    pub exact_state_attempts: usize,
}

/// Fixed-membership production-sampler isolation under co-tenant replacement
/// and row permutation.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProductionSamplingIsolationObservation {
    pub generated_tokens_per_instance: usize,
    pub request_seeds: [u64; 3],
    pub co_tenant_a_output_exact: bool,
    pub co_tenant_a_state_max_absolute_difference: f32,
    pub row_permutation_output_exact: [bool; 2],
    pub row_permutation_state_max_absolute_difference: [f32; 2],
    pub primary_rows_are_distinct: bool,
    pub exact: bool,
}

/// Mixed stop-token and output-limit completion at one batched token boundary.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProductionBatchLengthBoundaryObservation {
    pub max_new_tokens: usize,
    pub generated_token_counts: [usize; 2],
    pub finish_reasons: [GenerationFinishReason; 2],
    pub state_sequence_lengths: [usize; 2],
    pub state_length_accounting_exact: [bool; 2],
    pub mixed_stop_and_length: bool,
    pub exact: bool,
}

/// User-visible and parser-visible result of one two-request workflow turn.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProductionBatchWorkflowTurn {
    pub input_token_counts: [usize; 2],
    pub generated_token_counts: [usize; 2],
    pub finish_reasons: [GenerationFinishReason; 2],
    pub switched_survivor_to_single_row: bool,
    pub elapsed_nanos: u64,
    pub text: [String; 2],
    pub parser_warnings: [Option<String>; 2],
    pub parsed_output: [Vec<EchoOutputItem>; 2],
    pub state_sequence_lengths: [usize; 2],
    pub state_length_accounting_exact: [bool; 2],
}

/// One sampled two-turn tool loop using the exact admitted Qwen template.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProductionBatchWorkflowCase {
    pub case_index: usize,
    pub request_seeds: [u64; 2],
    pub tool_turn: ProductionBatchWorkflowTurn,
    pub expected_tool_calls_exact: [bool; 2],
    pub continuation_turn: ProductionBatchWorkflowTurn,
    pub continuation_messages_valid: [bool; 2],
    pub exact_state_accounting: bool,
    pub quality_checks_passed: bool,
}

/// Bounded evidence for deciding whether two-row continuous batching is safe
/// to promote into the production scheduler.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProductionBatchQualityDiagnostic {
    pub schema_version: u32,
    pub engine: ResidentEngineInfo,
    pub sampling: SamplingConfig,
    pub performance_context_tokens: usize,
    pub generated_tokens_per_performance_request: usize,
    pub warmup_rounds: usize,
    pub measured_rounds: usize,
    pub performance_attempts: Vec<ProductionBatchPerformanceAttempt>,
    pub performance_summaries: Vec<ProductionBatchPerformanceSummary>,
    pub batch_aggregate_throughput_gain_percent: f64,
    pub sampling_isolation: ProductionSamplingIsolationObservation,
    pub length_boundary: ProductionBatchLengthBoundaryObservation,
    pub workflow_cases: Vec<ProductionBatchWorkflowCase>,
    pub all_workflow_quality_checks_passed: bool,
    pub final_metal_memory: RuntimeMemoryStats,
    pub adoption_gate_passed: bool,
}

struct EngineParts<'a> {
    gpu: &'a Gpu,
    plan: &'a ModelPlan,
    weights: &'a BoundModelWeights,
    gdn_kernel: &'a GdnKernel,
    moe_kernel: &'a MoeKernel,
}

struct FixedGeneration {
    output: Vec<u32>,
    state: MlxInferenceState,
    decode_nanos: u64,
}

struct FixedPairGeneration {
    output: [Vec<u32>; 2],
    state: [MlxInferenceState; 2],
    pair_decode_nanos: u64,
    request_decode_nanos: [u64; 2],
}

struct FinishedGeneration {
    output: Vec<u32>,
    state: MlxInferenceState,
    finish_reason: GenerationFinishReason,
}

struct CompletedPair {
    rows: [FinishedGeneration; 2],
    switched_survivor_to_single_row: bool,
    elapsed_nanos: u64,
}

/// Runs production sampling, isolation, EOS membership-change, tool parsing,
/// and exact continuation probes without changing the production scheduler.
///
/// # Errors
///
/// Returns [`EngineError`] when model admission, prompt encoding, execution,
/// state accounting, parsing inputs, or metric denominators are invalid.
#[allow(clippy::too_many_lines)]
pub fn run_production_batch_quality_diagnostic(
    model_directory: &Path,
    warmup_rounds: usize,
    measured_rounds: usize,
    generated_tokens: usize,
    context_tokens: usize,
    workflow_case_count: usize,
) -> Result<ProductionBatchQualityDiagnostic, EngineError> {
    if measured_rounds == 0
        || generated_tokens == 0
        || context_tokens == 0
        || workflow_case_count == 0
    {
        return Err(EngineError::Unsupported(
            "production batch diagnostic requires measured rounds, generated tokens, context tokens, and workflow cases"
                .into(),
        ));
    }
    let tokenizer = Qwen35ChatTokenizer::load(model_directory).map_err(|error| {
        EngineError::Unsupported(format!("load production batch tokenizer: {error}"))
    })?;
    let engine = ResidentEngine::load(
        model_directory,
        ResidentEngineConfig {
            max_new_tokens_per_request: generated_tokens.max(256),
            ..ResidentEngineConfig::default()
        },
    )
    .map_err(|error| EngineError::Unsupported(format!("load production batch engine: {error}")))?;
    let parts = engine_parts(&engine);
    let benchmark_prompts = [
        encode_plain_prompt(&tokenizer, "production-batch-probe-A")?,
        encode_plain_prompt(&tokenizer, "production-batch-probe-B")?,
        encode_plain_prompt(&tokenizer, "production-batch-probe-C")?,
    ];
    let input_tokens = distinct_tokens(&benchmark_prompts)?;
    let context = repeated_context(&benchmark_prompts[0], context_tokens)?;
    let base_state = prefill_from_empty(&parts, &context)?;
    let configs = [
        SamplingConfig::echo_production(10_001),
        SamplingConfig::echo_production(20_003),
        SamplingConfig::echo_production(30_007),
    ];

    let performance_attempts = run_performance_attempts(
        &parts,
        &base_state,
        [input_tokens[0], input_tokens[1]],
        [configs[0], configs[1]],
        warmup_rounds,
        measured_rounds,
        generated_tokens,
        context_tokens,
    )?;
    let performance_summaries = [
        summarize_performance(&performance_attempts, ProductionBatchMode::FifoSerial)?,
        summarize_performance(&performance_attempts, ProductionBatchMode::FixedBatchTwo)?,
    ];
    let batch_aggregate_throughput_gain_percent =
        throughput_gain_percent(&performance_summaries[0], &performance_summaries[1])?;
    let sampling_isolation =
        run_sampling_isolation(&parts, &base_state, input_tokens, configs, generated_tokens)?;
    let length_boundary = run_length_boundary(&parts, &tokenizer)?;
    let workflow_cases = run_workflow_cases(
        &parts,
        &tokenizer,
        workflow_case_count,
        generated_tokens.max(256),
    )?;
    let all_workflow_quality_checks_passed =
        workflow_cases.iter().all(|case| case.quality_checks_passed);
    let final_metal_memory = metal_memory_stats().map_err(EngineError::Mlx)?.into();
    let adoption_gate_passed = performance_summaries
        .iter()
        .all(|summary| summary.exact_state_attempts == summary.count)
        && batch_aggregate_throughput_gain_percent > 0.0
        && sampling_isolation.exact
        && length_boundary.exact
        && all_workflow_quality_checks_passed;

    Ok(ProductionBatchQualityDiagnostic {
        schema_version: 1,
        engine: engine.info().clone(),
        sampling: SamplingConfig::echo_production(0),
        performance_context_tokens: context_tokens,
        generated_tokens_per_performance_request: generated_tokens,
        warmup_rounds,
        measured_rounds,
        performance_attempts,
        performance_summaries: performance_summaries.into(),
        batch_aggregate_throughput_gain_percent,
        sampling_isolation,
        length_boundary,
        workflow_cases,
        all_workflow_quality_checks_passed,
        final_metal_memory,
        adoption_gate_passed,
    })
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

#[allow(clippy::too_many_arguments)]
fn run_performance_attempts(
    parts: &EngineParts<'_>,
    base: &MlxInferenceState,
    input_tokens: [u32; 2],
    configs: [SamplingConfig; 2],
    warmup_rounds: usize,
    measured_rounds: usize,
    generated_tokens: usize,
    context_tokens: usize,
) -> Result<Vec<ProductionBatchPerformanceAttempt>, EngineError> {
    let mut attempts = Vec::with_capacity((warmup_rounds + measured_rounds) * 2);
    for ordinal in 0..warmup_rounds + measured_rounds {
        let (phase, round) = if ordinal < warmup_rounds {
            ("warmup", ordinal)
        } else {
            ("measured", ordinal - warmup_rounds)
        };
        let order = if ordinal.is_multiple_of(2) {
            [
                ProductionBatchMode::FifoSerial,
                ProductionBatchMode::FixedBatchTwo,
            ]
        } else {
            [
                ProductionBatchMode::FixedBatchTwo,
                ProductionBatchMode::FifoSerial,
            ]
        };
        for (order_position, mode) in order.into_iter().enumerate() {
            let execution =
                run_fixed_mode(parts, mode, base, input_tokens, configs, generated_tokens)?;
            let expected_length = context_tokens
                .checked_add(1)
                .and_then(|value| value.checked_add(generated_tokens))
                .ok_or_else(|| {
                    EngineError::Unsupported("performance state length overflow".into())
                })?;
            let state_sequence_lengths = [
                execution.state[0].sequence_length()?,
                execution.state[1].sequence_length()?,
            ];
            attempts.push(ProductionBatchPerformanceAttempt {
                phase,
                round,
                order_position,
                mode,
                pair_decode_nanos: execution.pair_decode_nanos,
                request_decode_nanos: execution.request_decode_nanos,
                aggregate_tokens_per_second: tokens_per_second(
                    generated_tokens.saturating_mul(2),
                    execution.pair_decode_nanos,
                )?,
                request_tokens_per_second: [
                    tokens_per_second(generated_tokens, execution.request_decode_nanos[0])?,
                    tokens_per_second(generated_tokens, execution.request_decode_nanos[1])?,
                ],
                output_token_sha256: [
                    token_digest(&execution.output[0]),
                    token_digest(&execution.output[1]),
                ],
                state_sequence_lengths,
                state_lengths_exact: state_sequence_lengths == [expected_length, expected_length],
            });
        }
    }
    Ok(attempts)
}

fn run_fixed_mode(
    parts: &EngineParts<'_>,
    mode: ProductionBatchMode,
    base: &MlxInferenceState,
    input_tokens: [u32; 2],
    configs: [SamplingConfig; 2],
    generated_tokens: usize,
) -> Result<FixedPairGeneration, EngineError> {
    match mode {
        ProductionBatchMode::FifoSerial => {
            let first =
                run_single_fixed(parts, base, input_tokens[0], configs[0], generated_tokens)?;
            let second =
                run_single_fixed(parts, base, input_tokens[1], configs[1], generated_tokens)?;
            Ok(FixedPairGeneration {
                output: [first.output, second.output],
                state: [first.state, second.state],
                pair_decode_nanos: first.decode_nanos.saturating_add(second.decode_nanos),
                request_decode_nanos: [first.decode_nanos, second.decode_nanos],
            })
        }
        ProductionBatchMode::FixedBatchTwo => {
            run_pair_fixed(parts, [base, base], input_tokens, configs, generated_tokens)
        }
    }
}

fn run_single_fixed(
    parts: &EngineParts<'_>,
    base: &MlxInferenceState,
    input_token: u32,
    config: SamplingConfig,
    generated_tokens: usize,
) -> Result<FixedGeneration, EngineError> {
    let state = prepare_runtime_state(
        parts.gpu,
        base,
        1,
        generated_tokens.saturating_add(1),
        parts.plan,
    )?;
    let input = token_array(&[input_token])?;
    let mut execution = execute(parts, &input, state)?;
    evaluate_runtime_execution(parts.gpu, &execution)?;
    let started = Instant::now();
    let mut output = Vec::with_capacity(generated_tokens);
    for _ in 0..generated_tokens {
        let RuntimeModelExecution { logits, state } = execution;
        let token = sample_token(
            parts.gpu,
            &logits,
            &output,
            output.len(),
            config,
            parts.plan.vocabulary_size,
        )?;
        let scalar = parts.gpu.reshape(&token, &[]).map_err(EngineError::Mlx)?;
        let next = execute(parts, &token, state)?;
        schedule_runtime_execution(parts.gpu, &scalar, &next)?;
        output.push(scalar.item_u32().map_err(EngineError::Mlx)?);
        execution = next;
    }
    evaluate_runtime_execution(parts.gpu, &execution)?;
    Ok(FixedGeneration {
        output,
        state: compact_runtime_state(parts.gpu, execution.state, parts.plan)?,
        decode_nanos: duration_nanos(started.elapsed()),
    })
}

fn run_pair_fixed(
    parts: &EngineParts<'_>,
    bases: [&MlxInferenceState; 2],
    input_tokens: [u32; 2],
    configs: [SamplingConfig; 2],
    generated_tokens: usize,
) -> Result<FixedPairGeneration, EngineError> {
    let state = prepare_merged_runtime_state(
        parts.gpu,
        &bases,
        generated_tokens.saturating_add(1),
        parts.plan,
    )?;
    let input = paired_token_array(input_tokens)?;
    let mut execution = execute(parts, &input, state)?;
    evaluate_runtime_execution(parts.gpu, &execution)?;
    let started = Instant::now();
    let mut output = [
        Vec::with_capacity(generated_tokens),
        Vec::with_capacity(generated_tokens),
    ];
    for _ in 0..generated_tokens {
        let RuntimeModelExecution { logits, state } = execution;
        let tokens = sample_token_rows(
            parts.gpu,
            &logits,
            &output,
            &configs,
            parts.plan.vocabulary_size,
        )?;
        let next = execute(parts, &tokens, state)?;
        schedule_runtime_execution(parts.gpu, &tokens, &next)?;
        let values = paired_token_values(parts.gpu, &tokens)?;
        output[0].push(values[0]);
        output[1].push(values[1]);
        execution = next;
    }
    evaluate_runtime_execution(parts.gpu, &execution)?;
    let pair_decode_nanos = duration_nanos(started.elapsed());
    let state = split_pair_state(parts, execution.state)?;
    Ok(FixedPairGeneration {
        output,
        state,
        pair_decode_nanos,
        request_decode_nanos: [pair_decode_nanos, pair_decode_nanos],
    })
}

fn summarize_performance(
    attempts: &[ProductionBatchPerformanceAttempt],
    mode: ProductionBatchMode,
) -> Result<ProductionBatchPerformanceSummary, EngineError> {
    let selected = attempts
        .iter()
        .filter(|attempt| attempt.phase == "measured" && attempt.mode == mode)
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(EngineError::Unsupported(
            "production batch summary requires measured attempts".into(),
        ));
    }
    Ok(ProductionBatchPerformanceSummary {
        mode,
        count: selected.len(),
        median_pair_decode_nanos: median_u64(
            selected
                .iter()
                .map(|attempt| attempt.pair_decode_nanos)
                .collect(),
        )?,
        median_request_decode_nanos: [
            median_u64(
                selected
                    .iter()
                    .map(|attempt| attempt.request_decode_nanos[0])
                    .collect(),
            )?,
            median_u64(
                selected
                    .iter()
                    .map(|attempt| attempt.request_decode_nanos[1])
                    .collect(),
            )?,
        ],
        median_aggregate_tokens_per_second: median_f64(
            selected
                .iter()
                .map(|attempt| attempt.aggregate_tokens_per_second)
                .collect(),
        )?,
        median_request_tokens_per_second: [
            median_f64(
                selected
                    .iter()
                    .map(|attempt| attempt.request_tokens_per_second[0])
                    .collect(),
            )?,
            median_f64(
                selected
                    .iter()
                    .map(|attempt| attempt.request_tokens_per_second[1])
                    .collect(),
            )?,
        ],
        exact_state_attempts: selected
            .iter()
            .filter(|attempt| attempt.state_lengths_exact)
            .count(),
    })
}

fn throughput_gain_percent(
    fifo: &ProductionBatchPerformanceSummary,
    batch: &ProductionBatchPerformanceSummary,
) -> Result<f64, EngineError> {
    if fifo.mode != ProductionBatchMode::FifoSerial
        || batch.mode != ProductionBatchMode::FixedBatchTwo
        || fifo.median_aggregate_tokens_per_second <= 0.0
    {
        return Err(EngineError::Unsupported(
            "production batch throughput summaries are not comparable".into(),
        ));
    }
    Ok(
        (batch.median_aggregate_tokens_per_second / fifo.median_aggregate_tokens_per_second - 1.0)
            * 100.0,
    )
}

fn run_sampling_isolation(
    parts: &EngineParts<'_>,
    base: &MlxInferenceState,
    input_tokens: [u32; 3],
    configs: [SamplingConfig; 3],
    generated_tokens: usize,
) -> Result<ProductionSamplingIsolationObservation, EngineError> {
    let primary = run_pair_fixed(
        parts,
        [base, base],
        [input_tokens[0], input_tokens[1]],
        [configs[0], configs[1]],
        generated_tokens,
    )?;
    let changed_co_tenant = run_pair_fixed(
        parts,
        [base, base],
        [input_tokens[0], input_tokens[2]],
        [configs[0], configs[2]],
        generated_tokens,
    )?;
    let permuted = run_pair_fixed(
        parts,
        [base, base],
        [input_tokens[1], input_tokens[0]],
        [configs[1], configs[0]],
        generated_tokens,
    )?;
    let co_tenant_a_output_exact = primary.output[0] == changed_co_tenant.output[0];
    let co_tenant_a_state_max_absolute_difference =
        state_max_absolute_difference(parts, &primary.state[0], &changed_co_tenant.state[0])?;
    let row_permutation_output_exact = [
        primary.output[0] == permuted.output[1],
        primary.output[1] == permuted.output[0],
    ];
    let row_permutation_state_max_absolute_difference = [
        state_max_absolute_difference(parts, &primary.state[0], &permuted.state[1])?,
        state_max_absolute_difference(parts, &primary.state[1], &permuted.state[0])?,
    ];
    let primary_rows_are_distinct = primary.output[0] != primary.output[1]
        && state_max_absolute_difference(parts, &primary.state[0], &primary.state[1])? > 0.0;
    let exact = co_tenant_a_output_exact
        && co_tenant_a_state_max_absolute_difference == 0.0
        && row_permutation_output_exact == [true, true]
        && row_permutation_state_max_absolute_difference == [0.0, 0.0]
        && primary_rows_are_distinct;
    Ok(ProductionSamplingIsolationObservation {
        generated_tokens_per_instance: generated_tokens,
        request_seeds: [configs[0].seed, configs[1].seed, configs[2].seed],
        co_tenant_a_output_exact,
        co_tenant_a_state_max_absolute_difference,
        row_permutation_output_exact,
        row_permutation_state_max_absolute_difference,
        primary_rows_are_distinct,
        exact,
    })
}

fn run_length_boundary(
    parts: &EngineParts<'_>,
    tokenizer: &Qwen35ChatTokenizer,
) -> Result<ProductionBatchLengthBoundaryObservation, EngineError> {
    const MAX_NEW_TOKENS: usize = 43;

    let (prompts, _) = tool_prompts();
    let encoded = [
        tokenizer
            .encode_prompt(&prompts[0])
            .map_err(|error| {
                EngineError::Unsupported(format!("encode length-boundary prompt A: {error}"))
            })?
            .token_ids,
        tokenizer
            .encode_prompt(&prompts[1])
            .map_err(|error| {
                EngineError::Unsupported(format!("encode length-boundary prompt B: {error}"))
            })?
            .token_ids,
    ];
    let bases = [
        prefill_from_empty(parts, prompt_prefix(&encoded[0])?)?,
        prefill_from_empty(parts, prompt_prefix(&encoded[1])?)?,
    ];
    let completed = complete_pair(
        parts,
        [&bases[0], &bases[1]],
        [last_token(&encoded[0])?, last_token(&encoded[1])?],
        [
            SamplingConfig::echo_production(50_021),
            SamplingConfig::echo_production(50_022),
        ],
        tokenizer.eos_token_id(),
        MAX_NEW_TOKENS,
    )?;
    let turn = observe_turn(
        tokenizer,
        &completed,
        [0, 0],
        [encoded[0].len(), encoded[1].len()],
    )?;
    let mixed_stop_and_length = turn
        .finish_reasons
        .contains(&GenerationFinishReason::StopToken)
        && turn
            .finish_reasons
            .contains(&GenerationFinishReason::Length);
    let exact = mixed_stop_and_length
        && turn
            .state_length_accounting_exact
            .into_iter()
            .all(|value| value);
    Ok(ProductionBatchLengthBoundaryObservation {
        max_new_tokens: MAX_NEW_TOKENS,
        generated_token_counts: turn.generated_token_counts,
        finish_reasons: turn.finish_reasons,
        state_sequence_lengths: turn.state_sequence_lengths,
        state_length_accounting_exact: turn.state_length_accounting_exact,
        mixed_stop_and_length,
        exact,
    })
}

#[allow(clippy::too_many_lines)]
fn run_workflow_cases(
    parts: &EngineParts<'_>,
    tokenizer: &Qwen35ChatTokenizer,
    case_count: usize,
    max_new_tokens: usize,
) -> Result<Vec<ProductionBatchWorkflowCase>, EngineError> {
    let (prompts, expected_inputs) = tool_prompts();
    let encoded = [
        tokenizer
            .encode_prompt(&prompts[0])
            .map_err(|error| EngineError::Unsupported(format!("encode tool prompt A: {error}")))?
            .token_ids,
        tokenizer
            .encode_prompt(&prompts[1])
            .map_err(|error| EngineError::Unsupported(format!("encode tool prompt B: {error}")))?
            .token_ids,
    ];
    let bases = [
        prefill_from_empty(parts, prompt_prefix(&encoded[0])?)?,
        prefill_from_empty(parts, prompt_prefix(&encoded[1])?)?,
    ];
    let input_tokens = [last_token(&encoded[0])?, last_token(&encoded[1])?];
    let continuation_prompts = tool_result_prompts();
    let continuation_tokens = [
        tokenizer
            .encode_continuation(&continuation_prompts[0])
            .map_err(|error| {
                EngineError::Unsupported(format!("encode tool continuation A: {error}"))
            })?
            .token_ids,
        tokenizer
            .encode_continuation(&continuation_prompts[1])
            .map_err(|error| {
                EngineError::Unsupported(format!("encode tool continuation B: {error}"))
            })?
            .token_ids,
    ];

    let mut cases = Vec::with_capacity(case_count);
    for case_index in 0..case_count {
        let case_offset = u64::try_from(case_index)
            .map_err(|error| {
                EngineError::Unsupported(format!("workflow case index overflow: {error}"))
            })?
            .saturating_mul(100);
        let configs = [
            SamplingConfig::echo_production(50_021_u64.saturating_add(case_offset)),
            SamplingConfig::echo_production(50_022_u64.saturating_add(case_offset)),
        ];
        let first = complete_pair(
            parts,
            [&bases[0], &bases[1]],
            input_tokens,
            configs,
            tokenizer.eos_token_id(),
            max_new_tokens,
        )?;
        let tool_turn = observe_turn(
            tokenizer,
            &first,
            [0, 0],
            [encoded[0].len(), encoded[1].len()],
        )?;
        let expected_tool_calls_exact = [
            expected_tool_call(&tool_turn.parsed_output[0], &expected_inputs[0]),
            expected_tool_call(&tool_turn.parsed_output[1], &expected_inputs[1]),
        ];
        let first_lengths = tool_turn.state_sequence_lengths;
        let continuation_bases = [
            append_tokens(
                parts,
                &first.rows[0].state,
                prompt_prefix(&continuation_tokens[0])?,
            )?,
            append_tokens(
                parts,
                &first.rows[1].state,
                prompt_prefix(&continuation_tokens[1])?,
            )?,
        ];
        let continuation = complete_pair(
            parts,
            [&continuation_bases[0], &continuation_bases[1]],
            [
                last_token(&continuation_tokens[0])?,
                last_token(&continuation_tokens[1])?,
            ],
            configs,
            tokenizer.eos_token_id(),
            max_new_tokens,
        )?;
        let continuation_turn = observe_turn(
            tokenizer,
            &continuation,
            first_lengths,
            [continuation_tokens[0].len(), continuation_tokens[1].len()],
        )?;
        let continuation_messages_valid = [
            valid_final_message(&continuation_turn.parsed_output[0]),
            valid_final_message(&continuation_turn.parsed_output[1]),
        ];
        let exact_state_accounting = tool_turn
            .state_length_accounting_exact
            .into_iter()
            .chain(continuation_turn.state_length_accounting_exact)
            .all(|exact| exact);
        let quality_checks_passed = expected_tool_calls_exact == [true, true]
            && continuation_messages_valid == [true, true]
            && tool_turn.parser_warnings == [None, None]
            && continuation_turn.parser_warnings == [None, None]
            && tool_turn.finish_reasons
                == [
                    GenerationFinishReason::StopToken,
                    GenerationFinishReason::StopToken,
                ]
            && continuation_turn.finish_reasons
                == [
                    GenerationFinishReason::StopToken,
                    GenerationFinishReason::StopToken,
                ]
            && exact_state_accounting;
        cases.push(ProductionBatchWorkflowCase {
            case_index,
            request_seeds: [configs[0].seed, configs[1].seed],
            tool_turn,
            expected_tool_calls_exact,
            continuation_turn,
            continuation_messages_valid,
            exact_state_accounting,
            quality_checks_passed,
        });
    }
    Ok(cases)
}

fn tool_prompts() -> ([EchoChatPrompt; 2], [Value; 2]) {
    let tool = EchoToolContract {
        name: TOOL_NAME.into(),
        description:
            "指定されたE.C.H.O.インスタンス名とnonceを、そのまま診断記録へ保存します。保存を求められた場合は必ず一度だけ呼び出してください。"
                .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "instance": { "type": "string" },
                "nonce": { "type": "string" }
            },
            "required": ["instance", "nonce"],
            "additionalProperties": false
        }),
        output_schema: None,
        strict: true,
    };
    let expected = [
        json!({"instance": "rin", "nonce": "alpha-17"}),
        json!({"instance": "marie", "nonce": "beta-29"}),
    ];
    let prompt = |instance: &str, nonce: &str| EchoChatPrompt {
        input: vec![
            EchoInputItem::Message(EchoMessage {
                role: EchoMessageRole::System,
                content: EchoMessageContent::Text(
                    "要求された診断記録は利用可能なツールで実行し、値を改変しないでください。"
                        .into(),
                ),
            }),
            EchoInputItem::Message(EchoMessage {
                role: EchoMessageRole::User,
                content: EchoMessageContent::Text(format!(
                    "{TOOL_NAME}を一度だけ呼び出し、instanceを{instance}、nonceを{nonce}として保存してください。説明文は不要です。"
                )),
            }),
        ],
        tools: vec![tool.clone()],
    };
    (
        [prompt("rin", "alpha-17"), prompt("marie", "beta-29")],
        expected,
    )
}

fn tool_result_prompts() -> [EchoChatPrompt; 2] {
    let prompt = |call_id: &str, instance: &str, nonce: &str| EchoChatPrompt {
        input: vec![EchoInputItem::ToolResult {
            kind: ToolResultKind::ToolResult,
            result: EchoToolResult {
                call_id: call_id.into(),
                output: format!(
                    "{{\"stored\":true,\"instance\":\"{instance}\",\"nonce\":\"{nonce}\"}}\n保存に成功しました。追加のツール呼び出しはせず、短く完了を伝えてください。"
                ),
            },
        }],
        tools: Vec::new(),
    };
    [
        prompt("batch-quality-a", "rin", "alpha-17"),
        prompt("batch-quality-b", "marie", "beta-29"),
    ]
}

fn observe_turn(
    tokenizer: &Qwen35ChatTokenizer,
    completed: &CompletedPair,
    starting_state_lengths: [usize; 2],
    input_token_counts: [usize; 2],
) -> Result<ProductionBatchWorkflowTurn, EngineError> {
    let text = [
        decode_generated(tokenizer, &completed.rows[0].output)?,
        decode_generated(tokenizer, &completed.rows[1].output)?,
    ];
    let parsed = [
        parse_qwen_output(&text[0], "batch-quality-a"),
        parse_qwen_output(&text[1], "batch-quality-b"),
    ];
    let state_sequence_lengths = [
        completed.rows[0].state.sequence_length()?,
        completed.rows[1].state.sequence_length()?,
    ];
    let generated_token_counts = [
        completed.rows[0].output.len(),
        completed.rows[1].output.len(),
    ];
    let finish_reasons = [
        completed.rows[0].finish_reason,
        completed.rows[1].finish_reason,
    ];
    let expected_lengths = [0, 1].map(|row| {
        starting_state_lengths[row]
            .saturating_add(input_token_counts[row])
            .saturating_add(generated_token_counts[row])
            .saturating_add(usize::from(
                finish_reasons[row] == GenerationFinishReason::Length,
            ))
    });
    Ok(ProductionBatchWorkflowTurn {
        input_token_counts,
        generated_token_counts,
        finish_reasons,
        switched_survivor_to_single_row: completed.switched_survivor_to_single_row,
        elapsed_nanos: completed.elapsed_nanos,
        text,
        parser_warnings: [parsed[0].warning.clone(), parsed[1].warning.clone()],
        parsed_output: [parsed[0].output.clone(), parsed[1].output.clone()],
        state_sequence_lengths,
        state_length_accounting_exact: [
            state_sequence_lengths[0] == expected_lengths[0],
            state_sequence_lengths[1] == expected_lengths[1],
        ],
    })
}

fn expected_tool_call(output: &[EchoOutputItem], expected_input: &Value) -> bool {
    let calls = output
        .iter()
        .filter_map(|item| match item {
            EchoOutputItem::ToolCall {
                tool_name, input, ..
            } => Some((tool_name, input)),
            EchoOutputItem::Message { .. } => None,
        })
        .collect::<Vec<_>>();
    calls.len() == 1
        && calls[0].0 == TOOL_NAME
        && serde_json::from_str::<Value>(calls[0].1).ok().as_ref() == Some(expected_input)
}

fn valid_final_message(output: &[EchoOutputItem]) -> bool {
    output.iter().any(|item| {
        matches!(item, EchoOutputItem::Message { content, .. } if !content.trim().is_empty())
    }) && !output
        .iter()
        .any(|item| matches!(item, EchoOutputItem::ToolCall { .. }))
}

fn decode_generated(
    tokenizer: &Qwen35ChatTokenizer,
    generated: &[u32],
) -> Result<String, EngineError> {
    let text_length = generated
        .iter()
        .position(|token| *token == tokenizer.eos_token_id())
        .unwrap_or(generated.len());
    tokenizer
        .decode(&generated[..text_length])
        .map_err(|error| EngineError::Unsupported(format!("decode batch output: {error}")))
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn complete_pair(
    parts: &EngineParts<'_>,
    bases: [&MlxInferenceState; 2],
    input_tokens: [u32; 2],
    configs: [SamplingConfig; 2],
    eos_token: u32,
    max_new_tokens: usize,
) -> Result<CompletedPair, EngineError> {
    let started = Instant::now();
    let state = prepare_merged_runtime_state(
        parts.gpu,
        &bases,
        max_new_tokens.saturating_add(2),
        parts.plan,
    )?;
    let input = paired_token_array(input_tokens)?;
    let mut execution = execute(parts, &input, state)?;
    evaluate_runtime_execution(parts.gpu, &execution)?;
    let mut output = [
        Vec::with_capacity(max_new_tokens),
        Vec::with_capacity(max_new_tokens),
    ];
    loop {
        let RuntimeModelExecution { logits, state } = execution;
        let tokens = sample_token_rows(
            parts.gpu,
            &logits,
            &output,
            &configs,
            parts.plan.vocabulary_size,
        )?;
        let next = execute(parts, &tokens, state)?;
        schedule_runtime_execution(parts.gpu, &tokens, &next)?;
        let values = paired_token_values(parts.gpu, &tokens)?;
        output[0].push(values[0]);
        output[1].push(values[1]);
        let stopped = [values[0] == eos_token, values[1] == eos_token];
        let reached_length = [
            !stopped[0] && output[0].len() == max_new_tokens,
            !stopped[1] && output[1].len() == max_new_tokens,
        ];
        if !stopped[0] && !stopped[1] && !reached_length[0] && !reached_length[1] {
            execution = next;
            continue;
        }

        evaluate_runtime_execution(parts.gpu, &next)?;
        let RuntimeModelExecution {
            logits: next_logits,
            state: next_state,
        } = next;
        let states = split_pair_state(parts, next_state)?;

        match (
            stopped[0] || reached_length[0],
            stopped[1] || reached_length[1],
        ) {
            (true, true) => {
                let [state_a, state_b] =
                    close_length_rows(parts, states, reached_length, eos_token)?;
                return Ok(CompletedPair {
                    rows: [
                        finish_row(&mut output, 0, state_a, stopped[0]),
                        finish_row(&mut output, 1, state_b, stopped[1]),
                    ],
                    switched_survivor_to_single_row: false,
                    elapsed_nanos: duration_nanos(started.elapsed()),
                });
            }
            (true, false) => {
                let [state_a, state_b] = states;
                let state_a = if reached_length[0] {
                    close_single_state(parts, &state_a, eos_token)?
                } else {
                    state_a
                };
                let survivor_logits = slice_logits_row(parts.gpu, &next_logits, 1, 2)?;
                let survivor = continue_single(
                    parts,
                    &state_b,
                    survivor_logits,
                    std::mem::take(&mut output[1]),
                    configs[1],
                    eos_token,
                    max_new_tokens,
                )?;
                return Ok(CompletedPair {
                    rows: [finish_row(&mut output, 0, state_a, stopped[0]), survivor],
                    switched_survivor_to_single_row: true,
                    elapsed_nanos: duration_nanos(started.elapsed()),
                });
            }
            (false, true) => {
                let [state_a, state_b] = states;
                let state_b = if reached_length[1] {
                    close_single_state(parts, &state_b, eos_token)?
                } else {
                    state_b
                };
                let survivor_logits = slice_logits_row(parts.gpu, &next_logits, 0, 2)?;
                let survivor = continue_single(
                    parts,
                    &state_a,
                    survivor_logits,
                    std::mem::take(&mut output[0]),
                    configs[0],
                    eos_token,
                    max_new_tokens,
                )?;
                return Ok(CompletedPair {
                    rows: [survivor, finish_row(&mut output, 1, state_b, stopped[1])],
                    switched_survivor_to_single_row: true,
                    elapsed_nanos: duration_nanos(started.elapsed()),
                });
            }
            (false, false) => {
                return Err(EngineError::Unsupported(
                    "batch completion reached an impossible active state".into(),
                ));
            }
        }
    }
}

fn finish_row(
    output: &mut [Vec<u32>; 2],
    row: usize,
    state: MlxInferenceState,
    stopped: bool,
) -> FinishedGeneration {
    FinishedGeneration {
        output: std::mem::take(&mut output[row]),
        state,
        finish_reason: if stopped {
            GenerationFinishReason::StopToken
        } else {
            GenerationFinishReason::Length
        },
    }
}

fn continue_single(
    parts: &EngineParts<'_>,
    state: &MlxInferenceState,
    mut logits: Array,
    mut output: Vec<u32>,
    config: SamplingConfig,
    eos_token: u32,
    max_new_tokens: usize,
) -> Result<FinishedGeneration, EngineError> {
    let remaining = max_new_tokens.saturating_sub(output.len());
    let mut state =
        prepare_runtime_state(parts.gpu, state, 1, remaining.saturating_add(1), parts.plan)?;
    while output.len() < max_new_tokens {
        let token = sample_token(
            parts.gpu,
            &logits,
            &output,
            output.len(),
            config,
            parts.plan.vocabulary_size,
        )?;
        let scalar = parts.gpu.reshape(&token, &[]).map_err(EngineError::Mlx)?;
        let next = execute(parts, &token, state)?;
        schedule_runtime_execution(parts.gpu, &scalar, &next)?;
        let value = scalar.item_u32().map_err(EngineError::Mlx)?;
        output.push(value);
        if value == eos_token {
            evaluate_runtime_execution(parts.gpu, &next)?;
            return Ok(FinishedGeneration {
                output,
                state: compact_runtime_state(parts.gpu, next.state, parts.plan)?,
                finish_reason: GenerationFinishReason::StopToken,
            });
        }
        let RuntimeModelExecution {
            logits: next_logits,
            state: next_state,
        } = next;
        logits = next_logits;
        state = next_state;
    }
    let state = close_runtime_state(parts, state, eos_token)?;
    Ok(FinishedGeneration {
        output,
        state,
        finish_reason: GenerationFinishReason::Length,
    })
}

fn close_length_rows(
    parts: &EngineParts<'_>,
    states: [MlxInferenceState; 2],
    reached_length: [bool; 2],
    eos_token: u32,
) -> Result<[MlxInferenceState; 2], EngineError> {
    let [state_a, state_b] = states;
    Ok([
        if reached_length[0] {
            close_single_state(parts, &state_a, eos_token)?
        } else {
            state_a
        },
        if reached_length[1] {
            close_single_state(parts, &state_b, eos_token)?
        } else {
            state_b
        },
    ])
}

fn close_single_state(
    parts: &EngineParts<'_>,
    state: &MlxInferenceState,
    eos_token: u32,
) -> Result<MlxInferenceState, EngineError> {
    let state = prepare_runtime_state(parts.gpu, state, 1, 1, parts.plan)?;
    close_runtime_state(parts, state, eos_token)
}

fn close_runtime_state(
    parts: &EngineParts<'_>,
    state: RuntimeInferenceState,
    eos_token: u32,
) -> Result<MlxInferenceState, EngineError> {
    let eos = token_array(&[eos_token])?;
    let execution = execute(parts, &eos, state)?;
    evaluate_runtime_execution(parts.gpu, &execution)?;
    compact_runtime_state(parts.gpu, execution.state, parts.plan)
}

fn execute(
    parts: &EngineParts<'_>,
    input: &Array,
    state: RuntimeInferenceState,
) -> Result<RuntimeModelExecution, EngineError> {
    execute_runtime_model(
        parts.gpu,
        input,
        state,
        parts.weights,
        parts.plan,
        parts.gdn_kernel,
        parts.moe_kernel,
    )
}

fn prefill_from_empty(
    parts: &EngineParts<'_>,
    tokens: &[u32],
) -> Result<MlxInferenceState, EngineError> {
    let initial = MlxInferenceState::empty(parts.gpu, 1, parts.plan)?;
    append_tokens(parts, &initial, tokens)
}

fn append_tokens(
    parts: &EngineParts<'_>,
    base: &MlxInferenceState,
    tokens: &[u32],
) -> Result<MlxInferenceState, EngineError> {
    if tokens.is_empty() {
        return Err(EngineError::Unsupported(
            "batch quality prefill requires at least one token".into(),
        ));
    }
    let mut state = prepare_runtime_state(parts.gpu, base, 1, tokens.len(), parts.plan)?;
    let mut final_execution = None;
    for chunk in tokens.chunks(PREFILL_CHUNK_SIZE) {
        let input = token_array(chunk)?;
        let execution = execute(parts, &input, state)?;
        evaluate_runtime_execution(parts.gpu, &execution)?;
        state = execution.state;
        final_execution = Some(());
    }
    if final_execution.is_none() {
        return Err(EngineError::Unsupported(
            "batch quality prefill produced no execution".into(),
        ));
    }
    compact_runtime_state(parts.gpu, state, parts.plan)
}

fn split_pair_state(
    parts: &EngineParts<'_>,
    state: RuntimeInferenceState,
) -> Result<[MlxInferenceState; 2], EngineError> {
    let rows = split_runtime_state(parts.gpu, state, parts.plan)?;
    let count = rows.len();
    rows.try_into().map_err(|_| {
        EngineError::Unsupported(format!(
            "production batch split produced {count} states instead of two"
        ))
    })
}

fn state_max_absolute_difference(
    parts: &EngineParts<'_>,
    left: &MlxInferenceState,
    right: &MlxInferenceState,
) -> Result<f32, EngineError> {
    left.validate(parts.plan, 1)?;
    right.validate(parts.plan, 1)?;
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
                    "production batch state layer {index} kinds differ"
                )));
            }
        };
        for (left_array, right_array) in pairs {
            if left_array.shape() != right_array.shape()
                || left_array.dtype() != right_array.dtype()
            {
                return Err(EngineError::Unsupported(format!(
                    "production batch state layer {index} tensor layout differs"
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

fn slice_logits_row(
    gpu: &Gpu,
    logits: &Array,
    row: usize,
    batch_size: usize,
) -> Result<Array, EngineError> {
    let shape = logits.shape();
    let [observed_batch, sequence_length, vocabulary_size] = <[usize; 3]>::try_from(shape.clone())
        .map_err(|shape| {
            EngineError::Unsupported(format!(
                "production batch logits must be rank 3, observed {shape:?}"
            ))
        })?;
    if observed_batch != batch_size || row >= batch_size {
        return Err(EngineError::Unsupported(format!(
            "production batch logits row {row} is invalid for shape {shape:?}"
        )));
    }
    gpu.slice(
        logits,
        &[dimension(row, "logit row")?, 0, 0],
        &[
            dimension(row + 1, "logit row stop")?,
            dimension(sequence_length, "logit sequence")?,
            dimension(vocabulary_size, "logit vocabulary")?,
        ],
        &[1, 1, 1],
    )
    .map_err(EngineError::Mlx)
}

fn token_array(tokens: &[u32]) -> Result<Array, EngineError> {
    let values = tokens
        .iter()
        .map(|token| {
            i32::try_from(*token).map_err(|error| {
                EngineError::Unsupported(format!(
                    "production batch token {token} does not fit int32: {error}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Array::from_i32_slice(&values, &[1, values.len()]).map_err(EngineError::Mlx)
}

fn paired_token_array(tokens: [u32; 2]) -> Result<Array, EngineError> {
    let values = tokens
        .into_iter()
        .map(|token| {
            i32::try_from(token).map_err(|error| {
                EngineError::Unsupported(format!(
                    "production batch token {token} does not fit int32: {error}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Array::from_i32_slice(&values, &[2, 1]).map_err(EngineError::Mlx)
}

fn paired_token_values(gpu: &Gpu, tokens: &Array) -> Result<[u32; 2], EngineError> {
    if tokens.shape() != vec![2, 1] {
        return Err(EngineError::Unsupported(format!(
            "production sampled tokens must have shape [2, 1], observed {:?}",
            tokens.shape()
        )));
    }
    Ok([
        gpu.slice(tokens, &[0, 0], &[1, 1], &[1, 1])
            .and_then(|value| gpu.reshape(&value, &[]))
            .and_then(|value| value.item_u32())
            .map_err(EngineError::Mlx)?,
        gpu.slice(tokens, &[1, 0], &[2, 1], &[1, 1])
            .and_then(|value| gpu.reshape(&value, &[]))
            .and_then(|value| value.item_u32())
            .map_err(EngineError::Mlx)?,
    ])
}

fn encode_plain_prompt(
    tokenizer: &Qwen35ChatTokenizer,
    label: &str,
) -> Result<Vec<u32>, EngineError> {
    tokenizer
        .encode_prompt(&EchoChatPrompt {
            input: vec![EchoInputItem::Message(EchoMessage {
                role: EchoMessageRole::User,
                content: EchoMessageContent::Text(format!(
                    "これは本番samplingの分離診断です。識別子は{label}です。"
                )),
            })],
            tools: Vec::new(),
        })
        .map(|encoded| encoded.token_ids)
        .map_err(|error| EngineError::Unsupported(format!("encode sampling prompt: {error}")))
}

fn distinct_tokens(prompts: &[Vec<u32>; 3]) -> Result<[u32; 3], EngineError> {
    let values = prompts
        .iter()
        .flat_map(|prompt| prompt.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .take(3)
        .collect::<Vec<_>>();
    values.try_into().map_err(|values: Vec<u32>| {
        EngineError::Unsupported(format!(
            "sampling prompts yielded only {} distinct token IDs",
            values.len()
        ))
    })
}

fn repeated_context(seed: &[u32], target: usize) -> Result<Vec<u32>, EngineError> {
    if seed.is_empty() || target == 0 {
        return Err(EngineError::Unsupported(
            "production sampling context requires a non-empty seed and target".into(),
        ));
    }
    Ok(seed.iter().copied().cycle().take(target).collect())
}

fn prompt_prefix(tokens: &[u32]) -> Result<&[u32], EngineError> {
    tokens
        .get(..tokens.len().saturating_sub(1))
        .filter(|prefix| !prefix.is_empty())
        .ok_or_else(|| {
            EngineError::Unsupported("workflow prompt must contain at least two tokens".into())
        })
}

fn last_token(tokens: &[u32]) -> Result<u32, EngineError> {
    tokens.last().copied().ok_or_else(|| {
        EngineError::Unsupported("workflow prompt unexpectedly contains no tokens".into())
    })
}

fn dimension(value: usize, label: &str) -> Result<i32, EngineError> {
    i32::try_from(value)
        .map_err(|error| EngineError::Unsupported(format!("{label} does not fit int32: {error}")))
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
            "production sampling throughput requires tokens and duration".into(),
        ));
    }
    let tokens = u32::try_from(tokens).map_err(|error| {
        EngineError::Unsupported(format!("production token count does not fit u32: {error}"))
    })?;
    Ok(f64::from(tokens) / Duration::from_nanos(nanos).as_secs_f64())
}

fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn median_u64(mut values: Vec<u64>) -> Result<u64, EngineError> {
    if values.is_empty() {
        return Err(EngineError::Unsupported(
            "production batch median requires samples".into(),
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

fn median_f64(mut values: Vec<f64>) -> Result<f64, EngineError> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::Unsupported(
            "production batch median requires finite samples".into(),
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
