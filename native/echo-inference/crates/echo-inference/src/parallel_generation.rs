use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use echo_inference_state::{
    CommittedState, ExpectedState, InstanceId, ModelIdentity, PreparedState, StateStore,
};
use echo_mlx::{Array, Gpu, MetalMemoryStats, SafeTensors, metal_memory_stats, reset_peak_memory};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::chat::{
    EchoChatPrompt, EchoInputItem, EchoMessage, EchoMessageContent, EchoMessageRole,
    Qwen35ChatTokenizer,
};
use super::decoder::MoeKernel;
use super::full_model::{
    RuntimeModelExecution, compact_runtime_state, evaluate_runtime_execution,
    execute_runtime_model, greedy_token, prepare_merged_runtime_state, prepare_runtime_state,
    runtime_execution_arrays, schedule_runtime_execution, split_runtime_state,
};
use super::gdn::GdnKernel;
use super::model_state::{LayerState, MlxInferenceState};
use super::runtime::{
    ResidentEngine, ResidentEngineConfig, ResidentEngineInfo, RuntimeMemoryStats,
};
use super::snapshot::restore_named_state;
use super::weights::BoundModelWeights;
use super::{EngineError, ModelPlan, sha256_file};

const BENCHMARK_PROMPT_PREFIX: &str = concat!(
    "これは2インスタンス同時生成の推論性能測定です。ツールは使わず、",
    "指定された固定文字列だけを独立した行へ繰り返してください。\n",
    "各行には前置き、番号、説明、後書き、省略記号を加えないでください。\n",
    "固定文字列: native-parallel-instance-"
);

/// Diagnostic execution strategy for two simultaneous E.C.H.O. instances.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelGenerationMode {
    /// Current production baseline: finish one request before starting the next.
    FifoSerial,
    /// Share weights but issue each request through a distinct MLX GPU stream.
    IndependentStreams,
    /// Execute the two active sequences as one fixed batch of two.
    FixedBatchTwo,
}

/// Process-wide Metal allocator observation around one two-request attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct ParallelMemoryObservation {
    pub start: RuntimeMemoryStats,
    pub end: RuntimeMemoryStats,
    pub active_delta_nbytes: i64,
    pub peak_above_start_active_nbytes: u64,
}

/// Correctness and isolation comparison against the serial reference pair.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ParallelCorrectnessObservation {
    pub output_matches_serial: [bool; 2],
    pub state_max_absolute_difference_from_own_serial: [f32; 2],
    pub state_max_absolute_difference_from_other_serial: [f32; 2],
    pub candidate_state_rows_max_absolute_difference: f32,
    pub exact_serial_equivalence: bool,
    pub each_state_is_closer_to_own_serial: bool,
}

/// Fixed-batch isolation under changed co-tenants and batch-row placement.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FixedBatchIsolationObservation {
    pub co_tenant_a_output_exact: bool,
    pub co_tenant_a_state_max_absolute_difference: f32,
    pub row_permutation_output_exact: [bool; 2],
    pub row_permutation_state_max_absolute_difference: [f32; 2],
    pub duplicate_a_output_rows_exact: bool,
    pub duplicate_a_state_rows_max_absolute_difference: f32,
    pub exact: bool,
}

/// Merge, split, and continuation behavior for already-resident unequal states.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ResidentBatchContinuationObservation {
    pub initial_state_sequence_lengths: [usize; 3],
    pub left_padding: [usize; 2],
    pub roundtrip_state_max_absolute_difference: [f32; 2],
    pub co_tenant_a_output_exact: bool,
    pub co_tenant_a_state_max_absolute_difference: f32,
    pub row_permutation_output_exact: [bool; 2],
    pub row_permutation_state_max_absolute_difference: [f32; 2],
    pub split_remerge_first_stage_output_exact: [bool; 2],
    pub split_remerge_second_stage_output_exact: [bool; 2],
    pub split_remerge_final_state_max_absolute_difference: [f32; 2],
    pub exact: bool,
}

/// State and transaction accounting across live batch membership changes.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DynamicBatchLifecycleObservation {
    pub base_state_sequence_lengths: [usize; 2],
    pub generated_tokens_before_join: usize,
    pub generated_tokens_while_joined: usize,
    pub generated_tokens_after_leave: usize,
    pub state_sequence_lengths_before_join: [usize; 2],
    pub state_sequence_lengths_at_leave: [usize; 2],
    pub final_survivor_sequence_length: usize,
    pub sequence_length_accounting_exact: bool,
    pub emitted_token_counts_exact: bool,
    pub survivor_commit_visible: bool,
    pub cancelled_base_preserved: bool,
    pub instance_leases_released: bool,
    pub exact: bool,
}

#[derive(Debug, Deserialize)]
struct ResidentBatchOracleManifest {
    schema_version: u32,
    model_type: String,
    config_sha256: String,
    history_token_rows: Vec<Vec<u32>>,
    continuation_token_ids: Vec<u32>,
    generation_steps: usize,
    expected_generated_token_rows: Vec<Vec<u32>>,
    expected_state_files: Vec<String>,
    expected_state_sha256: Vec<String>,
    expected_state_tensor_counts: Vec<usize>,
}

/// Direct comparison against an official MLX-LM unequal-cache merge oracle.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ResidentBatchOracleParity {
    pub schema_version: u32,
    pub engine: ResidentEngineInfo,
    pub manifest_path: PathBuf,
    pub history_sequence_lengths: [usize; 2],
    pub continuation_token_ids: [u32; 2],
    pub generation_steps: usize,
    pub expected_state_sha256: [String; 2],
    pub generated_token_rows: [Vec<u32>; 2],
    pub output_matches_oracle: [bool; 2],
    pub final_state_max_absolute_difference_from_oracle: [f32; 2],
    pub exact: bool,
}

/// One warmup or measured pair under one diagnostic strategy.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ParallelGenerationAttempt {
    pub phase: &'static str,
    pub round: usize,
    pub order_position: usize,
    pub mode: ParallelGenerationMode,
    pub pair_total_nanos: u64,
    pub pair_decode_nanos: u64,
    pub ttft_nanos: [u64; 2],
    pub completion_nanos: [u64; 2],
    pub request_decode_nanos: [u64; 2],
    pub aggregate_decode_tokens_per_second: f64,
    pub aggregate_end_to_end_tokens_per_second: f64,
    pub request_decode_tokens_per_second: [f64; 2],
    pub output_token_sha256: [String; 2],
    pub state_sequence_lengths: [usize; 2],
    pub state_logical_nbytes: [usize; 2],
    pub memory: ParallelMemoryObservation,
    pub correctness: ParallelCorrectnessObservation,
}

/// Median measured behavior for one strategy.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ParallelGenerationSummary {
    pub mode: ParallelGenerationMode,
    pub count: usize,
    pub median_pair_total_nanos: u64,
    pub median_pair_decode_nanos: u64,
    pub median_ttft_nanos: [u64; 2],
    pub median_completion_nanos: [u64; 2],
    pub median_aggregate_decode_tokens_per_second: f64,
    pub median_aggregate_end_to_end_tokens_per_second: f64,
    pub median_request_decode_tokens_per_second: [f64; 2],
    pub median_peak_above_start_active_nbytes: u64,
    pub exact_serial_equivalent_attempts: usize,
    pub own_serial_state_closer_attempts: usize,
}

/// Feature-gated real-model comparison of the current FIFO and two candidates.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ParallelGenerationDiagnostic {
    pub schema_version: u32,
    pub engine: ResidentEngineInfo,
    pub prompt_tokens_per_instance: usize,
    pub max_new_tokens: usize,
    pub warmup_rounds: usize,
    pub measured_rounds: usize,
    pub independent_stream_indices: [i32; 2],
    pub serial_reference_output_sha256: [String; 2],
    pub serial_reference_state_difference: f32,
    pub fixed_batch_isolation: FixedBatchIsolationObservation,
    pub resident_batch_continuation: ResidentBatchContinuationObservation,
    pub dynamic_batch_lifecycle: DynamicBatchLifecycleObservation,
    pub attempts: Vec<ParallelGenerationAttempt>,
    pub summaries: Vec<ParallelGenerationSummary>,
    pub final_metal_memory: RuntimeMemoryStats,
}

/// One resident-context FIFO or fixed-batch measurement.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ResidentBatchContextAttempt {
    pub target_context_tokens: usize,
    pub phase: &'static str,
    pub round: usize,
    pub order_position: usize,
    pub mode: ParallelGenerationMode,
    pub pair_total_nanos: u64,
    pub pair_decode_nanos: u64,
    pub ttft_nanos: [u64; 2],
    pub completion_nanos: [u64; 2],
    pub aggregate_decode_tokens_per_second: f64,
    pub request_decode_tokens_per_second: [f64; 2],
    pub state_sequence_lengths: [usize; 2],
    pub state_lengths_exact: bool,
    pub memory: ParallelMemoryObservation,
}

/// Median resident-context behavior for one execution mode.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ResidentBatchContextSummary {
    pub target_context_tokens: usize,
    pub mode: ParallelGenerationMode,
    pub count: usize,
    pub median_pair_total_nanos: u64,
    pub median_pair_decode_nanos: u64,
    pub median_ttft_nanos: [u64; 2],
    pub median_completion_nanos: [u64; 2],
    pub median_aggregate_decode_tokens_per_second: f64,
    pub median_request_decode_tokens_per_second: [f64; 2],
    pub median_peak_above_start_active_nbytes: u64,
    pub exact_state_attempts: usize,
}

/// Resident batch scaling across E.C.H.O.'s admitted context tiers.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ResidentBatchContextDiagnostic {
    pub schema_version: u32,
    pub engine: ResidentEngineInfo,
    pub context_targets: [usize; 3],
    pub prefill_chunk_size_tokens: usize,
    pub generated_tokens_per_instance: usize,
    pub warmup_rounds: usize,
    pub measured_rounds: usize,
    pub base_prefill_nanos: [u64; 3],
    pub base_state_logical_nbytes: [usize; 3],
    pub attempts: Vec<ResidentBatchContextAttempt>,
    pub summaries: Vec<ResidentBatchContextSummary>,
    pub final_metal_memory: RuntimeMemoryStats,
}

struct EngineParts<'a> {
    gpu: &'a Gpu,
    plan: &'a ModelPlan,
    weights: &'a BoundModelWeights,
    gdn_kernel: &'a GdnKernel,
    moe_kernel: &'a MoeKernel,
}

struct SequenceExecution {
    output: Vec<u32>,
    state: MlxInferenceState,
    ttft_nanos: u64,
    decode_nanos: u64,
    total_nanos: u64,
}

struct PairExecution {
    output: [Vec<u32>; 2],
    state: [MlxInferenceState; 2],
    pair_total_nanos: u64,
    pair_decode_nanos: u64,
    ttft_nanos: [u64; 2],
    completion_nanos: [u64; 2],
    request_decode_nanos: [u64; 2],
    memory: ParallelMemoryObservation,
}

struct ResidentBatchExecution {
    output: [Vec<u32>; 2],
    state: [MlxInferenceState; 2],
}

struct ResidentTwoStageExecution {
    first_output: [Vec<u32>; 2],
    second_output: [Vec<u32>; 2],
    state: [MlxInferenceState; 2],
}

struct DecodeCursor {
    state: MlxInferenceState,
    pending_token: u32,
}

/// Runs the bounded two-instance diagnostic without changing the production
/// scheduler or protocol.
///
/// `fixed_batch_two` is intentionally only the smallest active-batch kernel:
/// both requests arrive together, have equal prompt and output lengths, and
/// leave together. A winning result would justify a later continuous-batching
/// scheduler; it is not itself that production scheduler.
///
/// # Errors
///
/// Returns [`EngineError`] when the configuration, tokenizer, model, stream,
/// execution, state comparison, or metric denominator is invalid.
#[allow(clippy::too_many_lines)]
pub fn run_parallel_generation_diagnostic(
    model_directory: &Path,
    warmup_rounds: usize,
    measured_rounds: usize,
    max_new_tokens: usize,
) -> Result<ParallelGenerationDiagnostic, EngineError> {
    if measured_rounds == 0 || max_new_tokens == 0 {
        return Err(EngineError::Unsupported(
            "parallel diagnostic requires measured rounds and generated tokens".into(),
        ));
    }
    let tokenizer = Qwen35ChatTokenizer::load(model_directory).map_err(|error| {
        EngineError::Unsupported(format!("load parallel diagnostic tokenizer: {error}"))
    })?;
    let inputs = [
        encode_benchmark_prompt(&tokenizer, 'A')?,
        encode_benchmark_prompt(&tokenizer, 'B')?,
        encode_benchmark_prompt(&tokenizer, 'C')?,
    ];
    if inputs.iter().any(|input| input.len() != inputs[0].len()) {
        return Err(EngineError::Unsupported(format!(
            "parallel diagnostic prompts must have equal token lengths, observed {:?}",
            inputs.iter().map(Vec::len).collect::<Vec<_>>()
        )));
    }
    let primary_inputs = [inputs[0].clone(), inputs[1].clone()];
    let engine = ResidentEngine::load(
        model_directory,
        ResidentEngineConfig {
            max_new_tokens_per_request: max_new_tokens,
            ..ResidentEngineConfig::default()
        },
    )
    .map_err(|error| EngineError::Unsupported(format!("load parallel engine: {error}")))?;
    let stream_a = Gpu::new_independent().map_err(EngineError::Mlx)?;
    let stream_b = Gpu::new_independent().map_err(EngineError::Mlx)?;
    let stream_indices = [
        stream_a.stream_index().map_err(EngineError::Mlx)?,
        stream_b.stream_index().map_err(EngineError::Mlx)?,
    ];
    if stream_indices[0] == stream_indices[1] {
        return Err(EngineError::Unsupported(
            "parallel diagnostic received duplicate MLX stream indices".into(),
        ));
    }

    let reference = run_mode(
        ParallelGenerationMode::FifoSerial,
        &engine,
        [&stream_a, &stream_b],
        &primary_inputs,
        max_new_tokens,
    )?;
    let reference_state_difference = state_max_absolute_difference(
        engine.gpu(),
        &reference.state[0],
        &reference.state[1],
        engine.diagnostic_plan(),
    )?;
    if reference_state_difference == 0.0 {
        return Err(EngineError::Unsupported(
            "parallel diagnostic prompts did not produce distinct serial states".into(),
        ));
    }
    let reference_output_sha256 = [
        token_digest(&reference.output[0]),
        token_digest(&reference.output[1]),
    ];

    let mut attempts = Vec::with_capacity(
        warmup_rounds
            .saturating_add(measured_rounds)
            .saturating_mul(3),
    );
    for ordinal in 0..warmup_rounds.saturating_add(measured_rounds) {
        let measured = ordinal >= warmup_rounds;
        let round = if measured {
            ordinal - warmup_rounds + 1
        } else {
            ordinal + 1
        };
        for (order_position, mode) in mode_order(ordinal).into_iter().enumerate() {
            let candidate = run_mode(
                mode,
                &engine,
                [&stream_a, &stream_b],
                &primary_inputs,
                max_new_tokens,
            )?;
            let correctness = compare_to_reference(
                engine.gpu(),
                &candidate,
                &reference,
                engine.diagnostic_plan(),
            )?;
            attempts.push(to_attempt(
                if measured { "measured" } else { "warmup" },
                round,
                order_position + 1,
                mode,
                &candidate,
                correctness,
                max_new_tokens,
            )?);
        }
    }

    let summaries = [
        ParallelGenerationMode::FifoSerial,
        ParallelGenerationMode::IndependentStreams,
        ParallelGenerationMode::FixedBatchTwo,
    ]
    .into_iter()
    .map(|mode| summarize_mode(&attempts, mode))
    .collect::<Result<Vec<_>, _>>()?;
    let fixed_batch_isolation = run_fixed_batch_isolation(
        &engine_parts(&engine, engine.gpu()),
        &inputs,
        max_new_tokens,
    )?;
    let resident_batch_continuation = run_resident_batch_continuation(
        &engine_parts(&engine, engine.gpu()),
        &inputs,
        max_new_tokens.min(32),
    )?;
    let dynamic_batch_lifecycle = run_dynamic_batch_lifecycle(
        &engine_parts(&engine, engine.gpu()),
        &inputs,
        &engine.info().model,
    )?;
    let final_metal_memory = metal_memory_stats().map_err(EngineError::Mlx)?.into();
    Ok(ParallelGenerationDiagnostic {
        schema_version: 4,
        engine: engine.info().clone(),
        prompt_tokens_per_instance: inputs[0].len(),
        max_new_tokens,
        warmup_rounds,
        measured_rounds,
        independent_stream_indices: stream_indices,
        serial_reference_output_sha256: reference_output_sha256,
        serial_reference_state_difference: reference_state_difference,
        fixed_batch_isolation,
        resident_batch_continuation,
        dynamic_batch_lifecycle,
        attempts,
        summaries,
        final_metal_memory,
    })
}

/// Runs Native unequal resident-state batching against an official MLX-LM
/// generated fixture.
///
/// # Errors
///
/// Returns [`EngineError`] when the fixture, model identity, execution, or
/// complete final state does not satisfy the admitted oracle contract.
#[allow(clippy::too_many_lines)]
pub fn run_resident_batch_oracle_parity(
    model_directory: &Path,
    oracle_directory: &Path,
) -> Result<ResidentBatchOracleParity, EngineError> {
    let manifest_path = oracle_directory.join("resident-batch.manifest.json");
    let manifest_bytes = fs::read(&manifest_path).map_err(|source| EngineError::Io {
        path: manifest_path.clone(),
        source,
    })?;
    let manifest: ResidentBatchOracleManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|error| {
            EngineError::Unsupported(format!(
                "resident batch oracle manifest is malformed: {error}"
            ))
        })?;
    if manifest.schema_version != 1
        || manifest.history_token_rows.len() != 2
        || manifest.history_token_rows.iter().any(Vec::is_empty)
        || manifest.continuation_token_ids.len() != 2
        || manifest.generation_steps == 0
        || manifest.expected_generated_token_rows.len() != 2
        || manifest
            .expected_generated_token_rows
            .iter()
            .any(|row| row.len() != manifest.generation_steps)
        || manifest.expected_state_files.len() != 2
        || manifest.expected_state_sha256.len() != 2
        || manifest.expected_state_tensor_counts.len() != 2
    {
        return Err(EngineError::Unsupported(
            "resident batch oracle manifest contract is invalid".into(),
        ));
    }
    let engine = ResidentEngine::load(
        model_directory,
        ResidentEngineConfig {
            max_new_tokens_per_request: manifest.generation_steps,
            ..ResidentEngineConfig::default()
        },
    )
    .map_err(|error| EngineError::Unsupported(format!("load resident oracle engine: {error}")))?;
    if manifest.model_type != engine.diagnostic_plan().architecture
        || manifest.config_sha256 != sha256_file(&model_directory.join("config.json"))?
    {
        return Err(EngineError::Unsupported(
            "resident batch oracle model identity differs from the loaded model".into(),
        ));
    }
    let parts = engine_parts(&engine, engine.gpu());
    let states = [
        prefill_state(&parts, &manifest.history_token_rows[0])?,
        prefill_state(&parts, &manifest.history_token_rows[1])?,
    ];
    let continuation_token_ids = [
        manifest.continuation_token_ids[0],
        manifest.continuation_token_ids[1],
    ];
    let execution = run_resident_batch_once(
        &parts,
        [&states[0], &states[1]],
        continuation_token_ids,
        manifest.generation_steps,
    )?;

    let mut expected_states = Vec::with_capacity(2);
    for index in 0..2 {
        let relative = Path::new(&manifest.expected_state_files[index]);
        if relative.components().count() != 1 || relative.file_name().is_none() {
            return Err(EngineError::Unsupported(format!(
                "resident oracle state path is not a file name: {}",
                relative.display()
            )));
        }
        let path = oracle_directory.join(relative);
        if sha256_file(&path)? != manifest.expected_state_sha256[index] {
            return Err(EngineError::Unsupported(format!(
                "resident oracle state {index} digest mismatch"
            )));
        }
        let tensors = SafeTensors::load(&path).map_err(EngineError::Mlx)?;
        if tensors.len() != manifest.expected_state_tensor_counts[index]
            || tensors.len() != parts.plan.layer_count * 2
        {
            return Err(EngineError::Unsupported(format!(
                "resident oracle state {index} tensor count mismatch"
            )));
        }
        let state = restore_named_state(&tensors, parts.plan)?;
        state.validate(parts.plan, 1)?;
        expected_states.push(state);
    }
    let expected_states: [MlxInferenceState; 2] =
        expected_states.try_into().map_err(|states: Vec<_>| {
            EngineError::Unsupported(format!(
                "resident oracle restored {} states instead of two",
                states.len()
            ))
        })?;
    let output_matches_oracle = [
        execution.output[0] == manifest.expected_generated_token_rows[0],
        execution.output[1] == manifest.expected_generated_token_rows[1],
    ];
    let final_state_max_absolute_difference_from_oracle = [
        state_max_absolute_difference(
            parts.gpu,
            &execution.state[0],
            &expected_states[0],
            parts.plan,
        )?,
        state_max_absolute_difference(
            parts.gpu,
            &execution.state[1],
            &expected_states[1],
            parts.plan,
        )?,
    ];
    let exact = output_matches_oracle == [true, true]
        && final_state_max_absolute_difference_from_oracle == [0.0, 0.0];
    Ok(ResidentBatchOracleParity {
        schema_version: manifest.schema_version,
        engine: engine.info().clone(),
        manifest_path,
        history_sequence_lengths: [
            manifest.history_token_rows[0].len(),
            manifest.history_token_rows[1].len(),
        ],
        continuation_token_ids,
        generation_steps: manifest.generation_steps,
        expected_state_sha256: [
            manifest.expected_state_sha256[0].clone(),
            manifest.expected_state_sha256[1].clone(),
        ],
        generated_token_rows: execution.output,
        output_matches_oracle,
        final_state_max_absolute_difference_from_oracle,
        exact,
    })
}

/// Measures FIFO and fixed-batch decode from already-resident 4K, 16K, and
/// 32K states. Base-state prefill is recorded but excluded from request timing.
///
/// # Errors
///
/// Returns [`EngineError`] when model admission, production-shaped chunked
/// prefill, execution, timing, memory accounting, or state accounting fails.
#[allow(clippy::too_many_lines)]
pub fn run_resident_batch_context_diagnostic(
    model_directory: &Path,
    warmup_rounds: usize,
    measured_rounds: usize,
    generated_tokens_per_instance: usize,
) -> Result<ResidentBatchContextDiagnostic, EngineError> {
    const CONTEXT_TARGETS: [usize; 3] = [4_096, 16_384, 32_768];
    const PREFILL_CHUNK_SIZE: usize = 2_048;
    if measured_rounds == 0 || generated_tokens_per_instance == 0 {
        return Err(EngineError::Unsupported(
            "resident context diagnostic requires measured rounds and generated tokens".into(),
        ));
    }
    let tokenizer = Qwen35ChatTokenizer::load(model_directory).map_err(|error| {
        EngineError::Unsupported(format!("load resident context tokenizer: {error}"))
    })?;
    let seeds = [
        encode_benchmark_prompt(&tokenizer, 'A')?,
        encode_benchmark_prompt(&tokenizer, 'B')?,
    ];
    let engine = ResidentEngine::load(
        model_directory,
        ResidentEngineConfig {
            max_new_tokens_per_request: generated_tokens_per_instance,
            ..ResidentEngineConfig::default()
        },
    )
    .map_err(|error| {
        EngineError::Unsupported(format!("load context diagnostic engine: {error}"))
    })?;
    let parts = engine_parts(&engine, engine.gpu());
    let input_tokens = [
        *seeds[0].last().ok_or_else(|| {
            EngineError::Unsupported("context seed A is unexpectedly empty".into())
        })?,
        *seeds[1].last().ok_or_else(|| {
            EngineError::Unsupported("context seed B is unexpectedly empty".into())
        })?,
    ];
    let mut base_prefill_nanos = [0_u64; 3];
    let mut base_state_logical_nbytes = [0_usize; 3];
    let mut attempts = Vec::new();

    for (target_index, target_context_tokens) in CONTEXT_TARGETS.into_iter().enumerate() {
        let context_tokens = repeated_token_context(&seeds[0], target_context_tokens)?;
        let prefill_started = Instant::now();
        let base = prefill_context_state(
            &parts,
            &context_tokens,
            PREFILL_CHUNK_SIZE,
            generated_tokens_per_instance.saturating_add(1),
        )?;
        base_prefill_nanos[target_index] = duration_nanos(prefill_started.elapsed());
        base_state_logical_nbytes[target_index] = base.logical_nbytes()?;
        if base.sequence_length()? != target_context_tokens {
            return Err(EngineError::Unsupported(format!(
                "context prefill produced length {}, expected {target_context_tokens}",
                base.sequence_length()?
            )));
        }

        for round in 0..warmup_rounds.saturating_add(measured_rounds) {
            let measured = round >= warmup_rounds;
            let phase = if measured { "measured" } else { "warmup" };
            let order = if (target_index + round).is_multiple_of(2) {
                [
                    ParallelGenerationMode::FifoSerial,
                    ParallelGenerationMode::FixedBatchTwo,
                ]
            } else {
                [
                    ParallelGenerationMode::FixedBatchTwo,
                    ParallelGenerationMode::FifoSerial,
                ]
            };
            for (order_position, mode) in order.into_iter().enumerate() {
                let execution = run_resident_context_mode(
                    &parts,
                    mode,
                    &base,
                    input_tokens,
                    generated_tokens_per_instance,
                )?;
                attempts.push(to_resident_context_attempt(
                    target_context_tokens,
                    phase,
                    round,
                    order_position + 1,
                    mode,
                    &execution,
                    generated_tokens_per_instance,
                )?);
            }
        }
    }
    let mut summaries = Vec::with_capacity(CONTEXT_TARGETS.len() * 2);
    for target in CONTEXT_TARGETS {
        for mode in [
            ParallelGenerationMode::FifoSerial,
            ParallelGenerationMode::FixedBatchTwo,
        ] {
            summaries.push(summarize_resident_context(&attempts, target, mode)?);
        }
    }
    let final_metal_memory = metal_memory_stats().map_err(EngineError::Mlx)?.into();
    Ok(ResidentBatchContextDiagnostic {
        schema_version: 1,
        engine: engine.info().clone(),
        context_targets: CONTEXT_TARGETS,
        prefill_chunk_size_tokens: PREFILL_CHUNK_SIZE,
        generated_tokens_per_instance,
        warmup_rounds,
        measured_rounds,
        base_prefill_nanos,
        base_state_logical_nbytes,
        attempts,
        summaries,
        final_metal_memory,
    })
}

fn run_fixed_batch_isolation(
    parts: &EngineParts<'_>,
    inputs: &[Vec<u32>; 3],
    max_new_tokens: usize,
) -> Result<FixedBatchIsolationObservation, EngineError> {
    let ab = run_fixed_batch_pair(
        parts,
        &[inputs[0].clone(), inputs[1].clone()],
        max_new_tokens,
    )?;
    let ac = run_fixed_batch_pair(
        parts,
        &[inputs[0].clone(), inputs[2].clone()],
        max_new_tokens,
    )?;
    let ba = run_fixed_batch_pair(
        parts,
        &[inputs[1].clone(), inputs[0].clone()],
        max_new_tokens,
    )?;
    let aa = run_fixed_batch_pair(
        parts,
        &[inputs[0].clone(), inputs[0].clone()],
        max_new_tokens,
    )?;

    let co_tenant_a_output_exact = ab.output[0] == ac.output[0];
    let co_tenant_a_state_max_absolute_difference =
        state_max_absolute_difference(parts.gpu, &ab.state[0], &ac.state[0], parts.plan)?;
    let row_permutation_output_exact = [ab.output[0] == ba.output[1], ab.output[1] == ba.output[0]];
    let row_permutation_state_max_absolute_difference = [
        state_max_absolute_difference(parts.gpu, &ab.state[0], &ba.state[1], parts.plan)?,
        state_max_absolute_difference(parts.gpu, &ab.state[1], &ba.state[0], parts.plan)?,
    ];
    let duplicate_a_output_rows_exact = aa.output[0] == aa.output[1];
    let duplicate_a_state_rows_max_absolute_difference =
        state_max_absolute_difference(parts.gpu, &aa.state[0], &aa.state[1], parts.plan)?;
    let exact = co_tenant_a_output_exact
        && co_tenant_a_state_max_absolute_difference == 0.0
        && row_permutation_output_exact == [true, true]
        && row_permutation_state_max_absolute_difference == [0.0, 0.0]
        && duplicate_a_output_rows_exact
        && duplicate_a_state_rows_max_absolute_difference == 0.0;

    Ok(FixedBatchIsolationObservation {
        co_tenant_a_output_exact,
        co_tenant_a_state_max_absolute_difference,
        row_permutation_output_exact,
        row_permutation_state_max_absolute_difference,
        duplicate_a_output_rows_exact,
        duplicate_a_state_rows_max_absolute_difference,
        exact,
    })
}

#[allow(clippy::too_many_lines)]
fn run_resident_batch_continuation(
    parts: &EngineParts<'_>,
    inputs: &[Vec<u32>; 3],
    generated_tokens: usize,
) -> Result<ResidentBatchContinuationObservation, EngineError> {
    const EXTRA_HISTORY_TOKENS: usize = 9;

    if generated_tokens == 0 {
        return Err(EngineError::Unsupported(
            "resident batch continuation requires generated tokens".into(),
        ));
    }
    let mut resident_inputs = inputs.clone();
    for row in 1..3 {
        let source = inputs[row].clone();
        for index in 0..EXTRA_HISTORY_TOKENS {
            resident_inputs[row].push(source[1 + index % (source.len() - 2)]);
        }
    }
    let resident_states = [
        prefill_state(parts, &resident_inputs[0])?,
        prefill_state(parts, &resident_inputs[1])?,
        prefill_state(parts, &resident_inputs[2])?,
    ];
    let initial_state_sequence_lengths = [
        resident_states[0].sequence_length()?,
        resident_states[1].sequence_length()?,
        resident_states[2].sequence_length()?,
    ];
    if initial_state_sequence_lengths[0] >= initial_state_sequence_lengths[1]
        || initial_state_sequence_lengths[1] != initial_state_sequence_lengths[2]
    {
        return Err(EngineError::Unsupported(format!(
            "resident batch diagnostic requires A shorter than equal-length B/C, observed {initial_state_sequence_lengths:?}"
        )));
    }
    let left_padding = [
        initial_state_sequence_lengths[1] - initial_state_sequence_lengths[0],
        0,
    ];

    let roundtrip = split_pair_runtime_state(
        parts,
        prepare_merged_runtime_state(
            parts.gpu,
            &[&resident_states[0], &resident_states[1]],
            0,
            parts.plan,
        )?,
    )?;
    let roundtrip_state_max_absolute_difference = [
        state_max_absolute_difference(parts.gpu, &roundtrip[0], &resident_states[0], parts.plan)?,
        state_max_absolute_difference(parts.gpu, &roundtrip[1], &resident_states[1], parts.plan)?,
    ];

    let continuation_tokens = [
        *inputs[0].last().ok_or_else(|| {
            EngineError::Unsupported("resident input A is unexpectedly empty".into())
        })?,
        *inputs[1].last().ok_or_else(|| {
            EngineError::Unsupported("resident input B is unexpectedly empty".into())
        })?,
    ];
    let ab = run_resident_batch_once(
        parts,
        [&resident_states[0], &resident_states[1]],
        continuation_tokens,
        generated_tokens,
    )?;
    let ac = run_resident_batch_once(
        parts,
        [&resident_states[0], &resident_states[2]],
        continuation_tokens,
        generated_tokens,
    )?;
    let ba = run_resident_batch_once(
        parts,
        [&resident_states[1], &resident_states[0]],
        [continuation_tokens[1], continuation_tokens[0]],
        generated_tokens,
    )?;
    let co_tenant_a_output_exact = ab.output[0] == ac.output[0];
    let co_tenant_a_state_max_absolute_difference =
        state_max_absolute_difference(parts.gpu, &ab.state[0], &ac.state[0], parts.plan)?;
    let row_permutation_output_exact = [ab.output[0] == ba.output[1], ab.output[1] == ba.output[0]];
    let row_permutation_state_max_absolute_difference = [
        state_max_absolute_difference(parts.gpu, &ab.state[0], &ba.state[1], parts.plan)?,
        state_max_absolute_difference(parts.gpu, &ab.state[1], &ba.state[0], parts.plan)?,
    ];

    let second_stage_tokens = [continuation_tokens[1], continuation_tokens[0]];
    let second_stage_generated_tokens = generated_tokens.min(8);
    let direct = run_resident_two_stage(
        parts,
        [&resident_states[0], &resident_states[1]],
        continuation_tokens,
        generated_tokens,
        second_stage_tokens,
        second_stage_generated_tokens,
        false,
    )?;
    let split_remerge = run_resident_two_stage(
        parts,
        [&resident_states[0], &resident_states[1]],
        continuation_tokens,
        generated_tokens,
        second_stage_tokens,
        second_stage_generated_tokens,
        true,
    )?;
    let split_remerge_first_stage_output_exact = [
        direct.first_output[0] == split_remerge.first_output[0],
        direct.first_output[1] == split_remerge.first_output[1],
    ];
    let split_remerge_second_stage_output_exact = [
        direct.second_output[0] == split_remerge.second_output[0],
        direct.second_output[1] == split_remerge.second_output[1],
    ];
    let split_remerge_final_state_max_absolute_difference = [
        state_max_absolute_difference(
            parts.gpu,
            &direct.state[0],
            &split_remerge.state[0],
            parts.plan,
        )?,
        state_max_absolute_difference(
            parts.gpu,
            &direct.state[1],
            &split_remerge.state[1],
            parts.plan,
        )?,
    ];
    let exact = roundtrip_state_max_absolute_difference == [0.0, 0.0]
        && co_tenant_a_output_exact
        && co_tenant_a_state_max_absolute_difference == 0.0
        && row_permutation_output_exact == [true, true]
        && row_permutation_state_max_absolute_difference == [0.0, 0.0]
        && split_remerge_first_stage_output_exact == [true, true]
        && split_remerge_second_stage_output_exact == [true, true]
        && split_remerge_final_state_max_absolute_difference == [0.0, 0.0];

    Ok(ResidentBatchContinuationObservation {
        initial_state_sequence_lengths,
        left_padding,
        roundtrip_state_max_absolute_difference,
        co_tenant_a_output_exact,
        co_tenant_a_state_max_absolute_difference,
        row_permutation_output_exact,
        row_permutation_state_max_absolute_difference,
        split_remerge_first_stage_output_exact,
        split_remerge_second_stage_output_exact,
        split_remerge_final_state_max_absolute_difference,
        exact,
    })
}

#[allow(clippy::too_many_lines)]
fn run_dynamic_batch_lifecycle(
    parts: &EngineParts<'_>,
    inputs: &[Vec<u32>; 3],
    model: &ModelIdentity,
) -> Result<DynamicBatchLifecycleObservation, EngineError> {
    const BEFORE_JOIN: usize = 3;
    const WHILE_JOINED: usize = 5;
    const AFTER_LEAVE: usize = 4;

    let mut history_b = inputs[1].clone();
    for index in 0..9 {
        history_b.push(inputs[1][1 + index % (inputs[1].len() - 2)]);
    }
    let initial_states = [
        prefill_state(parts, &inputs[0])?,
        prefill_state(parts, &history_b)?,
    ];
    let base_state_sequence_lengths = [
        initial_states[0].sequence_length()?,
        initial_states[1].sequence_length()?,
    ];
    let instance_ids = [
        InstanceId::new("dynamic-rin").map_err(|error| {
            EngineError::Unsupported(format!("create dynamic instance A: {error}"))
        })?,
        InstanceId::new("dynamic-marie").map_err(|error| {
            EngineError::Unsupported(format!("create dynamic instance B: {error}"))
        })?,
    ];
    let store = StateStore::<MlxInferenceState>::default();
    let [state_a, state_b] = initial_states;
    let base_a = store
        .restore(CommittedState {
            instance_id: instance_ids[0].clone(),
            model: model.clone(),
            payload: state_a,
        })
        .map_err(|error| {
            EngineError::Unsupported(format!("restore dynamic instance A: {error}"))
        })?;
    let base_b = store
        .restore(CommittedState {
            instance_id: instance_ids[1].clone(),
            model: model.clone(),
            payload: state_b,
        })
        .map_err(|error| {
            EngineError::Unsupported(format!("restore dynamic instance B: {error}"))
        })?;
    let lease_a = store
        .begin(instance_ids[0].clone(), ExpectedState::Present)
        .map_err(|error| EngineError::Unsupported(format!("begin dynamic instance A: {error}")))?;
    let lease_b = store
        .begin(instance_ids[1].clone(), ExpectedState::Present)
        .map_err(|error| EngineError::Unsupported(format!("begin dynamic instance B: {error}")))?;
    let continuation_tokens = [
        *inputs[0].last().ok_or_else(|| {
            EngineError::Unsupported("dynamic input A is unexpectedly empty".into())
        })?,
        *inputs[1].last().ok_or_else(|| {
            EngineError::Unsupported("dynamic input B is unexpectedly empty".into())
        })?,
    ];
    let cursor_a = prepare_decode_cursor(
        parts,
        &lease_a
            .base()
            .ok_or_else(|| EngineError::Unsupported("dynamic lease A lost its base".into()))?
            .payload,
        continuation_tokens[0],
    )?;
    let cursor_b = prepare_decode_cursor(
        parts,
        &lease_b
            .base()
            .ok_or_else(|| EngineError::Unsupported("dynamic lease B lost its base".into()))?
            .payload,
        continuation_tokens[1],
    )?;
    let (before_join_output, cursor_a) = advance_single_cursor(parts, cursor_a, BEFORE_JOIN)?;
    let state_sequence_lengths_before_join = [
        cursor_a.state.sequence_length()?,
        cursor_b.state.sequence_length()?,
    ];
    let (joined_output, [cursor_a, cancelled_cursor_b]) =
        advance_joined_cursors(parts, [cursor_a, cursor_b], WHILE_JOINED)?;
    let state_sequence_lengths_at_leave = [
        cursor_a.state.sequence_length()?,
        cancelled_cursor_b.state.sequence_length()?,
    ];
    let (after_leave_output, survivor_cursor) =
        advance_single_cursor(parts, cursor_a, AFTER_LEAVE)?;
    let final_survivor_sequence_length = survivor_cursor.state.sequence_length()?;

    let expected_before_join = [
        base_state_sequence_lengths[0] + 1 + BEFORE_JOIN,
        base_state_sequence_lengths[1] + 1,
    ];
    let expected_at_leave = [
        expected_before_join[0] + WHILE_JOINED,
        expected_before_join[1] + WHILE_JOINED,
    ];
    let expected_final_survivor = expected_at_leave[0] + AFTER_LEAVE;
    let sequence_length_accounting_exact = state_sequence_lengths_before_join
        == expected_before_join
        && state_sequence_lengths_at_leave == expected_at_leave
        && final_survivor_sequence_length == expected_final_survivor;
    let emitted_token_counts_exact = before_join_output.len() == BEFORE_JOIN
        && joined_output[0].len() == WHILE_JOINED
        && joined_output[1].len() == WHILE_JOINED
        && after_leave_output.len() == AFTER_LEAVE;

    let committed_a = lease_a
        .commit(PreparedState {
            model: model.clone(),
            payload: survivor_cursor.state,
        })
        .map_err(|error| EngineError::Unsupported(format!("commit dynamic instance A: {error}")))?;
    drop(cancelled_cursor_b);
    lease_b.rollback();
    let current_a = store
        .current(&instance_ids[0])
        .ok_or_else(|| EngineError::Unsupported("dynamic survivor commit is not visible".into()))?;
    let current_b = store
        .current(&instance_ids[1])
        .ok_or_else(|| EngineError::Unsupported("dynamic cancelled base disappeared".into()))?;
    let survivor_commit_visible = Arc::ptr_eq(&current_a, &committed_a)
        && !Arc::ptr_eq(&current_a, &base_a)
        && current_a.payload.sequence_length()? == final_survivor_sequence_length;
    let cancelled_base_preserved = Arc::ptr_eq(&current_b, &base_b)
        && current_b.payload.sequence_length()? == base_state_sequence_lengths[1];
    let retry_a = store.begin(instance_ids[0].clone(), ExpectedState::Present);
    let retry_b = store.begin(instance_ids[1].clone(), ExpectedState::Present);
    let instance_leases_released = retry_a.is_ok() && retry_b.is_ok();
    if let Ok(lease) = retry_a {
        lease.rollback();
    }
    if let Ok(lease) = retry_b {
        lease.rollback();
    }
    let exact = sequence_length_accounting_exact
        && emitted_token_counts_exact
        && survivor_commit_visible
        && cancelled_base_preserved
        && instance_leases_released;

    Ok(DynamicBatchLifecycleObservation {
        base_state_sequence_lengths,
        generated_tokens_before_join: BEFORE_JOIN,
        generated_tokens_while_joined: WHILE_JOINED,
        generated_tokens_after_leave: AFTER_LEAVE,
        state_sequence_lengths_before_join,
        state_sequence_lengths_at_leave,
        final_survivor_sequence_length,
        sequence_length_accounting_exact,
        emitted_token_counts_exact,
        survivor_commit_visible,
        cancelled_base_preserved,
        instance_leases_released,
        exact,
    })
}

fn prefill_state(parts: &EngineParts<'_>, input: &[u32]) -> Result<MlxInferenceState, EngineError> {
    let execution = build_prefill(parts, input, 1, 0)?;
    evaluate_runtime_execution(parts.gpu, &execution)?;
    compact_runtime_state(parts.gpu, execution.state, parts.plan)
}

fn prepare_decode_cursor(
    parts: &EngineParts<'_>,
    base: &MlxInferenceState,
    input_token: u32,
) -> Result<DecodeCursor, EngineError> {
    let input = token_array(&[input_token])?;
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
    let pending_token = parts
        .gpu
        .reshape(&pending, &[])
        .and_then(|value| value.item_u32())
        .map_err(EngineError::Mlx)?;
    Ok(DecodeCursor {
        state: compact_runtime_state(parts.gpu, execution.state, parts.plan)?,
        pending_token,
    })
}

fn advance_single_cursor(
    parts: &EngineParts<'_>,
    cursor: DecodeCursor,
    generated_tokens: usize,
) -> Result<(Vec<u32>, DecodeCursor), EngineError> {
    if generated_tokens == 0 {
        return Ok((Vec::new(), cursor));
    }
    let mut runtime_state =
        prepare_runtime_state(parts.gpu, &cursor.state, 1, generated_tokens, parts.plan)?;
    let mut pending_token = cursor.pending_token;
    let mut output = Vec::with_capacity(generated_tokens);
    for _ in 0..generated_tokens {
        let token = token_array(&[pending_token])?;
        let execution = execute_runtime_model(
            parts.gpu,
            &token,
            runtime_state,
            parts.weights,
            parts.plan,
            parts.gdn_kernel,
            parts.moe_kernel,
        )?;
        evaluate_runtime_execution(parts.gpu, &execution)?;
        output.push(pending_token);
        let pending = greedy_token(parts.gpu, &execution.logits, 1, parts.plan)?;
        pending_token = parts
            .gpu
            .reshape(&pending, &[])
            .and_then(|value| value.item_u32())
            .map_err(EngineError::Mlx)?;
        runtime_state = execution.state;
    }
    Ok((
        output,
        DecodeCursor {
            state: compact_runtime_state(parts.gpu, runtime_state, parts.plan)?,
            pending_token,
        },
    ))
}

fn advance_joined_cursors(
    parts: &EngineParts<'_>,
    cursors: [DecodeCursor; 2],
    generated_tokens: usize,
) -> Result<([Vec<u32>; 2], [DecodeCursor; 2]), EngineError> {
    if generated_tokens == 0 {
        return Ok(([Vec::new(), Vec::new()], cursors));
    }
    let [cursor_a, cursor_b] = cursors;
    let mut runtime_state = prepare_merged_runtime_state(
        parts.gpu,
        &[&cursor_a.state, &cursor_b.state],
        generated_tokens,
        parts.plan,
    )?;
    let mut pending_tokens = [cursor_a.pending_token, cursor_b.pending_token];
    let mut output = [
        Vec::with_capacity(generated_tokens),
        Vec::with_capacity(generated_tokens),
    ];
    for _ in 0..generated_tokens {
        let token = paired_token_array(&[vec![pending_tokens[0]], vec![pending_tokens[1]]])?;
        let execution = execute_runtime_model(
            parts.gpu,
            &token,
            runtime_state,
            parts.weights,
            parts.plan,
            parts.gdn_kernel,
            parts.moe_kernel,
        )?;
        evaluate_runtime_execution(parts.gpu, &execution)?;
        output[0].push(pending_tokens[0]);
        output[1].push(pending_tokens[1]);
        let pending = greedy_token(parts.gpu, &execution.logits, 2, parts.plan)?;
        pending_tokens = paired_token_values(parts.gpu, &pending)?;
        runtime_state = execution.state;
    }
    let [state_a, state_b] = split_pair_runtime_state(parts, runtime_state)?;
    Ok((
        output,
        [
            DecodeCursor {
                state: state_a,
                pending_token: pending_tokens[0],
            },
            DecodeCursor {
                state: state_b,
                pending_token: pending_tokens[1],
            },
        ],
    ))
}

fn run_resident_batch_once(
    parts: &EngineParts<'_>,
    states: [&MlxInferenceState; 2],
    input_tokens: [u32; 2],
    generated_tokens: usize,
) -> Result<ResidentBatchExecution, EngineError> {
    let runtime_state = prepare_merged_runtime_state(
        parts.gpu,
        &states,
        generated_tokens.saturating_add(1),
        parts.plan,
    )?;
    let (output, runtime_state) =
        advance_resident_batch(parts, runtime_state, input_tokens, generated_tokens)?;
    Ok(ResidentBatchExecution {
        output,
        state: split_pair_runtime_state(parts, runtime_state)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_resident_two_stage(
    parts: &EngineParts<'_>,
    states: [&MlxInferenceState; 2],
    first_input_tokens: [u32; 2],
    first_generated_tokens: usize,
    second_input_tokens: [u32; 2],
    second_generated_tokens: usize,
    split_between_stages: bool,
) -> Result<ResidentTwoStageExecution, EngineError> {
    let additional_tokens = first_generated_tokens
        .checked_add(second_generated_tokens)
        .and_then(|count| count.checked_add(2))
        .ok_or_else(|| {
            EngineError::Unsupported("resident continuation capacity overflow".into())
        })?;
    let initial = prepare_merged_runtime_state(parts.gpu, &states, additional_tokens, parts.plan)?;
    let (first_output, first_state) =
        advance_resident_batch(parts, initial, first_input_tokens, first_generated_tokens)?;
    let second_state = if split_between_stages {
        let split = split_pair_runtime_state(parts, first_state)?;
        prepare_merged_runtime_state(
            parts.gpu,
            &[&split[0], &split[1]],
            second_generated_tokens.saturating_add(1),
            parts.plan,
        )?
    } else {
        first_state
    };
    let (second_output, final_state) = advance_resident_batch(
        parts,
        second_state,
        second_input_tokens,
        second_generated_tokens,
    )?;
    Ok(ResidentTwoStageExecution {
        first_output,
        second_output,
        state: split_pair_runtime_state(parts, final_state)?,
    })
}

fn advance_resident_batch(
    parts: &EngineParts<'_>,
    initial_state: super::full_model::RuntimeInferenceState,
    input_tokens: [u32; 2],
    generated_tokens: usize,
) -> Result<([Vec<u32>; 2], super::full_model::RuntimeInferenceState), EngineError> {
    let input = paired_token_array(&[vec![input_tokens[0]], vec![input_tokens[1]]])?;
    let mut execution = execute_runtime_model(
        parts.gpu,
        &input,
        initial_state,
        parts.weights,
        parts.plan,
        parts.gdn_kernel,
        parts.moe_kernel,
    )?;
    let mut output = [
        Vec::with_capacity(generated_tokens),
        Vec::with_capacity(generated_tokens),
    ];
    for _ in 0..generated_tokens {
        let RuntimeModelExecution { logits, state } = execution;
        let tokens = greedy_token(parts.gpu, &logits, 2, parts.plan)?;
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
        let values = paired_token_values(parts.gpu, &tokens)?;
        output[0].push(values[0]);
        output[1].push(values[1]);
        execution = next;
    }
    evaluate_runtime_execution(parts.gpu, &execution)?;
    Ok((output, execution.state))
}

fn split_pair_runtime_state(
    parts: &EngineParts<'_>,
    state: super::full_model::RuntimeInferenceState,
) -> Result<[MlxInferenceState; 2], EngineError> {
    let rows = split_runtime_state(parts.gpu, state, parts.plan)?;
    let count = rows.len();
    rows.try_into().map_err(|_| {
        EngineError::Unsupported(format!(
            "resident runtime split produced {count} states instead of two"
        ))
    })
}

fn encode_benchmark_prompt(
    tokenizer: &Qwen35ChatTokenizer,
    instance: char,
) -> Result<Vec<u32>, EngineError> {
    let prompt = format!("{BENCHMARK_PROMPT_PREFIX}{instance}");
    tokenizer
        .encode_prompt(&EchoChatPrompt {
            input: vec![EchoInputItem::Message(EchoMessage {
                role: EchoMessageRole::User,
                content: EchoMessageContent::Text(prompt),
            })],
            tools: Vec::new(),
        })
        .map(|encoded| encoded.token_ids)
        .map_err(|error| {
            EngineError::Unsupported(format!("encode parallel diagnostic prompt: {error}"))
        })
}

fn engine_parts<'a>(engine: &'a ResidentEngine, gpu: &'a Gpu) -> EngineParts<'a> {
    EngineParts {
        gpu,
        plan: engine.diagnostic_plan(),
        weights: engine.diagnostic_weights(),
        gdn_kernel: engine.diagnostic_gdn_kernel(),
        moe_kernel: engine.diagnostic_moe_kernel(),
    }
}

fn repeated_token_context(seed: &[u32], target: usize) -> Result<Vec<u32>, EngineError> {
    if seed.is_empty() || target == 0 {
        return Err(EngineError::Unsupported(
            "resident context seed and target must be non-empty".into(),
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
            "resident context prefill requires tokens and a positive chunk size".into(),
        ));
    }
    let initial = MlxInferenceState::empty(parts.gpu, 1, parts.plan)?;
    let capacity = tokens
        .len()
        .checked_add(additional_tokens)
        .ok_or_else(|| EngineError::Unsupported("resident context capacity overflow".into()))?;
    let mut state = prepare_runtime_state(parts.gpu, &initial, 1, capacity, parts.plan)?;
    for chunk in tokens.chunks(chunk_size) {
        let input = token_array(chunk)?;
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

fn run_resident_context_mode(
    parts: &EngineParts<'_>,
    mode: ParallelGenerationMode,
    base: &MlxInferenceState,
    input_tokens: [u32; 2],
    generated_tokens: usize,
) -> Result<PairExecution, EngineError> {
    reset_peak_memory().map_err(EngineError::Mlx)?;
    let start_memory = metal_memory_stats().map_err(EngineError::Mlx)?;
    let mut execution = match mode {
        ParallelGenerationMode::FifoSerial => {
            run_resident_fifo_context(parts, base, input_tokens, generated_tokens)?
        }
        ParallelGenerationMode::FixedBatchTwo => {
            run_resident_fixed_context(parts, base, input_tokens, generated_tokens)?
        }
        ParallelGenerationMode::IndependentStreams => {
            return Err(EngineError::Unsupported(
                "resident context diagnostic does not include independent streams".into(),
            ));
        }
    };
    let end_memory = metal_memory_stats().map_err(EngineError::Mlx)?;
    execution.memory = memory_observation(start_memory, end_memory)?;
    Ok(execution)
}

fn run_resident_fifo_context(
    parts: &EngineParts<'_>,
    base: &MlxInferenceState,
    input_tokens: [u32; 2],
    generated_tokens: usize,
) -> Result<PairExecution, EngineError> {
    let pair_started = Instant::now();
    let first = run_resident_single_context(parts, base, input_tokens[0], generated_tokens)?;
    let first_completed = duration_nanos(pair_started.elapsed());
    let second = run_resident_single_context(parts, base, input_tokens[1], generated_tokens)?;
    let pair_total_nanos = duration_nanos(pair_started.elapsed());
    let second_start_nanos = pair_total_nanos.saturating_sub(second.total_nanos);
    Ok(PairExecution {
        output: [first.output, second.output],
        state: [first.state, second.state],
        pair_total_nanos,
        pair_decode_nanos: first.decode_nanos.saturating_add(second.decode_nanos),
        ttft_nanos: [
            first.ttft_nanos,
            second_start_nanos.saturating_add(second.ttft_nanos),
        ],
        completion_nanos: [first_completed, pair_total_nanos],
        request_decode_nanos: [first.decode_nanos, second.decode_nanos],
        memory: empty_memory_observation(),
    })
}

fn run_resident_single_context(
    parts: &EngineParts<'_>,
    base: &MlxInferenceState,
    input_token: u32,
    generated_tokens: usize,
) -> Result<SequenceExecution, EngineError> {
    let started = Instant::now();
    let input = token_array(&[input_token])?;
    let runtime_state = prepare_runtime_state(
        parts.gpu,
        base,
        1,
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
    let mut output = Vec::with_capacity(generated_tokens);
    let mut ttft_nanos = None;
    for _ in 0..generated_tokens {
        let RuntimeModelExecution { logits, state } = execution;
        let token = greedy_token(parts.gpu, &logits, 1, parts.plan)?;
        let scalar = parts.gpu.reshape(&token, &[]).map_err(EngineError::Mlx)?;
        let next = execute_runtime_model(
            parts.gpu,
            &token,
            state,
            parts.weights,
            parts.plan,
            parts.gdn_kernel,
            parts.moe_kernel,
        )?;
        schedule_runtime_execution(parts.gpu, &scalar, &next)?;
        output.push(scalar.item_u32().map_err(EngineError::Mlx)?);
        ttft_nanos.get_or_insert_with(|| duration_nanos(started.elapsed()));
        execution = next;
    }
    evaluate_runtime_execution(parts.gpu, &execution)?;
    let decode_nanos = duration_nanos(decode_started.elapsed());
    let state = compact_runtime_state(parts.gpu, execution.state, parts.plan)?;
    Ok(SequenceExecution {
        output,
        state,
        ttft_nanos: ttft_nanos.ok_or_else(|| {
            EngineError::Unsupported("resident context sequence produced no first token".into())
        })?,
        decode_nanos,
        total_nanos: duration_nanos(started.elapsed()),
    })
}

fn run_resident_fixed_context(
    parts: &EngineParts<'_>,
    base: &MlxInferenceState,
    input_tokens: [u32; 2],
    generated_tokens: usize,
) -> Result<PairExecution, EngineError> {
    let pair_started = Instant::now();
    let input = paired_token_array(&[vec![input_tokens[0]], vec![input_tokens[1]]])?;
    let runtime_state = prepare_merged_runtime_state(
        parts.gpu,
        &[base, base],
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
    let mut output = [
        Vec::with_capacity(generated_tokens),
        Vec::with_capacity(generated_tokens),
    ];
    let mut ttft_nanos = [None, None];
    for _ in 0..generated_tokens {
        let RuntimeModelExecution { logits, state } = execution;
        let tokens = greedy_token(parts.gpu, &logits, 2, parts.plan)?;
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
        let values = paired_token_values(parts.gpu, &tokens)?;
        for index in 0..2 {
            output[index].push(values[index]);
            ttft_nanos[index].get_or_insert_with(|| duration_nanos(pair_started.elapsed()));
        }
        execution = next;
    }
    evaluate_runtime_execution(parts.gpu, &execution)?;
    let pair_decode_nanos = duration_nanos(decode_started.elapsed());
    let states = split_pair_runtime_state(parts, execution.state)?;
    let pair_total_nanos = duration_nanos(pair_started.elapsed());
    Ok(PairExecution {
        output,
        state: states,
        pair_total_nanos,
        pair_decode_nanos,
        ttft_nanos: [
            ttft_nanos[0].ok_or_else(|| {
                EngineError::Unsupported("resident batch row A produced no first token".into())
            })?,
            ttft_nanos[1].ok_or_else(|| {
                EngineError::Unsupported("resident batch row B produced no first token".into())
            })?,
        ],
        completion_nanos: [pair_total_nanos, pair_total_nanos],
        request_decode_nanos: [pair_decode_nanos, pair_decode_nanos],
        memory: empty_memory_observation(),
    })
}

fn to_resident_context_attempt(
    target_context_tokens: usize,
    phase: &'static str,
    round: usize,
    order_position: usize,
    mode: ParallelGenerationMode,
    execution: &PairExecution,
    generated_tokens: usize,
) -> Result<ResidentBatchContextAttempt, EngineError> {
    let total_tokens = generated_tokens
        .checked_mul(2)
        .ok_or_else(|| EngineError::Unsupported("context token denominator overflow".into()))?;
    if execution
        .output
        .iter()
        .any(|row| row.len() != generated_tokens)
    {
        return Err(EngineError::Unsupported(
            "resident context execution produced the wrong token count".into(),
        ));
    }
    let state_sequence_lengths = [
        execution.state[0].sequence_length()?,
        execution.state[1].sequence_length()?,
    ];
    let expected_state_length = target_context_tokens
        .checked_add(generated_tokens)
        .and_then(|length| length.checked_add(1))
        .ok_or_else(|| EngineError::Unsupported("context state length overflow".into()))?;
    Ok(ResidentBatchContextAttempt {
        target_context_tokens,
        phase,
        round,
        order_position,
        mode,
        pair_total_nanos: execution.pair_total_nanos,
        pair_decode_nanos: execution.pair_decode_nanos,
        ttft_nanos: execution.ttft_nanos,
        completion_nanos: execution.completion_nanos,
        aggregate_decode_tokens_per_second: tokens_per_second(
            total_tokens,
            execution.pair_decode_nanos,
        )?,
        request_decode_tokens_per_second: [
            tokens_per_second(generated_tokens, execution.request_decode_nanos[0])?,
            tokens_per_second(generated_tokens, execution.request_decode_nanos[1])?,
        ],
        state_sequence_lengths,
        state_lengths_exact: state_sequence_lengths == [expected_state_length; 2],
        memory: execution.memory,
    })
}

fn summarize_resident_context(
    attempts: &[ResidentBatchContextAttempt],
    target_context_tokens: usize,
    mode: ParallelGenerationMode,
) -> Result<ResidentBatchContextSummary, EngineError> {
    let measured = attempts
        .iter()
        .filter(|attempt| {
            attempt.phase == "measured"
                && attempt.target_context_tokens == target_context_tokens
                && attempt.mode == mode
        })
        .collect::<Vec<_>>();
    let integers = |extract: fn(&ResidentBatchContextAttempt) -> u64| {
        measured
            .iter()
            .map(|attempt| extract(attempt))
            .collect::<Vec<_>>()
    };
    let floats = |extract: fn(&ResidentBatchContextAttempt) -> f64| {
        measured
            .iter()
            .map(|attempt| extract(attempt))
            .collect::<Vec<_>>()
    };
    Ok(ResidentBatchContextSummary {
        target_context_tokens,
        mode,
        count: measured.len(),
        median_pair_total_nanos: median_u64(integers(|attempt| attempt.pair_total_nanos))?,
        median_pair_decode_nanos: median_u64(integers(|attempt| attempt.pair_decode_nanos))?,
        median_ttft_nanos: [
            median_u64(integers(|attempt| attempt.ttft_nanos[0]))?,
            median_u64(integers(|attempt| attempt.ttft_nanos[1]))?,
        ],
        median_completion_nanos: [
            median_u64(integers(|attempt| attempt.completion_nanos[0]))?,
            median_u64(integers(|attempt| attempt.completion_nanos[1]))?,
        ],
        median_aggregate_decode_tokens_per_second: median_f64(floats(|attempt| {
            attempt.aggregate_decode_tokens_per_second
        }))?,
        median_request_decode_tokens_per_second: [
            median_f64(floats(|attempt| {
                attempt.request_decode_tokens_per_second[0]
            }))?,
            median_f64(floats(|attempt| {
                attempt.request_decode_tokens_per_second[1]
            }))?,
        ],
        median_peak_above_start_active_nbytes: median_u64(integers(|attempt| {
            attempt.memory.peak_above_start_active_nbytes
        }))?,
        exact_state_attempts: measured
            .iter()
            .filter(|attempt| attempt.state_lengths_exact)
            .count(),
    })
}

fn run_mode(
    mode: ParallelGenerationMode,
    engine: &ResidentEngine,
    streams: [&Gpu; 2],
    inputs: &[Vec<u32>; 2],
    max_new_tokens: usize,
) -> Result<PairExecution, EngineError> {
    reset_peak_memory().map_err(EngineError::Mlx)?;
    let start_memory = metal_memory_stats().map_err(EngineError::Mlx)?;
    let mut pair = match mode {
        ParallelGenerationMode::FifoSerial => {
            run_fifo_pair(&engine_parts(engine, engine.gpu()), inputs, max_new_tokens)?
        }
        ParallelGenerationMode::IndependentStreams => {
            let stream_parts = [
                engine_parts(engine, streams[0]),
                engine_parts(engine, streams[1]),
            ];
            run_independent_stream_pair(&stream_parts, inputs, max_new_tokens)?
        }
        ParallelGenerationMode::FixedBatchTwo => {
            run_fixed_batch_pair(&engine_parts(engine, engine.gpu()), inputs, max_new_tokens)?
        }
    };
    let end_memory = metal_memory_stats().map_err(EngineError::Mlx)?;
    pair.memory = memory_observation(start_memory, end_memory)?;
    Ok(pair)
}

fn run_fifo_pair(
    parts: &EngineParts<'_>,
    inputs: &[Vec<u32>; 2],
    max_new_tokens: usize,
) -> Result<PairExecution, EngineError> {
    let pair_started = Instant::now();
    let first = run_single_sequence(parts, &inputs[0], max_new_tokens)?;
    let first_completed = duration_nanos(pair_started.elapsed());
    let second = run_single_sequence(parts, &inputs[1], max_new_tokens)?;
    let pair_total_nanos = duration_nanos(pair_started.elapsed());
    let second_start_nanos = pair_total_nanos.saturating_sub(second.total_nanos);
    Ok(PairExecution {
        output: [first.output, second.output],
        state: [first.state, second.state],
        pair_total_nanos,
        pair_decode_nanos: first.decode_nanos.saturating_add(second.decode_nanos),
        ttft_nanos: [
            first.ttft_nanos,
            second_start_nanos.saturating_add(second.ttft_nanos),
        ],
        completion_nanos: [first_completed, pair_total_nanos],
        request_decode_nanos: [first.decode_nanos, second.decode_nanos],
        memory: empty_memory_observation(),
    })
}

fn run_single_sequence(
    parts: &EngineParts<'_>,
    input: &[u32],
    max_new_tokens: usize,
) -> Result<SequenceExecution, EngineError> {
    let started = Instant::now();
    let mut execution = build_prefill(parts, input, 1, max_new_tokens)?;
    evaluate_runtime_execution(parts.gpu, &execution)?;
    let decode_started = Instant::now();
    let mut output = Vec::with_capacity(max_new_tokens);
    let mut ttft_nanos = None;
    for _ in 0..max_new_tokens {
        let RuntimeModelExecution { logits, state } = execution;
        let token = greedy_token(parts.gpu, &logits, 1, parts.plan)?;
        let scalar = parts.gpu.reshape(&token, &[]).map_err(EngineError::Mlx)?;
        let next = execute_runtime_model(
            parts.gpu,
            &token,
            state,
            parts.weights,
            parts.plan,
            parts.gdn_kernel,
            parts.moe_kernel,
        )?;
        schedule_runtime_execution(parts.gpu, &scalar, &next)?;
        output.push(scalar.item_u32().map_err(EngineError::Mlx)?);
        ttft_nanos.get_or_insert_with(|| duration_nanos(started.elapsed()));
        execution = next;
    }
    evaluate_runtime_execution(parts.gpu, &execution)?;
    let decode_nanos = duration_nanos(decode_started.elapsed());
    let state = compact_runtime_state(parts.gpu, execution.state, parts.plan)?;
    Ok(SequenceExecution {
        output,
        state,
        ttft_nanos: ttft_nanos.ok_or_else(|| {
            EngineError::Unsupported("single diagnostic produced no first token".into())
        })?,
        decode_nanos,
        total_nanos: duration_nanos(started.elapsed()),
    })
}

#[allow(clippy::too_many_lines)]
fn run_independent_stream_pair(
    parts: &[EngineParts<'_>; 2],
    inputs: &[Vec<u32>; 2],
    max_new_tokens: usize,
) -> Result<PairExecution, EngineError> {
    let pair_started = Instant::now();
    let mut executions = [
        build_prefill(&parts[0], &inputs[0], 1, max_new_tokens)?,
        build_prefill(&parts[1], &inputs[1], 1, max_new_tokens)?,
    ];
    schedule_complete_execution(parts[0].gpu, &executions[0])?;
    schedule_complete_execution(parts[1].gpu, &executions[1])?;
    parts[0].gpu.synchronize().map_err(EngineError::Mlx)?;
    parts[1].gpu.synchronize().map_err(EngineError::Mlx)?;
    let decode_started = Instant::now();
    let mut outputs = [
        Vec::with_capacity(max_new_tokens),
        Vec::with_capacity(max_new_tokens),
    ];
    let mut ttft_nanos = [None, None];
    for _ in 0..max_new_tokens {
        let [execution_a, execution_b] = executions;
        let (token_a, scalar_a, next_a) = build_single_decode_step(&parts[0], execution_a)?;
        let (token_b, scalar_b, next_b) = build_single_decode_step(&parts[1], execution_b)?;
        schedule_runtime_execution(parts[0].gpu, &scalar_a, &next_a)?;
        schedule_runtime_execution(parts[1].gpu, &scalar_b, &next_b)?;
        let value_a = scalar_a.item_u32().map_err(EngineError::Mlx)?;
        ttft_nanos[0].get_or_insert_with(|| duration_nanos(pair_started.elapsed()));
        let value_b = scalar_b.item_u32().map_err(EngineError::Mlx)?;
        ttft_nanos[1].get_or_insert_with(|| duration_nanos(pair_started.elapsed()));
        outputs[0].push(value_a);
        outputs[1].push(value_b);
        drop((token_a, token_b));
        executions = [next_a, next_b];
    }
    schedule_complete_execution(parts[0].gpu, &executions[0])?;
    schedule_complete_execution(parts[1].gpu, &executions[1])?;
    parts[0].gpu.synchronize().map_err(EngineError::Mlx)?;
    let completion_a = duration_nanos(pair_started.elapsed());
    parts[1].gpu.synchronize().map_err(EngineError::Mlx)?;
    let completion_b = duration_nanos(pair_started.elapsed());
    let pair_decode_nanos = duration_nanos(decode_started.elapsed());
    let [execution_a, execution_b] = executions;
    let states = [
        compact_runtime_state(parts[0].gpu, execution_a.state, parts[0].plan)?,
        compact_runtime_state(parts[1].gpu, execution_b.state, parts[1].plan)?,
    ];
    let pair_total_nanos = duration_nanos(pair_started.elapsed());
    Ok(PairExecution {
        output: outputs,
        state: states,
        pair_total_nanos,
        pair_decode_nanos,
        ttft_nanos: [
            ttft_nanos[0].ok_or_else(|| {
                EngineError::Unsupported("stream A produced no first token".into())
            })?,
            ttft_nanos[1].ok_or_else(|| {
                EngineError::Unsupported("stream B produced no first token".into())
            })?,
        ],
        completion_nanos: [completion_a, completion_b],
        request_decode_nanos: [pair_decode_nanos, pair_decode_nanos],
        memory: empty_memory_observation(),
    })
}

fn build_single_decode_step(
    parts: &EngineParts<'_>,
    execution: RuntimeModelExecution,
) -> Result<(Array, Array, RuntimeModelExecution), EngineError> {
    let RuntimeModelExecution { logits, state } = execution;
    let token = greedy_token(parts.gpu, &logits, 1, parts.plan)?;
    let scalar = parts.gpu.reshape(&token, &[]).map_err(EngineError::Mlx)?;
    let next = execute_runtime_model(
        parts.gpu,
        &token,
        state,
        parts.weights,
        parts.plan,
        parts.gdn_kernel,
        parts.moe_kernel,
    )?;
    Ok((token, scalar, next))
}

fn run_fixed_batch_pair(
    parts: &EngineParts<'_>,
    inputs: &[Vec<u32>; 2],
    max_new_tokens: usize,
) -> Result<PairExecution, EngineError> {
    let pair_started = Instant::now();
    let input = paired_token_array(inputs)?;
    let initial = MlxInferenceState::empty(parts.gpu, 2, parts.plan)?;
    let runtime_state = prepare_runtime_state(
        parts.gpu,
        &initial,
        2,
        inputs[0].len().saturating_add(max_new_tokens),
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
    let mut outputs = [
        Vec::with_capacity(max_new_tokens),
        Vec::with_capacity(max_new_tokens),
    ];
    let mut ttft_nanos = [None, None];
    for _ in 0..max_new_tokens {
        let RuntimeModelExecution { logits, state } = execution;
        let tokens = greedy_token(parts.gpu, &logits, 2, parts.plan)?;
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
        let values = paired_token_values(parts.gpu, &tokens)?;
        for index in 0..2 {
            outputs[index].push(values[index]);
            ttft_nanos[index].get_or_insert_with(|| duration_nanos(pair_started.elapsed()));
        }
        execution = next;
    }
    evaluate_runtime_execution(parts.gpu, &execution)?;
    let pair_decode_nanos = duration_nanos(decode_started.elapsed());
    let compact = compact_runtime_state(parts.gpu, execution.state, parts.plan)?;
    let states = split_batch_state(parts.gpu, &compact, parts.plan)?;
    let pair_total_nanos = duration_nanos(pair_started.elapsed());
    Ok(PairExecution {
        output: outputs,
        state: states,
        pair_total_nanos,
        pair_decode_nanos,
        ttft_nanos: [
            ttft_nanos[0].ok_or_else(|| {
                EngineError::Unsupported("batch row A produced no first token".into())
            })?,
            ttft_nanos[1].ok_or_else(|| {
                EngineError::Unsupported("batch row B produced no first token".into())
            })?,
        ],
        completion_nanos: [pair_total_nanos, pair_total_nanos],
        request_decode_nanos: [pair_decode_nanos, pair_decode_nanos],
        memory: empty_memory_observation(),
    })
}

fn build_prefill(
    parts: &EngineParts<'_>,
    input: &[u32],
    batch_size: usize,
    max_new_tokens: usize,
) -> Result<RuntimeModelExecution, EngineError> {
    if batch_size != 1 {
        return Err(EngineError::Unsupported(
            "single-input prefill helper requires batch size one".into(),
        ));
    }
    let input_array = token_array(input)?;
    let initial = MlxInferenceState::empty(parts.gpu, batch_size, parts.plan)?;
    let runtime_state = prepare_runtime_state(
        parts.gpu,
        &initial,
        batch_size,
        input.len().saturating_add(max_new_tokens),
        parts.plan,
    )?;
    execute_runtime_model(
        parts.gpu,
        &input_array,
        runtime_state,
        parts.weights,
        parts.plan,
        parts.gdn_kernel,
        parts.moe_kernel,
    )
}

fn schedule_complete_execution(
    gpu: &Gpu,
    execution: &RuntimeModelExecution,
) -> Result<(), EngineError> {
    gpu.async_eval(&runtime_execution_arrays(execution))
        .map_err(EngineError::Mlx)
}

fn token_array(tokens: &[u32]) -> Result<Array, EngineError> {
    let values = tokens
        .iter()
        .map(|token| {
            i32::try_from(*token).map_err(|error| {
                EngineError::Unsupported(format!("token ID {token} does not fit int32: {error}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Array::from_i32_slice(&values, &[1, values.len()]).map_err(EngineError::Mlx)
}

fn paired_token_array(inputs: &[Vec<u32>; 2]) -> Result<Array, EngineError> {
    if inputs[0].len() != inputs[1].len() || inputs[0].is_empty() {
        return Err(EngineError::Unsupported(
            "paired token input requires two non-empty equal lengths".into(),
        ));
    }
    let values = inputs
        .iter()
        .flatten()
        .map(|token| {
            i32::try_from(*token).map_err(|error| {
                EngineError::Unsupported(format!("token ID {token} does not fit int32: {error}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Array::from_i32_slice(&values, &[2, inputs[0].len()]).map_err(EngineError::Mlx)
}

fn paired_token_values(gpu: &Gpu, tokens: &Array) -> Result<[u32; 2], EngineError> {
    if tokens.shape() != [2, 1] {
        return Err(EngineError::Unsupported(format!(
            "paired sampled tokens must have shape [2, 1], observed {:?}",
            tokens.shape()
        )));
    }
    let mut values = [0_u32; 2];
    for (row, value) in values.iter_mut().enumerate() {
        let row_i32 = i32::try_from(row).map_err(|error| {
            EngineError::Unsupported(format!("batch row does not fit int32: {error}"))
        })?;
        let scalar = gpu
            .slice(tokens, &[row_i32, 0], &[row_i32 + 1, 1], &[1, 1])
            .and_then(|row| gpu.reshape(&row, &[]))
            .map_err(EngineError::Mlx)?;
        *value = scalar.item_u32().map_err(EngineError::Mlx)?;
    }
    Ok(values)
}

fn split_batch_state(
    gpu: &Gpu,
    state: &MlxInferenceState,
    plan: &ModelPlan,
) -> Result<[MlxInferenceState; 2], EngineError> {
    state.validate(plan, 2)?;
    let mut rows = [
        Vec::with_capacity(plan.layer_count),
        Vec::with_capacity(plan.layer_count),
    ];
    for layer in state.layers() {
        match layer {
            LayerState::Gdn {
                convolution,
                recurrent,
            } => {
                for (row, destination) in rows.iter_mut().enumerate() {
                    destination.push(LayerState::Gdn {
                        convolution: slice_batch_row(gpu, convolution, row)?,
                        recurrent: slice_batch_row(gpu, recurrent, row)?,
                    });
                }
            }
            LayerState::Attention { keys, values } => {
                for (row, destination) in rows.iter_mut().enumerate() {
                    destination.push(LayerState::Attention {
                        keys: slice_batch_row(gpu, keys, row)?,
                        values: slice_batch_row(gpu, values, row)?,
                    });
                }
            }
        }
    }
    let [row_a, row_b] = rows;
    let states = [MlxInferenceState::new(row_a), MlxInferenceState::new(row_b)];
    states[0].validate(plan, 1)?;
    states[1].validate(plan, 1)?;
    Ok(states)
}

fn slice_batch_row(gpu: &Gpu, array: &Array, row: usize) -> Result<Array, EngineError> {
    let shape = array.shape();
    if shape.first().copied() != Some(2) {
        return Err(EngineError::Unsupported(format!(
            "batched state must have leading dimension 2, observed {shape:?}"
        )));
    }
    let mut start = vec![0_i32; shape.len()];
    let mut stop = shape
        .iter()
        .map(|dimension| {
            i32::try_from(*dimension).map_err(|error| {
                EngineError::Unsupported(format!("state dimension does not fit int32: {error}"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    start[0] = i32::try_from(row).map_err(|error| {
        EngineError::Unsupported(format!("state batch row does not fit int32: {error}"))
    })?;
    stop[0] = start[0] + 1;
    gpu.slice(array, &start, &stop, &vec![1; shape.len()])
        .map_err(EngineError::Mlx)
}

fn compare_to_reference(
    gpu: &Gpu,
    candidate: &PairExecution,
    reference: &PairExecution,
    plan: &ModelPlan,
) -> Result<ParallelCorrectnessObservation, EngineError> {
    let own = [
        state_max_absolute_difference(gpu, &candidate.state[0], &reference.state[0], plan)?,
        state_max_absolute_difference(gpu, &candidate.state[1], &reference.state[1], plan)?,
    ];
    let other = [
        state_max_absolute_difference(gpu, &candidate.state[0], &reference.state[1], plan)?,
        state_max_absolute_difference(gpu, &candidate.state[1], &reference.state[0], plan)?,
    ];
    let output_matches_serial = [
        candidate.output[0] == reference.output[0],
        candidate.output[1] == reference.output[1],
    ];
    let candidate_state_rows_max_absolute_difference =
        state_max_absolute_difference(gpu, &candidate.state[0], &candidate.state[1], plan)?;
    Ok(ParallelCorrectnessObservation {
        output_matches_serial,
        state_max_absolute_difference_from_own_serial: own,
        state_max_absolute_difference_from_other_serial: other,
        candidate_state_rows_max_absolute_difference,
        exact_serial_equivalence: output_matches_serial.into_iter().all(|matches| matches)
            && own == [0.0, 0.0],
        each_state_is_closer_to_own_serial: own[0] < other[0]
            && own[1] < other[1]
            && candidate_state_rows_max_absolute_difference > 0.0,
    })
}

fn state_max_absolute_difference(
    gpu: &Gpu,
    left: &MlxInferenceState,
    right: &MlxInferenceState,
    plan: &ModelPlan,
) -> Result<f32, EngineError> {
    left.validate(plan, 1)?;
    right.validate(plan, 1)?;
    if left.layer_count() != right.layer_count() {
        return Err(EngineError::Unsupported(
            "state comparison layer counts differ".into(),
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
                    "state comparison layer {index} kinds differ"
                )));
            }
        };
        for (left_array, right_array) in pairs {
            if left_array.shape() != right_array.shape()
                || left_array.dtype() != right_array.dtype()
            {
                return Err(EngineError::Unsupported(format!(
                    "state comparison layer {index} tensor layout differs"
                )));
            }
            maximum = maximum.max(
                gpu.max_abs_difference(left_array, right_array)
                    .map_err(EngineError::Mlx)?,
            );
        }
    }
    Ok(maximum)
}

fn to_attempt(
    phase: &'static str,
    round: usize,
    order_position: usize,
    mode: ParallelGenerationMode,
    execution: &PairExecution,
    correctness: ParallelCorrectnessObservation,
    max_new_tokens: usize,
) -> Result<ParallelGenerationAttempt, EngineError> {
    let total_tokens = max_new_tokens
        .checked_mul(2)
        .ok_or_else(|| EngineError::Unsupported("parallel token denominator overflow".into()))?;
    let generated = [execution.output[0].len(), execution.output[1].len()];
    if generated != [max_new_tokens, max_new_tokens] {
        return Err(EngineError::Unsupported(format!(
            "parallel attempt generated {generated:?}, expected {max_new_tokens} each"
        )));
    }
    let output_token_sha256 = [
        token_digest(&execution.output[0]),
        token_digest(&execution.output[1]),
    ];
    let state_sequence_lengths = [
        execution.state[0].sequence_length()?,
        execution.state[1].sequence_length()?,
    ];
    let state_logical_nbytes = [
        execution.state[0].logical_nbytes()?,
        execution.state[1].logical_nbytes()?,
    ];
    Ok(ParallelGenerationAttempt {
        phase,
        round,
        order_position,
        mode,
        pair_total_nanos: execution.pair_total_nanos,
        pair_decode_nanos: execution.pair_decode_nanos,
        ttft_nanos: execution.ttft_nanos,
        completion_nanos: execution.completion_nanos,
        request_decode_nanos: execution.request_decode_nanos,
        aggregate_decode_tokens_per_second: tokens_per_second(
            total_tokens,
            execution.pair_decode_nanos,
        )?,
        aggregate_end_to_end_tokens_per_second: tokens_per_second(
            total_tokens,
            execution.pair_total_nanos,
        )?,
        request_decode_tokens_per_second: [
            tokens_per_second(max_new_tokens, execution.request_decode_nanos[0])?,
            tokens_per_second(max_new_tokens, execution.request_decode_nanos[1])?,
        ],
        output_token_sha256,
        state_sequence_lengths,
        state_logical_nbytes,
        memory: execution.memory,
        correctness,
    })
}

fn summarize_mode(
    attempts: &[ParallelGenerationAttempt],
    mode: ParallelGenerationMode,
) -> Result<ParallelGenerationSummary, EngineError> {
    let measured = attempts
        .iter()
        .filter(|attempt| attempt.phase == "measured" && attempt.mode == mode)
        .collect::<Vec<_>>();
    let floating_values = |extract: fn(&ParallelGenerationAttempt) -> f64| {
        measured
            .iter()
            .map(|attempt| extract(attempt))
            .collect::<Vec<_>>()
    };
    let integer_values = |extract: fn(&ParallelGenerationAttempt) -> u64| {
        measured
            .iter()
            .map(|attempt| extract(attempt))
            .collect::<Vec<_>>()
    };
    Ok(ParallelGenerationSummary {
        mode,
        count: measured.len(),
        median_pair_total_nanos: median_u64(integer_values(|attempt| attempt.pair_total_nanos))?,
        median_pair_decode_nanos: median_u64(integer_values(|attempt| attempt.pair_decode_nanos))?,
        median_ttft_nanos: [
            median_u64(integer_values(|attempt| attempt.ttft_nanos[0]))?,
            median_u64(integer_values(|attempt| attempt.ttft_nanos[1]))?,
        ],
        median_completion_nanos: [
            median_u64(integer_values(|attempt| attempt.completion_nanos[0]))?,
            median_u64(integer_values(|attempt| attempt.completion_nanos[1]))?,
        ],
        median_aggregate_decode_tokens_per_second: median_f64(floating_values(|attempt| {
            attempt.aggregate_decode_tokens_per_second
        }))?,
        median_aggregate_end_to_end_tokens_per_second: median_f64(floating_values(|attempt| {
            attempt.aggregate_end_to_end_tokens_per_second
        }))?,
        median_request_decode_tokens_per_second: [
            median_f64(floating_values(|attempt| {
                attempt.request_decode_tokens_per_second[0]
            }))?,
            median_f64(floating_values(|attempt| {
                attempt.request_decode_tokens_per_second[1]
            }))?,
        ],
        median_peak_above_start_active_nbytes: median_u64(integer_values(|attempt| {
            attempt.memory.peak_above_start_active_nbytes
        }))?,
        exact_serial_equivalent_attempts: measured
            .iter()
            .filter(|attempt| attempt.correctness.exact_serial_equivalence)
            .count(),
        own_serial_state_closer_attempts: measured
            .iter()
            .filter(|attempt| attempt.correctness.each_state_is_closer_to_own_serial)
            .count(),
    })
}

fn memory_observation(
    start: MetalMemoryStats,
    end: MetalMemoryStats,
) -> Result<ParallelMemoryObservation, EngineError> {
    let start_active = i128::try_from(start.active_nbytes).map_err(|error| {
        EngineError::Unsupported(format!("start active memory does not fit i128: {error}"))
    })?;
    let end_active = i128::try_from(end.active_nbytes).map_err(|error| {
        EngineError::Unsupported(format!("end active memory does not fit i128: {error}"))
    })?;
    let active_delta_nbytes = i64::try_from(end_active - start_active).map_err(|error| {
        EngineError::Unsupported(format!("active memory delta does not fit i64: {error}"))
    })?;
    Ok(ParallelMemoryObservation {
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

const fn empty_memory_observation() -> ParallelMemoryObservation {
    ParallelMemoryObservation {
        start: RuntimeMemoryStats {
            active_nbytes: 0,
            cache_nbytes: 0,
            peak_nbytes: 0,
        },
        end: RuntimeMemoryStats {
            active_nbytes: 0,
            cache_nbytes: 0,
            peak_nbytes: 0,
        },
        active_delta_nbytes: 0,
        peak_above_start_active_nbytes: 0,
    }
}

fn mode_order(ordinal: usize) -> [ParallelGenerationMode; 3] {
    const ORDERS: [[ParallelGenerationMode; 3]; 3] = [
        [
            ParallelGenerationMode::FifoSerial,
            ParallelGenerationMode::IndependentStreams,
            ParallelGenerationMode::FixedBatchTwo,
        ],
        [
            ParallelGenerationMode::IndependentStreams,
            ParallelGenerationMode::FixedBatchTwo,
            ParallelGenerationMode::FifoSerial,
        ],
        [
            ParallelGenerationMode::FixedBatchTwo,
            ParallelGenerationMode::FifoSerial,
            ParallelGenerationMode::IndependentStreams,
        ],
    ];
    ORDERS[ordinal % ORDERS.len()]
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
            "parallel throughput requires non-zero tokens and duration".into(),
        ));
    }
    let tokens_u32 = u32::try_from(tokens).map_err(|error| {
        EngineError::Unsupported(format!("parallel token count does not fit u32: {error}"))
    })?;
    Ok(f64::from(tokens_u32) / Duration::from_nanos(nanos).as_secs_f64())
}

fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn median_u64(mut values: Vec<u64>) -> Result<u64, EngineError> {
    if values.is_empty() {
        return Err(EngineError::Unsupported(
            "parallel diagnostic median requires samples".into(),
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
            "parallel diagnostic median requires finite samples".into(),
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
    fn rotates_each_mode_through_each_order_position() {
        assert_eq!(
            mode_order(0),
            [
                ParallelGenerationMode::FifoSerial,
                ParallelGenerationMode::IndependentStreams,
                ParallelGenerationMode::FixedBatchTwo,
            ]
        );
        assert_eq!(
            mode_order(1),
            [
                ParallelGenerationMode::IndependentStreams,
                ParallelGenerationMode::FixedBatchTwo,
                ParallelGenerationMode::FifoSerial,
            ]
        );
        assert_eq!(
            mode_order(2),
            [
                ParallelGenerationMode::FixedBatchTwo,
                ParallelGenerationMode::FifoSerial,
                ParallelGenerationMode::IndependentStreams,
            ]
        );
        assert_eq!(mode_order(3), mode_order(0));
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn medians_handle_odd_and_even_samples() {
        assert_eq!(
            median_f64(vec![3.0, 1.0, 2.0]).expect("odd float median"),
            2.0
        );
        assert_eq!(
            median_f64(vec![4.0, 1.0, 3.0, 2.0]).expect("even float median"),
            2.5
        );
        assert_eq!(median_u64(vec![3, 1, 2]).expect("odd integer median"), 2);
        assert_eq!(
            median_u64(vec![4, 1, 3, 2]).expect("even integer median"),
            2
        );
    }
}
