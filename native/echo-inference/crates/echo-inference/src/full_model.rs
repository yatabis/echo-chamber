use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use echo_inference_state::{
    CommittedState, ExpectedState, InstanceId, ModelIdentity, PreparedState, StateStore,
};
use echo_mlx::{Array, DType, DequantizeConfig, Gpu, SafeTensors};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::decoder::MoeKernel;
use super::gdn::GdnKernel;
use super::gdn::{
    apply_bound_quantized_linear, dimension_i32, quantized_linear_with_config, require_tensor,
    validate_array,
};
use super::layer::{
    execute_attention_decoder_layer, execute_buffered_attention_decoder_layer_with_bound_weights,
    execute_gdn_decoder_layer_with_bound_weights, execute_gdn_decoder_layer_with_kernel,
};
use super::model_state::{LayerState, MlxInferenceState};
use super::runtime::{
    GenerationDirective, GenerationFinishReason, GenerationObserver, InferenceRequest,
    InferenceResponse, RequestState, ResidentEngine, ResidentEngineConfig, RuntimeError,
    RuntimeMetrics, ScheduledInferenceOutcome, SingleGenerationScheduler,
};
use super::sampling::SamplingConfig;
use super::snapshot::{CURRENT_STATE_FILE, CurrentStateOwner};
use super::weights::{BoundModelWeights, BoundQuantizedWeights, ShardedWeights, TensorLookup};
use super::{EngineError, ModelPlan, identify_model, sha256_file};

const KV_CACHE_ALLOCATION_STEP: usize = 256;

#[derive(Debug, Deserialize)]
struct FixtureManifest {
    schema_version: u32,
    model_type: String,
    config_sha256: String,
    prompt_token_ids: Vec<u32>,
    prefix_length: usize,
    continuation_length: usize,
    generation_steps: usize,
    expected_generated_tokens: Vec<u32>,
    input_dtype: String,
    hidden_dtype: String,
    logits_dtype: String,
    prefix_attention_mask_mode: String,
    prefix_gdn_mask_mode: String,
    continuation_attention_mask_mode: String,
    continuation_gdn_mask_mode: String,
    layer_classes: Vec<String>,
    dimensions: FixtureDimensions,
    rope: FixtureRope,
    quantization: FixtureQuantization,
    norm_topk_prob: bool,
    tie_word_embeddings: bool,
    fixture_sha256: String,
    fixture_tensor_count: usize,
}

#[derive(Debug, Deserialize, Eq, PartialEq)]
struct FixtureDimensions {
    hidden_size: usize,
    layer_count: usize,
    full_attention_interval: usize,
    gdn_key_heads: usize,
    gdn_value_heads: usize,
    gdn_key_head_dim: usize,
    gdn_value_head_dim: usize,
    gdn_conv_kernel_size: usize,
    attention_heads: usize,
    key_value_heads: usize,
    attention_head_dim: usize,
    rotary_dim: usize,
    expert_count: usize,
    experts_per_token: usize,
    moe_intermediate_size: usize,
    shared_expert_intermediate_size: usize,
    vocabulary_size: usize,
}

#[derive(Debug, Deserialize, PartialEq)]
struct FixtureRope {
    base: f32,
    scale: f32,
    traditional: bool,
}

#[derive(Debug, Deserialize, Eq, PartialEq)]
struct FixtureQuantization {
    default: FixtureQuantizationParameters,
    router: FixtureQuantizationParameters,
    shared_expert_gate: FixtureQuantizationParameters,
}

#[derive(Debug, Deserialize, Eq, PartialEq)]
struct FixtureQuantizationParameters {
    group_size: usize,
    bits: usize,
    mode: String,
}

struct FullModelExecution {
    embedding: Array,
    layer_outputs: Vec<Array>,
    normalized_hidden: Array,
    logits: Array,
    state: MlxInferenceState,
}

pub(crate) struct RuntimeModelExecution {
    pub(crate) logits: Array,
    pub(crate) state: RuntimeInferenceState,
}

enum RuntimeLayerState {
    Gdn {
        convolution: Array,
        recurrent: Array,
    },
    Attention {
        key_buffer: Array,
        value_buffer: Array,
        offset: usize,
    },
}

pub(crate) struct RuntimeInferenceState {
    layers: Vec<RuntimeLayerState>,
    left_padding: Option<Vec<usize>>,
}

#[derive(Clone, Copy, Debug, Default)]
struct PhaseDifferences {
    embedding: f32,
    layer_output: f32,
    normalized_hidden: f32,
    logits: f32,
    state: f32,
}

impl PhaseDifferences {
    fn maximum(self) -> f32 {
        self.embedding
            .max(self.layer_output)
            .max(self.normalized_hidden)
            .max(self.logits)
            .max(self.state)
    }

    fn exact(self) -> bool {
        self.maximum() == 0.0
    }
}

/// Exact full-model and minimal greedy-generation comparison for the target
/// Qwen3.5-family `MoE` model.
#[derive(Clone, Debug, Serialize)]
pub struct FullModelParity {
    pub architecture: String,
    pub batch_size: usize,
    pub prefix_length: usize,
    pub continuation_length: usize,
    pub generation_steps: usize,
    pub weight_shard_count: usize,
    pub weight_tensor_count: usize,
    pub fixture_sha256: String,
    pub prefix_embedding_max_absolute_difference: f32,
    pub prefix_layer_output_max_absolute_difference: f32,
    pub prefix_normalized_hidden_max_absolute_difference: f32,
    pub prefix_logits_max_absolute_difference: f32,
    pub prefix_state_max_absolute_difference: f32,
    pub continuation_embedding_max_absolute_difference: f32,
    pub continuation_layer_output_max_absolute_difference: f32,
    pub continuation_normalized_hidden_max_absolute_difference: f32,
    pub continuation_logits_max_absolute_difference: f32,
    pub continuation_state_max_absolute_difference: f32,
    pub generation_logits_max_absolute_difference: f32,
    pub generation_state_max_absolute_difference: f32,
    pub generated_tokens_max_absolute_difference: f32,
    pub generated_tokens: Vec<u32>,
    pub generated_token_rows: Vec<Vec<u32>>,
    pub differences: BTreeMap<String, f32>,
    pub exact: bool,
}

/// Exact proof that live MLX state can be committed and restored by instance.
#[derive(Clone, Debug, Serialize)]
pub struct LiveStateParity {
    pub instance_id: InstanceId,
    pub model: ModelIdentity,
    pub fixture_sha256: String,
    pub prefix_sequence_length: usize,
    pub continuation_sequence_length: usize,
    pub initial_weight_shard_count: usize,
    pub restored_weight_shard_count: usize,
    pub initial_weight_tensor_count: usize,
    pub restored_weight_tensor_count: usize,
    pub state_layer_count: usize,
    pub state_tensor_count: usize,
    pub prefix_state_logical_nbytes: usize,
    pub continuation_state_logical_nbytes: usize,
    pub fixture_token_ids_max_absolute_difference: f32,
    pub prefix_vs_oracle_max_absolute_difference: f32,
    pub prefix_logits_vs_oracle_max_absolute_difference: f32,
    pub prefix_state_vs_oracle_max_absolute_difference: f32,
    pub direct_continuation_vs_oracle_max_absolute_difference: f32,
    pub direct_continuation_logits_vs_oracle_max_absolute_difference: f32,
    pub direct_continuation_state_vs_oracle_max_absolute_difference: f32,
    pub restored_continuation_vs_oracle_max_absolute_difference: f32,
    pub restored_continuation_logits_vs_oracle_max_absolute_difference: f32,
    pub restored_continuation_state_vs_oracle_max_absolute_difference: f32,
    pub restored_continuation_vs_direct_max_absolute_difference: f32,
    pub restored_continuation_logits_vs_direct_max_absolute_difference: f32,
    pub restored_continuation_state_vs_direct_max_absolute_difference: f32,
    pub store_current_is_continuation_commit: bool,
    pub exact: bool,
}

/// Producer-process evidence for one durable current-state checkpoint.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DurableStateProducer {
    pub process_id: u32,
    pub instance_id: InstanceId,
    pub model: ModelIdentity,
    pub fixture_sha256: String,
    pub prefix_sequence_length: usize,
    pub current_path: PathBuf,
    pub physical_nbytes: u64,
    pub logical_nbytes: usize,
    pub state_tensor_count: usize,
    pub weight_shard_count: usize,
    pub weight_tensor_count: usize,
    pub fixture_token_ids_max_absolute_difference: f32,
    pub prefix_vs_oracle_max_absolute_difference: f32,
    pub prefix_logits_vs_oracle_max_absolute_difference: f32,
    pub prefix_state_vs_oracle_max_absolute_difference: f32,
    pub direct_continuation_vs_oracle_max_absolute_difference: f32,
    pub direct_continuation_logits_vs_oracle_max_absolute_difference: f32,
    pub direct_continuation_state_vs_oracle_max_absolute_difference: f32,
    pub exact: bool,
}

/// Restorer-process evidence after loading and replacing one current state.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DurableStateRestorer {
    pub process_id: u32,
    pub instance_id: InstanceId,
    pub model: ModelIdentity,
    pub restored_sequence_length: usize,
    pub continuation_sequence_length: usize,
    pub current_path: PathBuf,
    pub continuation_physical_nbytes: u64,
    pub continuation_logical_nbytes: usize,
    pub state_tensor_count: usize,
    pub weight_shard_count: usize,
    pub weight_tensor_count: usize,
    pub restored_store_is_load_bearing: bool,
    pub restored_continuation_vs_oracle_max_absolute_difference: f32,
    pub restored_continuation_logits_vs_oracle_max_absolute_difference: f32,
    pub restored_continuation_state_vs_oracle_max_absolute_difference: f32,
    pub exact: bool,
}

/// End-to-end proof of crash-consistent state publication and process restart.
#[derive(Clone, Debug, Serialize)]
pub struct DurableStateParity {
    pub orchestrator_process_id: u32,
    pub producer: DurableStateProducer,
    pub restorer: DurableStateRestorer,
    pub assertions: DurableStateAssertions,
    pub exact: bool,
}

/// Cross-process assertions made after both worker processes have exited.
#[derive(Clone, Copy, Debug, Serialize)]
// Each boolean is independently inspectable evidence for one durability axis.
#[allow(clippy::struct_excessive_bools)]
pub struct DurableStateAssertions {
    pub distinct_producer_and_restorer_processes: bool,
    pub current_file_present_after_producer_exit: bool,
    pub current_file_present_after_restorer_exit: bool,
    pub restored_state_was_load_bearing: bool,
    pub continuation_exact_via_common_oracle: bool,
}

/// Metrics retained for one successful request in the resident-runtime proof.
#[derive(Clone, Debug, Serialize)]
pub struct ResidentRuntimeRequestParity {
    pub ticket: u64,
    pub phase: String,
    pub instance_id: InstanceId,
    pub state_sequence_length: usize,
    pub generated_tokens: Vec<u32>,
    pub metrics: RuntimeMetrics,
}

/// Exact proof of the resident model owner and FIFO state transaction boundary.
#[derive(Clone, Debug, Serialize)]
pub struct ResidentRuntimeParity {
    pub architecture: String,
    pub fixture_sha256: String,
    pub model: ModelIdentity,
    pub engine_id: u64,
    pub model_load_nanos: u64,
    pub resident_model_owner_count: usize,
    pub weight_shard_count: usize,
    pub weight_tensor_count: usize,
    pub scheduler_active_generation_limit: usize,
    pub queue_tickets: Vec<u64>,
    pub observed_execution_order: Vec<u64>,
    pub cancelled_ticket: u64,
    pub successful_requests: Vec<ResidentRuntimeRequestParity>,
    pub failed_request: ResidentRuntimeFailureParity,
    pub generated_tokens: Vec<u32>,
    pub assertions: ResidentRuntimeAssertions,
    pub rin_prefix_state_vs_oracle_max_absolute_difference: f32,
    pub marie_prefix_state_vs_oracle_max_absolute_difference: f32,
    pub rin_generated_state_vs_oracle_max_absolute_difference: f32,
    pub marie_continuation_state_vs_oracle_max_absolute_difference: f32,
    pub exact: bool,
}

/// Retained error and state boundary for the rollback request.
#[derive(Clone, Debug, Serialize)]
pub struct ResidentRuntimeFailureParity {
    pub ticket: u64,
    pub phase: String,
    pub instance_id: InstanceId,
    pub error: String,
    pub sequence_length_before: usize,
    pub sequence_length_after: usize,
    pub preserved_same_state: bool,
}

/// Structured resident-runtime invariants retained by the proof.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ResidentRuntimeAssertions {
    pub queue: ResidentRuntimeQueueAssertions,
    pub state: ResidentRuntimeStateAssertions,
    pub residency: ResidentRuntimeResidencyAssertions,
}

/// FIFO and waiting-request cancellation assertions.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ResidentRuntimeQueueAssertions {
    pub fifo_order_exact: bool,
    pub scheduler_continued_after_failed_request: bool,
    pub cancelled_request_never_executed: bool,
}

/// Per-instance transaction and rollback assertions.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ResidentRuntimeStateAssertions {
    pub rin_and_marie_states_are_distinct: bool,
    pub invalid_initial_transition_rejected: bool,
    pub failed_request_preserved_rin_state: bool,
}

/// Resident weight-owner assertion.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct ResidentRuntimeResidencyAssertions {
    pub same_resident_engine_for_every_success: bool,
}

/// Real-model proof of E.C.H.O.'s GDN-preserving session transition.
#[derive(Clone, Debug, Serialize)]
pub struct NewSessionParity {
    pub architecture: String,
    pub fixture_sha256: String,
    pub model: ModelIdentity,
    pub engine_id: u64,
    pub first_session_sequence_length: usize,
    pub first_session_generated_tokens: Vec<u32>,
    pub first_session_state_vs_oracle_max_absolute_difference: f32,
    pub transition_gdn_max_absolute_difference: f32,
    pub transitioned_gdn_vs_empty_max_absolute_difference: f32,
    pub new_session_sequence_length: usize,
    pub new_session_generated_tokens: Vec<u32>,
    pub new_session_metrics: RuntimeMetrics,
    pub empty_ablation_generated_tokens: Vec<u32>,
    pub new_session_state_vs_empty_ablation_max_absolute_difference: f32,
    pub continuation_sequence_length: usize,
    pub continuation_metrics: RuntimeMetrics,
    pub assertions: NewSessionAssertions,
    pub exact: bool,
}

/// State, lineage, and execution assertions made by [`run_new_session_parity`].
#[derive(Clone, Copy, Debug, Serialize)]
// A flat set of independently inspectable booleans is the machine-readable
// evidence contract; collapsing them into one status would hide failed axes.
#[allow(clippy::struct_excessive_bools)]
pub struct NewSessionAssertions {
    pub gdn_carried_exactly: bool,
    pub attention_kv_cleared_before_fresh_prompt: bool,
    pub carried_gdn_is_nonempty: bool,
    pub fresh_prompt_processed_without_cached_prefix: bool,
    pub shorter_token_lineage_replaced_the_old_lineage: bool,
    pub carried_state_is_load_bearing_against_empty_ablation: bool,
    pub continuation_reused_only_the_new_lineage: bool,
    pub every_success_used_one_resident_owner: bool,
}

struct StopAfterFirstGeneratedToken;

impl GenerationObserver for StopAfterFirstGeneratedToken {
    fn is_cancelled(&self) -> bool {
        false
    }

    fn on_token(&mut self, _token: u32) -> Result<GenerationDirective, String> {
        Ok(GenerationDirective::Stop)
    }
}

/// Runs multiple E.C.H.O. instances through one resident model and FIFO owner.
///
/// The retained official fixture remains the numerical oracle. This scenario
/// additionally proves that a continuation does not replay its cached prefix,
/// generated tokens advance state, an invalid state transition rolls back after
/// lease acquisition, a later instance still runs, and a cancelled waiting
/// request never executes.
///
/// # Errors
///
/// Returns [`EngineError`] on model, fixture, scheduler, request, state, metric,
/// or MLX drift.
#[allow(clippy::too_many_lines)]
pub fn run_resident_runtime_parity(
    model_directory: &Path,
    fixture_path: &Path,
    manifest_path: &Path,
) -> Result<ResidentRuntimeParity, EngineError> {
    let plan = ModelPlan::from_directory(model_directory)?;
    let manifest = load_manifest(manifest_path)?;
    validate_manifest(&plan, model_directory, fixture_path, &manifest)?;
    let fixture = SafeTensors::load(fixture_path).map_err(EngineError::Mlx)?;
    if fixture.len() != manifest.fixture_tensor_count {
        return Err(EngineError::Unsupported(format!(
            "full-model fixture tensor count drift: manifest {}, payload {}",
            manifest.fixture_tensor_count,
            fixture.len()
        )));
    }

    let engine = ResidentEngine::load(
        model_directory,
        ResidentEngineConfig {
            max_new_tokens_per_request: manifest.generation_steps,
            ..ResidentEngineConfig::default()
        },
    )
    .map_err(|error| state_operation_error("load resident runtime", error))?;
    let info = engine.info().clone();
    let mut scheduler = SingleGenerationScheduler::new(engine, 6)
        .map_err(|error| state_operation_error("create resident scheduler", error))?;

    let rin = InstanceId::new("echo-runtime-rin")
        .map_err(|error| state_operation_error("create Rin runtime instance", error))?;
    let marie = InstanceId::new("echo-runtime-marie")
        .map_err(|error| state_operation_error("create Marie runtime instance", error))?;
    let cancelled_instance = InstanceId::new("echo-runtime-cancelled")
        .map_err(|error| state_operation_error("create cancelled runtime instance", error))?;
    let prefix_tokens = manifest.prompt_token_ids[..manifest.prefix_length].to_vec();
    let continuation_input_tokens = manifest.prompt_token_ids[manifest.prefix_length..].to_vec();

    let rin_prefix_ticket = enqueue_runtime_request(
        &mut scheduler,
        InferenceRequest {
            instance_id: rin.clone(),
            state_transition: RequestState::Initial,
            input_tokens: prefix_tokens.clone(),
            max_new_tokens: 0,
            length_eos_token: None,
            sampling: SamplingConfig::default(),
        },
        "enqueue Rin prefix",
    )?;
    let marie_prefix_ticket = enqueue_runtime_request(
        &mut scheduler,
        InferenceRequest {
            instance_id: marie.clone(),
            state_transition: RequestState::Initial,
            input_tokens: prefix_tokens.clone(),
            max_new_tokens: 0,
            length_eos_token: None,
            sampling: SamplingConfig::default(),
        },
        "enqueue Marie prefix",
    )?;
    let rin_generation_ticket = enqueue_runtime_request(
        &mut scheduler,
        InferenceRequest {
            instance_id: rin.clone(),
            state_transition: RequestState::Continuation,
            input_tokens: continuation_input_tokens.clone(),
            max_new_tokens: manifest.generation_steps,
            length_eos_token: None,
            sampling: SamplingConfig::default(),
        },
        "enqueue Rin continuation and generation",
    )?;
    let invalid_initial_ticket = enqueue_runtime_request(
        &mut scheduler,
        InferenceRequest {
            instance_id: rin.clone(),
            state_transition: RequestState::Initial,
            input_tokens: continuation_input_tokens.clone(),
            max_new_tokens: 0,
            length_eos_token: None,
            sampling: SamplingConfig::default(),
        },
        "enqueue invalid Rin initial transition",
    )?;
    let marie_continuation_ticket = enqueue_runtime_request(
        &mut scheduler,
        InferenceRequest {
            instance_id: marie.clone(),
            state_transition: RequestState::Continuation,
            input_tokens: continuation_input_tokens,
            max_new_tokens: 0,
            length_eos_token: None,
            sampling: SamplingConfig::default(),
        },
        "enqueue Marie continuation",
    )?;
    let cancelled_ticket = enqueue_runtime_request(
        &mut scheduler,
        InferenceRequest {
            instance_id: cancelled_instance.clone(),
            state_transition: RequestState::Initial,
            input_tokens: prefix_tokens,
            max_new_tokens: 0,
            length_eos_token: None,
            sampling: SamplingConfig::default(),
        },
        "enqueue cancellable request",
    )?;
    let cancelled_while_waiting = scheduler.cancel(cancelled_ticket);

    let queue_tickets = vec![
        rin_prefix_ticket.get(),
        marie_prefix_ticket.get(),
        rin_generation_ticket.get(),
        invalid_initial_ticket.get(),
        marie_continuation_ticket.get(),
        cancelled_ticket.get(),
    ];
    let expected_execution_order = queue_tickets[..5].to_vec();
    let mut observed_execution_order = Vec::with_capacity(5);

    let (ticket, rin_prefix_response) = next_runtime_success(&mut scheduler, "execute Rin prefix")?;
    observed_execution_order.push(ticket);
    let rin_prefix_state = require_runtime_state(&scheduler, &rin, "Rin prefix")?;
    let rin_prefix_state_vs_oracle_max_absolute_difference = compare_runtime_state_to_fixture(
        &scheduler,
        &fixture,
        "prefix",
        &rin_prefix_state.payload,
        &plan,
    )?;

    let (ticket, marie_prefix_response) =
        next_runtime_success(&mut scheduler, "execute Marie prefix")?;
    observed_execution_order.push(ticket);
    let marie_prefix_state = require_runtime_state(&scheduler, &marie, "Marie prefix")?;
    let marie_prefix_state_vs_oracle_max_absolute_difference = compare_runtime_state_to_fixture(
        &scheduler,
        &fixture,
        "prefix",
        &marie_prefix_state.payload,
        &plan,
    )?;
    let rin_and_marie_states_are_distinct = !Arc::ptr_eq(&rin_prefix_state, &marie_prefix_state);
    drop(rin_prefix_state);
    drop(marie_prefix_state);

    let (ticket, rin_generation_response) =
        next_runtime_success(&mut scheduler, "execute Rin generation")?;
    observed_execution_order.push(ticket);
    let rin_generated_state = require_runtime_state(&scheduler, &rin, "Rin generated")?;
    let rin_generated_state_vs_oracle_max_absolute_difference = compare_runtime_state_to_fixture(
        &scheduler,
        &fixture,
        "generation.final",
        &rin_generated_state.payload,
        &plan,
    )?;
    let rin_before_failed_request = Arc::clone(&rin_generated_state);
    let sequence_length_before_failed_request =
        rin_before_failed_request.payload.sequence_length()?;
    drop(rin_generated_state);

    let failed_outcome = scheduler.run_next().ok_or_else(|| {
        EngineError::Unsupported("scheduler lost the invalid-initial request".into())
    })?;
    observed_execution_order.push(failed_outcome.ticket.get());
    let invalid_initial_transition_rejected = matches!(
        &failed_outcome.result,
        Err(RuntimeError::Begin(
            echo_inference_state::BeginError::UnexpectedExisting { .. }
        ))
    );
    let failed_request_error = match &failed_outcome.result {
        Ok(_) => "unexpected success".into(),
        Err(error) => error.to_string(),
    };
    let rin_after_failed_request = require_runtime_state(&scheduler, &rin, "Rin rollback")?;
    let sequence_length_after_failed_request =
        rin_after_failed_request.payload.sequence_length()?;
    let failed_request_preserved_rin_state =
        Arc::ptr_eq(&rin_before_failed_request, &rin_after_failed_request)
            && sequence_length_after_failed_request == sequence_length_before_failed_request;
    drop(rin_before_failed_request);
    drop(rin_after_failed_request);

    let (ticket, marie_continuation_response) =
        next_runtime_success(&mut scheduler, "execute Marie continuation")?;
    observed_execution_order.push(ticket);
    let marie_continuation_state = require_runtime_state(&scheduler, &marie, "Marie continuation")?;
    let marie_continuation_state_vs_oracle_max_absolute_difference =
        compare_runtime_state_to_fixture(
            &scheduler,
            &fixture,
            "continuation",
            &marie_continuation_state.payload,
            &plan,
        )?;
    drop(marie_continuation_state);

    let cancelled_request_never_executed = cancelled_while_waiting
        && scheduler
            .engine()
            .current_state(&cancelled_instance)
            .is_none()
        && scheduler.queued_request_count() == 0
        && scheduler.run_next().is_none();
    let fifo_order_exact = observed_execution_order == expected_execution_order;
    let responses = [
        &rin_prefix_response,
        &marie_prefix_response,
        &rin_generation_response,
        &marie_continuation_response,
    ];
    let same_resident_engine_for_every_success = responses
        .iter()
        .all(|response| response.engine_id == info.engine_id && response.model == info.model);
    let scheduler_continued_after_failed_request =
        marie_continuation_response.state_sequence_length == manifest.prompt_token_ids.len();
    let request_metrics_exact = rin_prefix_response.metrics.cached_prefix_tokens == 0
        && rin_prefix_response.metrics.input_tokens_processed == manifest.prefix_length
        && rin_prefix_response.metrics.generated_tokens == 0
        && marie_prefix_response.metrics.cached_prefix_tokens == 0
        && marie_prefix_response.metrics.input_tokens_processed == manifest.prefix_length
        && rin_generation_response.metrics.cached_prefix_tokens == manifest.prefix_length
        && rin_generation_response.metrics.input_tokens_processed == manifest.continuation_length
        && rin_generation_response.metrics.generated_tokens == manifest.generation_steps
        && rin_generation_response.metrics.model_step_count == manifest.generation_steps + 1
        && rin_generation_response
            .metrics
            .first_generated_token_nanos
            .is_some()
        && marie_continuation_response.metrics.cached_prefix_tokens == manifest.prefix_length
        && marie_continuation_response.metrics.input_tokens_processed
            == manifest.continuation_length
        && marie_continuation_response.metrics.generated_tokens == 0;
    let state_lengths_exact = rin_prefix_response.state_sequence_length == manifest.prefix_length
        && marie_prefix_response.state_sequence_length == manifest.prefix_length
        && rin_generation_response.state_sequence_length
            == manifest.prompt_token_ids.len() + manifest.generation_steps
        && marie_continuation_response.state_sequence_length == manifest.prompt_token_ids.len();
    let generated_tokens = rin_generation_response.generated_tokens.clone();
    let exact = fifo_order_exact
        && same_resident_engine_for_every_success
        && rin_and_marie_states_are_distinct
        && invalid_initial_transition_rejected
        && failed_request_preserved_rin_state
        && scheduler_continued_after_failed_request
        && cancelled_request_never_executed
        && request_metrics_exact
        && state_lengths_exact
        && generated_tokens == manifest.expected_generated_tokens
        && rin_prefix_state_vs_oracle_max_absolute_difference == 0.0
        && marie_prefix_state_vs_oracle_max_absolute_difference == 0.0
        && rin_generated_state_vs_oracle_max_absolute_difference == 0.0
        && marie_continuation_state_vs_oracle_max_absolute_difference == 0.0;
    let successful_requests = vec![
        runtime_request_parity(rin_prefix_ticket.get(), "rin_prefix", &rin_prefix_response),
        runtime_request_parity(
            marie_prefix_ticket.get(),
            "marie_prefix",
            &marie_prefix_response,
        ),
        runtime_request_parity(
            rin_generation_ticket.get(),
            "rin_continuation_and_generation",
            &rin_generation_response,
        ),
        runtime_request_parity(
            marie_continuation_ticket.get(),
            "marie_continuation",
            &marie_continuation_response,
        ),
    ];
    let failed_request = ResidentRuntimeFailureParity {
        ticket: failed_outcome.ticket.get(),
        phase: "rin_invalid_initial_transition".into(),
        instance_id: rin,
        error: failed_request_error,
        sequence_length_before: sequence_length_before_failed_request,
        sequence_length_after: sequence_length_after_failed_request,
        preserved_same_state: failed_request_preserved_rin_state,
    };
    let assertions = ResidentRuntimeAssertions {
        queue: ResidentRuntimeQueueAssertions {
            fifo_order_exact,
            scheduler_continued_after_failed_request,
            cancelled_request_never_executed,
        },
        state: ResidentRuntimeStateAssertions {
            rin_and_marie_states_are_distinct,
            invalid_initial_transition_rejected,
            failed_request_preserved_rin_state,
        },
        residency: ResidentRuntimeResidencyAssertions {
            same_resident_engine_for_every_success,
        },
    };

    Ok(ResidentRuntimeParity {
        architecture: plan.architecture,
        fixture_sha256: manifest.fixture_sha256,
        model: info.model,
        engine_id: info.engine_id,
        model_load_nanos: info.model_load_nanos,
        resident_model_owner_count: 1,
        weight_shard_count: info.weight_shard_count,
        weight_tensor_count: info.weight_tensor_count,
        scheduler_active_generation_limit: 1,
        queue_tickets,
        observed_execution_order,
        cancelled_ticket: cancelled_ticket.get(),
        successful_requests,
        failed_request,
        generated_tokens,
        assertions,
        rin_prefix_state_vs_oracle_max_absolute_difference,
        marie_prefix_state_vs_oracle_max_absolute_difference,
        rin_generated_state_vs_oracle_max_absolute_difference,
        marie_continuation_state_vs_oracle_max_absolute_difference,
        exact,
    })
}

/// Runs a real-model GDN-preserving transition into a fresh session.
///
/// The same resident instance first commits the retained oracle generation,
/// then starts a shorter session from that current state. Before processing
/// the fresh prompt, both GDN tensors are compared with the old state, the
/// attention cache is validated at zero tokens, and an empty-state ablation is
/// retained to prove that carried state affects the resulting model state.
/// The final request continues from only the newly committed session state.
///
/// # Errors
///
/// Returns [`EngineError`] on model, fixture, request, state, metric, or MLX
/// drift.
#[allow(clippy::too_many_lines)]
pub fn run_new_session_parity(
    model_directory: &Path,
    fixture_path: &Path,
    manifest_path: &Path,
) -> Result<NewSessionParity, EngineError> {
    let plan = ModelPlan::from_directory(model_directory)?;
    let manifest = load_manifest(manifest_path)?;
    validate_manifest(&plan, model_directory, fixture_path, &manifest)?;
    let fixture = SafeTensors::load(fixture_path).map_err(EngineError::Mlx)?;
    let mut engine = ResidentEngine::load(
        model_directory,
        ResidentEngineConfig {
            max_new_tokens_per_request: manifest.generation_steps.max(1),
            ..ResidentEngineConfig::default()
        },
    )
    .map_err(|error| state_operation_error("load new-session resident runtime", error))?;
    let info = engine.info().clone();
    let instance = InstanceId::new("echo-new-session-rin")
        .map_err(|error| state_operation_error("create new-session instance", error))?;
    let ablation_instance = InstanceId::new("echo-new-session-empty-ablation")
        .map_err(|error| state_operation_error("create empty-state ablation instance", error))?;
    let fresh_prompt_tokens = manifest.prompt_token_ids[..manifest.prefix_length].to_vec();
    let first_session_prefix_response = engine
        .execute(InferenceRequest {
            instance_id: instance.clone(),
            state_transition: RequestState::Initial,
            input_tokens: fresh_prompt_tokens.clone(),
            max_new_tokens: 0,
            length_eos_token: None,
            sampling: SamplingConfig::default(),
        })
        .map_err(|error| state_operation_error("execute first-session prefix", error))?;
    let first_session_response = engine
        .execute(InferenceRequest {
            instance_id: instance.clone(),
            state_transition: RequestState::Continuation,
            input_tokens: manifest.prompt_token_ids[manifest.prefix_length..].to_vec(),
            max_new_tokens: manifest.generation_steps,
            length_eos_token: None,
            sampling: SamplingConfig::default(),
        })
        .map_err(|error| state_operation_error("execute first session", error))?;
    let first_session_state = require_resident_state(&engine, &instance, "first session")?;
    let mut oracle_differences = BTreeMap::new();
    let first_session_state_vs_oracle_max_absolute_difference = compare_states(
        engine.gpu(),
        &fixture,
        "generation.final",
        &first_session_state.payload,
        &plan,
        &mut oracle_differences,
    )?;

    let transitioned = first_session_state.payload.begin_new_session(
        engine.gpu(),
        1,
        &plan,
        crate::NewSessionGdnPolicy::CarryAll,
    )?;
    transitioned.validate(&plan, 1)?;
    let transition_gdn_max_absolute_difference = compare_gdn_state_values(
        engine.gpu(),
        &first_session_state.payload,
        &transitioned,
        &plan,
    )?;
    let empty_state = MlxInferenceState::empty(engine.gpu(), 1, &plan)?;
    let transitioned_gdn_vs_empty_max_absolute_difference =
        compare_gdn_state_values(engine.gpu(), &transitioned, &empty_state, &plan)?;

    let mut stop_after_first = StopAfterFirstGeneratedToken;
    let new_session_response = engine
        .execute_observed(
            InferenceRequest {
                instance_id: instance.clone(),
                state_transition: RequestState::NewSession,
                input_tokens: fresh_prompt_tokens.clone(),
                max_new_tokens: 1,
                length_eos_token: None,
                sampling: SamplingConfig::default(),
            },
            &mut stop_after_first,
        )
        .map_err(|error| state_operation_error("execute new session", error))?;
    let new_session_state = require_resident_state(&engine, &instance, "new session")?;
    new_session_state.payload.validate(&plan, 1)?;

    let mut stop_after_first = StopAfterFirstGeneratedToken;
    let empty_ablation_response = engine
        .execute_observed(
            InferenceRequest {
                instance_id: ablation_instance.clone(),
                state_transition: RequestState::Initial,
                input_tokens: fresh_prompt_tokens.clone(),
                max_new_tokens: 1,
                length_eos_token: None,
                sampling: SamplingConfig::default(),
            },
            &mut stop_after_first,
        )
        .map_err(|error| state_operation_error("execute empty-state ablation", error))?;
    let empty_ablation_state =
        require_resident_state(&engine, &ablation_instance, "empty-state ablation")?;
    let new_session_state_vs_empty_ablation_max_absolute_difference = compare_state_values(
        engine.gpu(),
        &new_session_state.payload,
        &empty_ablation_state.payload,
        &plan,
    )?;

    let mut new_session_lineage = fresh_prompt_tokens;
    new_session_lineage.extend_from_slice(&new_session_response.generated_tokens);
    let continuation_input_token = manifest
        .prompt_token_ids
        .get(manifest.prefix_length)
        .copied()
        .ok_or_else(|| {
            EngineError::Unsupported(
                "new-session parity fixture has no continuation input token".into(),
            )
        })?;
    let mut continuation_lineage = new_session_lineage.clone();
    continuation_lineage.push(continuation_input_token);
    let continuation_response = engine
        .execute(InferenceRequest {
            instance_id: instance.clone(),
            state_transition: RequestState::Continuation,
            input_tokens: vec![continuation_input_token],
            max_new_tokens: 0,
            length_eos_token: None,
            sampling: SamplingConfig::default(),
        })
        .map_err(|error| state_operation_error("continue new session", error))?;

    let mut first_session_lineage = manifest.prompt_token_ids.clone();
    first_session_lineage.extend_from_slice(&manifest.expected_generated_tokens);
    let gdn_carried_exactly = transition_gdn_max_absolute_difference == 0.0;
    let attention_kv_cleared_before_fresh_prompt =
        transitioned.validate(&plan, 1).is_ok() && transitioned.sequence_length()? == 0;
    let carried_gdn_is_nonempty = transitioned_gdn_vs_empty_max_absolute_difference > 0.0;
    let fresh_prompt_processed_without_cached_prefix =
        new_session_response.metrics.cached_prefix_tokens == 0
            && new_session_response.metrics.input_tokens_processed == manifest.prefix_length
            && new_session_response.metrics.generated_tokens == 1
            && new_session_response.finish_reason == GenerationFinishReason::StopToken;
    let shorter_token_lineage_replaced_the_old_lineage = new_session_response.state_sequence_length
        == new_session_lineage.len()
        && new_session_response.state_sequence_length
            < first_session_response.state_sequence_length;
    let carried_state_is_load_bearing_against_empty_ablation =
        new_session_state_vs_empty_ablation_max_absolute_difference > 0.0;
    let continuation_reused_only_the_new_lineage = continuation_response.state_sequence_length
        == continuation_lineage.len()
        && continuation_response.metrics.cached_prefix_tokens == new_session_lineage.len()
        && continuation_response.metrics.input_tokens_processed == 1;
    let every_success_used_one_resident_owner = [
        &first_session_prefix_response,
        &first_session_response,
        &new_session_response,
        &empty_ablation_response,
        &continuation_response,
    ]
    .iter()
    .all(|response| response.engine_id == info.engine_id && response.model == info.model);
    let assertions = NewSessionAssertions {
        gdn_carried_exactly,
        attention_kv_cleared_before_fresh_prompt,
        carried_gdn_is_nonempty,
        fresh_prompt_processed_without_cached_prefix,
        shorter_token_lineage_replaced_the_old_lineage,
        carried_state_is_load_bearing_against_empty_ablation,
        continuation_reused_only_the_new_lineage,
        every_success_used_one_resident_owner,
    };
    let exact = first_session_prefix_response.state_sequence_length == manifest.prefix_length
        && first_session_response.state_sequence_length == first_session_lineage.len()
        && first_session_response.generated_tokens == manifest.expected_generated_tokens
        && first_session_state_vs_oracle_max_absolute_difference == 0.0
        && gdn_carried_exactly
        && attention_kv_cleared_before_fresh_prompt
        && carried_gdn_is_nonempty
        && fresh_prompt_processed_without_cached_prefix
        && shorter_token_lineage_replaced_the_old_lineage
        && carried_state_is_load_bearing_against_empty_ablation
        && continuation_reused_only_the_new_lineage
        && every_success_used_one_resident_owner;

    Ok(NewSessionParity {
        architecture: plan.architecture,
        fixture_sha256: manifest.fixture_sha256,
        model: info.model,
        engine_id: info.engine_id,
        first_session_sequence_length: first_session_response.state_sequence_length,
        first_session_generated_tokens: first_session_response.generated_tokens,
        first_session_state_vs_oracle_max_absolute_difference,
        transition_gdn_max_absolute_difference,
        transitioned_gdn_vs_empty_max_absolute_difference,
        new_session_sequence_length: new_session_response.state_sequence_length,
        new_session_generated_tokens: new_session_response.generated_tokens,
        new_session_metrics: new_session_response.metrics,
        empty_ablation_generated_tokens: empty_ablation_response.generated_tokens,
        new_session_state_vs_empty_ablation_max_absolute_difference,
        continuation_sequence_length: continuation_response.state_sequence_length,
        continuation_metrics: continuation_response.metrics,
        assertions,
        exact,
    })
}

fn require_resident_state(
    engine: &ResidentEngine,
    instance_id: &InstanceId,
    phase: &str,
) -> Result<Arc<CommittedState<MlxInferenceState>>, EngineError> {
    engine
        .current_state(instance_id)
        .ok_or_else(|| EngineError::Unsupported(format!("{phase} state disappeared after commit")))
}

fn enqueue_runtime_request(
    scheduler: &mut SingleGenerationScheduler,
    request: InferenceRequest,
    operation: &str,
) -> Result<super::runtime::ScheduleTicket, EngineError> {
    scheduler
        .enqueue(request)
        .map_err(|error| state_operation_error(operation, error))
}

fn next_runtime_success(
    scheduler: &mut SingleGenerationScheduler,
    operation: &str,
) -> Result<(u64, InferenceResponse), EngineError> {
    let ScheduledInferenceOutcome { ticket, result } = scheduler
        .run_next()
        .ok_or_else(|| EngineError::Unsupported(format!("{operation}: queue is empty")))?;
    let response = result.map_err(|error| state_operation_error(operation, error))?;
    Ok((ticket.get(), response))
}

fn require_runtime_state(
    scheduler: &SingleGenerationScheduler,
    instance_id: &InstanceId,
    phase: &str,
) -> Result<Arc<CommittedState<MlxInferenceState>>, EngineError> {
    scheduler
        .engine()
        .current_state(instance_id)
        .ok_or_else(|| EngineError::Unsupported(format!("{phase} state disappeared after commit")))
}

fn compare_runtime_state_to_fixture(
    scheduler: &SingleGenerationScheduler,
    fixture: &SafeTensors,
    phase: &str,
    state: &MlxInferenceState,
    plan: &ModelPlan,
) -> Result<f32, EngineError> {
    let mut differences = BTreeMap::new();
    compare_states(
        scheduler.engine().gpu(),
        fixture,
        phase,
        state,
        plan,
        &mut differences,
    )
}

fn runtime_request_parity(
    ticket: u64,
    phase: &str,
    response: &InferenceResponse,
) -> ResidentRuntimeRequestParity {
    ResidentRuntimeRequestParity {
        ticket,
        phase: phase.into(),
        instance_id: response.instance_id.clone(),
        state_sequence_length: response.state_sequence_length,
        generated_tokens: response.generated_tokens.clone(),
        metrics: response.metrics.clone(),
    }
}

/// Executes an empty-state prefix, a stateful continuation, and two or more
/// greedy decode steps through the complete target model.
///
/// # Errors
///
/// Returns [`EngineError`] when the model, oracle, sharded weights, state
/// shapes, dtypes, or any MLX operation differ from the admitted plan.
#[allow(clippy::too_many_lines)]
pub fn run_full_model_parity(
    model_directory: &Path,
    fixture_path: &Path,
    manifest_path: &Path,
) -> Result<FullModelParity, EngineError> {
    let plan = ModelPlan::from_directory(model_directory)?;
    if plan.tie_word_embeddings {
        return Err(EngineError::Unsupported(
            "the admitted full-model path requires an independent language-model head".into(),
        ));
    }
    let manifest = load_manifest(manifest_path)?;
    validate_manifest(&plan, model_directory, fixture_path, &manifest)?;

    let fixture = SafeTensors::load(fixture_path).map_err(EngineError::Mlx)?;
    if fixture.len() != manifest.fixture_tensor_count {
        return Err(EngineError::Unsupported(format!(
            "full-model fixture tensor count drift: manifest {}, payload {}",
            manifest.fixture_tensor_count,
            fixture.len()
        )));
    }
    let weights = ShardedWeights::load(model_directory)?;
    let gpu = Gpu::new();
    let mut differences = BTreeMap::new();

    let prefix_ids = require_tensor(&fixture, "prefix.input_ids")?;
    let batch_size =
        validate_batched_token_input(prefix_ids, manifest.prefix_length, "prefix.input_ids")?;
    let empty_state = MlxInferenceState::empty(&gpu, batch_size, &plan)?;
    let prefix = execute_full_model(&gpu, prefix_ids, &empty_state, &weights, &plan)?;
    evaluate_execution(&gpu, &prefix)?;
    let prefix_differences =
        compare_phase(&gpu, &fixture, "prefix", &prefix, &plan, &mut differences)?;

    let continuation_ids = require_tensor(&fixture, "continuation.input_ids")?;
    let continuation_batch_size = validate_batched_token_input(
        continuation_ids,
        manifest.continuation_length,
        "continuation.input_ids",
    )?;
    if continuation_batch_size != batch_size {
        return Err(EngineError::Unsupported(format!(
            "full-model continuation batch size differs from prefix: {continuation_batch_size} != {batch_size}"
        )));
    }
    validate_array(
        continuation_ids,
        &[batch_size, manifest.continuation_length],
        DType::Int32,
        "continuation.input_ids",
    )?;
    let continuation = execute_full_model(&gpu, continuation_ids, &prefix.state, &weights, &plan)?;
    evaluate_execution(&gpu, &continuation)?;
    let continuation_differences = compare_phase(
        &gpu,
        &fixture,
        "continuation",
        &continuation,
        &plan,
        &mut differences,
    )?;

    let mut current_logits = continuation.logits;
    let mut current_state = continuation.state;
    let mut generated_arrays = Vec::with_capacity(manifest.generation_steps);
    let mut generated_token_rows = vec![Vec::with_capacity(manifest.generation_steps); batch_size];
    let mut generation_logits_max_absolute_difference = 0.0_f32;
    for step in 0..manifest.generation_steps {
        let token = greedy_token(&gpu, &current_logits, batch_size, &plan)?;
        let values = batched_token_values(&gpu, &token, batch_size)?;
        for (row, value) in generated_token_rows.iter_mut().zip(values) {
            row.push(value);
        }
        generated_arrays.push(token.try_clone().map_err(EngineError::Mlx)?);

        let execution = execute_full_model(&gpu, &token, &current_state, &weights, &plan)?;
        evaluate_execution(&gpu, &execution)?;
        let expected_logits =
            require_tensor(&fixture, &format!("generation.step.{step}.expected_logits"))?;
        let difference = gpu
            .max_abs_difference(&execution.logits, expected_logits)
            .map_err(EngineError::Mlx)?;
        differences.insert(
            format!("generation.step.{step}.expected_logits"),
            difference,
        );
        generation_logits_max_absolute_difference =
            generation_logits_max_absolute_difference.max(difference);
        current_logits = execution.logits;
        current_state = execution.state;
    }

    let generated_references = generated_arrays.iter().collect::<Vec<_>>();
    let generated_array = gpu
        .concatenate(&generated_references, 1)
        .map_err(EngineError::Mlx)?;
    let expected_generated_tokens = require_tensor(&fixture, "generation.expected_tokens")?;
    let generated_tokens_max_absolute_difference = gpu
        .max_abs_difference(&generated_array, expected_generated_tokens)
        .map_err(EngineError::Mlx)?;
    differences.insert(
        "generation.expected_tokens".into(),
        generated_tokens_max_absolute_difference,
    );
    let generation_state_max_absolute_difference = compare_states(
        &gpu,
        &fixture,
        "generation.final",
        &current_state,
        &plan,
        &mut differences,
    )?;

    let generated_tokens = generated_token_rows.first().cloned().ok_or_else(|| {
        EngineError::Unsupported("full-model generation produced no batch rows".into())
    })?;
    let exact = differences.values().all(|difference| *difference == 0.0)
        && generated_tokens == manifest.expected_generated_tokens;
    Ok(FullModelParity {
        architecture: plan.architecture,
        batch_size,
        prefix_length: manifest.prefix_length,
        continuation_length: manifest.continuation_length,
        generation_steps: manifest.generation_steps,
        weight_shard_count: weights.shard_count(),
        weight_tensor_count: weights.tensor_count(),
        fixture_sha256: manifest.fixture_sha256,
        prefix_embedding_max_absolute_difference: prefix_differences.embedding,
        prefix_layer_output_max_absolute_difference: prefix_differences.layer_output,
        prefix_normalized_hidden_max_absolute_difference: prefix_differences.normalized_hidden,
        prefix_logits_max_absolute_difference: prefix_differences.logits,
        prefix_state_max_absolute_difference: prefix_differences.state,
        continuation_embedding_max_absolute_difference: continuation_differences.embedding,
        continuation_layer_output_max_absolute_difference: continuation_differences.layer_output,
        continuation_normalized_hidden_max_absolute_difference: continuation_differences
            .normalized_hidden,
        continuation_logits_max_absolute_difference: continuation_differences.logits,
        continuation_state_max_absolute_difference: continuation_differences.state,
        generation_logits_max_absolute_difference,
        generation_state_max_absolute_difference,
        generated_tokens_max_absolute_difference,
        generated_tokens,
        generated_token_rows,
        differences,
        exact,
    })
}

/// Commits real prefix MLX state, reloads the execution context, restores the
/// committed payload, and replaces the same instance's current state.
///
/// The direct continuation and restored continuation are each compared with
/// the retained official MLX-LM oracle, and are also compared with each other.
/// This is an in-process lifecycle proof; it does not serialize state or prove
/// recovery across a process restart.
///
/// # Errors
///
/// Returns [`EngineError`] on model, oracle, token, state, identity, shape,
/// dtype, or MLX drift.
#[allow(clippy::too_many_lines)]
pub fn run_live_state_parity(
    model_directory: &Path,
    fixture_path: &Path,
    manifest_path: &Path,
) -> Result<LiveStateParity, EngineError> {
    let plan = ModelPlan::from_directory(model_directory)?;
    if plan.tie_word_embeddings {
        return Err(EngineError::Unsupported(
            "the admitted live-state path requires an independent language-model head".into(),
        ));
    }
    let manifest = load_manifest(manifest_path)?;
    validate_manifest(&plan, model_directory, fixture_path, &manifest)?;
    let model = identify_model(model_directory)?;

    let fixture = SafeTensors::load(fixture_path).map_err(EngineError::Mlx)?;
    if fixture.len() != manifest.fixture_tensor_count {
        return Err(EngineError::Unsupported(format!(
            "full-model fixture tensor count drift: manifest {}, payload {}",
            manifest.fixture_tensor_count,
            fixture.len()
        )));
    }
    let prefix_ids = require_tensor(&fixture, "prefix.input_ids")?;
    validate_token_input(prefix_ids, manifest.prefix_length, "prefix.input_ids")?;
    let continuation_ids = require_tensor(&fixture, "continuation.input_ids")?;
    validate_token_input(
        continuation_ids,
        manifest.continuation_length,
        "continuation.input_ids",
    )?;

    let prefix_tokens = &manifest.prompt_token_ids[..manifest.prefix_length];
    let continuation_tokens = &manifest.prompt_token_ids[manifest.prefix_length..];
    let validation_gpu = Gpu::new();
    let prefix_token_difference =
        compare_fixture_token_ids(&validation_gpu, prefix_ids, prefix_tokens)?;
    let continuation_token_difference =
        compare_fixture_token_ids(&validation_gpu, continuation_ids, continuation_tokens)?;
    let fixture_token_ids_max_absolute_difference =
        prefix_token_difference.max(continuation_token_difference);
    if fixture_token_ids_max_absolute_difference != 0.0 {
        return Err(EngineError::Unsupported(format!(
            "fixture token IDs differ from the manifest by {fixture_token_ids_max_absolute_difference}"
        )));
    }
    drop(validation_gpu);

    let prefix_sequence_length = prefix_tokens.len();
    let continuation_sequence_length = manifest.prompt_token_ids.len();
    let batch_size = prefix_ids.shape()[0];
    let initial_gpu = Gpu::new();
    let initial_weights = ShardedWeights::load(model_directory)?;
    let initial_weight_shard_count = initial_weights.shard_count();
    let initial_weight_tensor_count = initial_weights.tensor_count();

    let empty_state = MlxInferenceState::empty(&initial_gpu, batch_size, &plan)?;
    let prefix = execute_full_model(
        &initial_gpu,
        prefix_ids,
        &empty_state,
        &initial_weights,
        &plan,
    )?;
    evaluate_execution(&initial_gpu, &prefix)?;
    prefix.state.validate(&plan, batch_size)?;
    let mut prefix_oracle_differences = BTreeMap::new();
    let prefix_vs_oracle = compare_phase(
        &initial_gpu,
        &fixture,
        "prefix",
        &prefix,
        &plan,
        &mut prefix_oracle_differences,
    )?;

    let direct_continuation = execute_full_model(
        &initial_gpu,
        continuation_ids,
        &prefix.state,
        &initial_weights,
        &plan,
    )?;
    evaluate_execution(&initial_gpu, &direct_continuation)?;
    direct_continuation.state.validate(&plan, batch_size)?;
    let mut direct_oracle_differences = BTreeMap::new();
    let direct_vs_oracle = compare_phase(
        &initial_gpu,
        &fixture,
        "continuation",
        &direct_continuation,
        &plan,
        &mut direct_oracle_differences,
    )?;

    let instance_id = InstanceId::new("echo-live-state-parity")
        .map_err(|error| state_operation_error("create parity instance", error))?;
    let store = StateStore::<MlxInferenceState>::default();
    let prefix_lease = store
        .begin(instance_id.clone(), ExpectedState::Absent)
        .map_err(|error| state_operation_error("begin prefix state transaction", error))?;
    let FullModelExecution {
        state: prefix_state,
        ..
    } = prefix;
    let prefix_commit = prefix_lease
        .commit(PreparedState {
            model: model.clone(),
            payload: prefix_state,
        })
        .map_err(|error| state_operation_error("commit prefix state", error))?;
    prefix_commit.payload.validate(&plan, batch_size)?;

    // The direct path is already materialized. Drop the original execution
    // owner and reload all weight handles before consuming committed state.
    drop(initial_weights);
    drop(initial_gpu);
    let restored_gpu = Gpu::new();
    let restored_weights = ShardedWeights::load(model_directory)?;
    let restored_weight_shard_count = restored_weights.shard_count();
    let restored_weight_tensor_count = restored_weights.tensor_count();
    let committed_prefix = store
        .current(&instance_id)
        .ok_or_else(|| EngineError::Unsupported("prefix state disappeared after commit".into()))?;
    if !Arc::ptr_eq(&committed_prefix, &prefix_commit) {
        return Err(EngineError::Unsupported(
            "state store did not return the atomically committed prefix".into(),
        ));
    }
    committed_prefix.payload.validate(&plan, batch_size)?;

    let restored_continuation = execute_full_model(
        &restored_gpu,
        continuation_ids,
        &committed_prefix.payload,
        &restored_weights,
        &plan,
    )?;
    evaluate_execution(&restored_gpu, &restored_continuation)?;
    restored_continuation.state.validate(&plan, batch_size)?;
    let mut restored_oracle_differences = BTreeMap::new();
    let restored_vs_oracle = compare_phase(
        &restored_gpu,
        &fixture,
        "continuation",
        &restored_continuation,
        &plan,
        &mut restored_oracle_differences,
    )?;
    let restored_vs_direct = compare_executions(
        &restored_gpu,
        &restored_continuation,
        &direct_continuation,
        &plan,
    )?;

    let continuation_lease = store
        .begin(instance_id.clone(), ExpectedState::Present)
        .map_err(|error| state_operation_error("begin continuation state transaction", error))?;
    let FullModelExecution {
        state: restored_state,
        ..
    } = restored_continuation;
    let continuation_commit = continuation_lease
        .commit(PreparedState {
            model: model.clone(),
            payload: restored_state,
        })
        .map_err(|error| state_operation_error("commit continuation state", error))?;
    continuation_commit.payload.validate(&plan, batch_size)?;

    let current = store.current(&instance_id).ok_or_else(|| {
        EngineError::Unsupported("continuation state disappeared after commit".into())
    })?;
    let store_current_is_continuation_commit = Arc::ptr_eq(&current, &continuation_commit);
    let prefix_state_logical_nbytes = prefix_commit.payload.logical_nbytes()?;
    let continuation_state_logical_nbytes = continuation_commit.payload.logical_nbytes()?;
    let state_layer_count = continuation_commit.payload.layer_count();
    let state_tensor_count = continuation_commit.payload.tensor_count();

    let exact = fixture_token_ids_max_absolute_difference == 0.0
        && prefix_vs_oracle.exact()
        && direct_vs_oracle.exact()
        && restored_vs_oracle.exact()
        && restored_vs_direct.exact()
        && prefix_commit.instance_id == instance_id
        && continuation_commit.instance_id == instance_id
        && prefix_commit.payload.sequence_length()? == prefix_sequence_length
        && continuation_commit.payload.sequence_length()? == continuation_sequence_length
        && prefix_commit.model == model
        && continuation_commit.model == model
        && initial_weight_shard_count == restored_weight_shard_count
        && initial_weight_tensor_count == restored_weight_tensor_count
        && state_layer_count == plan.layer_count
        && state_tensor_count == plan.layer_count * 2
        && store_current_is_continuation_commit;

    Ok(LiveStateParity {
        instance_id,
        model,
        fixture_sha256: manifest.fixture_sha256,
        prefix_sequence_length,
        continuation_sequence_length,
        initial_weight_shard_count,
        restored_weight_shard_count,
        initial_weight_tensor_count,
        restored_weight_tensor_count,
        state_layer_count,
        state_tensor_count,
        prefix_state_logical_nbytes,
        continuation_state_logical_nbytes,
        fixture_token_ids_max_absolute_difference,
        prefix_vs_oracle_max_absolute_difference: prefix_vs_oracle.maximum(),
        prefix_logits_vs_oracle_max_absolute_difference: prefix_vs_oracle.logits,
        prefix_state_vs_oracle_max_absolute_difference: prefix_vs_oracle.state,
        direct_continuation_vs_oracle_max_absolute_difference: direct_vs_oracle.maximum(),
        direct_continuation_logits_vs_oracle_max_absolute_difference: direct_vs_oracle.logits,
        direct_continuation_state_vs_oracle_max_absolute_difference: direct_vs_oracle.state,
        restored_continuation_vs_oracle_max_absolute_difference: restored_vs_oracle.maximum(),
        restored_continuation_logits_vs_oracle_max_absolute_difference: restored_vs_oracle.logits,
        restored_continuation_state_vs_oracle_max_absolute_difference: restored_vs_oracle.state,
        restored_continuation_vs_direct_max_absolute_difference: restored_vs_direct.maximum(),
        restored_continuation_logits_vs_direct_max_absolute_difference: restored_vs_direct.logits,
        restored_continuation_state_vs_direct_max_absolute_difference: restored_vs_direct.state,
        store_current_is_continuation_commit,
        exact,
    })
}

/// Runs producer and restorer phases in separate child processes.
///
/// The producer atomically publishes one current.safetensors and exits. The
/// restorer then acquires the same owner lock, authenticates that fixed file,
/// advances the state, atomically replaces it, and exits.
///
/// # Errors
///
/// Returns [`EngineError`] when either child fails, its JSON evidence is
/// malformed, the current file is missing, or a durability/parity invariant
/// differs.
pub fn run_durable_state_parity(
    executable: &Path,
    model_directory: &Path,
    fixture_path: &Path,
    manifest_path: &Path,
    snapshot_root: &Path,
) -> Result<DurableStateParity, EngineError> {
    let producer = run_json_child::<DurableStateProducer>(
        executable,
        "produce-durable-state-parity",
        [model_directory, fixture_path, manifest_path, snapshot_root],
    )?;
    let current_path = snapshot_root.join(CURRENT_STATE_FILE);
    let current_file_present_after_producer_exit =
        current_path
            .try_exists()
            .map_err(|source| EngineError::Io {
                path: current_path.clone(),
                source,
            })?;
    let restorer = run_json_child::<DurableStateRestorer>(
        executable,
        "restore-durable-state-parity",
        [model_directory, fixture_path, manifest_path, snapshot_root],
    )?;
    let current_file_present_after_restorer_exit =
        current_path
            .try_exists()
            .map_err(|source| EngineError::Io {
                path: current_path.clone(),
                source,
            })?;

    let orchestrator_process_id = std::process::id();
    let distinct_producer_and_restorer_processes = producer.process_id != restorer.process_id
        && producer.process_id != orchestrator_process_id
        && restorer.process_id != orchestrator_process_id;
    let continuation_exact_via_common_oracle =
        producer.direct_continuation_vs_oracle_max_absolute_difference == 0.0
            && restorer.restored_continuation_vs_oracle_max_absolute_difference == 0.0;
    let restored_state_was_load_bearing = restorer.restored_store_is_load_bearing;
    let exact = producer.exact
        && restorer.exact
        && distinct_producer_and_restorer_processes
        && current_file_present_after_producer_exit
        && current_file_present_after_restorer_exit
        && restored_state_was_load_bearing
        && continuation_exact_via_common_oracle
        && producer.instance_id == restorer.instance_id
        && producer.model == restorer.model
        && producer.prefix_sequence_length == restorer.restored_sequence_length
        && producer.current_path == current_path
        && restorer.current_path == current_path
        && restorer.continuation_sequence_length > restorer.restored_sequence_length;

    Ok(DurableStateParity {
        orchestrator_process_id,
        producer,
        restorer,
        assertions: DurableStateAssertions {
            distinct_producer_and_restorer_processes,
            current_file_present_after_producer_exit,
            current_file_present_after_restorer_exit,
            restored_state_was_load_bearing,
            continuation_exact_via_common_oracle,
        },
        exact,
    })
}

/// Produces direct/oracle evidence and atomically publishes one current state.
///
/// # Errors
///
/// Returns [`EngineError`] on model, fixture, numerical, state, serialization,
/// synchronization, or atomic-publication drift.
#[allow(clippy::too_many_lines)]
pub fn produce_durable_state_parity(
    model_directory: &Path,
    fixture_path: &Path,
    manifest_path: &Path,
    snapshot_root: &Path,
) -> Result<DurableStateProducer, EngineError> {
    let plan = ModelPlan::from_directory(model_directory)?;
    if plan.tie_word_embeddings {
        return Err(EngineError::Unsupported(
            "the durable-state path requires an independent language-model head".into(),
        ));
    }
    let manifest = load_manifest(manifest_path)?;
    validate_manifest(&plan, model_directory, fixture_path, &manifest)?;
    let model = identify_model(model_directory)?;
    let fixture = SafeTensors::load(fixture_path).map_err(EngineError::Mlx)?;
    if fixture.len() != manifest.fixture_tensor_count {
        return Err(EngineError::Unsupported(format!(
            "full-model fixture tensor count drift: manifest {}, payload {}",
            manifest.fixture_tensor_count,
            fixture.len()
        )));
    }

    let prefix_ids = require_tensor(&fixture, "prefix.input_ids")?;
    validate_token_input(prefix_ids, manifest.prefix_length, "prefix.input_ids")?;
    let continuation_ids = require_tensor(&fixture, "continuation.input_ids")?;
    validate_token_input(
        continuation_ids,
        manifest.continuation_length,
        "continuation.input_ids",
    )?;
    let prefix_tokens = &manifest.prompt_token_ids[..manifest.prefix_length];
    let continuation_tokens = &manifest.prompt_token_ids[manifest.prefix_length..];

    let gpu = Gpu::new();
    let fixture_token_ids_max_absolute_difference =
        compare_fixture_token_ids(&gpu, prefix_ids, prefix_tokens)?.max(compare_fixture_token_ids(
            &gpu,
            continuation_ids,
            continuation_tokens,
        )?);
    if fixture_token_ids_max_absolute_difference != 0.0 {
        return Err(EngineError::Unsupported(format!(
            "fixture token IDs differ from the manifest by {fixture_token_ids_max_absolute_difference}"
        )));
    }

    let batch_size = prefix_ids.shape()[0];
    let weights = ShardedWeights::load(model_directory)?;
    let weight_shard_count = weights.shard_count();
    let weight_tensor_count = weights.tensor_count();
    let empty_state = MlxInferenceState::empty(&gpu, batch_size, &plan)?;
    let prefix = execute_full_model(&gpu, prefix_ids, &empty_state, &weights, &plan)?;
    evaluate_execution(&gpu, &prefix)?;
    prefix.state.validate(&plan, batch_size)?;
    let mut prefix_differences = BTreeMap::new();
    let prefix_vs_oracle = compare_phase(
        &gpu,
        &fixture,
        "prefix",
        &prefix,
        &plan,
        &mut prefix_differences,
    )?;

    let direct_continuation =
        execute_full_model(&gpu, continuation_ids, &prefix.state, &weights, &plan)?;
    evaluate_execution(&gpu, &direct_continuation)?;
    direct_continuation.state.validate(&plan, batch_size)?;
    let mut continuation_differences = BTreeMap::new();
    let direct_vs_oracle = compare_phase(
        &gpu,
        &fixture,
        "continuation",
        &direct_continuation,
        &plan,
        &mut continuation_differences,
    )?;
    if !prefix_vs_oracle.exact() || !direct_vs_oracle.exact() {
        return Err(EngineError::Unsupported(
            "durable producer output differs from the admitted oracle".into(),
        ));
    }

    let instance_id = InstanceId::new("echo-durable-state-parity")
        .map_err(|error| state_operation_error("create durable parity instance", error))?;
    let store = StateStore::<MlxInferenceState>::default();
    let lease = store
        .begin(instance_id.clone(), ExpectedState::Absent)
        .map_err(|error| state_operation_error("begin durable prefix transaction", error))?;
    let FullModelExecution {
        state: prefix_state,
        ..
    } = prefix;
    let prefix_commit = lease
        .commit(PreparedState {
            model: model.clone(),
            payload: prefix_state,
        })
        .map_err(|error| state_operation_error("commit durable prefix state", error))?;
    let prefix_sequence_length = prefix_commit.payload.sequence_length()?;
    let owner = CurrentStateOwner::acquire(snapshot_root)?;
    let published = owner.publish(&prefix_commit, &plan, batch_size, &gpu)?;
    let reloaded = owner
        .load_current(&instance_id, &plan, &model, batch_size)?
        .ok_or_else(|| {
            EngineError::Unsupported("published current state was not visible to its owner".into())
        })?;
    let logical_nbytes = prefix_commit.payload.logical_nbytes()?;
    let state_tensor_count = prefix_commit.payload.tensor_count();
    let exact = fixture_token_ids_max_absolute_difference == 0.0
        && prefix_vs_oracle.exact()
        && direct_vs_oracle.exact()
        && prefix_sequence_length == manifest.prefix_length
        && reloaded.state.sequence_length()? == prefix_sequence_length
        && published.instance_id == instance_id
        && published.path == snapshot_root.join(CURRENT_STATE_FILE)
        && state_tensor_count == plan.layer_count * 2;

    Ok(DurableStateProducer {
        process_id: std::process::id(),
        instance_id,
        model,
        fixture_sha256: manifest.fixture_sha256,
        prefix_sequence_length,
        current_path: published.path,
        physical_nbytes: published.physical_nbytes,
        logical_nbytes,
        state_tensor_count,
        weight_shard_count,
        weight_tensor_count,
        fixture_token_ids_max_absolute_difference,
        prefix_vs_oracle_max_absolute_difference: prefix_vs_oracle.maximum(),
        prefix_logits_vs_oracle_max_absolute_difference: prefix_vs_oracle.logits,
        prefix_state_vs_oracle_max_absolute_difference: prefix_vs_oracle.state,
        direct_continuation_vs_oracle_max_absolute_difference: direct_vs_oracle.maximum(),
        direct_continuation_logits_vs_oracle_max_absolute_difference: direct_vs_oracle.logits,
        direct_continuation_state_vs_oracle_max_absolute_difference: direct_vs_oracle.state,
        exact,
    })
}

/// Restores current.safetensors in a fresh process and replaces it after one
/// exact continuation.
///
/// # Errors
///
/// Returns [`EngineError`] on durable authentication, model/fixture identity,
/// process-local restore, numerical continuation, or publication drift.
#[allow(clippy::too_many_lines)]
pub fn restore_durable_state_parity(
    model_directory: &Path,
    fixture_path: &Path,
    manifest_path: &Path,
    snapshot_root: &Path,
) -> Result<DurableStateRestorer, EngineError> {
    let plan = ModelPlan::from_directory(model_directory)?;
    if plan.tie_word_embeddings {
        return Err(EngineError::Unsupported(
            "the durable-state path requires an independent language-model head".into(),
        ));
    }
    let manifest = load_manifest(manifest_path)?;
    validate_manifest(&plan, model_directory, fixture_path, &manifest)?;
    let model = identify_model(model_directory)?;
    let fixture = SafeTensors::load(fixture_path).map_err(EngineError::Mlx)?;
    if fixture.len() != manifest.fixture_tensor_count {
        return Err(EngineError::Unsupported(format!(
            "full-model fixture tensor count drift: manifest {}, payload {}",
            manifest.fixture_tensor_count,
            fixture.len()
        )));
    }
    let prefix_ids = require_tensor(&fixture, "prefix.input_ids")?;
    validate_token_input(prefix_ids, manifest.prefix_length, "prefix.input_ids")?;
    let continuation_ids = require_tensor(&fixture, "continuation.input_ids")?;
    validate_token_input(
        continuation_ids,
        manifest.continuation_length,
        "continuation.input_ids",
    )?;
    let prefix_tokens = &manifest.prompt_token_ids[..manifest.prefix_length];
    let continuation_tokens = &manifest.prompt_token_ids[manifest.prefix_length..];
    let validation_gpu = Gpu::new();
    let fixture_token_difference =
        compare_fixture_token_ids(&validation_gpu, prefix_ids, prefix_tokens)?.max(
            compare_fixture_token_ids(&validation_gpu, continuation_ids, continuation_tokens)?,
        );
    if fixture_token_difference != 0.0 {
        return Err(EngineError::Unsupported(
            "restorer fixture token IDs differ from the manifest".into(),
        ));
    }
    drop(validation_gpu);

    let instance_id = InstanceId::new("echo-durable-state-parity")
        .map_err(|error| state_operation_error("create durable parity instance", error))?;
    let batch_size = prefix_ids.shape()[0];
    let owner = CurrentStateOwner::acquire(snapshot_root)?;
    let restored = owner
        .load_current(&instance_id, &plan, &model, batch_size)?
        .ok_or_else(|| EngineError::Unsupported("durable current state is missing".into()))?;
    let restored_sequence_length = restored.state.sequence_length()?;
    if restored_sequence_length != manifest.prefix_length {
        return Err(EngineError::Unsupported(format!(
            "durable state sequence length mismatch: expected {}, observed {restored_sequence_length}",
            manifest.prefix_length
        )));
    }

    let store = StateStore::<MlxInferenceState>::default();
    let restored_commit = store
        .restore(CommittedState {
            instance_id: restored.instance_id,
            model: restored.model,
            payload: restored.state,
        })
        .map_err(|error| state_operation_error("restore durable state into ownership", error))?;
    let current = store.current(&instance_id).ok_or_else(|| {
        EngineError::Unsupported("restored durable state did not become current".into())
    })?;
    let restored_store_is_load_bearing = Arc::ptr_eq(&current, &restored_commit);

    let gpu = Gpu::new();
    let weights = ShardedWeights::load(model_directory)?;
    let weight_shard_count = weights.shard_count();
    let weight_tensor_count = weights.tensor_count();
    let continuation =
        execute_full_model(&gpu, continuation_ids, &current.payload, &weights, &plan)?;
    evaluate_execution(&gpu, &continuation)?;
    continuation.state.validate(&plan, batch_size)?;
    let mut differences = BTreeMap::new();
    let restored_vs_oracle = compare_phase(
        &gpu,
        &fixture,
        "continuation",
        &continuation,
        &plan,
        &mut differences,
    )?;
    if !restored_vs_oracle.exact() {
        return Err(EngineError::Unsupported(
            "durable restorer output differs from the admitted oracle".into(),
        ));
    }

    let lease = store
        .begin(instance_id.clone(), ExpectedState::Present)
        .map_err(|error| state_operation_error("begin restored continuation transaction", error))?;
    let FullModelExecution {
        state: continuation_state,
        ..
    } = continuation;
    let continuation_commit = lease
        .commit(PreparedState {
            model: model.clone(),
            payload: continuation_state,
        })
        .map_err(|error| state_operation_error("commit restored continuation state", error))?;
    let continuation_sequence_length = continuation_commit.payload.sequence_length()?;
    let published = owner.publish(&continuation_commit, &plan, batch_size, &gpu)?;
    let reloaded = owner
        .load_current(&instance_id, &plan, &model, batch_size)?
        .ok_or_else(|| {
            EngineError::Unsupported(
                "replacement current state was not visible to its owner".into(),
            )
        })?;
    let continuation_logical_nbytes = continuation_commit.payload.logical_nbytes()?;
    let state_tensor_count = continuation_commit.payload.tensor_count();
    let current_is_continuation = store
        .current(&instance_id)
        .is_some_and(|state| Arc::ptr_eq(&state, &continuation_commit));
    let exact = restored_store_is_load_bearing
        && restored_vs_oracle.exact()
        && continuation_sequence_length == manifest.prompt_token_ids.len()
        && reloaded.state.sequence_length()? == continuation_sequence_length
        && continuation_commit.model == model
        && published.instance_id == instance_id
        && published.path == snapshot_root.join(CURRENT_STATE_FILE)
        && state_tensor_count == plan.layer_count * 2
        && current_is_continuation;

    Ok(DurableStateRestorer {
        process_id: std::process::id(),
        instance_id,
        model,
        restored_sequence_length,
        continuation_sequence_length,
        current_path: published.path,
        continuation_physical_nbytes: published.physical_nbytes,
        continuation_logical_nbytes,
        state_tensor_count,
        weight_shard_count,
        weight_tensor_count,
        restored_store_is_load_bearing,
        restored_continuation_vs_oracle_max_absolute_difference: restored_vs_oracle.maximum(),
        restored_continuation_logits_vs_oracle_max_absolute_difference: restored_vs_oracle.logits,
        restored_continuation_state_vs_oracle_max_absolute_difference: restored_vs_oracle.state,
        exact,
    })
}

fn run_json_child<T: DeserializeOwned>(
    executable: &Path,
    command: &str,
    arguments: [&Path; 4],
) -> Result<T, EngineError> {
    let output = Command::new(executable)
        .arg(command)
        .args(arguments)
        .output()
        .map_err(|source| EngineError::Io {
            path: executable.to_path_buf(),
            source,
        })?;
    if !output.status.success() {
        return Err(EngineError::Unsupported(format!(
            "{command} failed with status {:?}; stdout: {}; stderr: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout).trim(),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    serde_json::from_slice(&output.stdout).map_err(|error| {
        EngineError::Unsupported(format!(
            "{command} returned malformed JSON evidence: {error}"
        ))
    })
}

#[allow(clippy::too_many_lines)]
fn execute_full_model(
    gpu: &Gpu,
    input_ids: &Array,
    initial_state: &MlxInferenceState,
    weights: &dyn TensorLookup,
    plan: &ModelPlan,
) -> Result<FullModelExecution, EngineError> {
    let gdn_kernel = GdnKernel::new(gpu, plan)?;
    let moe_kernel = MoeKernel::new(plan)?;
    execute_full_model_with_trace(
        gpu,
        input_ids,
        initial_state,
        weights,
        plan,
        &gdn_kernel,
        &moe_kernel,
        true,
    )
}

#[allow(clippy::too_many_lines)]
pub(crate) fn execute_runtime_model(
    gpu: &Gpu,
    input_ids: &Array,
    initial_state: RuntimeInferenceState,
    weights: &BoundModelWeights,
    plan: &ModelPlan,
    gdn_kernel: &GdnKernel,
    moe_kernel: &MoeKernel,
) -> Result<RuntimeModelExecution, EngineError> {
    let shape = input_ids.shape();
    let [batch_size, sequence_length] = <[usize; 2]>::try_from(shape.clone()).map_err(|shape| {
        EngineError::Unsupported(format!("token input must be rank 2, observed {shape:?}"))
    })?;
    if initial_state.layers.len() != plan.layer_count || weights.layers.len() != plan.layer_count {
        return Err(EngineError::Unsupported(format!(
            "runtime layer count mismatch: expected {}, observed {} states and {} weight sets",
            plan.layer_count,
            initial_state.layers.len(),
            weights.layers.len()
        )));
    }

    let RuntimeInferenceState {
        layers: initial_layers,
        left_padding,
    } = initial_state;
    let (rope_offsets, attention_mask) = if let Some(padding) = left_padding.as_deref() {
        let offset = runtime_attention_offset(&initial_layers, plan)?;
        let metadata =
            prepare_left_padded_attention_metadata(batch_size, sequence_length, offset, padding)?;
        (Some(metadata.0), Some(metadata.1))
    } else {
        (None, None)
    };

    let embedding = execute_bound_quantized_embedding(gpu, input_ids, &weights.embedding, plan)?;
    let mut hidden = embedding.try_clone().map_err(EngineError::Mlx)?;
    let mut states = Vec::with_capacity(plan.layer_count);
    for (layer_index, (initial_layer_state, layer_weights)) in
        initial_layers.into_iter().zip(&weights.layers).enumerate()
    {
        let full_attention = (layer_index + 1).is_multiple_of(plan.full_attention_interval);
        match (full_attention, initial_layer_state) {
            (
                true,
                RuntimeLayerState::Attention {
                    key_buffer,
                    value_buffer,
                    offset,
                },
            ) => {
                let execution = execute_buffered_attention_decoder_layer_with_bound_weights(
                    gpu,
                    &hidden,
                    &key_buffer,
                    &value_buffer,
                    layer_weights,
                    plan,
                    offset,
                    sequence_length > 1,
                    rope_offsets.as_ref(),
                    attention_mask.as_ref(),
                    moe_kernel,
                )?;
                hidden = execution.output;
                states.push(RuntimeLayerState::Attention {
                    key_buffer: execution.first_state,
                    value_buffer: execution.second_state,
                    offset: offset.checked_add(sequence_length).ok_or_else(|| {
                        EngineError::Unsupported("runtime attention offset overflow".into())
                    })?,
                });
            }
            (
                false,
                RuntimeLayerState::Gdn {
                    convolution,
                    recurrent,
                },
            ) => {
                let execution = execute_gdn_decoder_layer_with_bound_weights(
                    gpu,
                    &hidden,
                    &convolution,
                    &recurrent,
                    layer_weights,
                    plan,
                    gdn_kernel,
                    moe_kernel,
                )?;
                hidden = execution.output;
                states.push(RuntimeLayerState::Gdn {
                    convolution: execution.first_state,
                    recurrent: execution.second_state,
                });
            }
            (true, RuntimeLayerState::Gdn { .. }) => {
                return Err(EngineError::Unsupported(format!(
                    "layer {layer_index} requires attention state, observed GDN state"
                )));
            }
            (false, RuntimeLayerState::Attention { .. }) => {
                return Err(EngineError::Unsupported(format!(
                    "layer {layer_index} requires GDN state, observed attention state"
                )));
            }
        }
    }

    let normalized_hidden = gpu
        .rms_norm(&hidden, Some(&weights.final_norm), plan.rms_norm_epsilon)
        .map_err(EngineError::Mlx)?;
    let logits = apply_bound_quantized_linear(gpu, &normalized_hidden, &weights.lm_head)?;
    validate_array(
        &logits,
        &[batch_size, sequence_length, plan.vocabulary_size],
        DType::BFloat16,
        "full-model logits",
    )?;

    Ok(RuntimeModelExecution {
        logits,
        state: RuntimeInferenceState {
            layers: states,
            left_padding,
        },
    })
}

pub(crate) fn prepare_runtime_state(
    gpu: &Gpu,
    state: &MlxInferenceState,
    batch_size: usize,
    additional_tokens: usize,
    plan: &ModelPlan,
) -> Result<RuntimeInferenceState, EngineError> {
    if state.layer_count() != plan.layer_count {
        return Err(EngineError::Unsupported(format!(
            "runtime preparation state count mismatch: expected {}, observed {}",
            plan.layer_count,
            state.layer_count()
        )));
    }
    let mut layers = Vec::with_capacity(plan.layer_count);
    for (layer_index, layer) in state.layers().iter().enumerate() {
        let full_attention = (layer_index + 1).is_multiple_of(plan.full_attention_interval);
        match (full_attention, layer) {
            (true, LayerState::Attention { keys, values }) => {
                let offset = attention_offset(keys, values, batch_size, plan)?;
                let required = offset.checked_add(additional_tokens).ok_or_else(|| {
                    EngineError::Unsupported("runtime KV capacity overflow".into())
                })?;
                let capacity = round_kv_capacity(required)?;
                let shape = [
                    batch_size,
                    plan.key_value_head_count,
                    capacity,
                    plan.attention_head_dimension,
                ];
                let empty_keys = gpu
                    .zeros(&shape, DType::BFloat16)
                    .map_err(EngineError::Mlx)?;
                let empty_values = gpu
                    .zeros(&shape, DType::BFloat16)
                    .map_err(EngineError::Mlx)?;
                let (key_buffer, value_buffer) = if offset == 0 {
                    (empty_keys, empty_values)
                } else {
                    let stop = [
                        dimension_i32(batch_size, "runtime KV batch size")?,
                        dimension_i32(plan.key_value_head_count, "runtime KV heads")?,
                        dimension_i32(offset, "runtime KV offset")?,
                        dimension_i32(plan.attention_head_dimension, "runtime KV head dimension")?,
                    ];
                    (
                        gpu.slice_update(&empty_keys, keys, &[0, 0, 0, 0], &stop, &[1, 1, 1, 1])
                            .map_err(EngineError::Mlx)?,
                        gpu.slice_update(
                            &empty_values,
                            values,
                            &[0, 0, 0, 0],
                            &stop,
                            &[1, 1, 1, 1],
                        )
                        .map_err(EngineError::Mlx)?,
                    )
                };
                layers.push(RuntimeLayerState::Attention {
                    key_buffer,
                    value_buffer,
                    offset,
                });
            }
            (
                false,
                LayerState::Gdn {
                    convolution,
                    recurrent,
                },
            ) => layers.push(RuntimeLayerState::Gdn {
                convolution: convolution.try_clone().map_err(EngineError::Mlx)?,
                recurrent: recurrent.try_clone().map_err(EngineError::Mlx)?,
            }),
            (true, LayerState::Gdn { .. }) => {
                return Err(EngineError::Unsupported(format!(
                    "layer {layer_index} requires attention state, observed GDN state"
                )));
            }
            (false, LayerState::Attention { .. }) => {
                return Err(EngineError::Unsupported(format!(
                    "layer {layer_index} requires GDN state, observed attention state"
                )));
            }
        }
    }
    Ok(RuntimeInferenceState {
        layers,
        left_padding: None,
    })
}

/// Merges independently owned single-instance caches into one runtime batch.
///
/// Full-attention KV rows are left-padded so their newest retained positions
/// line up at one physical append offset. GDN states are concatenated without
/// padding because they do not retain an absolute token axis.
#[allow(clippy::too_many_lines)]
pub(crate) fn prepare_merged_runtime_state(
    gpu: &Gpu,
    states: &[&MlxInferenceState],
    additional_tokens: usize,
    plan: &ModelPlan,
) -> Result<RuntimeInferenceState, EngineError> {
    if states.is_empty() {
        return Err(EngineError::Unsupported(
            "runtime state merge requires at least one instance".into(),
        ));
    }
    for state in states {
        state.validate(plan, 1)?;
    }
    let lengths = states
        .iter()
        .map(|state| state.sequence_length())
        .collect::<Result<Vec<_>, _>>()?;
    let maximum_length = lengths.iter().copied().max().ok_or_else(|| {
        EngineError::Unsupported("runtime state merge observed no sequence lengths".into())
    })?;
    let left_padding = lengths
        .iter()
        .map(|length| maximum_length - length)
        .collect::<Vec<_>>();
    let required = maximum_length
        .checked_add(additional_tokens)
        .ok_or_else(|| EngineError::Unsupported("merged runtime KV capacity overflow".into()))?;
    let capacity = round_kv_capacity(required)?;
    let mut layers = Vec::with_capacity(plan.layer_count);

    for layer_index in 0..plan.layer_count {
        let full_attention = (layer_index + 1).is_multiple_of(plan.full_attention_interval);
        if full_attention {
            let mut key_rows = Vec::with_capacity(states.len());
            let mut value_rows = Vec::with_capacity(states.len());
            for ((state, length), padding) in states.iter().zip(&lengths).zip(&left_padding) {
                let LayerState::Attention { keys, values } = &state.layers()[layer_index] else {
                    return Err(EngineError::Unsupported(format!(
                        "layer {layer_index} requires attention state while merging"
                    )));
                };
                let shape = [
                    1,
                    plan.key_value_head_count,
                    capacity,
                    plan.attention_head_dimension,
                ];
                let empty_keys = gpu
                    .zeros(&shape, DType::BFloat16)
                    .map_err(EngineError::Mlx)?;
                let empty_values = gpu
                    .zeros(&shape, DType::BFloat16)
                    .map_err(EngineError::Mlx)?;
                if *length == 0 {
                    key_rows.push(empty_keys);
                    value_rows.push(empty_values);
                    continue;
                }
                let start = [
                    0,
                    0,
                    dimension_i32(*padding, "merged runtime KV left padding")?,
                    0,
                ];
                let stop = [
                    1,
                    dimension_i32(plan.key_value_head_count, "merged runtime KV heads")?,
                    dimension_i32(
                        padding.checked_add(*length).ok_or_else(|| {
                            EngineError::Unsupported(
                                "merged runtime padded KV length overflow".into(),
                            )
                        })?,
                        "merged runtime KV row stop",
                    )?,
                    dimension_i32(
                        plan.attention_head_dimension,
                        "merged runtime KV head dimension",
                    )?,
                ];
                key_rows.push(
                    gpu.slice_update(&empty_keys, keys, &start, &stop, &[1, 1, 1, 1])
                        .map_err(EngineError::Mlx)?,
                );
                value_rows.push(
                    gpu.slice_update(&empty_values, values, &start, &stop, &[1, 1, 1, 1])
                        .map_err(EngineError::Mlx)?,
                );
            }
            let key_references = key_rows.iter().collect::<Vec<_>>();
            let value_references = value_rows.iter().collect::<Vec<_>>();
            layers.push(RuntimeLayerState::Attention {
                key_buffer: gpu
                    .concatenate(&key_references, 0)
                    .map_err(EngineError::Mlx)?,
                value_buffer: gpu
                    .concatenate(&value_references, 0)
                    .map_err(EngineError::Mlx)?,
                offset: maximum_length,
            });
        } else {
            let mut convolutions = Vec::with_capacity(states.len());
            let mut recurrents = Vec::with_capacity(states.len());
            for state in states {
                let LayerState::Gdn {
                    convolution,
                    recurrent,
                } = &state.layers()[layer_index]
                else {
                    return Err(EngineError::Unsupported(format!(
                        "layer {layer_index} requires GDN state while merging"
                    )));
                };
                convolutions.push(convolution);
                recurrents.push(recurrent);
            }
            layers.push(RuntimeLayerState::Gdn {
                convolution: gpu
                    .concatenate(&convolutions, 0)
                    .map_err(EngineError::Mlx)?,
                recurrent: gpu.concatenate(&recurrents, 0).map_err(EngineError::Mlx)?,
            });
        }
    }

    Ok(RuntimeInferenceState {
        layers,
        left_padding: left_padding
            .iter()
            .any(|padding| *padding != 0)
            .then_some(left_padding),
    })
}

/// Splits a runtime batch back into independently owned compact model states.
#[allow(clippy::too_many_lines)]
pub(crate) fn split_runtime_state(
    gpu: &Gpu,
    state: RuntimeInferenceState,
    plan: &ModelPlan,
) -> Result<Vec<MlxInferenceState>, EngineError> {
    if state.layers.len() != plan.layer_count {
        return Err(EngineError::Unsupported(format!(
            "runtime split state count mismatch: expected {}, observed {}",
            plan.layer_count,
            state.layers.len()
        )));
    }
    let batch_size = runtime_batch_size(&state.layers)?;
    let offset = runtime_attention_offset(&state.layers, plan)?;
    let left_padding = state.left_padding.unwrap_or_else(|| vec![0; batch_size]);
    if left_padding.len() != batch_size || left_padding.iter().any(|padding| *padding > offset) {
        return Err(EngineError::Unsupported(format!(
            "runtime split padding {left_padding:?} is invalid for batch {batch_size} and offset {offset}"
        )));
    }
    let mut rows = (0..batch_size)
        .map(|_| Vec::with_capacity(plan.layer_count))
        .collect::<Vec<_>>();

    for (layer_index, layer) in state.layers.into_iter().enumerate() {
        let full_attention = (layer_index + 1).is_multiple_of(plan.full_attention_interval);
        match (full_attention, layer) {
            (
                false,
                RuntimeLayerState::Gdn {
                    convolution,
                    recurrent,
                },
            ) => {
                for (row, destination) in rows.iter_mut().enumerate() {
                    destination.push(LayerState::Gdn {
                        convolution: slice_runtime_batch_row(gpu, &convolution, row, batch_size)?,
                        recurrent: slice_runtime_batch_row(gpu, &recurrent, row, batch_size)?,
                    });
                }
            }
            (
                true,
                RuntimeLayerState::Attention {
                    key_buffer,
                    value_buffer,
                    offset: layer_offset,
                },
            ) => {
                let shape = key_buffer.shape();
                let [observed_batch, heads, capacity, head_dimension] =
                    <[usize; 4]>::try_from(shape.clone()).map_err(|shape| {
                        EngineError::Unsupported(format!(
                            "layer {layer_index} runtime key buffer must be rank 4: {shape:?}"
                        ))
                    })?;
                if observed_batch != batch_size
                    || heads != plan.key_value_head_count
                    || capacity < offset
                    || head_dimension != plan.attention_head_dimension
                    || layer_offset != offset
                    || value_buffer.shape() != shape
                {
                    return Err(EngineError::Unsupported(format!(
                        "layer {layer_index} runtime KV layout drifted before split"
                    )));
                }
                for (row, destination) in rows.iter_mut().enumerate() {
                    let start = [
                        dimension_i32(row, "runtime split batch row")?,
                        0,
                        dimension_i32(left_padding[row], "runtime split left padding")?,
                        0,
                    ];
                    let stop = [
                        dimension_i32(row + 1, "runtime split batch row stop")?,
                        dimension_i32(heads, "runtime split KV heads")?,
                        dimension_i32(offset, "runtime split KV offset")?,
                        dimension_i32(head_dimension, "runtime split KV head dimension")?,
                    ];
                    destination.push(LayerState::Attention {
                        keys: gpu
                            .slice(&key_buffer, &start, &stop, &[1, 1, 1, 1])
                            .and_then(|array| gpu.contiguous(&array))
                            .map_err(EngineError::Mlx)?,
                        values: gpu
                            .slice(&value_buffer, &start, &stop, &[1, 1, 1, 1])
                            .and_then(|array| gpu.contiguous(&array))
                            .map_err(EngineError::Mlx)?,
                    });
                }
            }
            (true, RuntimeLayerState::Gdn { .. }) => {
                return Err(EngineError::Unsupported(format!(
                    "layer {layer_index} requires attention state while splitting"
                )));
            }
            (false, RuntimeLayerState::Attention { .. }) => {
                return Err(EngineError::Unsupported(format!(
                    "layer {layer_index} requires GDN state while splitting"
                )));
            }
        }
    }

    rows.into_iter()
        .map(|layers| {
            let state = MlxInferenceState::new(layers);
            state.validate(plan, 1)?;
            Ok(state)
        })
        .collect()
}

pub(crate) fn compact_runtime_state(
    gpu: &Gpu,
    state: RuntimeInferenceState,
    plan: &ModelPlan,
) -> Result<MlxInferenceState, EngineError> {
    if state.layers.len() != plan.layer_count {
        return Err(EngineError::Unsupported(format!(
            "runtime compaction state count mismatch: expected {}, observed {}",
            plan.layer_count,
            state.layers.len()
        )));
    }
    if state.left_padding.is_some() {
        return Err(EngineError::Unsupported(
            "left-padded runtime state must be split into instance-owned states before compaction"
                .into(),
        ));
    }
    let mut layers = Vec::with_capacity(plan.layer_count);
    for (layer_index, layer) in state.layers.into_iter().enumerate() {
        match layer {
            RuntimeLayerState::Gdn {
                convolution,
                recurrent,
            } => layers.push(LayerState::Gdn {
                convolution,
                recurrent,
            }),
            RuntimeLayerState::Attention {
                key_buffer,
                value_buffer,
                offset,
            } => {
                let shape = key_buffer.shape();
                let [batch_size, key_value_heads, capacity, head_dimension] =
                    <[usize; 4]>::try_from(shape.clone()).map_err(|shape| {
                        EngineError::Unsupported(format!(
                            "layer {layer_index} runtime key buffer must be rank 4: {shape:?}"
                        ))
                    })?;
                if capacity < offset
                    || key_value_heads != plan.key_value_head_count
                    || head_dimension != plan.attention_head_dimension
                    || value_buffer.shape() != shape
                {
                    return Err(EngineError::Unsupported(format!(
                        "layer {layer_index} runtime KV buffer shape or offset drifted"
                    )));
                }
                let stop = [
                    dimension_i32(batch_size, "compacted KV batch size")?,
                    dimension_i32(key_value_heads, "compacted KV heads")?,
                    dimension_i32(offset, "compacted KV offset")?,
                    dimension_i32(head_dimension, "compacted KV head dimension")?,
                ];
                layers.push(LayerState::Attention {
                    keys: gpu
                        .slice(&key_buffer, &[0, 0, 0, 0], &stop, &[1, 1, 1, 1])
                        .map_err(EngineError::Mlx)?,
                    values: gpu
                        .slice(&value_buffer, &[0, 0, 0, 0], &stop, &[1, 1, 1, 1])
                        .map_err(EngineError::Mlx)?,
                });
            }
        }
    }
    Ok(MlxInferenceState::new(layers))
}

fn round_kv_capacity(required: usize) -> Result<usize, EngineError> {
    if required == 0 {
        return Ok(0);
    }
    required
        .checked_add(KV_CACHE_ALLOCATION_STEP - 1)
        .map(|value| value / KV_CACHE_ALLOCATION_STEP * KV_CACHE_ALLOCATION_STEP)
        .ok_or_else(|| EngineError::Unsupported("runtime KV capacity overflow".into()))
}

fn runtime_batch_size(layers: &[RuntimeLayerState]) -> Result<usize, EngineError> {
    let shape = match layers.first() {
        Some(RuntimeLayerState::Gdn { convolution, .. }) => convolution.shape(),
        Some(RuntimeLayerState::Attention { key_buffer, .. }) => key_buffer.shape(),
        None => {
            return Err(EngineError::Unsupported(
                "runtime state contains no layers".into(),
            ));
        }
    };
    shape
        .first()
        .copied()
        .filter(|batch| *batch != 0)
        .ok_or_else(|| {
            EngineError::Unsupported(format!(
                "runtime state has no positive batch dimension: {shape:?}"
            ))
        })
}

fn runtime_attention_offset(
    layers: &[RuntimeLayerState],
    plan: &ModelPlan,
) -> Result<usize, EngineError> {
    let mut observed = None;
    for (layer_index, layer) in layers.iter().enumerate() {
        let full_attention = (layer_index + 1).is_multiple_of(plan.full_attention_interval);
        match (full_attention, layer) {
            (
                true,
                RuntimeLayerState::Attention {
                    key_buffer,
                    value_buffer,
                    offset,
                },
            ) => {
                let shape = key_buffer.shape();
                let capacity = shape.get(2).copied().ok_or_else(|| {
                    EngineError::Unsupported(format!(
                        "layer {layer_index} runtime key buffer must be rank 4: {shape:?}"
                    ))
                })?;
                if shape.len() != 4 || value_buffer.shape() != shape || *offset > capacity {
                    return Err(EngineError::Unsupported(format!(
                        "layer {layer_index} runtime KV buffer layout is invalid"
                    )));
                }
                if observed.is_some_and(|prior| prior != *offset) {
                    return Err(EngineError::Unsupported(format!(
                        "runtime attention offsets disagree: {observed:?} and {offset}"
                    )));
                }
                observed = Some(*offset);
            }
            (false, RuntimeLayerState::Gdn { .. }) => {}
            (true, RuntimeLayerState::Gdn { .. }) => {
                return Err(EngineError::Unsupported(format!(
                    "layer {layer_index} requires attention state"
                )));
            }
            (false, RuntimeLayerState::Attention { .. }) => {
                return Err(EngineError::Unsupported(format!(
                    "layer {layer_index} requires GDN state"
                )));
            }
        }
    }
    observed.ok_or_else(|| {
        EngineError::Unsupported("runtime state contains no full-attention layer".into())
    })
}

fn prepare_left_padded_attention_metadata(
    batch_size: usize,
    sequence_length: usize,
    offset: usize,
    left_padding: &[usize],
) -> Result<(Array, Array), EngineError> {
    if left_padding.len() != batch_size || left_padding.iter().any(|padding| *padding > offset) {
        return Err(EngineError::Unsupported(format!(
            "left-padded attention metadata {left_padding:?} is invalid for batch {batch_size} and offset {offset}"
        )));
    }
    let logical_length = offset
        .checked_add(sequence_length)
        .ok_or_else(|| EngineError::Unsupported("attention mask length overflow".into()))?;
    let rope_offset_values = left_padding
        .iter()
        .map(|padding| dimension_i32(offset - padding, "per-row RoPE offset"))
        .collect::<Result<Vec<_>, _>>()?;
    let rope_offsets =
        Array::from_i32_slice(&rope_offset_values, &[batch_size]).map_err(EngineError::Mlx)?;
    let mask_element_count = batch_size
        .checked_mul(sequence_length)
        .and_then(|count| count.checked_mul(logical_length))
        .ok_or_else(|| EngineError::Unsupported("attention mask size overflow".into()))?;
    let mut mask_values = Vec::with_capacity(mask_element_count);
    for padding in left_padding {
        for query in 0..sequence_length {
            let query_position = offset.checked_add(query).ok_or_else(|| {
                EngineError::Unsupported("attention query position overflow".into())
            })?;
            for key in 0..logical_length {
                mask_values.push(key >= *padding && key <= query_position);
            }
        }
    }
    let mask = Array::from_bool_slice(
        &mask_values,
        &[batch_size, 1, sequence_length, logical_length],
    )
    .map_err(EngineError::Mlx)?;
    Ok((rope_offsets, mask))
}

fn slice_runtime_batch_row(
    gpu: &Gpu,
    array: &Array,
    row: usize,
    batch_size: usize,
) -> Result<Array, EngineError> {
    let shape = array.shape();
    if shape.first().copied() != Some(batch_size) {
        return Err(EngineError::Unsupported(format!(
            "runtime state batch dimension mismatch: expected {batch_size}, observed {shape:?}"
        )));
    }
    let mut start = vec![0_i32; shape.len()];
    let mut stop = shape
        .iter()
        .map(|dimension| dimension_i32(*dimension, "runtime state dimension"))
        .collect::<Result<Vec<_>, _>>()?;
    start[0] = dimension_i32(row, "runtime state batch row")?;
    stop[0] = dimension_i32(row + 1, "runtime state batch row stop")?;
    gpu.slice(array, &start, &stop, &vec![1; shape.len()])
        .and_then(|value| gpu.contiguous(&value))
        .map_err(EngineError::Mlx)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn execute_full_model_with_trace(
    gpu: &Gpu,
    input_ids: &Array,
    initial_state: &MlxInferenceState,
    weights: &dyn TensorLookup,
    plan: &ModelPlan,
    gdn_kernel: &GdnKernel,
    moe_kernel: &MoeKernel,
    capture_layer_outputs: bool,
) -> Result<FullModelExecution, EngineError> {
    let shape = input_ids.shape();
    let [batch_size, sequence_length] = <[usize; 2]>::try_from(shape.clone()).map_err(|shape| {
        EngineError::Unsupported(format!("token input must be rank 2, observed {shape:?}"))
    })?;
    if initial_state.layer_count() != plan.layer_count {
        return Err(EngineError::Unsupported(format!(
            "full-model state count mismatch: expected {}, observed {}",
            plan.layer_count,
            initial_state.layer_count()
        )));
    }

    let embedding = execute_quantized_embedding(gpu, input_ids, weights, plan)?;
    let mut hidden = embedding.try_clone().map_err(EngineError::Mlx)?;
    let mut layer_outputs = if capture_layer_outputs {
        Vec::with_capacity(plan.layer_count)
    } else {
        Vec::new()
    };
    let mut states = Vec::with_capacity(plan.layer_count);
    for (layer_index, initial_layer_state) in initial_state.layers().iter().enumerate() {
        let full_attention = (layer_index + 1).is_multiple_of(plan.full_attention_interval);
        let execution = match (full_attention, initial_layer_state) {
            (true, LayerState::Attention { keys, values }) => {
                let offset = attention_offset(keys, values, batch_size, plan)?;
                execute_attention_decoder_layer(
                    gpu,
                    &hidden,
                    keys,
                    values,
                    weights,
                    layer_index,
                    plan,
                    offset,
                    sequence_length > 1,
                    moe_kernel,
                )?
            }
            (
                false,
                LayerState::Gdn {
                    convolution,
                    recurrent,
                },
            ) => execute_gdn_decoder_layer_with_kernel(
                gpu,
                &hidden,
                convolution,
                recurrent,
                weights,
                layer_index,
                plan,
                gdn_kernel,
                moe_kernel,
            )?,
            (true, LayerState::Gdn { .. }) => {
                return Err(EngineError::Unsupported(format!(
                    "layer {layer_index} requires attention state, observed GDN state"
                )));
            }
            (false, LayerState::Attention { .. }) => {
                return Err(EngineError::Unsupported(format!(
                    "layer {layer_index} requires GDN state, observed attention state"
                )));
            }
        };
        hidden = execution.output;
        if capture_layer_outputs {
            layer_outputs.push(hidden.try_clone().map_err(EngineError::Mlx)?);
        }
        if full_attention {
            states.push(LayerState::Attention {
                keys: execution.first_state,
                values: execution.second_state,
            });
        } else {
            states.push(LayerState::Gdn {
                convolution: execution.first_state,
                recurrent: execution.second_state,
            });
        }
    }

    let norm_weight = require_tensor(weights, "language_model.model.norm.weight")?;
    validate_array(
        norm_weight,
        &[plan.hidden_size],
        DType::BFloat16,
        "final normalization weight",
    )?;
    let normalized_hidden = gpu
        .rms_norm(&hidden, Some(norm_weight), plan.rms_norm_epsilon)
        .map_err(EngineError::Mlx)?;
    let logits = quantized_linear_with_config(
        gpu,
        &normalized_hidden,
        weights,
        "language_model.lm_head",
        plan.quantization_group_size,
        plan.quantization_bits,
        &plan.quantization_mode,
    )?;
    validate_array(
        &logits,
        &[batch_size, sequence_length, plan.vocabulary_size],
        DType::BFloat16,
        "full-model logits",
    )?;

    Ok(FullModelExecution {
        embedding,
        layer_outputs,
        normalized_hidden,
        logits,
        state: MlxInferenceState::new(states),
    })
}

fn evaluate_execution(gpu: &Gpu, execution: &FullModelExecution) -> Result<(), EngineError> {
    let mut outputs =
        Vec::with_capacity(3 + execution.layer_outputs.len() + execution.state.tensor_count());
    outputs.push(&execution.embedding);
    outputs.extend(execution.layer_outputs.iter());
    outputs.push(&execution.normalized_hidden);
    outputs.push(&execution.logits);
    for state in execution.state.layers() {
        outputs.extend(state.arrays());
    }
    gpu.eval(&outputs).map_err(EngineError::Mlx)
}

pub(crate) fn evaluate_runtime_execution(
    gpu: &Gpu,
    execution: &RuntimeModelExecution,
) -> Result<(), EngineError> {
    let outputs = runtime_execution_arrays(execution);
    gpu.eval(&outputs).map_err(EngineError::Mlx)?;
    Ok(())
}

pub(crate) fn schedule_runtime_execution(
    gpu: &Gpu,
    token: &Array,
    execution: &RuntimeModelExecution,
) -> Result<(), EngineError> {
    // Match mlx-lm's one-stage generation pipeline: schedule only the sampled
    // token and the look-ahead logits. State tensors remain in the same lazy
    // dependency graph and are forced either by the following decode step or
    // by the final commit materialization below.
    gpu.async_eval(&[token, &execution.logits])
        .map_err(EngineError::Mlx)
}

pub(crate) fn runtime_execution_arrays(execution: &RuntimeModelExecution) -> Vec<&Array> {
    let mut outputs = Vec::with_capacity(1 + execution.state.layers.len() * 2);
    outputs.push(&execution.logits);
    for state in &execution.state.layers {
        match state {
            RuntimeLayerState::Gdn {
                convolution,
                recurrent,
            } => {
                outputs.push(convolution);
                outputs.push(recurrent);
            }
            RuntimeLayerState::Attention {
                key_buffer,
                value_buffer,
                ..
            } => {
                outputs.push(key_buffer);
                outputs.push(value_buffer);
            }
        }
    }
    outputs
}

fn execute_bound_quantized_embedding(
    gpu: &Gpu,
    input_ids: &Array,
    weights: &BoundQuantizedWeights,
    plan: &ModelPlan,
) -> Result<Array, EngineError> {
    let shape = input_ids.shape();
    let [batch_size, sequence_length] = <[usize; 2]>::try_from(shape.clone()).map_err(|shape| {
        EngineError::Unsupported(format!(
            "embedding token input must be rank 2, observed {shape:?}"
        ))
    })?;
    if !matches!(input_ids.dtype(), DType::Int32 | DType::Uint32) {
        return Err(EngineError::Unsupported(format!(
            "embedding token input must be int32 or uint32, observed {}",
            input_ids.dtype_name()
        )));
    }
    let selected_weight = gpu
        .take_axis(&weights.weight, input_ids, 0)
        .map_err(EngineError::Mlx)?;
    let selected_scales = gpu
        .take_axis(&weights.scales, input_ids, 0)
        .map_err(EngineError::Mlx)?;
    let selected_biases = gpu
        .take_axis(&weights.biases, input_ids, 0)
        .map_err(EngineError::Mlx)?;
    let embedding = gpu
        .dequantize(
            &selected_weight,
            &selected_scales,
            DequantizeConfig {
                biases: Some(&selected_biases),
                group_size: weights.group_size,
                bits: weights.bits,
                mode: &weights.mode,
                dtype: None,
            },
        )
        .map_err(EngineError::Mlx)?;
    validate_array(
        &embedding,
        &[batch_size, sequence_length, plan.hidden_size],
        DType::BFloat16,
        "quantized embedding output",
    )?;
    Ok(embedding)
}

fn execute_quantized_embedding(
    gpu: &Gpu,
    input_ids: &Array,
    weights: &dyn TensorLookup,
    plan: &ModelPlan,
) -> Result<Array, EngineError> {
    let shape = input_ids.shape();
    let [batch_size, sequence_length] = <[usize; 2]>::try_from(shape.clone()).map_err(|shape| {
        EngineError::Unsupported(format!(
            "embedding token input must be rank 2, observed {shape:?}"
        ))
    })?;
    if !matches!(input_ids.dtype(), DType::Int32 | DType::Uint32) {
        return Err(EngineError::Unsupported(format!(
            "embedding token input must be int32 or uint32, observed {}",
            input_ids.dtype_name()
        )));
    }
    let prefix = "language_model.model.embed_tokens";
    let tensors = weights.quantized_tensors(prefix).ok_or_else(|| {
        EngineError::Unsupported(format!("missing quantized tensor set {prefix}"))
    })?;
    let weight = tensors.weight;
    let scales = tensors.scales;
    let biases = tensors.biases;
    let packed_hidden_size = plan
        .hidden_size
        .checked_mul(plan.quantization_bits)
        .filter(|bits| bits.is_multiple_of(32))
        .map(|bits| bits / 32)
        .ok_or_else(|| {
            EngineError::Unsupported("embedding packed hidden size is invalid".into())
        })?;
    validate_array(
        weight,
        &[plan.vocabulary_size, packed_hidden_size],
        DType::Uint32,
        "embedding weight",
    )?;
    if !plan
        .hidden_size
        .is_multiple_of(plan.quantization_group_size)
    {
        return Err(EngineError::Unsupported(
            "embedding hidden size is not divisible by its quantization group".into(),
        ));
    }
    let parameter_shape = [
        plan.vocabulary_size,
        plan.hidden_size / plan.quantization_group_size,
    ];
    validate_array(
        scales,
        &parameter_shape,
        DType::BFloat16,
        "embedding scales",
    )?;
    validate_array(
        biases,
        &parameter_shape,
        DType::BFloat16,
        "embedding biases",
    )?;

    let selected_weight = gpu
        .take_axis(weight, input_ids, 0)
        .map_err(EngineError::Mlx)?;
    let selected_scales = gpu
        .take_axis(scales, input_ids, 0)
        .map_err(EngineError::Mlx)?;
    let selected_biases = gpu
        .take_axis(biases, input_ids, 0)
        .map_err(EngineError::Mlx)?;
    let embedding = gpu
        .dequantize(
            &selected_weight,
            &selected_scales,
            DequantizeConfig {
                biases: Some(&selected_biases),
                group_size: dimension_i32(
                    plan.quantization_group_size,
                    "embedding quantization group size",
                )?,
                bits: dimension_i32(plan.quantization_bits, "embedding quantization bits")?,
                mode: &plan.quantization_mode,
                dtype: None,
            },
        )
        .map_err(EngineError::Mlx)?;
    validate_array(
        &embedding,
        &[batch_size, sequence_length, plan.hidden_size],
        DType::BFloat16,
        "quantized embedding output",
    )?;
    Ok(embedding)
}

fn attention_offset(
    keys: &Array,
    values: &Array,
    batch_size: usize,
    plan: &ModelPlan,
) -> Result<usize, EngineError> {
    let shape = keys.shape();
    let [observed_batch, key_value_heads, offset, head_dimension] =
        <[usize; 4]>::try_from(shape.clone()).map_err(|shape| {
            EngineError::Unsupported(format!(
                "attention key state must be rank 4, observed {shape:?}"
            ))
        })?;
    validate_array(
        keys,
        &[
            batch_size,
            plan.key_value_head_count,
            offset,
            plan.attention_head_dimension,
        ],
        DType::BFloat16,
        "attention key state",
    )?;
    validate_array(
        values,
        &[
            batch_size,
            plan.key_value_head_count,
            offset,
            plan.attention_head_dimension,
        ],
        DType::BFloat16,
        "attention value state",
    )?;
    if observed_batch != batch_size
        || key_value_heads != plan.key_value_head_count
        || head_dimension != plan.attention_head_dimension
    {
        return Err(EngineError::Unsupported(
            "attention state dimensions changed after validation".into(),
        ));
    }
    Ok(offset)
}

pub(crate) fn greedy_token(
    gpu: &Gpu,
    logits: &Array,
    batch_size: usize,
    plan: &ModelPlan,
) -> Result<Array, EngineError> {
    let shape = logits.shape();
    let [observed_batch, sequence_length, vocabulary_size] = <[usize; 3]>::try_from(shape.clone())
        .map_err(|shape| {
            EngineError::Unsupported(format!(
                "generation logits must be rank 3, observed {shape:?}"
            ))
        })?;
    if observed_batch != batch_size
        || sequence_length == 0
        || vocabulary_size != plan.vocabulary_size
    {
        return Err(EngineError::Unsupported(format!(
            "generation logits shape drift: observed {shape:?}"
        )));
    }
    let last_logits = gpu
        .slice(
            logits,
            &[
                0,
                dimension_i32(sequence_length - 1, "last logit position")?,
                0,
            ],
            &[
                dimension_i32(batch_size, "generation batch size")?,
                dimension_i32(sequence_length, "generation sequence length")?,
                dimension_i32(plan.vocabulary_size, "generation vocabulary size")?,
            ],
            &[1, 1, 1],
        )
        .and_then(|value| gpu.reshape(&value, &[batch_size, plan.vocabulary_size]))
        .map_err(EngineError::Mlx)?;
    gpu.argmax_axis(&last_logits, -1, true)
        .map_err(EngineError::Mlx)
}

fn compare_phase(
    gpu: &Gpu,
    fixture: &SafeTensors,
    phase: &str,
    execution: &FullModelExecution,
    plan: &ModelPlan,
    differences: &mut BTreeMap<String, f32>,
) -> Result<PhaseDifferences, EngineError> {
    let embedding = compare_fixture(
        gpu,
        fixture,
        &format!("{phase}.expected_embedding"),
        &execution.embedding,
        differences,
    )?;
    let normalized_hidden = compare_fixture(
        gpu,
        fixture,
        &format!("{phase}.expected_normalized_hidden"),
        &execution.normalized_hidden,
        differences,
    )?;
    let logits = compare_fixture(
        gpu,
        fixture,
        &format!("{phase}.expected_logits"),
        &execution.logits,
        differences,
    )?;
    let mut layer_output = 0.0_f32;
    for (layer_index, output) in execution.layer_outputs.iter().enumerate() {
        let difference = compare_fixture(
            gpu,
            fixture,
            &format!("{phase}.layer.{layer_index}.expected_output"),
            output,
            differences,
        )?;
        layer_output = layer_output.max(difference);
    }
    let state = compare_states(gpu, fixture, phase, &execution.state, plan, differences)?;
    Ok(PhaseDifferences {
        embedding,
        layer_output,
        normalized_hidden,
        logits,
        state,
    })
}

fn compare_executions(
    gpu: &Gpu,
    left: &FullModelExecution,
    right: &FullModelExecution,
    plan: &ModelPlan,
) -> Result<PhaseDifferences, EngineError> {
    if left.layer_outputs.len() != plan.layer_count || right.layer_outputs.len() != plan.layer_count
    {
        return Err(EngineError::Unsupported(format!(
            "execution layer output count mismatch: expected {}, observed left {} and right {}",
            plan.layer_count,
            left.layer_outputs.len(),
            right.layer_outputs.len()
        )));
    }
    let embedding = gpu
        .max_abs_difference(&left.embedding, &right.embedding)
        .map_err(EngineError::Mlx)?;
    let normalized_hidden = gpu
        .max_abs_difference(&left.normalized_hidden, &right.normalized_hidden)
        .map_err(EngineError::Mlx)?;
    let logits = gpu
        .max_abs_difference(&left.logits, &right.logits)
        .map_err(EngineError::Mlx)?;
    let mut layer_output = 0.0_f32;
    for (left_output, right_output) in left.layer_outputs.iter().zip(&right.layer_outputs) {
        layer_output = layer_output.max(
            gpu.max_abs_difference(left_output, right_output)
                .map_err(EngineError::Mlx)?,
        );
    }
    let state = compare_state_values(gpu, &left.state, &right.state, plan)?;
    Ok(PhaseDifferences {
        embedding,
        layer_output,
        normalized_hidden,
        logits,
        state,
    })
}

pub(crate) fn compare_state_values(
    gpu: &Gpu,
    left: &MlxInferenceState,
    right: &MlxInferenceState,
    plan: &ModelPlan,
) -> Result<f32, EngineError> {
    if left.layer_count() != plan.layer_count || right.layer_count() != plan.layer_count {
        return Err(EngineError::Unsupported(format!(
            "state comparison layer count mismatch: expected {}, observed left {} and right {}",
            plan.layer_count,
            left.layer_count(),
            right.layer_count()
        )));
    }
    let mut maximum = 0.0_f32;
    for (layer_index, (left_layer, right_layer)) in
        left.layers().iter().zip(right.layers()).enumerate()
    {
        let same_kind = matches!(
            (left_layer, right_layer),
            (LayerState::Gdn { .. }, LayerState::Gdn { .. })
                | (LayerState::Attention { .. }, LayerState::Attention { .. })
        );
        if !same_kind {
            return Err(EngineError::Unsupported(format!(
                "state comparison layer {layer_index} has different cache kinds"
            )));
        }
        for (left_array, right_array) in left_layer.arrays().into_iter().zip(right_layer.arrays()) {
            maximum = maximum.max(
                gpu.max_abs_difference(left_array, right_array)
                    .map_err(EngineError::Mlx)?,
            );
        }
    }
    Ok(maximum)
}

fn compare_gdn_state_values(
    gpu: &Gpu,
    left: &MlxInferenceState,
    right: &MlxInferenceState,
    plan: &ModelPlan,
) -> Result<f32, EngineError> {
    if left.layer_count() != plan.layer_count || right.layer_count() != plan.layer_count {
        return Err(EngineError::Unsupported(format!(
            "GDN state comparison layer count mismatch: expected {}, observed left {} and right {}",
            plan.layer_count,
            left.layer_count(),
            right.layer_count()
        )));
    }
    let mut maximum = 0.0_f32;
    for (layer_index, (left_layer, right_layer)) in
        left.layers().iter().zip(right.layers()).enumerate()
    {
        match (left_layer, right_layer) {
            (
                LayerState::Gdn {
                    convolution: left_convolution,
                    recurrent: left_recurrent,
                },
                LayerState::Gdn {
                    convolution: right_convolution,
                    recurrent: right_recurrent,
                },
            ) => {
                for (left_array, right_array) in [left_convolution, left_recurrent]
                    .into_iter()
                    .zip([right_convolution, right_recurrent])
                {
                    maximum = maximum.max(
                        gpu.max_abs_difference(left_array, right_array)
                            .map_err(EngineError::Mlx)?,
                    );
                }
            }
            (LayerState::Attention { .. }, LayerState::Attention { .. }) => {}
            _ => {
                return Err(EngineError::Unsupported(format!(
                    "GDN state comparison layer {layer_index} has different cache kinds"
                )));
            }
        }
    }
    Ok(maximum)
}

pub(crate) fn compare_states(
    gpu: &Gpu,
    fixture: &SafeTensors,
    phase: &str,
    state: &MlxInferenceState,
    plan: &ModelPlan,
    differences: &mut BTreeMap<String, f32>,
) -> Result<f32, EngineError> {
    if state.layer_count() != plan.layer_count {
        return Err(EngineError::Unsupported(format!(
            "{phase} state count mismatch: expected {}, observed {}",
            plan.layer_count,
            state.layer_count()
        )));
    }
    let mut maximum = 0.0_f32;
    for (layer_index, layer_state) in state.layers().iter().enumerate() {
        let full_attention = (layer_index + 1).is_multiple_of(plan.full_attention_interval);
        let (first_name, second_name) = match (full_attention, layer_state) {
            (true, LayerState::Attention { .. }) => ("expected_keys", "expected_values"),
            (false, LayerState::Gdn { .. }) => ("expected_conv_state", "expected_recurrent_state"),
            (true, LayerState::Gdn { .. }) => {
                return Err(EngineError::Unsupported(format!(
                    "{phase} layer {layer_index} requires attention state"
                )));
            }
            (false, LayerState::Attention { .. }) => {
                return Err(EngineError::Unsupported(format!(
                    "{phase} layer {layer_index} requires GDN state"
                )));
            }
        };
        for (name, actual) in [first_name, second_name]
            .into_iter()
            .zip(layer_state.arrays())
        {
            let difference = compare_fixture(
                gpu,
                fixture,
                &format!("{phase}.layer.{layer_index}.{name}"),
                actual,
                differences,
            )?;
            maximum = maximum.max(difference);
        }
    }
    Ok(maximum)
}

fn compare_fixture(
    gpu: &Gpu,
    fixture: &SafeTensors,
    name: &str,
    actual: &Array,
    differences: &mut BTreeMap<String, f32>,
) -> Result<f32, EngineError> {
    let expected = require_tensor(fixture, name)?;
    let difference = gpu
        .max_abs_difference(actual, expected)
        .map_err(EngineError::Mlx)?;
    differences.insert(name.to_owned(), difference);
    Ok(difference)
}

fn validate_token_input(
    input: &Array,
    sequence_length: usize,
    name: &str,
) -> Result<(), EngineError> {
    let shape = input.shape();
    let [batch_size, observed_length] = <[usize; 2]>::try_from(shape.clone()).map_err(|shape| {
        EngineError::Unsupported(format!("{name} must be rank 2, observed {shape:?}"))
    })?;
    if batch_size != 1 || observed_length != sequence_length || input.dtype() != DType::Int32 {
        return Err(EngineError::Unsupported(format!(
            "{name} must be int32 [1, {sequence_length}], observed {} {shape:?}",
            input.dtype_name()
        )));
    }
    Ok(())
}

fn validate_batched_token_input(
    input: &Array,
    sequence_length: usize,
    name: &str,
) -> Result<usize, EngineError> {
    let shape = input.shape();
    let [batch_size, observed_length] = <[usize; 2]>::try_from(shape.clone()).map_err(|shape| {
        EngineError::Unsupported(format!("{name} must be rank 2, observed {shape:?}"))
    })?;
    if batch_size == 0 || observed_length != sequence_length || input.dtype() != DType::Int32 {
        return Err(EngineError::Unsupported(format!(
            "{name} must be int32 [batch > 0, {sequence_length}], observed {} {shape:?}",
            input.dtype_name()
        )));
    }
    Ok(batch_size)
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
                EngineError::Unsupported(format!("batch row does not fit MLX int ABI: {error}"))
            })?;
            let scalar = gpu
                .slice(tokens, &[row, 0], &[row + 1, 1], &[1, 1])
                .and_then(|value| gpu.reshape(&value, &[]))
                .map_err(EngineError::Mlx)?;
            scalar.item_u32().map_err(EngineError::Mlx)
        })
        .collect()
}

fn compare_fixture_token_ids(
    gpu: &Gpu,
    input: &Array,
    expected_tokens: &[u32],
) -> Result<f32, EngineError> {
    let expected_tokens = expected_tokens
        .iter()
        .map(|token| {
            i32::try_from(*token).map_err(|error| {
                EngineError::Unsupported(format!(
                    "token ID {token} does not fit MLX int32 input: {error}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expected = Array::from_i32_slice(&expected_tokens, &[1, expected_tokens.len()])
        .map_err(EngineError::Mlx)?;
    gpu.max_abs_difference(input, &expected)
        .map_err(EngineError::Mlx)
}

fn state_operation_error(operation: &str, error: impl std::fmt::Display) -> EngineError {
    EngineError::Unsupported(format!("{operation}: {error}"))
}

fn load_manifest(path: &Path) -> Result<FixtureManifest, EngineError> {
    let bytes = fs::read(path).map_err(|source| EngineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| EngineError::Json {
        path: path.to_path_buf(),
        source,
    })
}

#[allow(clippy::too_many_lines)]
fn validate_manifest(
    plan: &ModelPlan,
    model_directory: &Path,
    fixture_path: &Path,
    manifest: &FixtureManifest,
) -> Result<(), EngineError> {
    if manifest.schema_version != 1 {
        return Err(EngineError::Unsupported(format!(
            "full-model fixture schema must be 1, observed {}",
            manifest.schema_version
        )));
    }
    if plan.architecture != "qwen3_5_moe" || manifest.model_type != plan.architecture {
        return Err(EngineError::Unsupported(format!(
            "full-model fixture requires qwen3_5_moe, observed fixture {} and model {}",
            manifest.model_type, plan.architecture
        )));
    }
    if manifest.prefix_length == 0
        || manifest.continuation_length <= 1
        || manifest.generation_steps <= 1
        || manifest.prompt_token_ids.len() != manifest.prefix_length + manifest.continuation_length
        || manifest.expected_generated_tokens.len() != manifest.generation_steps
        || manifest
            .prompt_token_ids
            .iter()
            .chain(&manifest.expected_generated_tokens)
            .any(|token| {
                usize::try_from(*token).map_or(true, |token| token >= plan.vocabulary_size)
            })
    {
        return Err(EngineError::Unsupported(
            "full-model fixture token boundaries or IDs are invalid".into(),
        ));
    }
    if manifest.input_dtype != "mlx.core.int32"
        || manifest.hidden_dtype != "mlx.core.bfloat16"
        || manifest.logits_dtype != "mlx.core.bfloat16"
        || manifest.prefix_attention_mask_mode != "causal"
        || manifest.prefix_gdn_mask_mode != "None"
        || manifest.continuation_attention_mask_mode != "causal"
        || manifest.continuation_gdn_mask_mode != "None"
    {
        return Err(EngineError::Unsupported(
            "full-model fixture dtype or mask modes differ from the admitted path".into(),
        ));
    }
    let expected_layer_classes = (0..plan.layer_count)
        .map(|layer_index| {
            if (layer_index + 1).is_multiple_of(plan.full_attention_interval) {
                "full_attention"
            } else {
                "gdn"
            }
            .to_owned()
        })
        .collect::<Vec<_>>();
    if manifest.layer_classes != expected_layer_classes {
        return Err(EngineError::Unsupported(
            "full-model fixture layer schedule differs from the model plan".into(),
        ));
    }
    let dimensions = FixtureDimensions {
        hidden_size: plan.hidden_size,
        layer_count: plan.layer_count,
        full_attention_interval: plan.full_attention_interval,
        gdn_key_heads: plan.key_head_count,
        gdn_value_heads: plan.value_head_count,
        gdn_key_head_dim: plan.key_head_dimension,
        gdn_value_head_dim: plan.value_head_dimension,
        gdn_conv_kernel_size: plan.convolution_kernel_size,
        attention_heads: plan.attention_head_count,
        key_value_heads: plan.key_value_head_count,
        attention_head_dim: plan.attention_head_dimension,
        rotary_dim: plan.rotary_dimension,
        expert_count: plan.expert_count,
        experts_per_token: plan.experts_per_token,
        moe_intermediate_size: plan.moe_intermediate_size,
        shared_expert_intermediate_size: plan.shared_expert_intermediate_size,
        vocabulary_size: plan.vocabulary_size,
    };
    if manifest.dimensions != dimensions {
        return Err(EngineError::Unsupported(format!(
            "full-model fixture dimension drift: fixture {:?}, model {:?}",
            manifest.dimensions, dimensions
        )));
    }
    let rope = FixtureRope {
        base: plan.rope_base,
        scale: 1.0,
        traditional: false,
    };
    if manifest.rope != rope {
        return Err(EngineError::Unsupported(format!(
            "full-model fixture RoPE drift: fixture {:?}, model {:?}",
            manifest.rope, rope
        )));
    }
    let quantization = FixtureQuantization {
        default: FixtureQuantizationParameters {
            group_size: plan.quantization_group_size,
            bits: plan.quantization_bits,
            mode: plan.quantization_mode.clone(),
        },
        router: FixtureQuantizationParameters {
            group_size: plan.router_quantization_group_size,
            bits: plan.router_quantization_bits,
            mode: plan.router_quantization_mode.clone(),
        },
        shared_expert_gate: FixtureQuantizationParameters {
            group_size: plan.shared_gate_quantization_group_size,
            bits: plan.shared_gate_quantization_bits,
            mode: plan.shared_gate_quantization_mode.clone(),
        },
    };
    if manifest.quantization != quantization
        || manifest.norm_topk_prob != plan.norm_topk_prob
        || manifest.tie_word_embeddings != plan.tie_word_embeddings
    {
        return Err(EngineError::Unsupported(
            "full-model fixture quantization, routing, or embedding tie drift".into(),
        ));
    }
    let config_path = model_directory.join("config.json");
    let config_digest = sha256_file(&config_path)?;
    if config_digest != manifest.config_sha256 {
        return Err(EngineError::Unsupported(format!(
            "full-model fixture config digest drift: fixture {}, model {}",
            manifest.config_sha256, config_digest
        )));
    }
    let fixture_digest = sha256_file(fixture_path)?;
    if fixture_digest != manifest.fixture_sha256 {
        return Err(EngineError::Unsupported(format!(
            "full-model fixture payload digest drift: manifest {}, payload {}",
            manifest.fixture_sha256, fixture_digest
        )));
    }
    Ok(())
}
