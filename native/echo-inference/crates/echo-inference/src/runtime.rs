use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::fmt;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use echo_inference_state::{
    BeginError, CommitError, CommittedState, ExpectedState, InstanceId, ModelIdentity,
    PreparedState, RestoreError, StateLease, StateStore,
};
use echo_mlx::{Array, Gpu, MetalMemoryStats, metal_memory_stats};
use serde::{Deserialize, Serialize};

use super::decoder::MoeKernel;
use super::full_model::{
    RuntimeModelExecution, compact_runtime_state, evaluate_runtime_execution,
    execute_runtime_model, prepare_runtime_state, schedule_runtime_execution,
};
use super::gdn::GdnKernel;
use super::model_state::{MlxInferenceState, NewSessionGdnPolicy};
use super::sampling::{SamplingConfig, sample_token};
use super::snapshot::{CurrentStateOwner, PublishedMlxCheckpoint, RestoredMlxCheckpoint};
use super::weights::{BoundModelWeights, ShardedWeights};
use super::{EngineError, ModelPlan, identify_model};

mod continuous_batch;

pub(crate) use continuous_batch::{BatchAdmission, BatchGenerationObserver};

static NEXT_ENGINE_ID: AtomicU64 = AtomicU64::new(1);
const DEFAULT_PREFILL_CHUNK_SIZE_TOKENS: usize = 2_048;
const DEFAULT_PREFILL_CHUNK_AT_OR_ABOVE_TOKENS: usize = 8_192;

/// State transition selected for one E.C.H.O. inference request.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestState {
    /// First request for an instance that has no current state.
    Initial,
    /// Continues the live thinking session from the current state.
    Continuation,
    /// Starts a fresh token lineage from the current state.
    ///
    /// GDN retention follows the engine's explicit boundary policy while
    /// full-attention KV is reset before the complete fresh prompt is
    /// processed.
    NewSession,
}

impl From<RequestState> for ExpectedState {
    fn from(value: RequestState) -> Self {
        match value {
            RequestState::Initial => Self::Absent,
            RequestState::Continuation | RequestState::NewSession => Self::Present,
        }
    }
}

/// One specialized E.C.H.O. inference request.
///
/// `input_tokens` contains only the tokens that this request must execute
/// before generation. The resident state owns every previously committed
/// token and appends this request atomically with its KV and GDN payload.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct InferenceRequest {
    /// Stable E.C.H.O. existence whose state is read and advanced.
    pub instance_id: InstanceId,
    /// Requested relation to the instance's one current state.
    pub state_transition: RequestState,
    /// Newly appended input tokens, or a complete fresh prompt for a new lineage.
    pub input_tokens: Vec<u32>,
    /// Number of tokens to generate and include in the new state.
    pub max_new_tokens: usize,
    /// EOS token used to close a length-limited production generation.
    ///
    /// `None` is reserved for low-level parity probes that intentionally
    /// inspect an open token boundary. The stdio production protocol always
    /// supplies the admitted tokenizer's EOS token.
    #[serde(default)]
    pub length_eos_token: Option<u32>,
    /// Request-owned generation profile.
    #[serde(default)]
    pub sampling: SamplingConfig,
}

/// Admission limits for one resident model owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentEngineConfig {
    /// Maximum generated tokens admitted for one request.
    pub max_new_tokens_per_request: usize,
    /// Size of each input execution when long prefill is chunked.
    ///
    /// `None` retains the single-execution fast path for every input length.
    pub prefill_chunk_size_tokens: Option<usize>,
    /// Inputs at or above this boundary use `prefill_chunk_size_tokens`.
    pub prefill_chunk_at_or_above_tokens: usize,
    /// GDN components retained when `new_session` starts a fresh token lineage.
    pub new_session_gdn_policy: NewSessionGdnPolicy,
}

impl Default for ResidentEngineConfig {
    fn default() -> Self {
        Self {
            max_new_tokens_per_request: 4_096,
            prefill_chunk_size_tokens: Some(DEFAULT_PREFILL_CHUNK_SIZE_TOKENS),
            prefill_chunk_at_or_above_tokens: DEFAULT_PREFILL_CHUNK_AT_OR_ABOVE_TOKENS,
            new_session_gdn_policy: NewSessionGdnPolicy::CarryAll,
        }
    }
}

/// Immutable facts about one loaded resident model owner.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ResidentEngineInfo {
    /// Process-local identity shared by every response from this owner.
    pub engine_id: u64,
    /// Exact model identity bound to every committed state.
    pub model: ModelIdentity,
    /// Number of model weight shards retained by the owner.
    pub weight_shard_count: usize,
    /// Number of model tensor handles retained by the owner.
    pub weight_tensor_count: usize,
    /// Wall-clock duration spent admitting, identifying, and loading the model.
    pub model_load_nanos: u64,
    /// GDN components retained at a `new_session` boundary.
    pub new_session_gdn_policy: NewSessionGdnPolicy,
    /// Process-wide Metal allocator observations immediately after admission.
    pub metal_memory: RuntimeMemoryStats,
}

/// Process-wide Metal allocator observations at one runtime boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[allow(clippy::struct_field_names)]
pub struct RuntimeMemoryStats {
    /// Bytes occupied by currently active MLX allocations.
    pub active_nbytes: usize,
    /// Bytes retained in MLX's reusable allocation cache.
    pub cache_nbytes: usize,
    /// Highest active-allocation byte count observed by this MLX process.
    pub peak_nbytes: usize,
}

impl From<MetalMemoryStats> for RuntimeMemoryStats {
    fn from(value: MetalMemoryStats) -> Self {
        Self {
            active_nbytes: value.active_nbytes,
            cache_nbytes: value.cache_nbytes,
            peak_nbytes: value.peak_nbytes,
        }
    }
}

/// Per-request timings and state-size observations.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RuntimeMetrics {
    /// Time spent waiting in the single-generation queue.
    pub queue_wait_nanos: u64,
    /// Number of resident lineage tokens reused without replay.
    pub cached_prefix_tokens: usize,
    /// Number of new caller-supplied tokens executed in this request.
    pub input_tokens_processed: usize,
    /// Number of tokens generated and advanced into state.
    pub generated_tokens: usize,
    /// Largest decode batch width in which this request advanced a token.
    pub maximum_decode_batch_size: usize,
    /// Number of times this request's decode batch width changed after its
    /// first state-advance step.
    pub decode_batch_membership_changes: usize,
    /// Input executions plus one state-advancing execution per generated token.
    pub model_step_count: usize,
    /// Number of model executions used to process this request's input tokens.
    pub input_model_execution_count: usize,
    /// Materialized input-suffix execution duration.
    pub input_execution_nanos: u64,
    /// CPU-side construction time for the input-suffix MLX graph.
    pub input_graph_construction_nanos: u64,
    /// Blocking materialization time for the input-suffix MLX graph.
    pub input_materialization_nanos: u64,
    /// Duration from input execution start until the first token was sampled.
    pub first_generated_token_nanos: Option<u64>,
    /// Materialized sampling and generated-token state-advance duration.
    pub decode_execution_nanos: u64,
    /// CPU-side lazy-graph construction time accumulated across decode steps.
    pub decode_graph_construction_nanos: u64,
    /// Time spent submitting look-ahead work through `async_eval`.
    ///
    /// This may include GPU queue backpressure and is not pure CPU overhead.
    pub decode_schedule_nanos: u64,
    /// Time blocked while reading sampled token scalars during decode.
    pub decode_token_wait_nanos: u64,
    /// Final blocking materialization time for the last state advance.
    pub decode_finalization_nanos: u64,
    /// Total model execution duration, including all synchronization points.
    pub model_execution_nanos: u64,
    /// Total runtime duration after leaving the queue, including validation and commit.
    pub request_nanos: u64,
    /// Logical bytes in the newly committed KV and GDN state.
    pub committed_state_logical_nbytes: usize,
    /// Process-wide Metal allocator observations after this state commits.
    ///
    /// `None` means the state committed successfully but MLX could not report
    /// its allocator counters. Monitoring failure must not turn a committed
    /// state transition into a failed inference response.
    pub metal_memory: Option<RuntimeMemoryStats>,
}

/// Successful state-advancing result.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct InferenceResponse {
    /// Process-local resident owner that produced this result.
    pub engine_id: u64,
    /// E.C.H.O. existence whose state was committed.
    pub instance_id: InstanceId,
    /// Exact model identity bound to the state.
    pub model: ModelIdentity,
    /// KV sequence length derived from the newly committed state tensors.
    pub state_sequence_length: usize,
    /// Token IDs generated by this request.
    pub generated_tokens: Vec<u32>,
    /// Why generation stopped after committing an exact token boundary.
    pub finish_reason: GenerationFinishReason,
    /// Request-level performance observations.
    pub metrics: RuntimeMetrics,
}

/// Reason a state-advancing generation stopped.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationFinishReason {
    /// The admitted output-token limit was reached.
    Length,
    /// The composition observer recognized a stop token such as Qwen EOS.
    StopToken,
}

/// Observer decision for one newly sampled token.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerationDirective {
    /// Continue sampling when the output limit permits.
    Continue,
    /// Advance this sampled token into state, then commit and stop.
    Stop,
}

/// Per-token streaming and cancellation boundary.
///
/// `on_token` runs immediately after sampling and before the corresponding
/// state-advance execution. The runtime still advances that token before
/// returning, including when the observer requests a stop.
pub trait GenerationObserver {
    /// Returns whether the request should stop at the next token boundary.
    fn is_cancelled(&self) -> bool;

    /// Observes one sampled token and decides whether it is a terminal token.
    ///
    /// # Errors
    ///
    /// Returns an error when the observer cannot deliver or process the token.
    /// The runtime then rolls the request state back.
    fn on_token(&mut self, token: u32) -> Result<GenerationDirective, String>;
}

/// Storage lifetime selected when one independently mutable state lane opens.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StatePersistence {
    /// State may be restored from and published to `current.safetensors`.
    Durable,
    /// State exists only inside the current resident process.
    Ephemeral,
}

/// Result of registering one independently mutable state lane.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct OpenedState {
    /// Native identity of the registered state lane.
    pub instance_id: InstanceId,
    /// Whether this lane has a durable authority or process-local lifetime.
    pub persistence: StatePersistence,
    /// Whether `current.safetensors` was present and restored.
    pub restored: bool,
    /// Fixed path at which subsequent snapshots are atomically published.
    pub current_path: Option<std::path::PathBuf>,
}

enum StateOwner {
    Durable(CurrentStateOwner),
    Ephemeral,
}

/// A resident, single-model Qwen3.5-family execution owner.
///
/// The engine owns one GPU handle, one sharded weight set, and one process-local
/// per-instance state store. Execution requires `&mut self`, so callers cannot
/// overlap generations through this owner.
pub struct ResidentEngine {
    info: ResidentEngineInfo,
    config: ResidentEngineConfig,
    plan: ModelPlan,
    gpu: Gpu,
    gdn_kernel: GdnKernel,
    moe_kernel: MoeKernel,
    weights: BoundModelWeights,
    states: StateStore<MlxInferenceState>,
    state_owners: HashMap<InstanceId, StateOwner>,
}

struct ModelRun {
    state: MlxInferenceState,
    generated_tokens: Vec<u32>,
    finish_reason: GenerationFinishReason,
    input_execution_nanos: u64,
    input_graph_construction_nanos: u64,
    input_materialization_nanos: u64,
    first_generated_token_nanos: Option<u64>,
    decode_execution_nanos: u64,
    decode_graph_construction_nanos: u64,
    decode_schedule_nanos: u64,
    decode_token_wait_nanos: u64,
    decode_finalization_nanos: u64,
    model_execution_nanos: u64,
    state_advance_steps: usize,
    input_model_execution_count: usize,
}

impl ResidentEngine {
    /// Admits and loads one specialized model exactly once.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the configuration is empty, the model is
    /// outside the admitted Qwen3.5-family path, identity calculation fails,
    /// or any weight shard cannot be loaded.
    pub fn load(
        model_directory: &Path,
        config: ResidentEngineConfig,
    ) -> Result<Self, RuntimeError> {
        if config.max_new_tokens_per_request == 0 {
            return Err(RuntimeError::InvalidRequest {
                detail: "max_new_tokens_per_request must be greater than zero".into(),
            });
        }
        if config.prefill_chunk_size_tokens == Some(0) {
            return Err(RuntimeError::InvalidRequest {
                detail: "prefill_chunk_size_tokens must be greater than zero when configured"
                    .into(),
            });
        }
        let load_started = Instant::now();
        let plan = ModelPlan::from_directory(model_directory)?;
        if plan.tie_word_embeddings {
            return Err(RuntimeError::InvalidRequest {
                detail: "the resident runtime requires an independent language-model head".into(),
            });
        }
        let model = identify_model(model_directory)?;
        let source_weights = ShardedWeights::load(model_directory)?;
        let weight_shard_count = source_weights.shard_count();
        let weight_tensor_count = source_weights.tensor_count();
        let weights = source_weights.bind_model(&plan)?;
        let gpu = Gpu::new();
        let gdn_kernel = GdnKernel::new(&gpu, &plan)?;
        let moe_kernel = MoeKernel::new(&plan)?;
        let metal_memory = metal_memory_stats().map_err(EngineError::Mlx)?;
        let info = ResidentEngineInfo {
            engine_id: NEXT_ENGINE_ID.fetch_add(1, Ordering::Relaxed),
            model,
            weight_shard_count,
            weight_tensor_count,
            model_load_nanos: duration_nanos(load_started.elapsed()),
            new_session_gdn_policy: config.new_session_gdn_policy,
            metal_memory: metal_memory.into(),
        };
        Ok(Self {
            info,
            config,
            plan,
            gpu,
            gdn_kernel,
            moe_kernel,
            weights,
            states: StateStore::default(),
            state_owners: HashMap::new(),
        })
    }

    /// Returns immutable facts about the resident model owner.
    #[must_use]
    pub const fn info(&self) -> &ResidentEngineInfo {
        &self.info
    }

    /// Returns the currently committed state for one instance.
    #[must_use]
    pub fn current_state(
        &self,
        instance_id: &InstanceId,
    ) -> Option<Arc<CommittedState<MlxInferenceState>>> {
        self.states.current(instance_id)
    }

    /// Restores one authenticated durable state into an empty resident slot.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] if its model identity or complete state layout
    /// differs from this resident owner, or the process-local slot is not empty.
    pub fn restore_state(
        &self,
        committed: CommittedState<MlxInferenceState>,
    ) -> Result<Arc<CommittedState<MlxInferenceState>>, RuntimeError> {
        if committed.model != self.info.model {
            return Err(RuntimeError::ModelMismatch {
                instance_id: committed.instance_id,
            });
        }
        committed.payload.validate(&self.plan, 1)?;
        self.states
            .restore(committed)
            .map_err(RuntimeError::Restore)
    }

    /// Binds one durable directory to an instance for this engine's lifetime.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] if the instance or directory is already open,
    /// another process holds the owner lock, or an existing current payload
    /// fails authentication and model-specific validation.
    pub fn open_state(
        &mut self,
        instance_id: InstanceId,
        root: &Path,
    ) -> Result<OpenedState, RuntimeError> {
        if self.state_owners.contains_key(&instance_id)
            || self.current_state(&instance_id).is_some()
        {
            return Err(RuntimeError::InvalidRequest {
                detail: format!(
                    "instance {} state is already open in this engine",
                    instance_id.as_str()
                ),
            });
        }
        let owner = CurrentStateOwner::acquire(root)?;
        let current_path = owner.current_path();
        let restored = owner.load_current(&instance_id, &self.plan, &self.info.model, 1)?;
        let was_restored = restored.is_some();
        if let Some(restored) = restored {
            self.restore_authenticated_snapshot(restored)?;
        }
        self.state_owners
            .insert(instance_id.clone(), StateOwner::Durable(owner));
        Ok(OpenedState {
            instance_id,
            persistence: StatePersistence::Durable,
            restored: was_restored,
            current_path: Some(current_path),
        })
    }

    /// Registers one process-local state lane without a filesystem authority.
    ///
    /// This is used for the memory and emotion modules: their committed state
    /// remains available for delta-prefill and retry during the resident
    /// process lifetime, but can never replace the main Echo checkpoint.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when this identity is already registered or
    /// already has process-local state.
    pub fn open_ephemeral_state(
        &mut self,
        instance_id: InstanceId,
    ) -> Result<OpenedState, RuntimeError> {
        if self.state_owners.contains_key(&instance_id)
            || self.current_state(&instance_id).is_some()
        {
            return Err(RuntimeError::InvalidRequest {
                detail: format!(
                    "instance {} state is already open in this engine",
                    instance_id.as_str()
                ),
            });
        }
        self.state_owners
            .insert(instance_id.clone(), StateOwner::Ephemeral);
        Ok(OpenedState {
            instance_id,
            persistence: StatePersistence::Ephemeral,
            restored: false,
            current_path: None,
        })
    }

    fn restore_authenticated_snapshot(
        &self,
        restored: RestoredMlxCheckpoint,
    ) -> Result<Arc<CommittedState<MlxInferenceState>>, RuntimeError> {
        self.restore_state(CommittedState {
            instance_id: restored.instance_id,
            model: restored.model,
            payload: restored.state,
        })
    }

    /// Whether this engine owns the durable directory for an instance.
    #[must_use]
    pub fn has_state_owner(&self, instance_id: &InstanceId) -> bool {
        self.state_owners.contains_key(instance_id)
    }

    /// Atomically replaces the owned instance's `current.safetensors`.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] when the instance has no committed state or
    /// snapshot validation/publication fails.
    pub fn publish_snapshot(
        &self,
        instance_id: &InstanceId,
    ) -> Result<PublishedMlxCheckpoint, RuntimeError> {
        let owner = match self.state_owners.get(instance_id) {
            Some(StateOwner::Durable(owner)) => owner,
            Some(StateOwner::Ephemeral) => {
                return Err(RuntimeError::InvalidRequest {
                    detail: format!(
                        "instance {} is process-local and cannot publish a snapshot",
                        instance_id.as_str()
                    ),
                });
            }
            None => {
                return Err(RuntimeError::InvalidRequest {
                    detail: format!(
                        "instance {} has no open durable state owner",
                        instance_id.as_str()
                    ),
                });
            }
        };
        let committed =
            self.current_state(instance_id)
                .ok_or_else(|| RuntimeError::InvalidRequest {
                    detail: format!(
                        "instance {} has no committed state to publish",
                        instance_id.as_str()
                    ),
                })?;
        owner
            .publish(committed.as_ref(), &self.plan, 1, &self.gpu)
            .map_err(RuntimeError::Engine)
    }

    /// Executes and atomically commits one request without queue delay.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] without changing committed state when request
    /// admission, state-presence checks, model execution, validation, or
    /// commit fails.
    pub fn execute(
        &mut self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, RuntimeError> {
        self.execute_with_queue_wait(request, Duration::ZERO)
    }

    /// Executes and atomically commits one request while exposing each sampled
    /// token and checking cancellation between model steps.
    ///
    /// Cancellation at any point rolls the entire request back. Consumers must
    /// discard token events unless a final `completed` event acknowledges the
    /// commit.
    ///
    /// # Errors
    ///
    /// Returns [`RuntimeError`] under the same fail-closed rules as
    /// [`Self::execute`], including cancellation after partial streaming.
    pub fn execute_observed<O: GenerationObserver>(
        &mut self,
        request: InferenceRequest,
        observer: &mut O,
    ) -> Result<InferenceResponse, RuntimeError> {
        self.execute_with_observer(request, Duration::ZERO, observer)
    }

    pub(crate) const fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    #[cfg(feature = "parallel-generation-diagnostics")]
    pub(crate) const fn diagnostic_plan(&self) -> &ModelPlan {
        &self.plan
    }

    #[cfg(feature = "parallel-generation-diagnostics")]
    pub(crate) const fn diagnostic_weights(&self) -> &BoundModelWeights {
        &self.weights
    }

    #[cfg(feature = "parallel-generation-diagnostics")]
    pub(crate) const fn diagnostic_gdn_kernel(&self) -> &GdnKernel {
        &self.gdn_kernel
    }

    #[cfg(feature = "parallel-generation-diagnostics")]
    pub(crate) const fn diagnostic_moe_kernel(&self) -> &MoeKernel {
        &self.moe_kernel
    }

    fn execute_with_queue_wait(
        &mut self,
        request: InferenceRequest,
        queue_wait: Duration,
    ) -> Result<InferenceResponse, RuntimeError> {
        self.execute_with_observer(request, queue_wait, &mut UnobservedGeneration)
    }

    fn execute_with_observer<O: GenerationObserver>(
        &mut self,
        request: InferenceRequest,
        queue_wait: Duration,
        observer: &mut O,
    ) -> Result<InferenceResponse, RuntimeError> {
        let request_started = Instant::now();
        self.validate_request(&request)?;
        if observer.is_cancelled() {
            return Err(RuntimeError::Cancelled {
                instance_id: request.instance_id,
            });
        }

        let request_state = request.state_transition;
        let lease = self
            .states
            .begin(request.instance_id.clone(), request_state.into())?;
        let (cached_prefix_tokens, owned_initial_state) =
            self.prepare_initial_state(&request, request_state, &lease)?;
        let input_tokens_processed = request.input_tokens.len();
        let input_ids = token_array(&request.input_tokens)?;

        let initial_state = if let Some(state) = owned_initial_state.as_ref() {
            state
        } else if let Some(base) = lease.base() {
            &base.payload
        } else {
            return Err(RuntimeError::InvalidRequest {
                detail: "continuation request lost its committed base".into(),
            });
        };
        let model_run = self.run_model(&request, &input_ids, initial_state, observer)?;
        if observer.is_cancelled() {
            return Err(RuntimeError::Cancelled {
                instance_id: request.instance_id,
            });
        }

        model_run.state.validate(&self.plan, 1)?;
        let state_sequence_length = model_run.state.sequence_length()?;
        let committed_state_logical_nbytes = model_run.state.logical_nbytes()?;
        let model = self.info.model.clone();
        let prepared = PreparedState {
            model,
            payload: model_run.state,
        };
        let (committed, metal_memory) =
            commit_with_optional_metal_memory(lease, prepared, metal_memory_stats)?;
        let request_nanos = duration_nanos(request_started.elapsed());
        let generated_token_count = model_run.generated_tokens.len();

        Ok(InferenceResponse {
            engine_id: self.info.engine_id,
            instance_id: committed.instance_id.clone(),
            model: committed.model.clone(),
            state_sequence_length,
            generated_tokens: model_run.generated_tokens,
            finish_reason: model_run.finish_reason,
            metrics: RuntimeMetrics {
                queue_wait_nanos: duration_nanos(queue_wait),
                cached_prefix_tokens,
                input_tokens_processed,
                generated_tokens: generated_token_count,
                maximum_decode_batch_size: 1,
                decode_batch_membership_changes: 0,
                model_step_count: model_run
                    .state_advance_steps
                    .saturating_add(model_run.input_model_execution_count),
                input_model_execution_count: model_run.input_model_execution_count,
                input_execution_nanos: model_run.input_execution_nanos,
                input_graph_construction_nanos: model_run.input_graph_construction_nanos,
                input_materialization_nanos: model_run.input_materialization_nanos,
                first_generated_token_nanos: model_run.first_generated_token_nanos,
                decode_execution_nanos: model_run.decode_execution_nanos,
                decode_graph_construction_nanos: model_run.decode_graph_construction_nanos,
                decode_schedule_nanos: model_run.decode_schedule_nanos,
                decode_token_wait_nanos: model_run.decode_token_wait_nanos,
                decode_finalization_nanos: model_run.decode_finalization_nanos,
                model_execution_nanos: model_run.model_execution_nanos,
                request_nanos,
                committed_state_logical_nbytes,
                metal_memory,
            },
        })
    }

    fn prepare_initial_state(
        &self,
        request: &InferenceRequest,
        request_state: RequestState,
        lease: &StateLease<MlxInferenceState>,
    ) -> Result<(usize, Option<MlxInferenceState>), RuntimeError> {
        let base = lease.base();
        if let Some(base) = base {
            if base.model != self.info.model {
                return Err(RuntimeError::ModelMismatch {
                    instance_id: request.instance_id.clone(),
                });
            }
            base.payload.validate(&self.plan, 1)?;
        }
        match (request_state, base) {
            (RequestState::Initial, None) => {
                Ok((0, Some(MlxInferenceState::empty(&self.gpu, 1, &self.plan)?)))
            }
            (RequestState::Continuation, Some(base)) => Ok((base.payload.sequence_length()?, None)),
            (RequestState::NewSession, Some(base)) => Ok((
                0,
                Some(base.payload.begin_new_session(
                    &self.gpu,
                    1,
                    &self.plan,
                    self.config.new_session_gdn_policy,
                )?),
            )),
            (RequestState::Initial, Some(_))
            | (RequestState::Continuation | RequestState::NewSession, None) => {
                Err(RuntimeError::InvalidRequest {
                    detail: "state-store precondition produced an inconsistent base".into(),
                })
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn run_model<O: GenerationObserver>(
        &self,
        request: &InferenceRequest,
        input_ids: &Array,
        initial_state: &MlxInferenceState,
        observer: &mut O,
    ) -> Result<ModelRun, RuntimeError> {
        if observer.is_cancelled() {
            return Err(RuntimeError::Cancelled {
                instance_id: request.instance_id.clone(),
            });
        }
        let model_started = Instant::now();
        let input_started = Instant::now();
        let input_shape = input_ids.shape();
        let [input_batch_size, input_token_count] = <[usize; 2]>::try_from(input_shape.clone())
            .map_err(|input_shape| {
                EngineError::Unsupported(format!(
                    "runtime token input must be rank 2, observed {input_shape:?}"
                ))
            })?;
        let closing_capacity = usize::from(request.length_eos_token.is_some());
        let additional_tokens = input_token_count
            .checked_add(request.max_new_tokens)
            .and_then(|tokens| tokens.checked_add(closing_capacity))
            .ok_or_else(|| {
                EngineError::Unsupported("runtime request token capacity overflow".into())
            })?;
        let runtime_state =
            prepare_runtime_state(&self.gpu, initial_state, 1, additional_tokens, &self.plan)?;
        let chunk_size = selected_prefill_chunk_size(self.config, input_token_count);
        let mut input_graph_construction_nanos = 0_u64;
        let mut input_materialization_nanos = 0_u64;
        let (mut execution, input_model_execution_count) = if let Some(chunk_size) = chunk_size {
            let mut state = runtime_state;
            let mut final_execution = None;
            let mut execution_count = 0_usize;
            for chunk_start in (0..input_token_count).step_by(chunk_size) {
                if observer.is_cancelled() {
                    return Err(RuntimeError::Cancelled {
                        instance_id: request.instance_id.clone(),
                    });
                }
                let chunk_stop = chunk_start
                    .saturating_add(chunk_size)
                    .min(input_token_count);
                let graph_started = Instant::now();
                let chunk = slice_token_chunk(
                    &self.gpu,
                    input_ids,
                    input_batch_size,
                    chunk_start,
                    chunk_stop,
                )?;
                let chunk_execution = execute_runtime_model(
                    &self.gpu,
                    &chunk,
                    state,
                    &self.weights,
                    &self.plan,
                    &self.gdn_kernel,
                    &self.moe_kernel,
                )?;
                input_graph_construction_nanos = input_graph_construction_nanos
                    .saturating_add(duration_nanos(graph_started.elapsed()));
                let materialization_started = Instant::now();
                evaluate_runtime_execution(&self.gpu, &chunk_execution)?;
                input_materialization_nanos = input_materialization_nanos
                    .saturating_add(duration_nanos(materialization_started.elapsed()));
                execution_count = execution_count.saturating_add(1);
                if chunk_stop == input_token_count {
                    final_execution = Some(chunk_execution);
                    break;
                }
                state = chunk_execution.state;
            }
            let execution = final_execution.ok_or_else(|| {
                EngineError::Unsupported("chunked prefill produced no execution".into())
            })?;
            (execution, execution_count)
        } else {
            let graph_started = Instant::now();
            let execution = execute_runtime_model(
                &self.gpu,
                input_ids,
                runtime_state,
                &self.weights,
                &self.plan,
                &self.gdn_kernel,
                &self.moe_kernel,
            )?;
            input_graph_construction_nanos = duration_nanos(graph_started.elapsed());
            let materialization_started = Instant::now();
            evaluate_runtime_execution(&self.gpu, &execution)?;
            input_materialization_nanos = duration_nanos(materialization_started.elapsed());
            (execution, 1)
        };
        let input_execution_nanos = duration_nanos(input_started.elapsed());

        let decode_started = Instant::now();
        let mut generated_tokens = Vec::with_capacity(request.max_new_tokens);
        let mut first_generated_token_nanos = None;
        let mut finish_reason = GenerationFinishReason::Length;
        let mut decode_graph_construction_nanos = 0_u64;
        let mut decode_schedule_nanos = 0_u64;
        let mut decode_token_wait_nanos = 0_u64;
        for _ in 0..request.max_new_tokens {
            if observer.is_cancelled() {
                return Err(RuntimeError::Cancelled {
                    instance_id: request.instance_id.clone(),
                });
            }
            let graph_started = Instant::now();
            let RuntimeModelExecution { logits, state } = execution;
            let token = sample_token(
                &self.gpu,
                &logits,
                &generated_tokens,
                generated_tokens.len(),
                request.sampling,
                self.plan.vocabulary_size,
            )?;
            let scalar_token = self.gpu.reshape(&token, &[]).map_err(EngineError::Mlx)?;
            let next_execution = execute_runtime_model(
                &self.gpu,
                &token,
                state,
                &self.weights,
                &self.plan,
                &self.gdn_kernel,
                &self.moe_kernel,
            )?;
            decode_graph_construction_nanos = decode_graph_construction_nanos
                .saturating_add(duration_nanos(graph_started.elapsed()));
            let schedule_started = Instant::now();
            schedule_runtime_execution(&self.gpu, &scalar_token, &next_execution)?;
            decode_schedule_nanos =
                decode_schedule_nanos.saturating_add(duration_nanos(schedule_started.elapsed()));
            let token_wait_started = Instant::now();
            let scalar = scalar_token.item_u32().map_err(EngineError::Mlx)?;
            decode_token_wait_nanos = decode_token_wait_nanos
                .saturating_add(duration_nanos(token_wait_started.elapsed()));
            if first_generated_token_nanos.is_none() {
                first_generated_token_nanos = Some(duration_nanos(model_started.elapsed()));
            }
            generated_tokens.push(scalar);
            let directive = observer
                .on_token(scalar)
                .map_err(|detail| RuntimeError::Observer {
                    instance_id: request.instance_id.clone(),
                    detail,
                })?;
            execution = next_execution;
            if directive == GenerationDirective::Stop {
                finish_reason = GenerationFinishReason::StopToken;
                break;
            }
        }
        let mut state_advance_steps = generated_tokens.len();
        if finish_reason == GenerationFinishReason::Length {
            if observer.is_cancelled() {
                return Err(RuntimeError::Cancelled {
                    instance_id: request.instance_id.clone(),
                });
            }
            if let Some(eos_token) = request.length_eos_token {
                let graph_started = Instant::now();
                let RuntimeModelExecution { state, .. } = execution;
                let eos = token_array(&[eos_token])?;
                execution = execute_runtime_model(
                    &self.gpu,
                    &eos,
                    state,
                    &self.weights,
                    &self.plan,
                    &self.gdn_kernel,
                    &self.moe_kernel,
                )?;
                decode_graph_construction_nanos = decode_graph_construction_nanos
                    .saturating_add(duration_nanos(graph_started.elapsed()));
                state_advance_steps = state_advance_steps.saturating_add(1);
            }
        }
        if observer.is_cancelled() {
            return Err(RuntimeError::Cancelled {
                instance_id: request.instance_id.clone(),
            });
        }
        let decode_finalization_started = Instant::now();
        evaluate_runtime_execution(&self.gpu, &execution)?;
        let decode_finalization_nanos = duration_nanos(decode_finalization_started.elapsed());
        let state = compact_runtime_state(&self.gpu, execution.state, &self.plan)?;

        Ok(ModelRun {
            state,
            generated_tokens,
            finish_reason,
            input_execution_nanos,
            input_graph_construction_nanos,
            input_materialization_nanos,
            first_generated_token_nanos,
            decode_execution_nanos: duration_nanos(decode_started.elapsed()),
            decode_graph_construction_nanos,
            decode_schedule_nanos,
            decode_token_wait_nanos,
            decode_finalization_nanos,
            model_execution_nanos: duration_nanos(model_started.elapsed()),
            state_advance_steps,
            input_model_execution_count,
        })
    }

    fn validate_request(&self, request: &InferenceRequest) -> Result<(), RuntimeError> {
        if request.input_tokens.is_empty() {
            return Err(RuntimeError::InvalidRequest {
                detail: "input_tokens must contain at least one new input token".into(),
            });
        }
        if request.max_new_tokens > self.config.max_new_tokens_per_request {
            return Err(RuntimeError::InvalidRequest {
                detail: format!(
                    "max_new_tokens {} exceeds resident limit {}",
                    request.max_new_tokens, self.config.max_new_tokens_per_request
                ),
            });
        }
        request.sampling.validate(self.plan.vocabulary_size)?;
        if let Some(token) = request.length_eos_token
            && usize::try_from(token).map_or(true, |token| token >= self.plan.vocabulary_size)
        {
            return Err(RuntimeError::InvalidRequest {
                detail: format!(
                    "length-closing EOS token {token} is outside vocabulary 0..{}",
                    self.plan.vocabulary_size
                ),
            });
        }
        for (position, token) in request.input_tokens.iter().copied().enumerate() {
            if usize::try_from(token).map_or(true, |token| token >= self.plan.vocabulary_size) {
                return Err(RuntimeError::InvalidToken {
                    position,
                    token,
                    vocabulary_size: self.plan.vocabulary_size,
                });
            }
        }
        Ok(())
    }
}

fn token_array(tokens: &[u32]) -> Result<Array, RuntimeError> {
    let tokens = tokens
        .iter()
        .map(|token| {
            i32::try_from(*token).map_err(|error| RuntimeError::InvalidRequest {
                detail: format!("token ID {token} does not fit MLX int32 input: {error}"),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Array::from_i32_slice(&tokens, &[1, tokens.len()])
        .map_err(EngineError::Mlx)
        .map_err(RuntimeError::Engine)
}

fn slice_token_chunk(
    gpu: &Gpu,
    input_ids: &Array,
    batch_size: usize,
    start: usize,
    stop: usize,
) -> Result<Array, RuntimeError> {
    let batch_size = i32::try_from(batch_size).map_err(|error| {
        EngineError::Unsupported(format!("input batch size does not fit int32: {error}"))
    })?;
    let start = i32::try_from(start).map_err(|error| {
        EngineError::Unsupported(format!("prefill chunk start does not fit int32: {error}"))
    })?;
    let stop = i32::try_from(stop).map_err(|error| {
        EngineError::Unsupported(format!("prefill chunk stop does not fit int32: {error}"))
    })?;
    gpu.slice(input_ids, &[0, start], &[batch_size, stop], &[1, 1])
        .map_err(EngineError::Mlx)
        .map_err(RuntimeError::Engine)
}

fn selected_prefill_chunk_size(
    config: ResidentEngineConfig,
    input_token_count: usize,
) -> Option<usize> {
    config.prefill_chunk_size_tokens.filter(|chunk_size| {
        input_token_count >= config.prefill_chunk_at_or_above_tokens
            && input_token_count > *chunk_size
    })
}

fn commit_with_optional_metal_memory<P, F, E>(
    lease: StateLease<P>,
    prepared: PreparedState<P>,
    observe: F,
) -> Result<(Arc<CommittedState<P>>, Option<RuntimeMemoryStats>), CommitError>
where
    F: FnOnce() -> Result<MetalMemoryStats, E>,
{
    let committed = lease.commit(prepared)?;
    let metal_memory = observe().ok().map(RuntimeMemoryStats::from);
    Ok((committed, metal_memory))
}

fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// Failure before or during one state transaction.
#[derive(Debug)]
pub enum RuntimeError {
    /// The request or resident configuration is structurally invalid.
    InvalidRequest {
        /// Human-readable admission detail.
        detail: String,
    },
    /// One caller token falls outside the admitted model vocabulary.
    InvalidToken {
        /// Position in the complete supplied lineage.
        position: usize,
        /// Rejected token ID.
        token: u32,
        /// Exclusive upper bound admitted by the model.
        vocabulary_size: usize,
    },
    /// Restored or committed model identity differs from the resident owner.
    ModelMismatch {
        /// State owner whose model identity did not match.
        instance_id: InstanceId,
    },
    /// Cancellation was observed before any generated output became visible,
    /// so the state transaction was rolled back.
    Cancelled {
        /// State owner whose request was cancelled.
        instance_id: InstanceId,
    },
    /// A streaming observer failed, so the request was rolled back.
    Observer {
        /// State owner whose request could not be streamed safely.
        instance_id: InstanceId,
        /// Observer-provided failure detail.
        detail: String,
    },
    /// State presence precondition or writer ownership failed.
    Begin(BeginError),
    /// Durable-state restoration into the process-local store failed.
    Restore(RestoreError),
    /// Model admission or MLX execution failed.
    Engine(EngineError),
    /// Atomic state commit failed.
    Commit(CommitError),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest { detail } => formatter.write_str(detail),
            Self::InvalidToken {
                position,
                token,
                vocabulary_size,
            } => write!(
                formatter,
                "token at position {position} is {token}, outside vocabulary 0..{vocabulary_size}"
            ),
            Self::ModelMismatch { instance_id } => write!(
                formatter,
                "instance {} state model differs from the resident model",
                instance_id.as_str()
            ),
            Self::Cancelled { instance_id } => {
                write!(
                    formatter,
                    "instance {} inference was cancelled",
                    instance_id.as_str()
                )
            }
            Self::Observer {
                instance_id,
                detail,
            } => write!(
                formatter,
                "instance {} generation observer failed: {detail}",
                instance_id.as_str()
            ),
            Self::Begin(error) => error.fmt(formatter),
            Self::Restore(error) => error.fmt(formatter),
            Self::Engine(error) => error.fmt(formatter),
            Self::Commit(error) => error.fmt(formatter),
        }
    }
}

impl Error for RuntimeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Begin(error) => Some(error),
            Self::Restore(error) => Some(error),
            Self::Engine(error) => Some(error),
            Self::Commit(error) => Some(error),
            Self::InvalidRequest { .. }
            | Self::InvalidToken { .. }
            | Self::ModelMismatch { .. }
            | Self::Cancelled { .. }
            | Self::Observer { .. } => None,
        }
    }
}

struct UnobservedGeneration;

impl GenerationObserver for UnobservedGeneration {
    fn is_cancelled(&self) -> bool {
        false
    }

    fn on_token(&mut self, _token: u32) -> Result<GenerationDirective, String> {
        Ok(GenerationDirective::Continue)
    }
}

impl From<BeginError> for RuntimeError {
    fn from(value: BeginError) -> Self {
        Self::Begin(value)
    }
}

impl From<CommitError> for RuntimeError {
    fn from(value: CommitError) -> Self {
        Self::Commit(value)
    }
}

impl From<EngineError> for RuntimeError {
    fn from(value: EngineError) -> Self {
        Self::Engine(value)
    }
}

/// Stable queue identity allocated in FIFO order.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct ScheduleTicket(u64);

impl ScheduleTicket {
    /// Returns the process-local ticket value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

struct QueuedInference {
    ticket: ScheduleTicket,
    enqueued_at: Instant,
    request: InferenceRequest,
}

struct InferenceQueue {
    pending: VecDeque<QueuedInference>,
    capacity: usize,
    next_ticket: u64,
}

impl InferenceQueue {
    fn new(capacity: usize) -> Result<Self, SchedulerError> {
        if capacity == 0 {
            return Err(SchedulerError::InvalidCapacity);
        }
        Ok(Self {
            pending: VecDeque::with_capacity(capacity),
            capacity,
            next_ticket: 1,
        })
    }

    fn enqueue(&mut self, request: InferenceRequest) -> Result<ScheduleTicket, SchedulerError> {
        if self.pending.len() >= self.capacity {
            return Err(SchedulerError::QueueFull {
                capacity: self.capacity,
            });
        }
        let next_ticket = self
            .next_ticket
            .checked_add(1)
            .ok_or(SchedulerError::TicketOverflow)?;
        let ticket = ScheduleTicket(self.next_ticket);
        self.next_ticket = next_ticket;
        self.pending.push_back(QueuedInference {
            ticket,
            enqueued_at: Instant::now(),
            request,
        });
        Ok(ticket)
    }

    fn cancel(&mut self, ticket: ScheduleTicket) -> bool {
        self.pending
            .iter()
            .position(|queued| queued.ticket == ticket)
            .and_then(|position| self.pending.remove(position))
            .is_some()
    }

    fn pop(&mut self) -> Option<QueuedInference> {
        self.pending.pop_front()
    }

    fn len(&self) -> usize {
        self.pending.len()
    }
}

/// Result associated with one queue ticket.
pub struct ScheduledInferenceOutcome {
    /// Ticket allocated when the request entered the FIFO queue.
    pub ticket: ScheduleTicket,
    /// Successful response or fail-closed request error.
    pub result: Result<InferenceResponse, RuntimeError>,
}

/// FIFO scheduler that permits only one active generation for one resident model.
pub struct SingleGenerationScheduler {
    engine: ResidentEngine,
    queue: InferenceQueue,
}

impl SingleGenerationScheduler {
    /// Creates a bounded queue around one resident model owner.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::InvalidCapacity`] when `queue_capacity` is zero.
    pub fn new(engine: ResidentEngine, queue_capacity: usize) -> Result<Self, SchedulerError> {
        Ok(Self {
            engine,
            queue: InferenceQueue::new(queue_capacity)?,
        })
    }

    /// Adds one request to the tail of the FIFO queue.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError`] when the bounded queue is full or its
    /// process-local ticket counter is exhausted.
    pub fn enqueue(&mut self, request: InferenceRequest) -> Result<ScheduleTicket, SchedulerError> {
        self.queue.enqueue(request)
    }

    /// Removes one request that has not started.
    ///
    /// Returns `false` when the ticket is unknown or already active/completed.
    pub fn cancel(&mut self, ticket: ScheduleTicket) -> bool {
        self.queue.cancel(ticket)
    }

    /// Executes the oldest queued request to completion.
    ///
    /// A failed request leaves its prior committed state unchanged and does
    /// not prevent the next queued request from running.
    pub fn run_next(&mut self) -> Option<ScheduledInferenceOutcome> {
        let queued = self.queue.pop()?;
        let queue_wait = queued.enqueued_at.elapsed();
        Some(ScheduledInferenceOutcome {
            ticket: queued.ticket,
            result: self
                .engine
                .execute_with_queue_wait(queued.request, queue_wait),
        })
    }

    /// Returns the number of requests that have not started.
    #[must_use]
    pub fn queued_request_count(&self) -> usize {
        self.queue.len()
    }

    /// Returns the resident owner for state and model inspection.
    #[must_use]
    pub const fn engine(&self) -> &ResidentEngine {
        &self.engine
    }
}

/// Queue-admission failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SchedulerError {
    /// A scheduler cannot be constructed with an empty queue.
    InvalidCapacity,
    /// The bounded FIFO queue has no remaining slot.
    QueueFull {
        /// Configured maximum number of waiting requests.
        capacity: usize,
    },
    /// The process-local ticket sequence cannot advance.
    TicketOverflow,
}

impl fmt::Display for SchedulerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCapacity => {
                formatter.write_str("queue capacity must be greater than zero")
            }
            Self::QueueFull { capacity } => {
                write!(formatter, "inference queue is full at capacity {capacity}")
            }
            Self::TicketOverflow => formatter.write_str("inference queue ticket overflow"),
        }
    }
}

impl Error for SchedulerError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(instance: &str, token: u32) -> InferenceRequest {
        InferenceRequest {
            instance_id: InstanceId::new(instance).expect("valid test instance"),
            state_transition: RequestState::Initial,
            input_tokens: vec![token],
            max_new_tokens: 0,
            length_eos_token: None,
            sampling: SamplingConfig::default(),
        }
    }

    fn model_identity() -> ModelIdentity {
        ModelIdentity {
            architecture: "qwen3_5_moe".into(),
            config_digest: "config".into(),
            weights_digest: "weights".into(),
            tokenizer_digest: "tokenizer".into(),
            template_digest: "template".into(),
        }
    }

    #[test]
    fn queue_is_fifo_and_cancellation_removes_only_waiting_ticket() {
        let mut queue = InferenceQueue::new(3).expect("valid queue");
        let first = queue.enqueue(request("rin", 1)).expect("first ticket");
        let cancelled = queue.enqueue(request("marie", 2)).expect("second ticket");
        let third = queue.enqueue(request("echo", 3)).expect("third ticket");

        assert!(queue.cancel(cancelled));
        assert!(!queue.cancel(cancelled));
        assert_eq!(queue.pop().expect("first queued request").ticket, first);
        assert_eq!(queue.pop().expect("third queued request").ticket, third);
        assert!(queue.pop().is_none());
    }

    #[test]
    fn queue_rejects_zero_capacity_and_excess_requests() {
        assert!(matches!(
            InferenceQueue::new(0),
            Err(SchedulerError::InvalidCapacity)
        ));
        let mut queue = InferenceQueue::new(1).expect("valid queue");
        queue.enqueue(request("rin", 1)).expect("first ticket");
        assert!(matches!(
            queue.enqueue(request("marie", 2)),
            Err(SchedulerError::QueueFull { capacity: 1 })
        ));
    }

    #[test]
    fn prefill_chunks_at_the_8k_boundary() {
        let config = ResidentEngineConfig::default();
        assert_eq!(selected_prefill_chunk_size(config, 8_191), None);
        assert_eq!(selected_prefill_chunk_size(config, 8_192), Some(2_048));
        assert_eq!(selected_prefill_chunk_size(config, 16_384), Some(2_048));
        assert_eq!(selected_prefill_chunk_size(config, 32_768), Some(2_048));
        assert_eq!(
            selected_prefill_chunk_size(
                ResidentEngineConfig {
                    prefill_chunk_size_tokens: None,
                    ..config
                },
                32_768,
            ),
            None
        );
    }

    #[test]
    fn metal_memory_observation_failure_keeps_committed_state_usable() {
        let store = StateStore::default();
        let instance_id = InstanceId::new("echo:rin").expect("valid instance");
        let lease = store
            .begin(instance_id.clone(), ExpectedState::Absent)
            .expect("initial lease");
        let prepared = PreparedState {
            model: model_identity(),
            payload: 1_u32,
        };

        let (committed, metal_memory) = commit_with_optional_metal_memory(lease, prepared, || {
            Err::<MetalMemoryStats, _>("allocator counters unavailable")
        })
        .expect("state commit must not depend on monitoring");

        assert_eq!(committed.payload, 1);
        assert_eq!(metal_memory, None);
        assert_eq!(
            store
                .current(&instance_id)
                .expect("committed state remains current")
                .payload,
            1
        );

        let next_lease = store
            .begin(instance_id.clone(), ExpectedState::Present)
            .expect("committed lane remains continuable");
        let next_prepared = PreparedState {
            model: model_identity(),
            payload: 2_u32,
        };
        let expected_memory = RuntimeMemoryStats {
            active_nbytes: 10,
            cache_nbytes: 20,
            peak_nbytes: 30,
        };
        let (next_committed, next_memory) =
            commit_with_optional_metal_memory(next_lease, next_prepared, || {
                Ok::<_, &str>(MetalMemoryStats {
                    active_nbytes: 10,
                    cache_nbytes: 20,
                    peak_nbytes: 30,
                })
            })
            .expect("next state commit");

        assert_eq!(next_committed.payload, 2);
        assert_eq!(next_memory, Some(expected_memory));
    }
}
