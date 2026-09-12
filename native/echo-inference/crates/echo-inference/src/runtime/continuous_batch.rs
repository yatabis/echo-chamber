use std::time::{Duration, Instant};

use echo_inference_state::{PreparedState, StateLease};
use echo_mlx::Array;

use super::{
    EngineError, GenerationDirective, GenerationFinishReason, Gpu, InferenceRequest,
    InferenceResponse, MlxInferenceState, ResidentEngine, RuntimeError, RuntimeMetrics,
    commit_with_optional_metal_memory, duration_nanos, selected_prefill_chunk_size,
    slice_token_chunk, token_array,
};
use crate::MAX_ACTIVE_BATCH_SIZE;
use crate::full_model::{
    RuntimeModelExecution, compact_runtime_state, evaluate_runtime_execution,
    execute_runtime_model, prepare_merged_runtime_state, prepare_runtime_state,
    schedule_runtime_execution, split_runtime_state,
};
use crate::sampling::{sample_token, sample_token_rows};

/// One request that has completed protocol preparation and can enter a model
/// batch without blocking for more caller input.
pub(crate) struct BatchAdmission {
    pub(crate) request: InferenceRequest,
    pub(crate) queue_wait: Duration,
}

#[derive(Clone, Copy)]
struct BatchExecutionPolicy {
    max_active_batch_size: usize,
    max_late_join_batch_size: usize,
    admit_late_requests: bool,
    require_state_owner: bool,
    notify_outcomes: bool,
}

/// Per-request observation surface for one ready generation cohort.
///
/// The logical request index remains stable even when physical batch rows
/// leave and the remaining execution is compacted.
pub(crate) trait BatchGenerationObserver {
    /// Returns whether one logical request should roll back at the next exact
    /// token boundary.
    fn is_cancelled(&self, request_index: usize) -> bool;

    /// Observes one provisional generated token for one logical request.
    ///
    /// # Errors
    ///
    /// Returning an error rolls back only that request. Other batch rows may
    /// continue from the same completed model boundary.
    fn on_token(&mut self, request_index: usize, token: u32)
    -> Result<GenerationDirective, String>;

    /// Takes already-ready requests that may join at the next exact model
    /// boundary. The returned order defines their stable logical indices.
    fn take_ready(
        &mut self,
        _first_request_index: usize,
        _capacity: usize,
    ) -> Result<Vec<BatchAdmission>, String> {
        Ok(Vec::new())
    }

    /// Observes a terminal per-request outcome as soon as its transaction has
    /// committed or rolled back.
    fn on_outcome(
        &mut self,
        _request_index: usize,
        _outcome: &Result<InferenceResponse, RuntimeError>,
    ) -> Result<(), String> {
        Ok(())
    }
}

struct BatchRow {
    request_index: usize,
    request: InferenceRequest,
    lease: Option<StateLease<MlxInferenceState>>,
    state: Option<MlxInferenceState>,
    logits: Array,
    queue_wait: Duration,
    request_started: Instant,
    model_started: Instant,
    decode_started: Instant,
    cached_prefix_tokens: usize,
    input_execution_nanos: u64,
    input_graph_construction_nanos: u64,
    input_materialization_nanos: u64,
    input_model_execution_count: usize,
    first_generated_token_nanos: Option<u64>,
    generated_tokens: Vec<u32>,
    last_decode_batch_size: Option<usize>,
    maximum_decode_batch_size: usize,
    decode_batch_membership_changes: usize,
    decode_graph_construction_nanos: u64,
    decode_schedule_nanos: u64,
    decode_token_wait_nanos: u64,
    decode_finalization_nanos: u64,
    state_advance_steps: usize,
}

enum RowDisposition {
    Continue,
    Finish(GenerationFinishReason),
    Fail(RuntimeError),
}

struct BoundarySample {
    rows: Vec<BatchRow>,
    tokens: Vec<u32>,
    dispositions: Vec<RowDisposition>,
}

struct DecodeStep {
    execution: RuntimeModelExecution,
    dispositions: Vec<RowDisposition>,
}

impl ResidentEngine {
    /// Runs an immediate, variable-width production batch. Requests already
    /// ready when the batch starts may fill `max_active_batch_size`; later
    /// arrivals enter only at completed token/state boundaries and only up to
    /// `max_late_join_batch_size` active rows.
    pub(crate) fn execute_continuous_batch_observed<O: BatchGenerationObserver>(
        &mut self,
        initial: Vec<BatchAdmission>,
        max_active_batch_size: usize,
        max_late_join_batch_size: usize,
        observer: &mut O,
    ) -> Result<(), RuntimeError> {
        if !(1..=MAX_ACTIVE_BATCH_SIZE).contains(&max_active_batch_size) {
            return Err(RuntimeError::InvalidRequest {
                detail: format!(
                    "continuous batch maximum must be within 1..={MAX_ACTIVE_BATCH_SIZE}, observed {max_active_batch_size}"
                ),
            });
        }
        if max_late_join_batch_size == 0 || max_late_join_batch_size > max_active_batch_size {
            return Err(RuntimeError::InvalidRequest {
                detail: format!(
                    "late-join maximum must be within 1..={max_active_batch_size}, observed {max_late_join_batch_size}"
                ),
            });
        }
        if initial.is_empty() || initial.len() > max_active_batch_size {
            return Err(RuntimeError::InvalidRequest {
                detail: format!(
                    "initial continuous cohort must contain 1..={max_active_batch_size} requests, observed {}",
                    initial.len()
                ),
            });
        }

        self.execute_batch_admissions(
            initial,
            BatchExecutionPolicy {
                max_active_batch_size,
                max_late_join_batch_size,
                admit_late_requests: true,
                require_state_owner: true,
                notify_outcomes: true,
            },
            observer,
        )?;
        Ok(())
    }

    fn execute_batch_admissions<O: BatchGenerationObserver>(
        &self,
        admissions: Vec<BatchAdmission>,
        policy: BatchExecutionPolicy,
        observer: &mut O,
    ) -> Result<Vec<Result<InferenceResponse, RuntimeError>>, RuntimeError> {
        let mut outcomes = Vec::with_capacity(policy.max_active_batch_size);
        let mut reported = Vec::with_capacity(policy.max_active_batch_size);
        let mut rows = Vec::with_capacity(policy.max_active_batch_size);
        self.append_admissions(
            admissions,
            policy,
            observer,
            &mut rows,
            &mut outcomes,
            &mut reported,
        );
        Self::report_outcomes(observer, &outcomes, &mut reported, policy)?;

        if !rows.is_empty()
            && let Err(error) =
                self.run_ready_batch(rows, policy, observer, &mut outcomes, &mut reported)
        {
            let detail = error.to_string();
            for outcome in &mut outcomes {
                if outcome.is_none() {
                    *outcome = Some(Err(RuntimeError::InvalidRequest {
                        detail: format!("shared batch execution failed: {detail}"),
                    }));
                }
            }
        }
        Self::report_outcomes(observer, &outcomes, &mut reported, policy)?;

        Ok(outcomes
            .into_iter()
            .map(|outcome| {
                outcome.unwrap_or_else(|| {
                    Err(RuntimeError::InvalidRequest {
                        detail: "batch execution lost one logical request outcome".into(),
                    })
                })
            })
            .collect())
    }

    fn append_admissions<O: BatchGenerationObserver>(
        &self,
        admissions: Vec<BatchAdmission>,
        policy: BatchExecutionPolicy,
        observer: &O,
        rows: &mut Vec<BatchRow>,
        outcomes: &mut Vec<Option<Result<InferenceResponse, RuntimeError>>>,
        reported: &mut Vec<bool>,
    ) {
        let admission_started = Instant::now();
        for BatchAdmission {
            request,
            queue_wait,
        } in admissions
        {
            let request_index = outcomes.len();
            outcomes.push(None);
            reported.push(false);
            if policy.require_state_owner && !self.has_state_owner(&request.instance_id) {
                outcomes[request_index] = Some(Err(RuntimeError::InvalidRequest {
                    detail: format!(
                        "instance {} must be opened before generation",
                        request.instance_id.as_str()
                    ),
                }));
                continue;
            }
            if observer.is_cancelled(request_index) {
                outcomes[request_index] = Some(Err(RuntimeError::Cancelled {
                    instance_id: request.instance_id,
                }));
                continue;
            }
            match self.prepare_batch_row(
                request_index,
                request,
                queue_wait,
                admission_started,
                observer,
            ) {
                Ok(row) if row.request.max_new_tokens == 0 => {
                    outcomes[request_index] = Some(self.finish_zero_visible_tokens(row));
                }
                Ok(row) => rows.push(row),
                Err(error) => outcomes[request_index] = Some(Err(error)),
            }
        }
    }

    fn take_ready_admissions<O: BatchGenerationObserver>(
        active_rows: usize,
        before_first_decode_step: bool,
        policy: BatchExecutionPolicy,
        observer: &mut O,
        next_request_index: usize,
    ) -> Result<Vec<BatchAdmission>, RuntimeError> {
        if !policy.admit_late_requests {
            return Ok(Vec::new());
        }
        let admitted_width = if active_rows == 0 || before_first_decode_step {
            policy.max_active_batch_size
        } else {
            policy.max_late_join_batch_size
        };
        let capacity = admitted_width.saturating_sub(active_rows);
        if capacity == 0 {
            return Ok(Vec::new());
        }
        let admissions = observer
            .take_ready(next_request_index, capacity)
            .map_err(|detail| RuntimeError::InvalidRequest {
                detail: format!("continuous batch admission failed: {detail}"),
            })?;
        if admissions.len() > capacity {
            return Err(RuntimeError::InvalidRequest {
                detail: format!(
                    "continuous batch observer returned {} requests for capacity {capacity}",
                    admissions.len()
                ),
            });
        }
        Ok(admissions)
    }

    fn report_outcomes<O: BatchGenerationObserver>(
        observer: &mut O,
        outcomes: &[Option<Result<InferenceResponse, RuntimeError>>],
        reported: &mut [bool],
        policy: BatchExecutionPolicy,
    ) -> Result<(), RuntimeError> {
        if !policy.notify_outcomes {
            return Ok(());
        }
        for (request_index, (outcome, was_reported)) in
            outcomes.iter().zip(reported.iter_mut()).enumerate()
        {
            if *was_reported {
                continue;
            }
            let Some(outcome) = outcome.as_ref() else {
                continue;
            };
            observer
                .on_outcome(request_index, outcome)
                .map_err(|detail| RuntimeError::InvalidRequest {
                    detail: format!(
                        "continuous batch outcome delivery failed for request {request_index}: {detail}"
                    ),
                })?;
            *was_reported = true;
        }
        Ok(())
    }

    fn prepare_batch_row<O: BatchGenerationObserver>(
        &self,
        request_index: usize,
        request: InferenceRequest,
        queue_wait: Duration,
        request_started: Instant,
        observer: &O,
    ) -> Result<BatchRow, RuntimeError> {
        self.validate_request(&request)?;
        if observer.is_cancelled(request_index) {
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
        let initial_state = if let Some(state) = owned_initial_state.as_ref() {
            state
        } else if let Some(base) = lease.base() {
            &base.payload
        } else {
            return Err(RuntimeError::InvalidRequest {
                detail: "continuation request lost its committed base".into(),
            });
        };
        let input_ids = token_array(&request.input_tokens)?;
        let model_started = Instant::now();
        let (
            execution,
            input_execution_nanos,
            input_graph_construction_nanos,
            input_materialization_nanos,
            input_model_execution_count,
        ) = self.run_batch_prefill(request_index, &request, &input_ids, initial_state, observer)?;
        let RuntimeModelExecution { logits, state } = execution;
        let state = compact_runtime_state(&self.gpu, state, &self.plan)?;
        let decode_started = Instant::now();
        Ok(BatchRow {
            request_index,
            request,
            lease: Some(lease),
            state: Some(state),
            logits,
            queue_wait,
            request_started,
            model_started,
            decode_started,
            cached_prefix_tokens,
            input_execution_nanos,
            input_graph_construction_nanos,
            input_materialization_nanos,
            input_model_execution_count,
            first_generated_token_nanos: None,
            generated_tokens: Vec::new(),
            last_decode_batch_size: None,
            maximum_decode_batch_size: 0,
            decode_batch_membership_changes: 0,
            decode_graph_construction_nanos: 0,
            decode_schedule_nanos: 0,
            decode_token_wait_nanos: 0,
            decode_finalization_nanos: 0,
            state_advance_steps: 0,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn run_batch_prefill<O: BatchGenerationObserver>(
        &self,
        request_index: usize,
        request: &InferenceRequest,
        input_ids: &Array,
        initial_state: &MlxInferenceState,
        observer: &O,
    ) -> Result<(RuntimeModelExecution, u64, u64, u64, usize), RuntimeError> {
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
        let (execution, execution_count) = if let Some(chunk_size) = chunk_size {
            let mut state = runtime_state;
            let mut final_execution = None;
            let mut execution_count = 0_usize;
            for chunk_start in (0..input_token_count).step_by(chunk_size) {
                if observer.is_cancelled(request_index) {
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
        Ok((
            execution,
            duration_nanos(input_started.elapsed()),
            input_graph_construction_nanos,
            input_materialization_nanos,
            execution_count,
        ))
    }

    fn finish_zero_visible_tokens(
        &self,
        mut row: BatchRow,
    ) -> Result<InferenceResponse, RuntimeError> {
        let mut state = row.take_state()?;
        if let Some(eos_token) = row.request.length_eos_token {
            let started = Instant::now();
            state = self.close_length_state(&state, eos_token)?;
            row.decode_finalization_nanos = row
                .decode_finalization_nanos
                .saturating_add(duration_nanos(started.elapsed()));
            row.state_advance_steps = row.state_advance_steps.saturating_add(1);
        }
        self.commit_batch_row(row, state, GenerationFinishReason::Length)
    }

    fn run_ready_batch<O: BatchGenerationObserver>(
        &self,
        mut rows: Vec<BatchRow>,
        policy: BatchExecutionPolicy,
        observer: &mut O,
        outcomes: &mut Vec<Option<Result<InferenceResponse, RuntimeError>>>,
        reported: &mut Vec<bool>,
    ) -> Result<(), RuntimeError> {
        let mut before_first_decode_step = true;
        loop {
            rows = Self::remove_cancelled_rows_without_execution(rows, observer, outcomes);
            Self::report_outcomes(observer, outcomes, reported, policy)?;
            if rows.is_empty() {
                before_first_decode_step = true;
            }

            let admissions = Self::take_ready_admissions(
                rows.len(),
                before_first_decode_step,
                policy,
                observer,
                outcomes.len(),
            )?;
            let admitted_any = !admissions.is_empty();
            self.append_admissions(admissions, policy, observer, &mut rows, outcomes, reported);
            Self::report_outcomes(observer, outcomes, reported, policy)?;
            if rows.is_empty() {
                if admitted_any {
                    continue;
                }
                return Ok(());
            }

            let BoundarySample {
                rows: mut rows_for_execution,
                tokens: pending_tokens,
                dispositions,
            } = self.sample_boundary_tokens(rows, observer, outcomes)?;
            Self::report_outcomes(observer, outcomes, reported, policy)?;
            if rows_for_execution.is_empty() {
                rows = rows_for_execution;
                continue;
            }
            let additional_tokens = rows_for_execution
                .iter()
                .map(BatchRow::remaining_state_capacity)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .max()
                .unwrap_or(1);
            record_decode_batch_width(&mut rows_for_execution);
            let state_refs = rows_for_execution
                .iter()
                .map(BatchRow::state_ref)
                .collect::<Result<Vec<_>, _>>()?;
            let graph_started = Instant::now();
            let runtime_state = if state_refs.len() == 1 {
                prepare_runtime_state(&self.gpu, state_refs[0], 1, additional_tokens, &self.plan)?
            } else {
                prepare_merged_runtime_state(&self.gpu, &state_refs, additional_tokens, &self.plan)?
            };
            drop(state_refs);
            for row in &mut rows_for_execution {
                row.state = None;
            }
            let input = batched_token_array(&pending_tokens)?;
            let execution = execute_runtime_model(
                &self.gpu,
                &input,
                runtime_state,
                &self.weights,
                &self.plan,
                &self.gdn_kernel,
                &self.moe_kernel,
            )?;
            add_graph_time(
                &mut rows_for_execution,
                duration_nanos(graph_started.elapsed()),
            );
            let schedule_started = Instant::now();
            schedule_runtime_execution(&self.gpu, &input, &execution)?;
            add_schedule_time(
                &mut rows_for_execution,
                duration_nanos(schedule_started.elapsed()),
            );
            before_first_decode_step = false;

            if has_terminal_disposition(&dispositions) {
                rows = self.split_and_apply_dispositions(
                    rows_for_execution,
                    execution,
                    dispositions,
                    outcomes,
                )?;
                continue;
            }
            rows = self.continue_ready_execution(
                rows_for_execution,
                execution,
                policy,
                observer,
                outcomes,
                reported,
            )?;
        }
    }

    fn continue_ready_execution<O: BatchGenerationObserver>(
        &self,
        mut rows: Vec<BatchRow>,
        mut execution: RuntimeModelExecution,
        policy: BatchExecutionPolicy,
        observer: &mut O,
        outcomes: &mut Vec<Option<Result<InferenceResponse, RuntimeError>>>,
        reported: &mut Vec<bool>,
    ) -> Result<Vec<BatchRow>, RuntimeError> {
        loop {
            let cancellation = rows
                .iter()
                .map(|row| {
                    if observer.is_cancelled(row.request_index) {
                        RowDisposition::Fail(RuntimeError::Cancelled {
                            instance_id: row.request.instance_id.clone(),
                        })
                    } else {
                        RowDisposition::Continue
                    }
                })
                .collect::<Vec<_>>();
            if has_terminal_disposition(&cancellation) {
                return self.split_and_apply_dispositions(rows, execution, cancellation, outcomes);
            }

            let DecodeStep {
                execution: next_execution,
                dispositions,
            } = self.advance_decode_step(&mut rows, execution, observer)?;
            execution = next_execution;
            if has_terminal_disposition(&dispositions) {
                return self.split_and_apply_dispositions(rows, execution, dispositions, outcomes);
            }

            let admissions =
                Self::take_ready_admissions(rows.len(), false, policy, observer, outcomes.len())?;
            if !admissions.is_empty() {
                let continue_dispositions =
                    (0..rows.len()).map(|_| RowDisposition::Continue).collect();
                let mut survivors = self.split_and_apply_dispositions(
                    rows,
                    execution,
                    continue_dispositions,
                    outcomes,
                )?;
                self.append_admissions(
                    admissions,
                    policy,
                    observer,
                    &mut survivors,
                    outcomes,
                    reported,
                );
                Self::report_outcomes(observer, outcomes, reported, policy)?;
                return Ok(survivors);
            }
        }
    }

    fn advance_decode_step<O: BatchGenerationObserver>(
        &self,
        rows: &mut [BatchRow],
        execution: RuntimeModelExecution,
        observer: &mut O,
    ) -> Result<DecodeStep, RuntimeError> {
        let graph_started = Instant::now();
        let sampled = if rows.len() == 1 {
            let row = &rows[0];
            sample_token(
                &self.gpu,
                &execution.logits,
                &row.generated_tokens,
                row.generated_tokens.len(),
                row.request.sampling,
                self.plan.vocabulary_size,
            )?
        } else {
            let histories = rows
                .iter()
                .map(|row| row.generated_tokens.clone())
                .collect::<Vec<_>>();
            let configs = rows
                .iter()
                .map(|row| row.request.sampling)
                .collect::<Vec<_>>();
            sample_token_rows(
                &self.gpu,
                &execution.logits,
                &histories,
                &configs,
                self.plan.vocabulary_size,
            )?
        };
        let next_execution = execute_runtime_model(
            &self.gpu,
            &sampled,
            execution.state,
            &self.weights,
            &self.plan,
            &self.gdn_kernel,
            &self.moe_kernel,
        )?;
        add_graph_time(rows, duration_nanos(graph_started.elapsed()));
        let schedule_started = Instant::now();
        schedule_runtime_execution(&self.gpu, &sampled, &next_execution)?;
        add_schedule_time(rows, duration_nanos(schedule_started.elapsed()));
        let token_wait_started = Instant::now();
        let tokens = batched_token_values(&self.gpu, &sampled, rows.len())?;
        add_token_wait_time(rows, duration_nanos(token_wait_started.elapsed()));
        let dispositions = observe_tokens(rows, &tokens, observer);
        Ok(DecodeStep {
            execution: next_execution,
            dispositions,
        })
    }

    fn remove_cancelled_rows_without_execution<O: BatchGenerationObserver>(
        rows: Vec<BatchRow>,
        observer: &O,
        outcomes: &mut [Option<Result<InferenceResponse, RuntimeError>>],
    ) -> Vec<BatchRow> {
        let mut survivors = Vec::with_capacity(rows.len());
        for row in rows {
            if observer.is_cancelled(row.request_index) {
                let request_index = row.request_index;
                outcomes[request_index] = Some(Err(RuntimeError::Cancelled {
                    instance_id: row.request.instance_id.clone(),
                }));
            } else {
                survivors.push(row);
            }
        }
        survivors
    }

    fn sample_boundary_tokens<O: BatchGenerationObserver>(
        &self,
        rows: Vec<BatchRow>,
        observer: &mut O,
        outcomes: &mut [Option<Result<InferenceResponse, RuntimeError>>],
    ) -> Result<BoundarySample, RuntimeError> {
        let mut active = Vec::with_capacity(rows.len());
        let mut tokens = Vec::with_capacity(rows.len());
        let mut dispositions = Vec::with_capacity(rows.len());
        for mut row in rows {
            let graph_started = Instant::now();
            let sampled = sample_token(
                &self.gpu,
                &row.logits,
                &row.generated_tokens,
                row.generated_tokens.len(),
                row.request.sampling,
                self.plan.vocabulary_size,
            )?;
            row.decode_graph_construction_nanos = row
                .decode_graph_construction_nanos
                .saturating_add(duration_nanos(graph_started.elapsed()));
            let token_wait_started = Instant::now();
            let token = self
                .gpu
                .reshape(&sampled, &[])
                .and_then(|value| value.item_u32())
                .map_err(EngineError::Mlx)?;
            row.decode_token_wait_nanos = row
                .decode_token_wait_nanos
                .saturating_add(duration_nanos(token_wait_started.elapsed()));
            if row.first_generated_token_nanos.is_none() {
                row.first_generated_token_nanos = Some(duration_nanos(row.model_started.elapsed()));
            }
            row.generated_tokens.push(token);
            row.state_advance_steps = row.state_advance_steps.saturating_add(1);
            match observer.on_token(row.request_index, token) {
                Ok(directive) => {
                    dispositions.push(row.disposition_after_token(directive, observer));
                    tokens.push(token);
                    active.push(row);
                }
                Err(detail) => {
                    let request_index = row.request_index;
                    outcomes[request_index] = Some(Err(RuntimeError::Observer {
                        instance_id: row.request.instance_id.clone(),
                        detail,
                    }));
                }
            }
        }
        Ok(BoundarySample {
            rows: active,
            tokens,
            dispositions,
        })
    }

    fn split_and_apply_dispositions(
        &self,
        rows: Vec<BatchRow>,
        execution: RuntimeModelExecution,
        dispositions: Vec<RowDisposition>,
        outcomes: &mut [Option<Result<InferenceResponse, RuntimeError>>],
    ) -> Result<Vec<BatchRow>, RuntimeError> {
        if rows.len() != dispositions.len() {
            return Err(RuntimeError::InvalidRequest {
                detail: "batch row dispositions changed cardinality".into(),
            });
        }
        let finalization_started = Instant::now();
        evaluate_runtime_execution(&self.gpu, &execution)?;
        let RuntimeModelExecution { logits, state } = execution;
        let states = if rows.len() == 1 {
            vec![compact_runtime_state(&self.gpu, state, &self.plan)?]
        } else {
            split_runtime_state(&self.gpu, state, &self.plan)?
        };
        let finalization_nanos = duration_nanos(finalization_started.elapsed());
        let batch_size = rows.len();
        let mut survivors = Vec::with_capacity(batch_size);
        for (row_number, ((mut row, state), disposition)) in
            rows.into_iter().zip(states).zip(dispositions).enumerate()
        {
            row.decode_finalization_nanos = row
                .decode_finalization_nanos
                .saturating_add(finalization_nanos);
            match disposition {
                RowDisposition::Continue => {
                    row.logits = slice_logits_row(
                        &self.gpu,
                        &logits,
                        row_number,
                        batch_size,
                        self.plan.vocabulary_size,
                    )?;
                    row.state = Some(state);
                    survivors.push(row);
                }
                RowDisposition::Finish(finish_reason) => {
                    let request_index = row.request_index;
                    outcomes[request_index] = Some(match finish_reason {
                        GenerationFinishReason::Length => self.finish_length_row(row, state),
                        GenerationFinishReason::StopToken => {
                            self.commit_batch_row(row, state, finish_reason)
                        }
                    });
                }
                RowDisposition::Fail(error) => {
                    outcomes[row.request_index] = Some(Err(error));
                }
            }
        }
        Ok(survivors)
    }

    fn finish_length_row(
        &self,
        mut row: BatchRow,
        mut state: MlxInferenceState,
    ) -> Result<InferenceResponse, RuntimeError> {
        if let Some(eos_token) = row.request.length_eos_token {
            let started = Instant::now();
            state = self.close_length_state(&state, eos_token)?;
            row.decode_finalization_nanos = row
                .decode_finalization_nanos
                .saturating_add(duration_nanos(started.elapsed()));
            row.state_advance_steps = row.state_advance_steps.saturating_add(1);
        }
        self.commit_batch_row(row, state, GenerationFinishReason::Length)
    }

    fn close_length_state(
        &self,
        state: &MlxInferenceState,
        eos_token: u32,
    ) -> Result<MlxInferenceState, RuntimeError> {
        let state = prepare_runtime_state(&self.gpu, state, 1, 1, &self.plan)?;
        let eos = token_array(&[eos_token])?;
        let execution = execute_runtime_model(
            &self.gpu,
            &eos,
            state,
            &self.weights,
            &self.plan,
            &self.gdn_kernel,
            &self.moe_kernel,
        )?;
        evaluate_runtime_execution(&self.gpu, &execution)?;
        compact_runtime_state(&self.gpu, execution.state, &self.plan).map_err(Into::into)
    }

    fn commit_batch_row(
        &self,
        mut row: BatchRow,
        state: MlxInferenceState,
        finish_reason: GenerationFinishReason,
    ) -> Result<InferenceResponse, RuntimeError> {
        state.validate(&self.plan, 1)?;
        let state_sequence_length = state.sequence_length()?;
        let committed_state_logical_nbytes = state.logical_nbytes()?;
        let model = self.info.model.clone();
        let prepared = PreparedState {
            model,
            payload: state,
        };
        let lease = row
            .lease
            .take()
            .ok_or_else(|| RuntimeError::InvalidRequest {
                detail: "batch request lost its state lease before commit".into(),
            })?;
        let (committed, metal_memory) =
            commit_with_optional_metal_memory(lease, prepared, echo_mlx::metal_memory_stats)?;
        let generated_token_count = row.generated_tokens.len();
        Ok(InferenceResponse {
            engine_id: self.info.engine_id,
            instance_id: committed.instance_id.clone(),
            model: committed.model.clone(),
            state_sequence_length,
            generated_tokens: row.generated_tokens,
            finish_reason,
            metrics: RuntimeMetrics {
                queue_wait_nanos: duration_nanos(row.queue_wait),
                cached_prefix_tokens: row.cached_prefix_tokens,
                input_tokens_processed: row.request.input_tokens.len(),
                generated_tokens: generated_token_count,
                maximum_decode_batch_size: row.maximum_decode_batch_size.max(1),
                decode_batch_membership_changes: row.decode_batch_membership_changes,
                model_step_count: row
                    .state_advance_steps
                    .saturating_add(row.input_model_execution_count),
                input_model_execution_count: row.input_model_execution_count,
                input_execution_nanos: row.input_execution_nanos,
                input_graph_construction_nanos: row.input_graph_construction_nanos,
                input_materialization_nanos: row.input_materialization_nanos,
                first_generated_token_nanos: row.first_generated_token_nanos,
                decode_execution_nanos: duration_nanos(row.decode_started.elapsed()),
                decode_graph_construction_nanos: row.decode_graph_construction_nanos,
                decode_schedule_nanos: row.decode_schedule_nanos,
                decode_token_wait_nanos: row.decode_token_wait_nanos,
                decode_finalization_nanos: row.decode_finalization_nanos,
                model_execution_nanos: duration_nanos(row.model_started.elapsed()),
                request_nanos: duration_nanos(row.request_started.elapsed()),
                committed_state_logical_nbytes,
                metal_memory,
            },
        })
    }
}

impl BatchRow {
    fn state_ref(&self) -> Result<&MlxInferenceState, RuntimeError> {
        self.state
            .as_ref()
            .ok_or_else(|| RuntimeError::InvalidRequest {
                detail: "batch request lost its compact state at a membership boundary".into(),
            })
    }

    fn take_state(&mut self) -> Result<MlxInferenceState, RuntimeError> {
        self.state
            .take()
            .ok_or_else(|| RuntimeError::InvalidRequest {
                detail: "batch request lost its compact state before finalization".into(),
            })
    }

    fn remaining_state_capacity(&self) -> Result<usize, RuntimeError> {
        self.request
            .max_new_tokens
            .checked_sub(self.generated_tokens.len())
            .and_then(|remaining| remaining.checked_add(1))
            .and_then(|remaining| {
                remaining.checked_add(usize::from(self.request.length_eos_token.is_some()))
            })
            .ok_or_else(|| RuntimeError::InvalidRequest {
                detail: "batch request remaining state capacity overflow".into(),
            })
    }

    fn disposition_after_token<O: BatchGenerationObserver>(
        &self,
        directive: GenerationDirective,
        observer: &O,
    ) -> RowDisposition {
        if observer.is_cancelled(self.request_index) {
            return RowDisposition::Fail(RuntimeError::Cancelled {
                instance_id: self.request.instance_id.clone(),
            });
        }
        if directive == GenerationDirective::Stop {
            RowDisposition::Finish(GenerationFinishReason::StopToken)
        } else if self.generated_tokens.len() >= self.request.max_new_tokens {
            RowDisposition::Finish(GenerationFinishReason::Length)
        } else {
            RowDisposition::Continue
        }
    }
}

fn observe_tokens<O: BatchGenerationObserver>(
    rows: &mut [BatchRow],
    tokens: &[u32],
    observer: &mut O,
) -> Vec<RowDisposition> {
    rows.iter_mut()
        .zip(tokens.iter().copied())
        .map(|(row, token)| {
            if row.first_generated_token_nanos.is_none() {
                row.first_generated_token_nanos = Some(duration_nanos(row.model_started.elapsed()));
            }
            row.generated_tokens.push(token);
            row.state_advance_steps = row.state_advance_steps.saturating_add(1);
            match observer.on_token(row.request_index, token) {
                Ok(directive) => row.disposition_after_token(directive, observer),
                Err(detail) => RowDisposition::Fail(RuntimeError::Observer {
                    instance_id: row.request.instance_id.clone(),
                    detail,
                }),
            }
        })
        .collect()
}

fn has_terminal_disposition(dispositions: &[RowDisposition]) -> bool {
    dispositions
        .iter()
        .any(|disposition| !matches!(disposition, RowDisposition::Continue))
}

fn add_graph_time(rows: &mut [BatchRow], elapsed: u64) {
    for row in rows {
        row.decode_graph_construction_nanos =
            row.decode_graph_construction_nanos.saturating_add(elapsed);
    }
}

fn record_decode_batch_width(rows: &mut [BatchRow]) {
    let batch_size = rows.len();
    for row in rows {
        if row
            .last_decode_batch_size
            .is_some_and(|previous| previous != batch_size)
        {
            row.decode_batch_membership_changes =
                row.decode_batch_membership_changes.saturating_add(1);
        }
        row.last_decode_batch_size = Some(batch_size);
        row.maximum_decode_batch_size = row.maximum_decode_batch_size.max(batch_size);
    }
}

fn add_schedule_time(rows: &mut [BatchRow], elapsed: u64) {
    for row in rows {
        row.decode_schedule_nanos = row.decode_schedule_nanos.saturating_add(elapsed);
    }
}

fn add_token_wait_time(rows: &mut [BatchRow], elapsed: u64) {
    for row in rows {
        row.decode_token_wait_nanos = row.decode_token_wait_nanos.saturating_add(elapsed);
    }
}

fn batched_token_array(tokens: &[u32]) -> Result<Array, RuntimeError> {
    if tokens.is_empty() {
        return Err(RuntimeError::InvalidRequest {
            detail: "batched token array requires at least one row".into(),
        });
    }
    let values = tokens
        .iter()
        .map(|token| {
            i32::try_from(*token).map_err(|error| RuntimeError::InvalidRequest {
                detail: format!("token ID {token} does not fit MLX int32: {error}"),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Array::from_i32_slice(&values, &[values.len(), 1])
        .map_err(EngineError::Mlx)
        .map_err(Into::into)
}

fn batched_token_values(
    gpu: &Gpu,
    tokens: &Array,
    batch_size: usize,
) -> Result<Vec<u32>, RuntimeError> {
    if tokens.shape() != [batch_size, 1] {
        return Err(RuntimeError::InvalidRequest {
            detail: format!(
                "sampled tokens must have shape [{batch_size}, 1], observed {:?}",
                tokens.shape()
            ),
        });
    }
    if batch_size == 1 {
        return gpu
            .reshape(tokens, &[])
            .and_then(|value| value.item_u32())
            .map(|token| vec![token])
            .map_err(EngineError::Mlx)
            .map_err(Into::into);
    }
    (0..batch_size)
        .map(|row| {
            let row = i32::try_from(row).map_err(|error| RuntimeError::InvalidRequest {
                detail: format!("batch row does not fit int32: {error}"),
            })?;
            gpu.slice(tokens, &[row, 0], &[row + 1, 1], &[1, 1])
                .and_then(|value| gpu.reshape(&value, &[]))
                .and_then(|value| value.item_u32())
                .map_err(EngineError::Mlx)
                .map_err(Into::into)
        })
        .collect()
}

fn slice_logits_row(
    gpu: &Gpu,
    logits: &Array,
    row: usize,
    batch_size: usize,
    vocabulary_size: usize,
) -> Result<Array, RuntimeError> {
    let shape = logits.shape();
    let [observed_batch, sequence_length, observed_vocabulary] =
        <[usize; 3]>::try_from(shape.clone()).map_err(|shape| RuntimeError::InvalidRequest {
            detail: format!("batch logits must be rank 3, observed {shape:?}"),
        })?;
    if observed_batch != batch_size
        || row >= batch_size
        || sequence_length == 0
        || observed_vocabulary != vocabulary_size
    {
        return Err(RuntimeError::InvalidRequest {
            detail: format!("batch logits row {row} is invalid for shape {shape:?}"),
        });
    }
    let row = i32::try_from(row).map_err(|error| RuntimeError::InvalidRequest {
        detail: format!("logit row does not fit int32: {error}"),
    })?;
    let sequence_length =
        i32::try_from(sequence_length).map_err(|error| RuntimeError::InvalidRequest {
            detail: format!("logit sequence length does not fit int32: {error}"),
        })?;
    let vocabulary_size =
        i32::try_from(vocabulary_size).map_err(|error| RuntimeError::InvalidRequest {
            detail: format!("logit vocabulary does not fit int32: {error}"),
        })?;
    gpu.slice(
        logits,
        &[row, 0, 0],
        &[row + 1, sequence_length, vocabulary_size],
        &[1, 1, 1],
    )
    .map_err(EngineError::Mlx)
    .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use crate::MAX_ACTIVE_BATCH_SIZE;

    #[test]
    fn production_batch_width_is_bounded_at_six() {
        assert_eq!(MAX_ACTIVE_BATCH_SIZE, 6);
    }
}
