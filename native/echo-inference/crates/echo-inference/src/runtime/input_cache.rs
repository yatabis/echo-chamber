//! Session-bounded checkpoints before the generation header, never after output.

use std::rc::Rc;
use std::time::Instant;

use echo_inference_state::InstanceId;
use serde::{Deserialize, Serialize};

use super::{
    EngineError, InferenceRequest, MlxInferenceState, RequestState, ResidentEngine, RuntimeError,
    RuntimeTokenUsage, StateOwner, duration_nanos, selected_prefill_chunk_size, token_array,
};
use crate::full_model::{
    RuntimeModelExecution, compact_runtime_state, evaluate_runtime_execution,
    execute_runtime_model, prepare_runtime_state,
};

/// Tokenizer-established stable input boundary within a full prompt.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct InputCacheRequest {
    /// Session identity supplied by the local coordinator, scoped to one lane.
    pub scope: String,
    /// Exact rendered/tokenized prefix before the assistant generation header.
    pub prefix_tokens: Vec<u32>,
}

pub(super) struct InputCheckpoint {
    identity: InputCacheRequest,
    state: Rc<MlxInferenceState>,
}

pub(super) struct PrefillExecution {
    pub execution: RuntimeModelExecution,
    pub elapsed_nanos: u64,
    pub graph_nanos: u64,
    pub materialization_nanos: u64,
    pub execution_count: usize,
}

impl ResidentEngine {
    #[cfg(test)]
    pub(crate) fn input_checkpoint(&self, id: &InstanceId) -> Option<Rc<MlxInferenceState>> {
        self.input_caches
            .borrow()
            .get(id)
            .map(|cache| Rc::clone(&cache.state))
    }

    /// Frees the process-local input checkpoint without altering generated state.
    ///
    /// # Errors
    /// Returns an error unless the lane was explicitly opened as ephemeral.
    pub fn clear_input_cache(&self, instance_id: &InstanceId) -> Result<(), RuntimeError> {
        self.require_input_cache_owner(instance_id)?;
        self.input_caches.borrow_mut().remove(instance_id);
        Ok(())
    }

    fn require_input_cache_owner(&self, instance_id: &InstanceId) -> Result<(), RuntimeError> {
        if !matches!(
            self.state_owners.get(instance_id),
            Some(StateOwner::Ephemeral)
        ) {
            return Err(RuntimeError::InvalidRequest {
                detail: "input caching requires an opened ephemeral lane".into(),
            });
        }
        Ok(())
    }

    pub(super) fn validate_input_cache(
        &self,
        request: &InferenceRequest,
    ) -> Result<(), RuntimeError> {
        let Some(cache) = &request.input_cache else {
            return Ok(());
        };
        self.require_input_cache_owner(&request.instance_id)?;
        if !matches!(
            request.state_transition,
            RequestState::Initial | RequestState::Reset
        ) || cache.scope.trim().is_empty()
            || cache.scope.len() > 256
            || cache.prefix_tokens.is_empty()
            || cache.prefix_tokens.len() >= request.input_tokens.len()
            || !request.input_tokens.starts_with(&cache.prefix_tokens)
        {
            return Err(RuntimeError::InvalidRequest {
                detail: "input caching requires an independent full prompt, a session scope, and an exact prefix before a nonempty generation suffix".into(),
            });
        }
        Ok(())
    }

    pub(super) fn cached_input(&self, request: &InferenceRequest) -> Option<Rc<MlxInferenceState>> {
        let identity = request.input_cache.as_ref()?;
        let caches = self.input_caches.borrow();
        let checkpoint = caches.get(&request.instance_id)?;
        compatible_prefix(&checkpoint.identity, identity).then(|| Rc::clone(&checkpoint.state))
    }

    /// Both single and batched generation use this path, so checkpoint ownership
    /// and cancellation accounting cannot diverge between scheduler modes.
    pub(super) fn prefill_input(
        &self,
        request: &InferenceRequest,
        initial_state: &MlxInferenceState,
        is_cancelled: impl Fn() -> bool,
    ) -> Result<PrefillExecution, RuntimeError> {
        let started = Instant::now();
        let cached = initial_state.sequence_length()?;
        let token_count = request.input_tokens.len();
        let additional = token_count
            .checked_add(request.max_new_tokens)
            .and_then(|count| count.checked_add(usize::from(request.length_eos_token.is_some())))
            .ok_or_else(|| EngineError::Unsupported("runtime token capacity overflow".into()))?;
        let mut state = prepare_runtime_state(&self.gpu, initial_state, 1, additional, &self.plan)?;
        let checkpoint_at = request
            .input_cache
            .as_ref()
            .map(|cache| cache.prefix_tokens.len() - cached);
        let chunk_size =
            selected_prefill_chunk_size(self.config, token_count).unwrap_or(token_count);
        let mut result = None;
        let mut offset = 0;
        let mut graph_nanos = 0_u64;
        let mut materialization_nanos = 0_u64;
        let mut execution_count = 0;
        while offset < token_count {
            if is_cancelled() {
                return Err(RuntimeError::Cancelled {
                    usage: RuntimeTokenUsage::observed(cached, offset, 0),
                    instance_id: request.instance_id.clone(),
                });
            }
            let mut stop = offset.saturating_add(chunk_size).min(token_count);
            if let Some(boundary) = checkpoint_at.filter(|boundary| *boundary > offset) {
                stop = stop.min(boundary);
            }
            let graph_started = Instant::now();
            let tokens = token_array(&request.input_tokens[offset..stop])?;
            let execution = execute_runtime_model(
                &self.gpu,
                &tokens,
                state,
                &self.weights,
                &self.plan,
                &self.gdn_kernel,
                &self.moe_kernel,
            )?;
            graph_nanos = graph_nanos.saturating_add(duration_nanos(graph_started.elapsed()));
            let materialization_started = Instant::now();
            evaluate_runtime_execution(&self.gpu, &execution)?;
            materialization_nanos = materialization_nanos
                .saturating_add(duration_nanos(materialization_started.elapsed()));
            execution_count += 1;
            offset = stop;
            if stop == token_count {
                result = Some(execution);
                break;
            }
            if checkpoint_at == Some(stop) {
                let snapshot = compact_runtime_state(&self.gpu, execution.state, &self.plan)?;
                snapshot.validate(&self.plan, 1)?;
                // The input boundary is complete, even if generation later fails.
                // Prepare fresh KV buffers before any suffix/decode can mutate it.
                let snapshot = Rc::new(snapshot);
                state =
                    prepare_runtime_state(&self.gpu, &snapshot, 1, additional - stop, &self.plan)?;
                self.input_caches.borrow_mut().insert(
                    request.instance_id.clone(),
                    InputCheckpoint {
                        identity: request.input_cache.clone().ok_or_else(|| {
                            RuntimeError::InvalidRequest {
                                detail: "input checkpoint lost its identity".into(),
                            }
                        })?,
                        state: snapshot,
                    },
                );
            } else {
                state = execution.state;
            }
        }
        Ok(PrefillExecution {
            execution: result
                .ok_or_else(|| EngineError::Unsupported("prefill produced no execution".into()))?,
            elapsed_nanos: duration_nanos(started.elapsed()),
            graph_nanos,
            materialization_nanos,
            execution_count,
        })
    }
}

fn compatible_prefix(cached: &InputCacheRequest, requested: &InputCacheRequest) -> bool {
    cached.scope == requested.scope && requested.prefix_tokens.starts_with(&cached.prefix_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuse_requires_the_entire_cached_prefix_and_the_same_session() {
        let cached = InputCacheRequest {
            scope: "session-a".into(),
            prefix_tokens: vec![1, 2, 3],
        };
        for (scope, tokens, expected) in [
            ("session-a", vec![1, 2, 3], true),
            ("session-a", vec![1, 2, 3, 4], true),
            ("session-b", vec![1, 2, 3, 4], false),
            ("session-a", vec![1, 2], false),
            ("session-a", vec![1, 9, 3, 4], false),
        ] {
            assert_eq!(
                compatible_prefix(
                    &cached,
                    &InputCacheRequest {
                        scope: scope.into(),
                        prefix_tokens: tokens
                    }
                ),
                expected
            );
        }
    }
}
