//! Opt-in real-model regression for transaction payloads, including GPU aliases.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use echo_inference_state::{CommittedState, InstanceId};
use echo_mlx::SafeTensors;

use crate::chat::{EchoChatPrompt, Qwen35ChatTokenizer};
use crate::model_state::{LayerState, MlxInferenceState};
use crate::runtime::{
    BatchAdmission, BatchGenerationObserver, GenerationDirective, GenerationObserver,
    InferenceRequest, InferenceResponse, InputCacheRequest, RequestState, ResidentEngine,
    ResidentEngineConfig, RuntimeError, RuntimeTokenUsage,
};
use crate::sampling::SamplingConfig;

/// Files freeze values independently of MLX handles: retaining another Arc or
/// graph handle would miss accidental mutation of a shared backing buffer.
struct FrozenState {
    path: PathBuf,
    gdn_layers: Vec<bool>,
}

impl FrozenState {
    fn capture(engine: &ResidentEngine, id: &InstanceId, directory: &Path) -> Self {
        let current = engine.current_state(id).expect("committed state");
        Self::capture_payload(engine, &current.payload, id.as_str(), directory)
    }

    fn capture_payload(
        engine: &ResidentEngine,
        state: &MlxInferenceState,
        name: &str,
        directory: &Path,
    ) -> Self {
        let arrays: Vec<_> = state.layers().iter().flat_map(LayerState::arrays).collect();
        engine
            .gpu()
            .eval(&arrays)
            .expect("materialize frozen state");
        engine
            .gpu()
            .synchronize()
            .expect("synchronize frozen state");
        let names: Vec<_> = (0..arrays.len())
            .map(|index| format!("tensor.{index}"))
            .collect();
        let tensors: Vec<_> = names
            .iter()
            .zip(arrays)
            .map(|(name, array)| (name.as_str(), array))
            .collect();
        let path = directory.join(format!("{name}.safetensors"));
        assert!(!path.exists(), "must not overwrite a frozen reference");
        SafeTensors::save(&path, &tensors, &[]).expect("freeze tensor values");
        Self {
            path,
            gdn_layers: state
                .layers()
                .iter()
                .map(|layer| matches!(layer, LayerState::Gdn { .. }))
                .collect(),
        }
    }

    fn assert_matches(&self, engine: &ResidentEngine, id: &InstanceId) {
        let current = engine.current_state(id).expect("state after operation");
        self.assert_payload(engine, &current.payload);
    }

    fn assert_payload(&self, engine: &ResidentEngine, state: &MlxInferenceState) {
        let reference = SafeTensors::load(&self.path).expect("load independent tensor values");
        assert_eq!(state.layer_count(), self.gdn_layers.len());
        for (index, layer) in state.layers().iter().enumerate() {
            assert_eq!(
                matches!(layer, LayerState::Gdn { .. }),
                self.gdn_layers[index]
            );
            for (component, actual) in layer.arrays().into_iter().enumerate() {
                let name = format!("tensor.{}", index * 2 + component);
                let expected = reference.tensor(&name).expect("reference tensor");
                assert_eq!(actual.dtype(), expected.dtype());
                assert_eq!(actual.shape(), expected.shape());
                let difference = engine
                    .gpu()
                    .max_abs_difference(actual, expected)
                    .expect("tensor difference");
                assert!(difference.is_finite());
                assert_eq!(
                    difference.to_bits(),
                    0.0_f32.to_bits(),
                    "{} {name}: {difference}",
                    self.path.display()
                );
            }
        }
    }

    fn restore_as(&self, engine: &mut ResidentEngine, id: &InstanceId) {
        let reference = SafeTensors::load(&self.path).expect("load fresh baseline tensors");
        let layers = self
            .gdn_layers
            .iter()
            .enumerate()
            .map(|(index, gdn)| {
                let first = reference
                    .tensor(&format!("tensor.{}", index * 2))
                    .expect("first component")
                    .try_clone()
                    .expect("retain first component");
                let second = reference
                    .tensor(&format!("tensor.{}", index * 2 + 1))
                    .expect("second component")
                    .try_clone()
                    .expect("retain second component");
                if *gdn {
                    LayerState::Gdn {
                        convolution: first,
                        recurrent: second,
                    }
                } else {
                    LayerState::Attention {
                        keys: first,
                        values: second,
                    }
                }
            })
            .collect();
        engine
            .open_ephemeral_state(id.clone())
            .expect("open fresh reference lane");
        engine
            .restore_state(CommittedState {
                instance_id: id.clone(),
                model: engine.info().model.clone(),
                payload: MlxInferenceState::new(layers),
            })
            .expect("restore independent baseline");
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Interruption {
    Cancel,
    ObserverFailure,
    Stop,
}

struct Observer {
    interruption: Interruption,
    tokens: Vec<Vec<u32>>,
    outcomes: Vec<Option<bool>>,
    cancelled_usage: Vec<Option<RuntimeTokenUsage>>,
}

impl Observer {
    fn new(interruption: Interruption, width: usize) -> Self {
        Self {
            interruption,
            tokens: vec![Vec::new(); width],
            outcomes: vec![None; width],
            cancelled_usage: vec![None; width],
        }
    }
}

impl BatchGenerationObserver for Observer {
    fn is_cancelled(&self, request_index: usize) -> bool {
        self.interruption == Interruption::Cancel && request_index == 0 && self.tokens[0].len() >= 3
    }

    fn on_token(
        &mut self,
        request_index: usize,
        token: u32,
    ) -> Result<GenerationDirective, String> {
        self.tokens[request_index].push(token);
        if request_index == 0 && self.tokens[0].len() == 3 {
            if self.interruption == Interruption::ObserverFailure {
                return Err("injected stream failure after model execution".into());
            }
            if self.interruption == Interruption::Stop {
                return Ok(GenerationDirective::Stop);
            }
        }
        Ok(GenerationDirective::Continue)
    }

    fn on_outcome(
        &mut self,
        index: usize,
        outcome: &Result<InferenceResponse, RuntimeError>,
    ) -> Result<(), String> {
        assert!(
            self.outcomes[index].is_none(),
            "one terminal outcome per request"
        );
        self.outcomes[index] = Some(outcome.is_ok());
        if let Err(RuntimeError::Cancelled { usage, .. }) = outcome {
            self.cancelled_usage[index] = Some(*usage);
        }
        Ok(())
    }
}

impl GenerationObserver for Observer {
    fn is_cancelled(&self) -> bool {
        BatchGenerationObserver::is_cancelled(self, 0)
    }

    fn on_token(&mut self, token: u32) -> Result<GenerationDirective, String> {
        BatchGenerationObserver::on_token(self, 0, token)
    }
}

fn state_id(name: &str) -> InstanceId {
    InstanceId::new(name).expect("test lane ID")
}

fn prompt(tokenizer: &Qwen35ChatTokenizer, label: &str) -> Vec<u32> {
    let prompt: EchoChatPrompt = serde_json::from_value(serde_json::json!({
        "input": [{ "role": "user", "content": format!("Continue writing the sequence {label} with a short explanation.") }],
        "tools": [],
    })).expect("test prompt");
    tokenizer
        .encode_prompt(&prompt)
        .expect("tokenize test prompt")
        .token_ids
}

fn request(id: &InstanceId, transition: RequestState, input: &[u32], eos: u32) -> InferenceRequest {
    InferenceRequest {
        input_cache: None,
        response_format: None,
        instance_id: id.clone(),
        state_transition: transition,
        input_tokens: input.to_vec(),
        max_new_tokens: 8,
        length_eos_token: Some(eos),
        sampling: SamplingConfig {
            temperature: 0.0,
            top_p: 0.0,
            top_k: 0,
            presence_penalty: 0.0,
            seed: 42,
            ..SamplingConfig::default()
        },
    }
}

/// Compare retries to a fresh, uninterrupted execution from the same frozen
/// input state and the same batch shape; cross-width numerical parity is not assumed.
fn check_single_interruption(
    engine: &mut ResidentEngine,
    frozen: &FrozenState,
    directory: &Path,
    input: &[u32],
    eos: u32,
    transition: RequestState,
    interruption: Interruption,
) {
    let id = state_id(&format!("single.{transition:?}.{interruption:?}"));
    let control = state_id(&format!("control.{}", id.as_str()));
    frozen.restore_as(engine, &id);
    frozen.restore_as(engine, &control);
    let mut observer = Observer::new(interruption, 1);
    let result = engine.execute_observed(request(&id, transition, input, eos), &mut observer);
    assert_eq!(
        observer.tokens[0].len(),
        3,
        "must interrupt after real GPU work"
    );
    match interruption {
        Interruption::Cancel => {
            let Err(RuntimeError::Cancelled { usage, .. }) = result else {
                panic!("expected cancellation")
            };
            assert_eq!(usage.generated_tokens, 3);
            assert_eq!(usage.input_tokens_processed, input.len());
            let cached = if transition == RequestState::Continuation {
                engine
                    .current_state(&id)
                    .expect("committed base")
                    .payload
                    .sequence_length()
                    .expect("prefix length")
            } else {
                0
            };
            assert_eq!(usage.cached_prefix_tokens, cached);
        }
        Interruption::ObserverFailure => {
            assert!(matches!(result, Err(RuntimeError::Observer { .. })));
        }
        Interruption::Stop => panic!("expected an injected failure"),
    }
    frozen.assert_matches(engine, &id);
    let retry = engine
        .execute(request(&id, transition, input, eos))
        .expect("retry releases writer");
    let expected = engine
        .execute(request(&control, transition, input, eos))
        .expect("uninterrupted control");
    assert_eq!(retry.generated_tokens, expected.generated_tokens);
    FrozenState::capture(engine, &control, directory).assert_matches(engine, &id);
    eprintln!(
        "PASS {transition:?} {interruption:?}: all KV/GDN tensors unchanged on failure; retry equals control"
    );
}

fn run_batch(
    engine: &mut ResidentEngine,
    requests: Vec<InferenceRequest>,
    interruption: Interruption,
) -> Observer {
    let mut observer = Observer::new(interruption, requests.len());
    let admissions = requests
        .into_iter()
        .map(|request| BatchAdmission {
            request,
            queue_wait: Duration::ZERO,
        })
        .collect();
    engine
        .execute_continuous_batch_observed(admissions, 6, 4, &mut observer)
        .expect("production batch execution");
    observer
}

fn check_batch_interruption(
    engine: &mut ResidentEngine,
    tokenizer: &Qwen35ChatTokenizer,
    directory: &Path,
    sentinel: &InstanceId,
    sentinel_frozen: &FrozenState,
    transition: RequestState,
) {
    let eos = tokenizer.eos_token_id();
    let mut originals = Vec::new();
    let mut cancelled = Vec::new();
    let mut controls = Vec::new();
    for index in 0..6 {
        let id = state_id(&format!("auxiliary.{transition:?}.{index}"));
        let control = state_id(&format!("reference.{transition:?}.{index}"));
        engine
            .open_ephemeral_state(id.clone())
            .expect("open auxiliary");
        let tokens = prompt(tokenizer, &format!("memory/emotion {index}"));
        engine
            .execute(request(&id, RequestState::Initial, &tokens, eos))
            .expect("prepare auxiliary state");
        let frozen = FrozenState::capture(engine, &id, directory);
        frozen.restore_as(engine, &control);
        originals.push(frozen);
        let suffix = prompt(tokenizer, &format!("additional observation {index}"));
        cancelled.push(request(&id, transition, &suffix, eos));
        controls.push(request(&control, transition, &suffix, eos));
    }
    // Stopping the control row at the same boundary preserves the six-to-five
    // batch shape seen by every survivor of the cancelled cohort.
    let baseline = run_batch(engine, controls.clone(), Interruption::Stop);
    let observed = run_batch(engine, cancelled.clone(), Interruption::Cancel);
    assert_eq!(observed.tokens[0].len(), 3);
    assert_eq!(observed.outcomes[0], Some(false));
    let usage = observed.cancelled_usage[0].expect("cancelled batch usage");
    assert_eq!(usage.generated_tokens, 3);
    assert_eq!(
        usage.input_tokens_processed,
        cancelled[0].input_tokens.len()
    );
    assert_eq!(
        usage.cached_prefix_tokens,
        if transition == RequestState::Continuation {
            engine
                .current_state(&cancelled[0].instance_id)
                .expect("cancelled base")
                .payload
                .sequence_length()
                .expect("prefix length")
        } else {
            0
        }
    );
    assert!(
        baseline
            .outcomes
            .iter()
            .all(|outcome| *outcome == Some(true))
    );
    originals[0].assert_matches(engine, &cancelled[0].instance_id);
    sentinel_frozen.assert_matches(engine, sentinel);
    for index in 1..6 {
        assert_eq!(observed.outcomes[index], Some(true));
        assert_eq!(observed.tokens[index], baseline.tokens[index]);
        FrozenState::capture(engine, &controls[index].instance_id, directory)
            .assert_matches(engine, &cancelled[index].instance_id);
    }
    let fresh = state_id(&format!("retry.reference.{transition:?}"));
    originals[0].restore_as(engine, &fresh);
    let mut control_request = cancelled[0].clone();
    control_request.instance_id = fresh.clone();
    let expected = engine
        .execute(control_request)
        .expect("uninterrupted retry reference");
    let retry = engine
        .execute(cancelled[0].clone())
        .expect("cancelled auxiliary retry");
    assert_eq!(retry.generated_tokens, expected.generated_tokens);
    FrozenState::capture(engine, &fresh, directory)
        .assert_matches(engine, &cancelled[0].instance_id);
    sentinel_frozen.assert_matches(engine, sentinel);
    eprintln!(
        "PASS six-row {transition:?} cancellation: five survivors match same-shape controls, Main is unchanged, retry matches uninterrupted state"
    );
}

/// Exercise the single-request grammar path against independently frozen state.
fn check_structured_interruption(
    engine: &mut ResidentEngine,
    tokenizer: &Qwen35ChatTokenizer,
    frozen: &FrozenState,
    directory: &Path,
) {
    let id = state_id("structured.cancel");
    let control = state_id("structured.control");
    frozen.restore_as(engine, &id);
    frozen.restore_as(engine, &control);
    let expected = serde_json::json!({"text": "猫と散歩。引用と改行を含めても、出力は文法に従う。", "answer": 7});
    let input = prompt(tokenizer, "answer in prose without JSON");
    let mut constrained = request(
        &id,
        RequestState::NewSession,
        &input,
        tokenizer.eos_token_id(),
    );
    constrained.max_new_tokens = 128;
    constrained.response_format = Some(crate::StructuredOutputFormat {
        kind: "json_schema".into(),
        name: "single_state".into(),
        strict: true,
        schema: serde_json::json!({"const": expected}),
    });
    let mut observer = Observer::new(Interruption::Cancel, 1);
    assert!(matches!(
        engine.execute_observed(constrained.clone(), &mut observer),
        Err(RuntimeError::Cancelled { .. })
    ));
    assert_eq!(observer.tokens[0].len(), 3);
    frozen.assert_matches(engine, &id);
    let retry = engine
        .execute(constrained.clone())
        .expect("fresh matcher after rollback");
    assert_eq!(
        retry.finish_reason,
        crate::GenerationFinishReason::StopToken
    );
    let text = tokenizer
        .decode(&retry.generated_tokens[..retry.generated_tokens.len() - 1])
        .unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&text).unwrap(),
        expected
    );
    constrained.instance_id = control.clone();
    let reference = engine
        .execute(constrained)
        .expect("uninterrupted grammar control");
    assert_eq!(retry.generated_tokens, reference.generated_tokens);
    FrozenState::capture(engine, &control, directory).assert_matches(engine, &id);
    eprintln!(
        "PASS structured cancellation: all KV/GDN tensors unchanged; fresh matcher retry equals uninterrupted control"
    );
}

/// A reset must erase both KV and GDN influence, not merely restart position IDs.
fn check_reset_matches_initial(
    engine: &mut ResidentEngine,
    tokenizer: &Qwen35ChatTokenizer,
    frozen: &FrozenState,
    directory: &Path,
) {
    for width in [1, 2] {
        let mut resets = Vec::new();
        let mut controls = Vec::new();
        for row in 0..width {
            let id = state_id(&format!("reset.{width}.{row}"));
            let control = state_id(&format!("cold.{width}.{row}"));
            frozen.restore_as(engine, &id);
            engine
                .open_ephemeral_state(control.clone())
                .expect("empty control");
            let input = prompt(tokenizer, &format!("Independent module input {row}"));
            resets.push(request(
                &id,
                RequestState::Reset,
                &input,
                tokenizer.eos_token_id(),
            ));
            controls.push(request(
                &control,
                RequestState::Initial,
                &input,
                tokenizer.eos_token_id(),
            ));
        }
        if width == 1 {
            let actual = engine.execute(resets[0].clone()).expect("reset");
            let expected = engine.execute(controls[0].clone()).expect("cold input");
            assert_eq!(actual.generated_tokens, expected.generated_tokens);
            assert_eq!(actual.state_sequence_length, expected.state_sequence_length);
            assert_eq!(actual.metrics.cached_prefix_tokens, 0);
        } else {
            // Both cohorts stop row zero at the same boundary, so their shapes match.
            let actual = run_batch(engine, resets.clone(), Interruption::Stop);
            let expected = run_batch(engine, controls.clone(), Interruption::Stop);
            assert_eq!(actual.tokens, expected.tokens);
            assert!(actual.outcomes.iter().all(|outcome| *outcome == Some(true)));
            assert!(
                expected
                    .outcomes
                    .iter()
                    .all(|outcome| *outcome == Some(true))
            );
        }
        for row in 0..width {
            FrozenState::capture(engine, &controls[row].instance_id, directory)
                .assert_matches(engine, &resets[row].instance_id);
        }
    }
    eprintln!(
        "PASS reset: single and batched generation match cold-start output and every KV/GDN tensor"
    );
}

#[test]
#[ignore = "requires ECHO_NATIVE_TEST_MODEL, local model weights and a Metal GPU"]
fn real_model_transactions_preserve_all_committed_state_tensors() {
    let model = std::env::var("ECHO_NATIVE_TEST_MODEL").expect("explicit local model path");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let directory = std::env::temp_dir().join(format!(
        "echo-state-integrity-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir(&directory).expect("new test-owned directory");
    eprintln!("Independent state references: {}", directory.display());
    let tokenizer = Qwen35ChatTokenizer::load(Path::new(&model)).expect("local tokenizer");
    let mut engine = ResidentEngine::load(Path::new(&model), ResidentEngineConfig::default())
        .expect("resident model");
    engine
        .enable_structured_output(Path::new(&model), tokenizer.eos_token_id())
        .expect("grammar compiler");
    let sentinel = state_id("main");
    engine
        .open_ephemeral_state(sentinel.clone())
        .expect("test-only Main state");
    let tokens = prompt(
        &tokenizer,
        "Main state that auxiliary work must never mutate",
    );
    engine
        .execute(request(
            &sentinel,
            RequestState::Initial,
            &tokens,
            tokenizer.eos_token_id(),
        ))
        .expect("initial Main state");
    let frozen = FrozenState::capture(&engine, &sentinel, &directory);
    for (transition, interruption) in [
        (RequestState::Continuation, Interruption::Cancel),
        (RequestState::NewSession, Interruption::Cancel),
        (RequestState::Reset, Interruption::Cancel),
        (RequestState::Reset, Interruption::ObserverFailure),
        (RequestState::Continuation, Interruption::ObserverFailure),
    ] {
        check_single_interruption(
            &mut engine,
            &frozen,
            &directory,
            &tokens,
            tokenizer.eos_token_id(),
            transition,
            interruption,
        );
        frozen.assert_matches(&engine, &sentinel);
    }
    for transition in [RequestState::Continuation, RequestState::Reset] {
        check_batch_interruption(
            &mut engine,
            &tokenizer,
            &directory,
            &sentinel,
            &frozen,
            transition,
        );
    }
    check_structured_interruption(&mut engine, &tokenizer, &frozen, &directory);
    check_reset_matches_initial(&mut engine, &tokenizer, &frozen, &directory);
    frozen.assert_matches(&engine, &sentinel);
    std::fs::remove_dir_all(&directory).expect("remove only test-owned references after success");
}

/// Cold references use the same chunk boundaries as incremental prefill, so
/// exact comparison measures cache correctness independently of kernel rounding.
fn cached_request(id: &InstanceId, prefix: &[u32], scope: &str) -> InferenceRequest {
    let mut input = prefix.to_vec();
    input.extend_from_slice(&[1, 2, 3, 4]);
    let mut request = request(id, RequestState::Reset, &input, 248_046);
    request.input_cache = Some(InputCacheRequest {
        scope: scope.into(),
        prefix_tokens: prefix.to_vec(),
    });
    request
}

struct CancelDuringPrefill(std::cell::Cell<usize>);

impl GenerationObserver for CancelDuringPrefill {
    fn is_cancelled(&self) -> bool {
        let count = self.0.get() + 1;
        self.0.set(count);
        count >= 5
    }
    fn on_token(&mut self, _: u32) -> Result<GenerationDirective, String> {
        panic!("must cancel before generation")
    }
}

fn check_cache_cancellation(
    engine: &mut ResidentEngine,
    id: &InstanceId,
    prefix: &[u32],
    directory: &Path,
) {
    let previous = FrozenState::capture(engine, id, directory);
    let checkpoint = engine.input_checkpoint(id).expect("prior checkpoint");
    let frozen =
        FrozenState::capture_payload(engine, &checkpoint, "input-before-cancel", directory);
    let request = cached_request(id, prefix, "session-one");
    let cancelled = engine.execute_observed(
        request.clone(),
        &mut CancelDuringPrefill(std::cell::Cell::new(0)),
    );
    let Err(RuntimeError::Cancelled { usage, .. }) = cancelled else {
        panic!("prefill cancellation")
    };
    assert!(usage.input_tokens_processed > 0 && usage.input_tokens_processed < prefix.len() - 128);
    assert_eq!(usage.cached_prefix_tokens, 128);
    frozen.assert_payload(
        engine,
        &engine.input_checkpoint(id).expect("preserved prefix"),
    );
    previous.assert_matches(engine, id);

    let cancelled =
        engine.execute_observed(request.clone(), &mut Observer::new(Interruption::Cancel, 1));
    assert!(matches!(cancelled, Err(RuntimeError::Cancelled { .. })));
    previous.assert_matches(engine, id);
    assert_eq!(
        engine
            .input_checkpoint(id)
            .expect("completed input retained")
            .sequence_length()
            .expect("length"),
        prefix.len()
    );
    let retry = engine
        .execute(request)
        .expect("retry from input checkpoint");
    assert_eq!(retry.metrics.cached_prefix_tokens, prefix.len());
    assert_eq!(retry.metrics.input_tokens_processed, 4);
    eprintln!(
        "PASS input cache: partial prefill keeps old checkpoint; decode cancellation retains completed input and rolls generated state back"
    );
}

fn check_cache_batch(engine: &mut ResidentEngine, prefix: &[u32], directory: &Path) {
    let mut requests = Vec::new();
    let mut references = Vec::new();
    for index in 0..2 {
        let id = state_id(&format!("cache.batch.{index}"));
        engine.open_ephemeral_state(id.clone()).expect("batch lane");
        let mut request = cached_request(&id, prefix, "batch-session");
        request.state_transition = RequestState::Initial;
        let cold = engine.execute(request.clone()).expect("cold input");
        assert_eq!(cold.metrics.cached_prefix_tokens, 0);
        let checkpoint = engine.input_checkpoint(&id).expect("batch input");
        references.push(FrozenState::capture_payload(
            engine,
            &checkpoint,
            &format!("batch-input-{index}"),
            directory,
        ));
        request.state_transition = RequestState::Reset;
        requests.push(request);
    }
    let outcomes = run_batch(engine, requests.clone(), Interruption::Cancel);
    assert_eq!(outcomes.outcomes, vec![Some(false), Some(true)]);
    assert_eq!(
        outcomes.cancelled_usage[0]
            .expect("cancel usage")
            .cached_prefix_tokens,
        prefix.len()
    );
    for (request, frozen) in requests.iter().zip(references) {
        frozen.assert_payload(
            engine,
            &engine
                .input_checkpoint(&request.instance_id)
                .expect("unchanged batch input"),
        );
    }
    eprintln!(
        "PASS input cache: independently owned checkpoints survive batched cancellation and sibling completion"
    );
}

#[test]
#[ignore = "requires ECHO_NATIVE_TEST_MODEL, local model weights and a Metal GPU"]
fn real_model_input_cache_reuses_only_exact_session_prefixes() {
    let model = std::env::var("ECHO_NATIVE_TEST_MODEL").expect("explicit model");
    let directory = std::env::temp_dir().join(format!(
        "echo-input-cache-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::create_dir(&directory).expect("test-owned directory");
    let mut engine = ResidentEngine::load(
        Path::new(&model),
        ResidentEngineConfig {
            prefill_chunk_size_tokens: Some(32),
            prefill_chunk_at_or_above_tokens: 0,
            ..ResidentEngineConfig::default()
        },
    )
    .expect("model");
    let tokenizer = Qwen35ChatTokenizer::load(Path::new(&model)).expect("tokenizer");
    let prefix: Vec<_> = prompt(&tokenizer, "input prefix")
        .into_iter()
        .cycle()
        .take(256)
        .collect();
    let warm = state_id("cache.warm");
    let cold = state_id("cache.cold");
    let prefill = state_id("cache.reference");
    for id in [&warm, &cold, &prefill] {
        engine.open_ephemeral_state(id.clone()).expect("open");
    }
    let mut first = cached_request(&warm, &prefix[..64], "session-one");
    first.state_transition = RequestState::Initial;
    assert_eq!(
        engine
            .execute(first)
            .expect("first")
            .metrics
            .cached_prefix_tokens,
        0
    );
    let second = engine
        .execute(cached_request(&warm, &prefix[..128], "session-one"))
        .expect("extended");
    assert_eq!(second.metrics.cached_prefix_tokens, 64);
    assert_eq!(second.metrics.input_tokens_processed, 68);
    let mut baseline = cached_request(&cold, &prefix[..128], "session-one");
    baseline.state_transition = RequestState::Initial;
    let expected = engine.execute(baseline).expect("cold reference");
    assert_eq!(second.generated_tokens, expected.generated_tokens);
    FrozenState::capture(&engine, &cold, &directory).assert_matches(&engine, &warm);
    let mut input_only = request(
        &prefill,
        RequestState::Initial,
        &prefix[..128],
        tokenizer.eos_token_id(),
    );
    input_only.max_new_tokens = 0;
    input_only.length_eos_token = None;
    engine.execute(input_only).expect("input-only reference");
    FrozenState::capture(&engine, &prefill, &directory).assert_payload(
        &engine,
        &engine.input_checkpoint(&warm).expect("input cache"),
    );
    check_cache_cancellation(&mut engine, &warm, &prefix, &directory);
    check_cache_batch(&mut engine, &prefix[..128], &directory);
    let changed_session = engine
        .execute(cached_request(&warm, &prefix, "session-two"))
        .expect("new session");
    assert_eq!(changed_session.metrics.cached_prefix_tokens, 0);
    let mut changed_prefix = prefix.clone();
    changed_prefix[0] = 42;
    assert_eq!(
        engine
            .execute(cached_request(&warm, &changed_prefix, "session-two"))
            .expect("phase change")
            .metrics
            .cached_prefix_tokens,
        0
    );
    engine.clear_input_cache(&warm).expect("session cleanup");
    assert!(engine.input_checkpoint(&warm).is_none());
    assert_eq!(
        engine
            .execute(cached_request(&warm, &changed_prefix, "session-two"))
            .expect("after cleanup")
            .metrics
            .cached_prefix_tokens,
        0
    );
    eprintln!(
        "PASS input cache: exact cold/incremental output and KV/GDN parity; no generated-state contamination; session/phase invalidation and explicit release"
    );
    std::fs::remove_dir_all(directory).expect("test cleanup");
}
