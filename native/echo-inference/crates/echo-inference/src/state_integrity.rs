//! Opt-in real-model regression for transaction payloads, including GPU aliases.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use echo_inference_state::{CommittedState, InstanceId};
use echo_mlx::SafeTensors;

use crate::chat::{EchoChatPrompt, Qwen35ChatTokenizer};
use crate::model_state::{LayerState, MlxInferenceState};
use crate::runtime::{
    BatchAdmission, BatchGenerationObserver, GenerationDirective, GenerationObserver,
    InferenceRequest, InferenceResponse, RequestState, ResidentEngine, ResidentEngineConfig,
    RuntimeError,
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
        let arrays: Vec<_> = current
            .payload
            .layers()
            .iter()
            .flat_map(LayerState::arrays)
            .collect();
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
        let path = directory.join(format!("{}.safetensors", id.as_str()));
        assert!(!path.exists(), "must not overwrite a frozen reference");
        SafeTensors::save(&path, &tensors, &[]).expect("freeze tensor values");
        Self {
            path,
            gdn_layers: current
                .payload
                .layers()
                .iter()
                .map(|layer| matches!(layer, LayerState::Gdn { .. }))
                .collect(),
        }
    }

    fn assert_matches(&self, engine: &ResidentEngine, id: &InstanceId) {
        let reference = SafeTensors::load(&self.path).expect("load independent tensor values");
        let current = engine.current_state(id).expect("state after operation");
        assert_eq!(current.payload.layer_count(), self.gdn_layers.len());
        for (index, layer) in current.payload.layers().iter().enumerate() {
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
                    id.as_str()
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
}

impl Observer {
    fn new(interruption: Interruption, width: usize) -> Self {
        Self {
            interruption,
            tokens: vec![Vec::new(); width],
            outcomes: vec![None; width],
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
        Interruption::Cancel => assert!(matches!(result, Err(RuntimeError::Cancelled { .. }))),
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
) {
    let eos = tokenizer.eos_token_id();
    let mut originals = Vec::new();
    let mut cancelled = Vec::new();
    let mut controls = Vec::new();
    for index in 0..6 {
        let id = state_id(&format!("auxiliary.{index}"));
        let control = state_id(&format!("reference.{index}"));
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
        cancelled.push(request(&id, RequestState::Continuation, &suffix, eos));
        controls.push(request(&control, RequestState::Continuation, &suffix, eos));
    }
    // Stopping the control row at the same boundary preserves the six-to-five
    // batch shape seen by every survivor of the cancelled cohort.
    let baseline = run_batch(engine, controls.clone(), Interruption::Stop);
    let observed = run_batch(engine, cancelled.clone(), Interruption::Cancel);
    assert_eq!(observed.tokens[0].len(), 3);
    assert_eq!(observed.outcomes[0], Some(false));
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
    let fresh = state_id("retry.reference");
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
        "PASS six-row cancellation: five survivors match same-shape controls, Main is unchanged, retry matches uninterrupted state"
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
    check_batch_interruption(&mut engine, &tokenizer, &directory, &sentinel, &frozen);
    std::fs::remove_dir_all(&directory).expect("remove only test-owned references after success");
}
