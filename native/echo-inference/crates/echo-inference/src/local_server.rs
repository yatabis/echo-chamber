use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::fmt;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use echo_inference_state::InstanceId;
use serde::{Deserialize, Serialize};

use super::chat::{
    ChatError, EchoChatPrompt, EchoInputItem, EchoToolContract, EncodedChatPrompt,
    Qwen35ChatTokenizer, Qwen35DecodeStream,
};
use super::runtime::{
    BatchAdmission, BatchGenerationObserver, GenerationDirective, GenerationObserver,
    InferenceRequest, InferenceResponse, RequestState, ResidentEngine, ResidentEngineConfig,
    ResidentEngineInfo, RuntimeError, StatePersistence,
};
use super::sampling::SamplingConfig;
use super::tool_output::{EchoOutputItem, parse_qwen_output};

const PROTOCOL_VERSION: u32 = 9;

/// Admission and backpressure limits for the dedicated local stdio server.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalServerConfig {
    /// Active plus waiting generate requests admitted at once.
    pub max_outstanding_requests: usize,
    /// Maximum independently owned rows executed as one model batch.
    pub max_active_batch_size: usize,
    /// Maximum active width after admitting a request that arrived mid-decode.
    pub max_late_join_batch_size: usize,
    /// Number of serialized events buffered before generation backpressures.
    pub event_buffer_capacity: usize,
    /// Resident-engine generation limits.
    pub engine: ResidentEngineConfig,
}

impl Default for LocalServerConfig {
    fn default() -> Self {
        Self {
            max_outstanding_requests: 8,
            max_active_batch_size: 6,
            max_late_join_batch_size: 4,
            event_buffer_capacity: 128,
            engine: ResidentEngineConfig::default(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum WireCommand {
    Generate {
        request_id: String,
        instance_id: InstanceId,
        state_transition: RequestState,
        stream_tokens: bool,
        input: Vec<EchoInputItem>,
        #[serde(default)]
        tools: Vec<EchoToolContract>,
        max_new_tokens: usize,
        #[serde(default)]
        sampling: SamplingConfig,
    },
    Cancel {
        request_id: String,
    },
    OpenState {
        request_id: String,
        instance_id: InstanceId,
        persistence: StatePersistence,
        #[serde(default)]
        snapshot_root: Option<PathBuf>,
    },
    Snapshot {
        request_id: String,
        instance_id: InstanceId,
    },
    Shutdown,
}

struct AcceptedGenerate {
    request_id: String,
    instance_id: InstanceId,
    state_transition: RequestState,
    stream_tokens: bool,
    prompt: EchoChatPrompt,
    max_new_tokens: usize,
    sampling: SamplingConfig,
    cancellation: Arc<AtomicBool>,
    enqueued_at: Instant,
}

enum AcceptedCommand {
    Generate(AcceptedGenerate),
    OpenState {
        request_id: String,
        instance_id: InstanceId,
        persistence: StatePersistence,
        snapshot_root: Option<PathBuf>,
    },
    Snapshot {
        request_id: String,
        instance_id: InstanceId,
    },
    Shutdown,
}

#[derive(Debug, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
enum WireEvent {
    Ready {
        protocol_version: u32,
        engine: ResidentEngineInfo,
        eos_token_id: u32,
        chat_template_sha256: String,
        max_outstanding_requests: usize,
        max_active_batch_size: usize,
        max_late_join_batch_size: usize,
    },
    Queued {
        request_id: String,
        outstanding_requests: usize,
    },
    Started {
        request_id: String,
        prompt_tokens: usize,
    },
    Token {
        request_id: String,
        index: usize,
        token_id: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        terminal: bool,
    },
    Completed {
        request_id: String,
        response: Box<InferenceResponse>,
        text: String,
        output: Vec<EchoOutputItem>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_parse_warning: Option<String>,
    },
    CancelAcknowledged {
        request_id: String,
        accepted: bool,
    },
    Cancelled {
        request_id: String,
    },
    StateOpened {
        request_id: String,
        instance_id: InstanceId,
        persistence: StatePersistence,
        restored: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        current_path: Option<PathBuf>,
    },
    SnapshotPublished {
        request_id: String,
        instance_id: InstanceId,
        path: PathBuf,
        physical_nbytes: u64,
    },
    Failed {
        #[serde(skip_serializing_if = "Option::is_none")]
        request_id: Option<String>,
        phase: &'static str,
        error: String,
    },
    Shutdown,
}

struct RequestRegistration {
    cancellation: Arc<AtomicBool>,
}

type RequestRegistry = Arc<Mutex<HashMap<String, RequestRegistration>>>;

/// Loads one model/tokenizer owner and serves E.C.H.O.-specific NDJSON over
/// stdin/stdout until `shutdown` or EOF.
///
/// A dedicated reader thread keeps cancellation available while the main
/// thread owns synchronous MLX execution. A bounded event channel applies
/// output backpressure without buffering an unbounded token stream.
/// Individual token events optimize streaming latency and precede that token's
/// state-advance step; only `completed` is a commit acknowledgement.
///
/// # Errors
///
/// Returns [`LocalServerError`] for invalid limits, model/tokenizer admission,
/// stdio failure, internal channel failure, or a panicked protocol thread.
pub fn serve_local_stdio(
    model_directory: &Path,
    config: LocalServerConfig,
) -> Result<(), LocalServerError> {
    validate_config(config)?;
    let tokenizer = Qwen35ChatTokenizer::load(model_directory)?;
    let mut engine = ResidentEngine::load(model_directory, config.engine)?;
    let registry = RequestRegistry::default();
    let (command_sender, command_receiver) = mpsc::channel();
    let (event_sender, event_receiver) = mpsc::sync_channel(config.event_buffer_capacity);

    let writer = thread::Builder::new()
        .name("echo-inference-stdio-writer".into())
        .spawn(move || write_events(&event_receiver))
        .map_err(LocalServerError::ThreadSpawn)?;
    send_event(
        &event_sender,
        WireEvent::Ready {
            protocol_version: PROTOCOL_VERSION,
            engine: engine.info().clone(),
            eos_token_id: tokenizer.eos_token_id(),
            chat_template_sha256: tokenizer.chat_template_sha256().into(),
            max_outstanding_requests: config.max_outstanding_requests,
            max_active_batch_size: config.max_active_batch_size,
            max_late_join_batch_size: config.max_late_join_batch_size,
        },
    )?;

    let reader_registry = Arc::clone(&registry);
    let reader_events = event_sender.clone();
    let reader = thread::Builder::new()
        .name("echo-inference-stdio-reader".into())
        .spawn(move || {
            read_commands(
                &command_sender,
                &reader_events,
                &reader_registry,
                config.max_outstanding_requests,
            )
        })
        .map_err(LocalServerError::ThreadSpawn)?;

    run_command_loop(
        &mut engine,
        &tokenizer,
        &command_receiver,
        &event_sender,
        &registry,
        config.max_active_batch_size,
        config.max_late_join_batch_size,
    )?;
    send_event(&event_sender, WireEvent::Shutdown)?;
    drop(event_sender);

    join_protocol_thread(reader, "reader")?;
    join_protocol_thread(writer, "writer")?;
    Ok(())
}

fn validate_config(config: LocalServerConfig) -> Result<(), LocalServerError> {
    if config.max_outstanding_requests == 0 {
        return Err(LocalServerError::InvalidConfiguration(
            "max_outstanding_requests must be greater than zero".into(),
        ));
    }
    if !(1..=6).contains(&config.max_active_batch_size) {
        return Err(LocalServerError::InvalidConfiguration(
            "max_active_batch_size must be within 1..=6".into(),
        ));
    }
    if config.max_late_join_batch_size == 0
        || config.max_late_join_batch_size > config.max_active_batch_size
    {
        return Err(LocalServerError::InvalidConfiguration(format!(
            "max_late_join_batch_size must be within 1..={}, observed {}",
            config.max_active_batch_size, config.max_late_join_batch_size
        )));
    }
    if config.event_buffer_capacity == 0 {
        return Err(LocalServerError::InvalidConfiguration(
            "event_buffer_capacity must be greater than zero".into(),
        ));
    }
    Ok(())
}

fn read_commands(
    command_sender: &Sender<AcceptedCommand>,
    event_sender: &SyncSender<WireEvent>,
    registry: &RequestRegistry,
    max_outstanding_requests: usize,
) -> Result<(), LocalServerError> {
    let input = io::stdin();
    let lines = BufReader::new(input.lock()).lines();
    for line in lines {
        let line = line.map_err(LocalServerError::Stdio)?;
        if line.trim().is_empty() {
            continue;
        }
        let command: WireCommand = match serde_json::from_str(&line) {
            Ok(command) => command,
            Err(error) => {
                send_event(
                    event_sender,
                    WireEvent::Failed {
                        request_id: None,
                        phase: "parse",
                        error: error.to_string(),
                    },
                )?;
                continue;
            }
        };
        if dispatch_wire_command(
            command,
            command_sender,
            event_sender,
            registry,
            max_outstanding_requests,
        )? {
            return Ok(());
        }
    }
    command_sender
        .send(AcceptedCommand::Shutdown)
        .map_err(|_| LocalServerError::ChannelClosed("command receiver"))?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn dispatch_wire_command(
    command: WireCommand,
    command_sender: &Sender<AcceptedCommand>,
    event_sender: &SyncSender<WireEvent>,
    registry: &RequestRegistry,
    max_outstanding_requests: usize,
) -> Result<bool, LocalServerError> {
    match command {
        WireCommand::Generate {
            request_id,
            instance_id,
            state_transition,
            stream_tokens,
            input,
            tools,
            max_new_tokens,
            sampling,
        } => {
            if !admit_request_id(&request_id, event_sender)? {
                return Ok(false);
            }
            let Some(cancellation) = register_request(
                registry,
                &request_id,
                max_outstanding_requests,
                event_sender,
            )?
            else {
                return Ok(false);
            };
            let accepted = AcceptedCommand::Generate(AcceptedGenerate {
                request_id: request_id.clone(),
                instance_id,
                state_transition,
                stream_tokens,
                prompt: EchoChatPrompt { input, tools },
                max_new_tokens,
                sampling,
                cancellation,
                enqueued_at: Instant::now(),
            });
            if command_sender.send(accepted).is_err() {
                remove_registration(registry, &request_id);
                return Err(LocalServerError::ChannelClosed("command receiver"));
            }
        }
        WireCommand::Cancel { request_id } => {
            let accepted = cancel_request(registry, &request_id);
            send_event(
                event_sender,
                WireEvent::CancelAcknowledged {
                    request_id,
                    accepted,
                },
            )?;
        }
        WireCommand::OpenState {
            request_id,
            instance_id,
            persistence,
            snapshot_root,
        } => {
            if !admit_request_id(&request_id, event_sender)? {
                return Ok(false);
            }
            command_sender
                .send(AcceptedCommand::OpenState {
                    request_id,
                    instance_id,
                    persistence,
                    snapshot_root,
                })
                .map_err(|_| LocalServerError::ChannelClosed("command receiver"))?;
        }
        WireCommand::Snapshot {
            request_id,
            instance_id,
        } => {
            if !admit_request_id(&request_id, event_sender)? {
                return Ok(false);
            }
            command_sender
                .send(AcceptedCommand::Snapshot {
                    request_id,
                    instance_id,
                })
                .map_err(|_| LocalServerError::ChannelClosed("command receiver"))?;
        }
        WireCommand::Shutdown => {
            command_sender
                .send(AcceptedCommand::Shutdown)
                .map_err(|_| LocalServerError::ChannelClosed("command receiver"))?;
            return Ok(true);
        }
    }
    Ok(false)
}

fn register_request(
    registry: &RequestRegistry,
    request_id: &str,
    max_outstanding_requests: usize,
    event_sender: &SyncSender<WireEvent>,
) -> Result<Option<Arc<AtomicBool>>, LocalServerError> {
    let mut requests = registry
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if requests.contains_key(request_id) {
        send_event(
            event_sender,
            WireEvent::Failed {
                request_id: Some(request_id.into()),
                phase: "admission",
                error: "request_id is already active or waiting".into(),
            },
        )?;
        return Ok(None);
    }
    if requests.len() >= max_outstanding_requests {
        send_event(
            event_sender,
            WireEvent::Failed {
                request_id: Some(request_id.into()),
                phase: "admission",
                error: format!("outstanding generation limit {max_outstanding_requests} reached"),
            },
        )?;
        return Ok(None);
    }
    let cancellation = Arc::new(AtomicBool::new(false));
    requests.insert(
        request_id.into(),
        RequestRegistration {
            cancellation: Arc::clone(&cancellation),
        },
    );
    send_event(
        event_sender,
        WireEvent::Queued {
            request_id: request_id.into(),
            outstanding_requests: requests.len(),
        },
    )?;
    Ok(Some(cancellation))
}

fn cancel_request(registry: &RequestRegistry, request_id: &str) -> bool {
    let requests = registry
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    requests
        .get(request_id)
        .is_some_and(|registration| !registration.cancellation.swap(true, Ordering::AcqRel))
}

fn remove_registration(registry: &RequestRegistry, request_id: &str) {
    registry
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(request_id);
}

fn run_command_loop(
    engine: &mut ResidentEngine,
    tokenizer: &Qwen35ChatTokenizer,
    commands: &Receiver<AcceptedCommand>,
    events: &SyncSender<WireEvent>,
    registry: &RequestRegistry,
    max_active_batch_size: usize,
    max_late_join_batch_size: usize,
) -> Result<(), LocalServerError> {
    let mut deferred = VecDeque::new();
    loop {
        let command = if let Some(command) = deferred.pop_front() {
            command
        } else {
            let Ok(command) = commands.recv() else {
                return Ok(());
            };
            command
        };
        match command {
            AcceptedCommand::Generate(first) => {
                let mut cohort = Vec::with_capacity(max_active_batch_size);
                cohort.push(first);
                while cohort.len() < max_active_batch_size {
                    let Some(generate) = try_take_ready_generate(commands, &mut deferred) else {
                        break;
                    };
                    cohort.push(generate);
                }
                run_generate_cohort(
                    engine,
                    tokenizer,
                    commands,
                    events,
                    cohort,
                    registry,
                    &mut deferred,
                    max_active_batch_size,
                    max_late_join_batch_size,
                )?;
            }
            AcceptedCommand::OpenState {
                request_id,
                instance_id,
                persistence,
                snapshot_root,
            } => {
                run_open_state(
                    engine,
                    events,
                    request_id,
                    instance_id,
                    persistence,
                    snapshot_root.as_deref(),
                )?;
            }
            AcceptedCommand::Snapshot {
                request_id,
                instance_id,
            } => {
                run_snapshot(engine, events, request_id, &instance_id)?;
            }
            AcceptedCommand::Shutdown => return Ok(()),
        }
    }
}

fn try_take_ready_generate(
    commands: &Receiver<AcceptedCommand>,
    deferred: &mut VecDeque<AcceptedCommand>,
) -> Option<AcceptedGenerate> {
    if !deferred.is_empty() {
        return None;
    }
    match commands.try_recv() {
        Ok(AcceptedCommand::Generate(generate)) => Some(generate),
        Ok(barrier) => {
            deferred.push_back(barrier);
            None
        }
        Err(TryRecvError::Empty | TryRecvError::Disconnected) => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_generate_cohort(
    engine: &mut ResidentEngine,
    tokenizer: &Qwen35ChatTokenizer,
    commands: &Receiver<AcceptedCommand>,
    events: &SyncSender<WireEvent>,
    cohort: Vec<AcceptedGenerate>,
    registry: &RequestRegistry,
    deferred: &mut VecDeque<AcceptedCommand>,
    max_active_batch_size: usize,
    max_late_join_batch_size: usize,
) -> Result<(), LocalServerError> {
    let mut coordinator = StdioBatchGenerationCoordinator {
        tokenizer,
        commands,
        events,
        registry,
        deferred,
        rows: Vec::with_capacity(max_active_batch_size),
    };
    let initial = coordinator.prepare_cohort(cohort)?;
    if initial.is_empty() {
        return Ok(());
    }
    engine
        .execute_continuous_batch_observed(
            initial,
            max_active_batch_size,
            max_late_join_batch_size,
            &mut coordinator,
        )
        .map_err(Into::into)
}

fn send_generation_outcome(
    events: &SyncSender<WireEvent>,
    tokenizer: &Qwen35ChatTokenizer,
    request_id: &str,
    outcome: &Result<InferenceResponse, RuntimeError>,
) -> Result<(), LocalServerError> {
    match outcome {
        Ok(response) => {
            let text_token_count = response
                .generated_tokens
                .iter()
                .position(|token| *token == tokenizer.eos_token_id())
                .unwrap_or(response.generated_tokens.len());
            let text = tokenizer.decode(&response.generated_tokens[..text_token_count])?;
            let parsed = parse_qwen_output(&text, request_id);
            send_event(
                events,
                WireEvent::Completed {
                    request_id: request_id.into(),
                    response: Box::new(response.clone()),
                    text,
                    output: parsed.output,
                    tool_parse_warning: parsed.warning,
                },
            )
        }
        Err(RuntimeError::Cancelled { .. }) => send_event(
            events,
            WireEvent::Cancelled {
                request_id: request_id.into(),
            },
        ),
        Err(error) => send_request_failure(events, request_id, "inference", &error),
    }
}

fn encode_generate_input(
    tokenizer: &Qwen35ChatTokenizer,
    state_transition: RequestState,
    prompt: &EchoChatPrompt,
) -> Result<EncodedChatPrompt, LocalServerError> {
    match state_transition {
        RequestState::Initial | RequestState::NewSession => {
            tokenizer.encode_prompt(prompt).map_err(Into::into)
        }
        RequestState::Continuation => tokenizer.encode_continuation(prompt).map_err(Into::into),
    }
}

fn run_open_state(
    engine: &mut ResidentEngine,
    events: &SyncSender<WireEvent>,
    request_id: String,
    instance_id: InstanceId,
    persistence: StatePersistence,
    snapshot_root: Option<&Path>,
) -> Result<(), LocalServerError> {
    let opened = match (persistence, snapshot_root) {
        (StatePersistence::Durable, Some(snapshot_root)) => {
            engine.open_state(instance_id, snapshot_root)
        }
        (StatePersistence::Ephemeral, None) => engine.open_ephemeral_state(instance_id),
        (StatePersistence::Durable, None) => Err(RuntimeError::InvalidRequest {
            detail: "durable state requires snapshot_root".into(),
        }),
        (StatePersistence::Ephemeral, Some(_)) => Err(RuntimeError::InvalidRequest {
            detail: "ephemeral state must not specify snapshot_root".into(),
        }),
    };
    match opened {
        Ok(opened) => send_event(
            events,
            WireEvent::StateOpened {
                request_id,
                instance_id: opened.instance_id,
                persistence: opened.persistence,
                restored: opened.restored,
                current_path: opened.current_path,
            },
        ),
        Err(error) => send_request_failure(events, &request_id, "open_state", &error),
    }
}

fn run_snapshot(
    engine: &ResidentEngine,
    events: &SyncSender<WireEvent>,
    request_id: String,
    instance_id: &InstanceId,
) -> Result<(), LocalServerError> {
    match engine.publish_snapshot(instance_id) {
        Ok(published) => send_event(
            events,
            WireEvent::SnapshotPublished {
                request_id,
                instance_id: published.instance_id,
                path: published.path,
                physical_nbytes: published.physical_nbytes,
            },
        ),
        Err(error) => send_request_failure(events, &request_id, "snapshot", &error),
    }
}

struct StdioGenerationObserver<'a> {
    request_id: String,
    cancellation: Arc<AtomicBool>,
    eos_token_id: u32,
    decoder: Option<Qwen35DecodeStream<'a>>,
    events: &'a SyncSender<WireEvent>,
    token_index: usize,
    finished: bool,
}

impl<'a> StdioGenerationObserver<'a> {
    fn new(
        request_id: String,
        cancellation: Arc<AtomicBool>,
        eos_token_id: u32,
        decoder: Option<Qwen35DecodeStream<'a>>,
        events: &'a SyncSender<WireEvent>,
    ) -> Self {
        Self {
            request_id,
            cancellation,
            eos_token_id,
            decoder,
            events,
            token_index: 0,
            finished: false,
        }
    }
}

impl GenerationObserver for StdioGenerationObserver<'_> {
    fn is_cancelled(&self) -> bool {
        self.cancellation.load(Ordering::Acquire)
    }

    fn on_token(&mut self, token: u32) -> Result<GenerationDirective, String> {
        let terminal = token == self.eos_token_id;
        if let Some(decoder) = &mut self.decoder {
            let text = if terminal {
                None
            } else {
                decoder.step(token).map_err(|error| error.to_string())?
            };
            send_event(
                self.events,
                WireEvent::Token {
                    request_id: self.request_id.clone(),
                    index: self.token_index,
                    token_id: token,
                    text,
                    terminal,
                },
            )
            .map_err(|error| error.to_string())?;
            self.token_index += 1;
        }
        if terminal {
            Ok(GenerationDirective::Stop)
        } else {
            Ok(GenerationDirective::Continue)
        }
    }
}

struct StdioBatchGenerationCoordinator<'a> {
    tokenizer: &'a Qwen35ChatTokenizer,
    commands: &'a Receiver<AcceptedCommand>,
    events: &'a SyncSender<WireEvent>,
    registry: &'a RequestRegistry,
    deferred: &'a mut VecDeque<AcceptedCommand>,
    rows: Vec<StdioGenerationObserver<'a>>,
}

impl StdioBatchGenerationCoordinator<'_> {
    fn prepare_cohort(
        &mut self,
        cohort: Vec<AcceptedGenerate>,
    ) -> Result<Vec<BatchAdmission>, LocalServerError> {
        let mut admissions = Vec::with_capacity(cohort.len());
        for generate in cohort {
            if let Some(admission) = self.prepare_generate(generate)? {
                admissions.push(admission);
            }
        }
        Ok(admissions)
    }

    fn prepare_generate(
        &mut self,
        generate: AcceptedGenerate,
    ) -> Result<Option<BatchAdmission>, LocalServerError> {
        let AcceptedGenerate {
            request_id,
            instance_id,
            state_transition,
            stream_tokens,
            prompt,
            max_new_tokens,
            sampling,
            cancellation,
            enqueued_at,
        } = generate;
        let encoded = match encode_generate_input(self.tokenizer, state_transition, &prompt) {
            Ok(encoded) => encoded,
            Err(error) => {
                remove_registration(self.registry, &request_id);
                send_request_failure(self.events, &request_id, "chat", &error)?;
                return Ok(None);
            }
        };
        send_event(
            self.events,
            WireEvent::Started {
                request_id: request_id.clone(),
                prompt_tokens: encoded.token_ids.len(),
            },
        )?;
        self.rows.push(StdioGenerationObserver::new(
            request_id,
            cancellation,
            self.tokenizer.eos_token_id(),
            stream_tokens.then(|| self.tokenizer.decode_stream()),
            self.events,
        ));
        Ok(Some(BatchAdmission {
            request: InferenceRequest {
                instance_id,
                state_transition,
                input_tokens: encoded.token_ids,
                max_new_tokens,
                length_eos_token: Some(self.tokenizer.eos_token_id()),
                sampling,
            },
            queue_wait: enqueued_at.elapsed(),
        }))
    }
}

impl BatchGenerationObserver for StdioBatchGenerationCoordinator<'_> {
    fn is_cancelled(&self, request_index: usize) -> bool {
        self.rows
            .get(request_index)
            .is_none_or(GenerationObserver::is_cancelled)
    }

    fn on_token(
        &mut self,
        request_index: usize,
        token: u32,
    ) -> Result<GenerationDirective, String> {
        self.rows
            .get_mut(request_index)
            .ok_or_else(|| format!("missing stdio observer for batch row {request_index}"))?
            .on_token(token)
    }

    fn take_ready(
        &mut self,
        first_request_index: usize,
        capacity: usize,
    ) -> Result<Vec<BatchAdmission>, String> {
        if self.rows.len() != first_request_index {
            return Err(format!(
                "stdio/runtime request index drift: observers={}, next={first_request_index}",
                self.rows.len()
            ));
        }
        let mut admissions = Vec::with_capacity(capacity);
        while admissions.len() < capacity {
            let Some(generate) = try_take_ready_generate(self.commands, self.deferred) else {
                break;
            };
            if let Some(admission) = self
                .prepare_generate(generate)
                .map_err(|error| error.to_string())?
            {
                admissions.push(admission);
            }
        }
        Ok(admissions)
    }

    fn on_outcome(
        &mut self,
        request_index: usize,
        outcome: &Result<InferenceResponse, RuntimeError>,
    ) -> Result<(), String> {
        let row = self
            .rows
            .get_mut(request_index)
            .ok_or_else(|| format!("missing stdio observer for outcome row {request_index}"))?;
        if row.finished {
            return Err(format!(
                "stdio outcome for request {} was delivered more than once",
                row.request_id
            ));
        }
        let request_id = row.request_id.clone();
        row.finished = true;
        remove_registration(self.registry, &request_id);
        send_generation_outcome(self.events, self.tokenizer, &request_id, outcome)
            .map_err(|error| error.to_string())
    }
}

fn send_request_failure(
    events: &SyncSender<WireEvent>,
    request_id: &str,
    phase: &'static str,
    error: &dyn fmt::Display,
) -> Result<(), LocalServerError> {
    send_event(
        events,
        WireEvent::Failed {
            request_id: Some(request_id.into()),
            phase,
            error: error.to_string(),
        },
    )
}

fn send_event(sender: &SyncSender<WireEvent>, event: WireEvent) -> Result<(), LocalServerError> {
    sender
        .send(event)
        .map_err(|_| LocalServerError::ChannelClosed("event writer"))
}

fn write_events(receiver: &Receiver<WireEvent>) -> Result<(), LocalServerError> {
    let output = io::stdout();
    let mut output = BufWriter::new(output.lock());
    while let Ok(event) = receiver.recv() {
        serde_json::to_writer(&mut output, &event).map_err(LocalServerError::EventJson)?;
        output.write_all(b"\n").map_err(LocalServerError::Stdio)?;
        output.flush().map_err(LocalServerError::Stdio)?;
    }
    Ok(())
}

fn validate_request_id(request_id: &str) -> Result<(), LocalServerError> {
    if request_id.is_empty() || request_id.len() > 128 {
        return Err(LocalServerError::InvalidRequestId);
    }
    if !request_id
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':'))
    {
        return Err(LocalServerError::InvalidRequestId);
    }
    Ok(())
}

fn admit_request_id(
    request_id: &str,
    events: &SyncSender<WireEvent>,
) -> Result<bool, LocalServerError> {
    match validate_request_id(request_id) {
        Ok(()) => Ok(true),
        Err(error) => {
            send_event(
                events,
                WireEvent::Failed {
                    request_id: Some(request_id.into()),
                    phase: "admission",
                    error: error.to_string(),
                },
            )?;
            Ok(false)
        }
    }
}

fn join_protocol_thread(
    handle: thread::JoinHandle<Result<(), LocalServerError>>,
    name: &'static str,
) -> Result<(), LocalServerError> {
    handle
        .join()
        .map_err(|_| LocalServerError::ThreadPanicked(name))?
}

/// Local composition-server failure.
#[derive(Debug)]
pub enum LocalServerError {
    /// Server limit is invalid.
    InvalidConfiguration(String),
    /// Model/tokenizer chat admission failed.
    Chat(ChatError),
    /// Resident model admission failed.
    Runtime(RuntimeError),
    /// stdin/stdout operation failed.
    Stdio(std::io::Error),
    /// Event serialization failed.
    EventJson(serde_json::Error),
    /// A protocol thread could not be spawned.
    ThreadSpawn(std::io::Error),
    /// A protocol thread panicked.
    ThreadPanicked(&'static str),
    /// Main or writer channel disappeared.
    ChannelClosed(&'static str),
    /// Request identity is empty, too long, or unsafe.
    InvalidRequestId,
}

impl fmt::Display for LocalServerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(detail) => formatter.write_str(detail),
            Self::Chat(error) => error.fmt(formatter),
            Self::Runtime(error) => error.fmt(formatter),
            Self::Stdio(error) | Self::ThreadSpawn(error) => error.fmt(formatter),
            Self::EventJson(error) => error.fmt(formatter),
            Self::ThreadPanicked(name) => write!(formatter, "{name} protocol thread panicked"),
            Self::ChannelClosed(name) => write!(formatter, "{name} channel closed"),
            Self::InvalidRequestId => formatter.write_str(
                "request_id must be 1-128 ASCII alphanumeric, underscore, dash, dot, or colon characters",
            ),
        }
    }
}

impl Error for LocalServerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Chat(error) => Some(error),
            Self::Runtime(error) => Some(error),
            Self::Stdio(error) | Self::ThreadSpawn(error) => Some(error),
            Self::EventJson(error) => Some(error),
            Self::InvalidConfiguration(_)
            | Self::ThreadPanicked(_)
            | Self::ChannelClosed(_)
            | Self::InvalidRequestId => None,
        }
    }
}

impl From<ChatError> for LocalServerError {
    fn from(value: ChatError) -> Self {
        Self::Chat(value)
    }
}

impl From<RuntimeError> for LocalServerError {
    fn from(value: RuntimeError) -> Self {
        Self::Runtime(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accepted_generate(request_id: &str) -> AcceptedCommand {
        AcceptedCommand::Generate(AcceptedGenerate {
            request_id: request_id.into(),
            instance_id: InstanceId::new(format!("instance-{request_id}")).expect("valid instance"),
            state_transition: RequestState::Initial,
            stream_tokens: false,
            prompt: EchoChatPrompt {
                input: vec![EchoInputItem::Message(super::super::chat::EchoMessage {
                    role: super::super::chat::EchoMessageRole::Developer,
                    content: super::super::chat::EchoMessageContent::Text("test".into()),
                })],
                tools: Vec::new(),
            },
            max_new_tokens: 1,
            sampling: SamplingConfig::default(),
            cancellation: Arc::new(AtomicBool::new(false)),
            enqueued_at: Instant::now(),
        })
    }

    #[test]
    fn request_ids_are_bounded_and_protocol_safe() {
        for valid in ["rin:1", "request_2", "abc.def-3"] {
            validate_request_id(valid).expect("valid request ID");
        }
        for invalid in ["", "has space", "slash/path", &"a".repeat(129)] {
            assert!(matches!(
                validate_request_id(invalid),
                Err(LocalServerError::InvalidRequestId)
            ));
        }
    }

    #[test]
    fn cancellation_is_visible_without_waiting_for_main_execution() {
        let registry = RequestRegistry::default();
        registry.lock().expect("registry").insert(
            "rin:1".into(),
            RequestRegistration {
                cancellation: Arc::new(AtomicBool::new(false)),
            },
        );
        assert!(cancel_request(&registry, "rin:1"));
        assert!(!cancel_request(&registry, "rin:1"));
        assert!(!cancel_request(&registry, "missing"));
    }

    #[test]
    fn lifecycle_barrier_prevents_later_generation_from_overtaking() {
        let (sender, receiver) = mpsc::channel();
        sender
            .send(accepted_generate("before"))
            .expect("queue before");
        sender
            .send(AcceptedCommand::Snapshot {
                request_id: "snapshot".into(),
                instance_id: InstanceId::new("instance-before").expect("valid instance"),
            })
            .expect("queue barrier");
        sender
            .send(accepted_generate("after"))
            .expect("queue after");
        let mut deferred = VecDeque::new();

        let before = try_take_ready_generate(&receiver, &mut deferred).expect("before barrier");
        assert_eq!(before.request_id, "before");
        assert!(try_take_ready_generate(&receiver, &mut deferred).is_none());
        assert!(matches!(
            deferred.front(),
            Some(AcceptedCommand::Snapshot { request_id, .. }) if request_id == "snapshot"
        ));
        assert!(try_take_ready_generate(&receiver, &mut deferred).is_none());

        deferred.pop_front();
        let after = try_take_ready_generate(&receiver, &mut deferred).expect("after barrier");
        assert_eq!(after.request_id, "after");
    }

    #[test]
    fn late_join_limit_cannot_exceed_active_batch_limit() {
        let error = validate_config(LocalServerConfig {
            max_active_batch_size: 3,
            max_late_join_batch_size: 4,
            ..LocalServerConfig::default()
        })
        .expect_err("late join wider than active batch must fail");
        assert!(error.to_string().contains("within 1..=3"));
    }

    #[test]
    fn wire_generate_matches_echo_model_request_shape() {
        let command: WireCommand = serde_json::from_str(
            r#"{
              "type": "generate",
              "request_id": "rin:1",
              "instance_id": "rin",
              "state_transition": "initial",
              "stream_tokens": true,
              "input": [{"role": "developer", "content": "continue"}],
              "tools": [],
              "max_new_tokens": 2
            }"#,
        )
        .expect("valid generate command");
        assert!(matches!(
            command,
            WireCommand::Generate {
                stream_tokens: true,
                ..
            }
        ));
    }

    #[test]
    fn wire_generate_can_disable_provisional_token_events() {
        let command: WireCommand = serde_json::from_str(
            r#"{
              "type": "generate",
              "request_id": "rin:no-stream",
              "instance_id": "rin",
              "state_transition": "initial",
              "stream_tokens": false,
              "input": [{"role": "developer", "content": "continue"}],
              "tools": [],
              "max_new_tokens": 2
            }"#,
        )
        .expect("valid non-streaming generate command");
        assert!(matches!(
            command,
            WireCommand::Generate {
                stream_tokens: false,
                ..
            }
        ));
    }

    #[test]
    fn wire_generate_requires_an_explicit_token_stream_policy() {
        let error = serde_json::from_str::<WireCommand>(
            r#"{
              "type": "generate",
              "request_id": "rin:missing-stream-policy",
              "instance_id": "rin",
              "state_transition": "initial",
              "input": [{"role": "developer", "content": "continue"}],
              "tools": [],
              "max_new_tokens": 2
            }"#,
        )
        .expect_err("protocol requires an explicit token stream policy");
        assert!(error.to_string().contains("stream_tokens"));
    }

    #[test]
    fn generate_rejects_the_removed_caller_owned_prefix() {
        let error = serde_json::from_str::<WireCommand>(
            r#"{
              "type": "generate",
              "request_id": "rin:2",
              "instance_id": "rin",
              "state_transition": "continuation",
              "stream_tokens": false,
              "input": [],
              "tools": [],
              "prefix_lineage_tokens": [1, 2, 3],
              "max_new_tokens": 2
            }"#,
        )
        .expect_err("protocol must reject caller-owned prefix tokens");
        assert!(error.to_string().contains("prefix_lineage_tokens"));
    }

    #[test]
    fn wire_new_session_carries_only_the_state_transition() {
        let command: WireCommand = serde_json::from_str(
            r#"{
              "type": "generate",
              "request_id": "rin:2",
              "instance_id": "rin",
              "state_transition": "new_session",
              "stream_tokens": false,
              "input": [{"role": "developer", "content": "fresh"}],
              "tools": [],
              "max_new_tokens": 2
            }"#,
        )
        .expect("valid new-session command");
        assert!(matches!(
            command,
            WireCommand::Generate {
                state_transition: RequestState::NewSession,
                ..
            }
        ));
    }

    #[test]
    fn wire_open_state_binds_instance_and_directory() {
        let command: WireCommand = serde_json::from_str(
            r#"{
              "type": "open_state",
              "request_id": "rin:startup",
              "instance_id": "rin",
              "persistence": "durable",
              "snapshot_root": "/state/rin"
            }"#,
        )
        .expect("valid current restore command");
        assert!(matches!(
            command,
            WireCommand::OpenState {
                request_id,
                instance_id,
                persistence: StatePersistence::Durable,
                snapshot_root,
            } if request_id == "rin:startup"
                && instance_id.as_str() == "rin"
                && snapshot_root.as_deref() == Some(Path::new("/state/rin"))
        ));
    }

    #[test]
    fn wire_ephemeral_state_has_no_snapshot_root() {
        let command: WireCommand = serde_json::from_str(
            r#"{
              "type": "open_state",
              "request_id": "rin:memory:startup",
              "instance_id": "rin.memory",
              "persistence": "ephemeral"
            }"#,
        )
        .expect("valid ephemeral state command");
        assert!(matches!(
            command,
            WireCommand::OpenState {
                persistence: StatePersistence::Ephemeral,
                snapshot_root: None,
                ..
            }
        ));
    }

    #[test]
    fn restore_rejects_external_token_or_prompt_reconstruction() {
        for extra in [
            r#", "lineage_tokens": [1]"#,
            r#", "input": [], "tools": []"#,
        ] {
            let command = format!(
                r#"{{"type":"open_state","request_id":"rin:restore","instance_id":"rin","persistence":"durable","snapshot_root":"/state/rin"{extra}}}"#
            );
            assert!(serde_json::from_str::<WireCommand>(&command).is_err());
        }
    }
}
