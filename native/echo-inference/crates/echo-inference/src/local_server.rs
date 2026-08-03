use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread;

use echo_inference_state::InstanceId;
use serde::{Deserialize, Serialize};

use super::chat::{
    ChatError, EchoChatPrompt, EchoInputItem, EchoToolContract, EncodedChatPrompt,
    Qwen35ChatTokenizer, Qwen35DecodeStream,
};
use super::runtime::{
    GenerationDirective, GenerationObserver, InferenceRequest, InferenceResponse, RequestState,
    ResidentEngine, ResidentEngineConfig, ResidentEngineInfo, RuntimeError,
};
use super::sampling::SamplingConfig;
use super::tool_output::{EchoOutputItem, parse_qwen_output};

const PROTOCOL_VERSION: u32 = 7;

/// Admission and backpressure limits for the dedicated local stdio server.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalServerConfig {
    /// Active plus waiting generate requests admitted at once.
    pub max_outstanding_requests: usize,
    /// Number of serialized events buffered before generation backpressures.
    pub event_buffer_capacity: usize,
    /// Resident-engine generation limits.
    pub engine: ResidentEngineConfig,
}

impl Default for LocalServerConfig {
    fn default() -> Self {
        Self {
            max_outstanding_requests: 8,
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
        snapshot_root: PathBuf,
    },
    Snapshot {
        request_id: String,
        instance_id: InstanceId,
    },
    Shutdown,
}

enum AcceptedCommand {
    Generate {
        request_id: String,
        instance_id: InstanceId,
        state_transition: RequestState,
        stream_tokens: bool,
        prompt: EchoChatPrompt,
        max_new_tokens: usize,
        sampling: SamplingConfig,
        cancellation: Arc<AtomicBool>,
    },
    OpenState {
        request_id: String,
        instance_id: InstanceId,
        snapshot_root: PathBuf,
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
        restored: bool,
        current_path: PathBuf,
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
            let accepted = AcceptedCommand::Generate {
                request_id: request_id.clone(),
                instance_id,
                state_transition,
                stream_tokens,
                prompt: EchoChatPrompt { input, tools },
                max_new_tokens,
                sampling,
                cancellation,
            };
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
            snapshot_root,
        } => {
            if !admit_request_id(&request_id, event_sender)? {
                return Ok(false);
            }
            command_sender
                .send(AcceptedCommand::OpenState {
                    request_id,
                    instance_id,
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
) -> Result<(), LocalServerError> {
    while let Ok(command) = commands.recv() {
        match command {
            AcceptedCommand::Generate {
                request_id,
                instance_id,
                state_transition,
                stream_tokens,
                prompt,
                max_new_tokens,
                sampling,
                cancellation,
            } => {
                run_generate(
                    engine,
                    tokenizer,
                    events,
                    &request_id,
                    instance_id,
                    state_transition,
                    stream_tokens,
                    &prompt,
                    max_new_tokens,
                    sampling,
                    cancellation,
                    registry,
                )?;
            }
            AcceptedCommand::OpenState {
                request_id,
                instance_id,
                snapshot_root,
            } => {
                run_open_state(engine, events, request_id, instance_id, &snapshot_root)?;
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
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_generate(
    engine: &mut ResidentEngine,
    tokenizer: &Qwen35ChatTokenizer,
    events: &SyncSender<WireEvent>,
    request_id: &str,
    instance_id: InstanceId,
    state_transition: RequestState,
    stream_tokens: bool,
    prompt: &EchoChatPrompt,
    max_new_tokens: usize,
    sampling: SamplingConfig,
    cancellation: Arc<AtomicBool>,
    registry: &RequestRegistry,
) -> Result<(), LocalServerError> {
    if !engine.has_state_owner(&instance_id) {
        remove_registration(registry, request_id);
        return send_request_failure(
            events,
            request_id,
            "state",
            &format_args!(
                "instance {} must be opened before generation",
                instance_id.as_str()
            ),
        );
    }
    let encoded =
        match encode_generate_input(engine, tokenizer, &instance_id, state_transition, prompt) {
            Ok(encoded) => encoded,
            Err(error) => {
                remove_registration(registry, request_id);
                return send_request_failure(events, request_id, "chat", &error);
            }
        };
    send_event(
        events,
        WireEvent::Started {
            request_id: request_id.into(),
            prompt_tokens: encoded.token_ids.len(),
        },
    )?;
    let request = InferenceRequest {
        instance_id,
        state_transition,
        input_tokens: encoded.token_ids,
        max_new_tokens,
        length_eos_token: Some(tokenizer.eos_token_id()),
        sampling,
    };
    let mut observer = StdioGenerationObserver::new(
        request_id,
        cancellation,
        tokenizer.eos_token_id(),
        stream_tokens.then(|| tokenizer.decode_stream()),
        events,
    );
    let result = engine.execute_observed(request, &mut observer);
    drop(observer);
    remove_registration(registry, request_id);

    match result {
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
                    response: Box::new(response),
                    text,
                    output: parsed.output,
                    tool_parse_warning: parsed.warning,
                },
            )?;
        }
        Err(RuntimeError::Cancelled { .. }) => {
            send_event(
                events,
                WireEvent::Cancelled {
                    request_id: request_id.into(),
                },
            )?;
        }
        Err(error) => return send_request_failure(events, request_id, "inference", &error),
    }
    Ok(())
}

fn encode_generate_input(
    engine: &ResidentEngine,
    tokenizer: &Qwen35ChatTokenizer,
    instance_id: &InstanceId,
    state_transition: RequestState,
    prompt: &EchoChatPrompt,
) -> Result<EncodedChatPrompt, LocalServerError> {
    match state_transition {
        RequestState::Initial | RequestState::NewSession => {
            tokenizer.encode_prompt(prompt).map_err(Into::into)
        }
        RequestState::Continuation => {
            engine
                .current_state(instance_id)
                .ok_or_else(|| RuntimeError::InvalidRequest {
                    detail: format!(
                        "instance {} has no resident state for continuation",
                        instance_id.as_str()
                    ),
                })?;
            tokenizer.encode_continuation(prompt).map_err(Into::into)
        }
    }
}

fn run_open_state(
    engine: &mut ResidentEngine,
    events: &SyncSender<WireEvent>,
    request_id: String,
    instance_id: InstanceId,
    snapshot_root: &Path,
) -> Result<(), LocalServerError> {
    match engine.open_state(instance_id, snapshot_root) {
        Ok(opened) => send_event(
            events,
            WireEvent::StateOpened {
                request_id,
                instance_id: opened.instance_id,
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
    request_id: &'a str,
    cancellation: Arc<AtomicBool>,
    eos_token_id: u32,
    decoder: Option<Qwen35DecodeStream<'a>>,
    events: &'a SyncSender<WireEvent>,
    token_index: usize,
}

impl<'a> StdioGenerationObserver<'a> {
    fn new(
        request_id: &'a str,
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
                    request_id: self.request_id.into(),
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
        .expect_err("protocol v7 requires an explicit token stream policy");
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
        .expect_err("protocol v7 must reject caller-owned prefix tokens");
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
              "snapshot_root": "/state/rin"
            }"#,
        )
        .expect("valid current restore command");
        assert!(matches!(
            command,
            WireCommand::OpenState {
                request_id,
                instance_id,
                snapshot_root,
            } if request_id == "rin:startup"
                && instance_id.as_str() == "rin"
                && snapshot_root == Path::new("/state/rin")
        ));
    }

    #[test]
    fn restore_rejects_external_token_or_prompt_reconstruction() {
        for extra in [
            r#", "lineage_tokens": [1]"#,
            r#", "input": [], "tools": []"#,
        ] {
            let command = format!(
                r#"{{"type":"open_state","request_id":"rin:restore","instance_id":"rin","snapshot_root":"/state/rin"{extra}}}"#
            );
            assert!(serde_json::from_str::<WireCommand>(&command).is_err());
        }
    }
}
