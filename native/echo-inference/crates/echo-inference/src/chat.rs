use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::ModelWrapper;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::tokenizer::DecodeStream;

const SUPPORTED_TEMPLATE_SHA256: &str =
    "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259";
const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const NON_THINKING_GENERATION_PROMPT: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
const TOOL_PREAMBLE: &str =
    "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>";
const TOOL_INSTRUCTIONS: &str = r"
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>";

/// E.C.H.O. message role accepted by the local Qwen composition boundary.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EchoMessageRole {
    /// Leading system instruction.
    System,
    /// Provider-neutral instruction mapped to `user`, matching the current
    /// E.C.H.O. Chat Completions adapter.
    Developer,
    /// User message.
    User,
    /// Assistant message retained in history.
    Assistant,
}

/// One text or image content part from E.C.H.O.'s model port.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EchoContentPart {
    /// Plain text content.
    Text {
        /// Text appended exactly at this position.
        text: String,
    },
    /// Vision content, deliberately rejected until the native vision encoder
    /// is load-bearing.
    Image {
        /// URL retained only for a precise admission error.
        image_url: String,
        /// Provider-neutral detail hint.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
}

/// String or ordered content-parts representation used by E.C.H.O.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(untagged)]
pub enum EchoMessageContent {
    /// Existing text-only path.
    Text(String),
    /// Ordered multimodal parts.
    Parts(Vec<EchoContentPart>),
}

/// Provider-neutral conversation message.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EchoMessage {
    /// Message role.
    pub role: EchoMessageRole,
    /// Message content.
    pub content: EchoMessageContent,
}

/// Tool call already present in E.C.H.O. input history.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EchoToolCall {
    /// Stable call identity used by E.C.H.O.; Qwen's template does not render it.
    pub call_id: String,
    /// Function name.
    pub tool_name: String,
    /// Raw JSON object passed to the function.
    pub input: String,
}

/// Tool result already present in E.C.H.O. input history.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EchoToolResult {
    /// Stable call identity used by E.C.H.O.; Qwen's template does not render it.
    pub call_id: String,
    /// Sanitized tool output.
    pub output: String,
}

/// One item from E.C.H.O.'s provider-neutral `ModelRequest.input`.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(untagged)]
pub enum EchoInputItem {
    /// Conversation message.
    Message(EchoMessage),
    /// Assistant tool call.
    ToolCall {
        /// Fixed discriminator from the TypeScript model port.
        #[serde(rename = "type")]
        kind: ToolCallKind,
        /// Tool call fields.
        #[serde(flatten)]
        call: EchoToolCall,
    },
    /// Tool result.
    ToolResult {
        /// Fixed discriminator from the TypeScript model port.
        #[serde(rename = "type")]
        kind: ToolResultKind,
        /// Tool result fields.
        #[serde(flatten)]
        result: EchoToolResult,
    },
}

/// Exact `tool_call` discriminator.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallKind {
    /// Tool-call input item.
    ToolCall,
}

/// Exact `tool_result` discriminator.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolResultKind {
    /// Tool-result input item.
    ToolResult,
}

/// E.C.H.O. tool contract rendered into Qwen's function-tool preamble.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EchoToolContract {
    /// Function name.
    pub name: String,
    /// Function description.
    pub description: String,
    /// JSON Schema from the model port.
    pub input_schema: Value,
    /// Output schema is intentionally not sent to the model, matching the
    /// existing Chat Completions adapter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
    /// Strict-schema hint.
    #[serde(default)]
    pub strict: bool,
}

/// Complete history rendered for one Qwen generation boundary.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EchoChatPrompt {
    /// Provider-neutral input history.
    pub input: Vec<EchoInputItem>,
    /// Tools available for this generation.
    #[serde(default)]
    pub tools: Vec<EchoToolContract>,
}

/// Exact rendered prompt and its ordered token IDs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedChatPrompt {
    /// Qwen chat-template output, or only the newly appended fragment for an
    /// exact resident-state continuation.
    pub rendered: String,
    /// Newly executed token IDs fed to the resident runtime.
    pub token_ids: Vec<u32>,
}

/// Per-case exact chat-template and tokenization comparison.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ChatTemplateCaseParity {
    /// Oracle case name.
    pub name: String,
    /// Rust renderer exactly matches the official Jinja output.
    pub rendered_exact: bool,
    /// Official Rust tokenizer output exactly matches the Python tokenizer IDs.
    pub token_ids_exact: bool,
    /// Number of prompt tokens observed by Rust.
    pub token_count: usize,
}

/// Aggregate official-Python versus native-Rust chat-boundary comparison.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ChatTemplateParity {
    /// Fixture schema.
    pub schema_version: u32,
    /// Exact admitted template source digest.
    pub chat_template_sha256: String,
    /// Qwen EOS token ID.
    pub eos_token_id: u32,
    /// Case-level comparisons.
    pub cases: Vec<ChatTemplateCaseParity>,
    /// `true` only when every rendered byte and token ID matches.
    pub exact: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct TokenizerConfiguration {
    chat_template: String,
    eos_token: String,
}

#[derive(Clone, Debug, Deserialize)]
struct ChatTemplateFixture {
    schema_version: u32,
    chat_template_sha256: String,
    eos_token_id: u32,
    cases: Vec<ChatTemplateFixtureCase>,
}

#[derive(Clone, Debug, Deserialize)]
struct ChatTemplateFixtureCase {
    name: String,
    input: Vec<EchoInputItem>,
    tools: Vec<EchoToolContract>,
    rendered: String,
    token_ids: Vec<u32>,
}

/// Qwen3.5-family tokenizer plus the one admitted E.C.H.O. chat-template path.
pub struct Qwen35ChatTokenizer {
    tokenizer: Tokenizer,
    eos_token_id: u32,
    chat_template_sha256: String,
}

/// UTF-8-safe incremental decoder backed by the official tokenizer stream.
pub struct Qwen35DecodeStream<'a> {
    inner: DecodeStream<
        'a,
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
}

impl Qwen35DecodeStream<'_> {
    /// Decodes one token, returning `None` until enough bytes form a valid
    /// chunk.
    ///
    /// # Errors
    ///
    /// Returns [`ChatError`] when the official streaming decoder rejects the
    /// token sequence.
    pub fn step(&mut self, token: u32) -> Result<Option<String>, ChatError> {
        self.inner
            .step(token)
            .map_err(|source| ChatError::Tokenizer {
                detail: source.to_string(),
            })
    }
}

impl Qwen35ChatTokenizer {
    /// Loads `tokenizer.json`, verifies the exact admitted template, and
    /// resolves Qwen's end-of-message token.
    ///
    /// # Errors
    ///
    /// Returns [`ChatError`] when either file is unreadable, tokenizer loading
    /// fails, the template is not the oracle-proven target, or its EOS token is
    /// absent from the vocabulary.
    pub fn load(model_directory: &Path) -> Result<Self, ChatError> {
        let config_path = model_directory.join("tokenizer_config.json");
        let config_bytes = fs::read(&config_path).map_err(|source| ChatError::Io {
            path: config_path.clone(),
            source,
        })?;
        let configuration: TokenizerConfiguration =
            serde_json::from_slice(&config_bytes).map_err(|source| ChatError::Json {
                path: config_path,
                source,
            })?;
        let chat_template_sha256 = sha256_text(&configuration.chat_template);
        if chat_template_sha256 != SUPPORTED_TEMPLATE_SHA256 {
            return Err(ChatError::UnsupportedTemplate {
                observed_sha256: chat_template_sha256,
            });
        }

        let tokenizer_path = model_directory.join("tokenizer.json");
        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|source| ChatError::Tokenizer {
                detail: format!("{}: {source}", tokenizer_path.display()),
            })?;
        let eos_token_id = tokenizer
            .token_to_id(&configuration.eos_token)
            .ok_or_else(|| ChatError::Tokenizer {
                detail: format!(
                    "EOS token {:?} is absent from {}",
                    configuration.eos_token,
                    tokenizer_path.display()
                ),
            })?;
        Ok(Self {
            tokenizer,
            eos_token_id,
            chat_template_sha256: SUPPORTED_TEMPLATE_SHA256.into(),
        })
    }

    /// Renders and tokenizes one complete E.C.H.O. input history.
    ///
    /// `developer` messages become Qwen `user` messages, exactly matching the
    /// current Chat Completions adapter. Thinking is disabled, and images fail
    /// closed because the native runtime currently owns only the language path.
    ///
    /// # Errors
    ///
    /// Returns [`ChatError`] for a structurally invalid history, unsupported
    /// vision input, malformed tool arguments, or tokenizer failure.
    pub fn encode_prompt(&self, prompt: &EchoChatPrompt) -> Result<EncodedChatPrompt, ChatError> {
        let rendered = render_chat_prompt(prompt)?;
        let encoding = self
            .tokenizer
            .encode(rendered.as_str(), false)
            .map_err(|source| ChatError::Tokenizer {
                detail: source.to_string(),
            })?;
        Ok(EncodedChatPrompt {
            rendered,
            token_ids: encoding.get_ids().to_vec(),
        })
    }

    /// Encodes one E.C.H.O. tool-result continuation against resident state.
    ///
    /// This path never reconstructs the preceding assistant output from
    /// parsed values. The runtime guarantees that a successfully committed
    /// production turn ends in Qwen EOS, so the complete preceding token
    /// sequence does not need to be stored separately. The new input may
    /// contain only tool results (or be empty for the existing no-tool retry
    /// behavior). Ordinary user/developer input requires the separate
    /// new-session transition because Qwen may rewrite old thinking during a
    /// full template render.
    ///
    /// # Errors
    ///
    /// Returns [`ChatError`] when tools are redefined, the delta contains a
    /// non-tool-result item, or tokenization fails.
    pub fn encode_continuation(
        &self,
        prompt: &EchoChatPrompt,
    ) -> Result<EncodedChatPrompt, ChatError> {
        if !prompt.tools.is_empty() {
            return Err(ChatError::InvalidPrompt {
                detail: "exact continuation cannot redefine the committed tool catalog".into(),
            });
        }
        let rendered = render_chat_continuation(&prompt.input)?;
        let encoding = self
            .tokenizer
            .encode(rendered.as_str(), false)
            .map_err(|source| ChatError::Tokenizer {
                detail: source.to_string(),
            })?;
        Ok(EncodedChatPrompt {
            rendered,
            token_ids: encoding.get_ids().to_vec(),
        })
    }

    /// Decodes generated IDs without dropping Qwen special tokens.
    ///
    /// # Errors
    ///
    /// Returns [`ChatError`] when the tokenizer decoder rejects the sequence.
    pub fn decode(&self, token_ids: &[u32]) -> Result<String, ChatError> {
        self.tokenizer
            .decode(token_ids, false)
            .map_err(|source| ChatError::Tokenizer {
                detail: source.to_string(),
            })
    }

    /// Creates a UTF-8-safe incremental decoder that retains Qwen special
    /// tokens.
    #[must_use]
    pub fn decode_stream(&self) -> Qwen35DecodeStream<'_> {
        Qwen35DecodeStream {
            inner: self.tokenizer.decode_stream(false),
        }
    }

    /// Qwen end-of-message token that terminates chat generation.
    #[must_use]
    pub const fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// SHA-256 of the exact admitted Jinja source.
    #[must_use]
    pub fn chat_template_sha256(&self) -> &str {
        &self.chat_template_sha256
    }
}

/// Compares the native specialized renderer and official Rust tokenizer with
/// an official Python/Transformers oracle fixture.
///
/// # Errors
///
/// Returns [`ChatError`] when the fixture cannot be read, its schema or
/// template identity differs, or local tokenizer admission fails.
pub fn run_chat_template_parity(
    model_directory: &Path,
    fixture_path: &Path,
) -> Result<ChatTemplateParity, ChatError> {
    let bytes = fs::read(fixture_path).map_err(|source| ChatError::Io {
        path: fixture_path.to_path_buf(),
        source,
    })?;
    let fixture: ChatTemplateFixture =
        serde_json::from_slice(&bytes).map_err(|source| ChatError::Json {
            path: fixture_path.to_path_buf(),
            source,
        })?;
    if fixture.schema_version != 1 {
        return Err(ChatError::InvalidPrompt {
            detail: format!(
                "chat-template fixture schema must be 1, observed {}",
                fixture.schema_version
            ),
        });
    }
    let tokenizer = Qwen35ChatTokenizer::load(model_directory)?;
    if fixture.chat_template_sha256 != tokenizer.chat_template_sha256()
        || fixture.eos_token_id != tokenizer.eos_token_id()
    {
        return Err(ChatError::InvalidPrompt {
            detail: "chat-template fixture identity differs from the admitted tokenizer".into(),
        });
    }

    let cases = fixture
        .cases
        .into_iter()
        .map(|case| {
            let encoded = tokenizer.encode_prompt(&EchoChatPrompt {
                input: case.input,
                tools: case.tools,
            })?;
            Ok(ChatTemplateCaseParity {
                name: case.name,
                rendered_exact: encoded.rendered == case.rendered,
                token_ids_exact: encoded.token_ids == case.token_ids,
                token_count: encoded.token_ids.len(),
            })
        })
        .collect::<Result<Vec<_>, ChatError>>()?;
    let exact = cases
        .iter()
        .all(|case| case.rendered_exact && case.token_ids_exact);
    Ok(ChatTemplateParity {
        schema_version: fixture.schema_version,
        chat_template_sha256: fixture.chat_template_sha256,
        eos_token_id: fixture.eos_token_id,
        cases,
        exact,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TemplateRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Clone, Debug)]
struct TemplateToolCall {
    name: String,
    arguments: Map<String, Value>,
}

#[derive(Clone, Debug)]
struct TemplateMessage {
    role: TemplateRole,
    content: String,
    tool_calls: Vec<TemplateToolCall>,
}

fn render_chat_prompt(prompt: &EchoChatPrompt) -> Result<String, ChatError> {
    let messages = normalize_messages(&prompt.input)?;
    if messages.is_empty() {
        return Err(ChatError::InvalidPrompt {
            detail: "no messages provided".into(),
        });
    }
    let last_query_index = last_query_index(&messages)?;
    let mut output = String::new();
    render_system_preamble(&mut output, &messages, &prompt.tools)?;
    for (index, message) in messages.iter().enumerate() {
        render_message(&mut output, &messages, index, message, last_query_index)?;
    }
    output.push_str(NON_THINKING_GENERATION_PROMPT);
    Ok(output)
}

fn render_chat_continuation(input: &[EchoInputItem]) -> Result<String, ChatError> {
    let messages = input
        .iter()
        .map(|item| match item {
            EchoInputItem::ToolResult { result, .. } => Ok(TemplateMessage {
                role: TemplateRole::Tool,
                content: result.output.clone(),
                tool_calls: Vec::new(),
            }),
            EchoInputItem::Message(_) | EchoInputItem::ToolCall { .. } => {
                Err(ChatError::InvalidPrompt {
                    detail:
                        "exact continuation accepts only tool_result items within one E.C.H.O. session"
                            .into(),
                })
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut output = String::from("\n");
    for (index, message) in messages.iter().enumerate() {
        render_tool_result(&mut output, &messages, index, message);
    }
    output.push_str(NON_THINKING_GENERATION_PROMPT);
    Ok(output)
}

fn normalize_messages(input: &[EchoInputItem]) -> Result<Vec<TemplateMessage>, ChatError> {
    input
        .iter()
        .map(|item| match item {
            EchoInputItem::Message(message) => {
                let role = match message.role {
                    EchoMessageRole::System => TemplateRole::System,
                    EchoMessageRole::Developer | EchoMessageRole::User => TemplateRole::User,
                    EchoMessageRole::Assistant => TemplateRole::Assistant,
                };
                Ok(TemplateMessage {
                    role,
                    content: render_content(&message.content)?,
                    tool_calls: Vec::new(),
                })
            }
            EchoInputItem::ToolCall { call, .. } => {
                validate_xml_name(&call.tool_name, "tool name")?;
                let value: Value =
                    serde_json::from_str(&call.input).map_err(|source| ChatError::ToolInput {
                        call_id: call.call_id.clone(),
                        detail: source.to_string(),
                    })?;
                let arguments = value
                    .as_object()
                    .cloned()
                    .ok_or_else(|| ChatError::ToolInput {
                        call_id: call.call_id.clone(),
                        detail: "tool input must be a JSON object".into(),
                    })?;
                for name in arguments.keys() {
                    validate_xml_name(name, "tool argument name")?;
                }
                Ok(TemplateMessage {
                    role: TemplateRole::Assistant,
                    content: String::new(),
                    tool_calls: vec![TemplateToolCall {
                        name: call.tool_name.clone(),
                        arguments,
                    }],
                })
            }
            EchoInputItem::ToolResult { result, .. } => Ok(TemplateMessage {
                role: TemplateRole::Tool,
                content: result.output.clone(),
                tool_calls: Vec::new(),
            }),
        })
        .collect()
}

fn render_content(content: &EchoMessageContent) -> Result<String, ChatError> {
    match content {
        EchoMessageContent::Text(text) => Ok(text.clone()),
        EchoMessageContent::Parts(parts) => {
            let mut output = String::new();
            for part in parts {
                match part {
                    EchoContentPart::Text { text } => output.push_str(text),
                    EchoContentPart::Image { .. } => {
                        return Err(ChatError::UnsupportedVision);
                    }
                }
            }
            Ok(output)
        }
    }
}

fn last_query_index(messages: &[TemplateMessage]) -> Result<usize, ChatError> {
    messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| {
            if message.role != TemplateRole::User {
                return None;
            }
            let content = message.content.trim();
            if content.starts_with("<tool_response>") && content.ends_with("</tool_response>") {
                None
            } else {
                Some(index)
            }
        })
        .ok_or_else(|| ChatError::InvalidPrompt {
            detail: "no user query found in messages".into(),
        })
}

fn render_system_preamble(
    output: &mut String,
    messages: &[TemplateMessage],
    tools: &[EchoToolContract],
) -> Result<(), ChatError> {
    if tools.is_empty() {
        if messages[0].role == TemplateRole::System {
            output.push_str(IM_START);
            output.push_str("system\n");
            output.push_str(messages[0].content.trim());
            output.push_str(IM_END);
            output.push('\n');
        }
        return Ok(());
    }

    output.push_str(TOOL_PREAMBLE);
    for tool in tools {
        validate_xml_name(&tool.name, "tool name")?;
        output.push('\n');
        output.push_str(&render_tool_contract(tool));
    }
    output.push_str(TOOL_INSTRUCTIONS);
    if messages[0].role == TemplateRole::System {
        let system = messages[0].content.trim();
        if !system.is_empty() {
            output.push_str("\n\n");
            output.push_str(system);
        }
    }
    output.push_str(IM_END);
    output.push('\n');
    Ok(())
}

fn render_message(
    output: &mut String,
    messages: &[TemplateMessage],
    index: usize,
    message: &TemplateMessage,
    last_query_index: usize,
) -> Result<(), ChatError> {
    match message.role {
        TemplateRole::System => {
            if index != 0 {
                return Err(ChatError::InvalidPrompt {
                    detail: "system message must be at the beginning".into(),
                });
            }
        }
        TemplateRole::User => {
            output.push_str(IM_START);
            output.push_str("user\n");
            output.push_str(message.content.trim());
            output.push_str(IM_END);
            output.push('\n');
        }
        TemplateRole::Assistant => {
            render_assistant(output, message, index > last_query_index);
        }
        TemplateRole::Tool => render_tool_result(output, messages, index, message),
    }
    Ok(())
}

fn render_assistant(output: &mut String, message: &TemplateMessage, preserve_thinking: bool) {
    let (reasoning, content) = split_reasoning(message.content.trim());
    output.push_str(IM_START);
    output.push_str("assistant\n");
    if preserve_thinking {
        output.push_str("<think>\n");
        output.push_str(reasoning.trim());
        output.push_str("\n</think>\n\n");
    }
    output.push_str(content);
    for (index, tool_call) in message.tool_calls.iter().enumerate() {
        if index > 0 || !content.trim().is_empty() {
            output.push_str("\n\n");
        }
        output.push_str("<tool_call>\n<function=");
        output.push_str(&tool_call.name);
        output.push_str(">\n");
        for (name, value) in &tool_call.arguments {
            output.push_str("<parameter=");
            output.push_str(name);
            output.push_str(">\n");
            if let Value::String(value) = value {
                output.push_str(value);
            } else {
                output.push_str(&python_json(value));
            }
            output.push_str("\n</parameter>\n");
        }
        output.push_str("</function>\n</tool_call>");
    }
    output.push_str(IM_END);
    output.push('\n');
}

fn render_tool_result(
    output: &mut String,
    messages: &[TemplateMessage],
    index: usize,
    message: &TemplateMessage,
) {
    if index == 0 || messages[index - 1].role != TemplateRole::Tool {
        output.push_str(IM_START);
        output.push_str("user");
    }
    output.push_str("\n<tool_response>\n");
    output.push_str(message.content.trim());
    output.push_str("\n</tool_response>");
    if index + 1 == messages.len() || messages[index + 1].role != TemplateRole::Tool {
        output.push_str(IM_END);
        output.push('\n');
    }
}

fn split_reasoning(content: &str) -> (&str, &str) {
    let Some((before_end, after_end)) = content.split_once("</think>") else {
        return ("", content);
    };
    let reasoning = before_end
        .trim_end_matches('\n')
        .rsplit_once("<think>")
        .map_or(before_end, |(_, reasoning)| reasoning)
        .trim_start_matches('\n');
    (reasoning, after_end.trim_start_matches('\n'))
}

fn render_tool_contract(tool: &EchoToolContract) -> String {
    let mut function = Map::new();
    function.insert("name".into(), Value::String(tool.name.clone()));
    function.insert(
        "description".into(),
        Value::String(tool.description.clone()),
    );
    if tool.input_schema.is_object() {
        function.insert("parameters".into(), tool.input_schema.clone());
    }
    function.insert("strict".into(), Value::Bool(tool.strict));

    let mut contract = Map::new();
    contract.insert("type".into(), Value::String("function".into()));
    contract.insert("function".into(), Value::Object(function));
    python_json(&Value::Object(contract))
}

fn python_json(value: &Value) -> String {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            serde_json::to_string(value).expect("serializing an in-memory JSON value cannot fail")
        }
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_json)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(name, value)| format!(
                    "{}: {}",
                    serde_json::to_string(name)
                        .expect("serializing an in-memory JSON key cannot fail"),
                    python_json(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn validate_xml_name(value: &str, label: &str) -> Result<(), ChatError> {
    if !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
    {
        return Ok(());
    }
    Err(ChatError::InvalidPrompt {
        detail: format!("{label} {value:?} is not safe for the Qwen tool-call envelope"),
    })
}

fn sha256_text(value: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest: [u8; 32] = Sha256::digest(value.as_bytes()).into();
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

/// Chat-template, tokenizer, or prompt-admission failure.
#[derive(Debug)]
pub enum ChatError {
    /// Local tokenizer file I/O failed.
    Io {
        /// Exact path.
        path: PathBuf,
        /// Operating-system error.
        source: std::io::Error,
    },
    /// `tokenizer_config.json` was invalid.
    Json {
        /// Exact path.
        path: PathBuf,
        /// JSON error.
        source: serde_json::Error,
    },
    /// The local model carries a chat template that has not passed the exact
    /// oracle fixture.
    UnsupportedTemplate {
        /// SHA-256 of the observed Jinja source.
        observed_sha256: String,
    },
    /// Official tokenizer loading, encoding, or decoding failed.
    Tokenizer {
        /// Failure detail.
        detail: String,
    },
    /// Input history violates the admitted Qwen template contract.
    InvalidPrompt {
        /// Failure detail.
        detail: String,
    },
    /// Tool-call `input` was not a JSON object.
    ToolInput {
        /// E.C.H.O. call identity.
        call_id: String,
        /// Parse or shape detail.
        detail: String,
    },
    /// The native text runtime cannot yet consume vision embeddings.
    UnsupportedVision,
}

impl fmt::Display for ChatError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(formatter, "{}: {source}", path.display()),
            Self::Json { path, source } => write!(formatter, "{}: {source}", path.display()),
            Self::UnsupportedTemplate { observed_sha256 } => write!(
                formatter,
                "chat template {observed_sha256} is not the admitted Qwen3.5 E.C.H.O. template"
            ),
            Self::Tokenizer { detail } | Self::InvalidPrompt { detail } => {
                formatter.write_str(detail)
            }
            Self::ToolInput { call_id, detail } => {
                write!(formatter, "tool call {call_id} input: {detail}")
            }
            Self::UnsupportedVision => formatter.write_str(
                "image input requires the native vision encoder, which is not admitted yet",
            ),
        }
    }
}

impl Error for ChatError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::UnsupportedTemplate { .. }
            | Self::Tokenizer { .. }
            | Self::InvalidPrompt { .. }
            | Self::ToolInput { .. }
            | Self::UnsupportedVision => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(role: EchoMessageRole, content: &str) -> EchoInputItem {
        EchoInputItem::Message(EchoMessage {
            role,
            content: EchoMessageContent::Text(content.into()),
        })
    }

    #[test]
    fn non_thinking_text_prompt_matches_qwen_template_shape() {
        let rendered = render_chat_prompt(&EchoChatPrompt {
            input: vec![text(EchoMessageRole::User, "hello")],
            tools: Vec::new(),
        })
        .expect("valid prompt");
        assert_eq!(
            rendered,
            "<|im_start|>user\nhello<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn developer_maps_to_user_for_echo_startup_compatibility() {
        let rendered = render_chat_prompt(&EchoChatPrompt {
            input: vec![text(EchoMessageRole::Developer, "persona")],
            tools: Vec::new(),
        })
        .expect("valid prompt");
        assert!(rendered.starts_with("<|im_start|>user\npersona<|im_end|>\n"));
    }

    #[test]
    fn consecutive_tool_results_share_one_user_envelope() {
        let prompt: EchoChatPrompt = serde_json::from_value(serde_json::json!({
            "input": [
                {"role": "user", "content": "query"},
                {
                    "type": "tool_result",
                    "call_id": "first",
                    "output": "{\"first\":1}"
                },
                {
                    "type": "tool_result",
                    "call_id": "second",
                    "output": "{\"second\":2}"
                }
            ]
        }))
        .expect("valid fixture");
        let rendered = render_chat_prompt(&prompt).expect("valid prompt");
        assert!(rendered.contains(
            "<|im_start|>user\n<tool_response>\n{\"first\":1}\n</tool_response>\
             \n<tool_response>\n{\"second\":2}\n</tool_response><|im_end|>\n"
        ));
    }

    #[test]
    fn image_parts_fail_closed() {
        let error = render_chat_prompt(&EchoChatPrompt {
            input: vec![EchoInputItem::Message(EchoMessage {
                role: EchoMessageRole::User,
                content: EchoMessageContent::Parts(vec![EchoContentPart::Image {
                    image_url: "data:image/png;base64,AA==".into(),
                    detail: None,
                }]),
            })],
            tools: Vec::new(),
        })
        .expect_err("vision must not reach a text-only runtime");
        assert!(matches!(error, ChatError::UnsupportedVision));
    }

    #[test]
    fn exact_tool_continuation_has_only_the_new_suffix() {
        let prompt: Vec<EchoInputItem> = serde_json::from_value(serde_json::json!([
            {
                "type": "tool_result",
                "call_id": "check",
                "output": "{\"unread\":0}"
            }
        ]))
        .expect("valid tool result");
        assert_eq!(
            render_chat_continuation(&prompt).expect("append-only continuation"),
            "\n<|im_start|>user\n<tool_response>\n{\"unread\":0}\n</tool_response>\
             <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn exact_continuation_rejects_new_user_query() {
        let error = render_chat_continuation(&[text(EchoMessageRole::User, "new query")])
            .expect_err("new query needs a session transition");
        assert!(matches!(error, ChatError::InvalidPrompt { .. }));
    }

    #[test]
    fn python_json_keeps_wire_order_and_python_spacing() {
        let value: Value =
            serde_json::from_str(r#"{"channels":["discord","email"],"limit":20}"#).expect("JSON");
        assert_eq!(
            python_json(&value),
            r#"{"channels": ["discord", "email"], "limit": 20}"#
        );
    }
}
