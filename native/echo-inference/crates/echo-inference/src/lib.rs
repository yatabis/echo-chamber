//! Qwen3.5-specific engine bootstrap and checkpoint validation.

mod attention;
mod chat;
mod decoder;
mod full_model;
mod gdn;
mod hybrid_block;
mod layer;
mod local_server;
mod model_state;
#[cfg(feature = "moe-performance-diagnostics")]
mod moe_performance;
mod runtime;
mod sampling;
mod snapshot;
mod tool_output;
mod weights;

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use echo_inference_state::ModelIdentity;
use echo_mlx::{MlxError, SafeTensors};
use serde::Deserialize;
use sha2::{Digest, Sha256};

pub use attention::{AttentionLayerParity, run_attention_layer_parity};
pub use chat::{
    ChatError, ChatTemplateCaseParity, ChatTemplateParity, EchoChatPrompt, EchoContentPart,
    EchoInputItem, EchoMessage, EchoMessageContent, EchoMessageRole, EchoToolCall,
    EchoToolContract, EchoToolResult, EncodedChatPrompt, Qwen35ChatTokenizer, Qwen35DecodeStream,
    ToolCallKind, ToolResultKind, run_chat_template_parity,
};
pub use decoder::{DecoderLayerParity, run_decoder_layer_parity};
pub use full_model::{
    DurableStateAssertions, DurableStateParity, DurableStateProducer, DurableStateRestorer,
    FullModelParity, LiveStateParity, NewSessionAssertions, NewSessionParity,
    ResidentRuntimeAssertions, ResidentRuntimeFailureParity, ResidentRuntimeParity,
    ResidentRuntimeQueueAssertions, ResidentRuntimeRequestParity,
    ResidentRuntimeResidencyAssertions, ResidentRuntimeStateAssertions,
    produce_durable_state_parity, restore_durable_state_parity, run_durable_state_parity,
    run_full_model_parity, run_live_state_parity, run_new_session_parity,
    run_resident_runtime_parity,
};
pub use gdn::{GdnLayerParity, run_gdn_layer_parity};
pub use hybrid_block::{HybridBlockParity, run_hybrid_block_parity};
pub use local_server::{LocalServerConfig, LocalServerError, serve_local_stdio};
pub use model_state::MlxInferenceState;
#[cfg(feature = "moe-performance-diagnostics")]
pub use moe_performance::{MoePerformanceDiagnostic, run_moe_performance_diagnostic};
pub use runtime::{
    GenerationDirective, GenerationFinishReason, GenerationObserver, InferenceRequest,
    InferenceResponse, OpenedState, RequestState, ResidentEngine, ResidentEngineConfig,
    ResidentEngineInfo, RuntimeError, RuntimeMetrics, ScheduleTicket, ScheduledInferenceOutcome,
    SchedulerError, SingleGenerationScheduler,
};
pub use sampling::{SamplingConfig, SamplingParity, SamplingParityCase, run_sampling_parity};
pub use snapshot::{
    CURRENT_STATE_FILE, CurrentStateOwner, PublishedMlxCheckpoint, RestoredMlxCheckpoint,
};
pub use tool_output::{EchoAssistantRole, EchoOutputItem, ParsedQwenOutput, parse_qwen_output};

/// Qwen3.5 architecture facts resolved once when the model is admitted.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelPlan {
    pub architecture: String,
    pub hidden_size: usize,
    pub vocabulary_size: usize,
    pub layer_count: usize,
    pub tie_word_embeddings: bool,
    pub full_attention_interval: usize,
    pub key_head_count: usize,
    pub value_head_count: usize,
    pub key_head_dimension: usize,
    pub value_head_dimension: usize,
    pub convolution_kernel_size: usize,
    pub rms_norm_epsilon: f32,
    pub recurrent_layer_count: usize,
    pub full_attention_layer_count: usize,
    pub expert_count: usize,
    pub experts_per_token: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub attention_head_count: usize,
    pub key_value_head_count: usize,
    pub attention_head_dimension: usize,
    pub rotary_dimension: usize,
    pub rope_base: f32,
    pub quantization_bits: usize,
    pub quantization_group_size: usize,
    pub quantization_mode: String,
    pub router_quantization_bits: usize,
    pub router_quantization_group_size: usize,
    pub router_quantization_mode: String,
    pub shared_gate_quantization_bits: usize,
    pub shared_gate_quantization_group_size: usize,
    pub shared_gate_quantization_mode: String,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    model_type: String,
    text_config: TextConfig,
    quantization: Quantization,
}

#[derive(Debug, Deserialize)]
struct TextConfig {
    hidden_size: usize,
    #[serde(default)]
    vocab_size: usize,
    num_hidden_layers: usize,
    #[serde(default)]
    tie_word_embeddings: bool,
    full_attention_interval: usize,
    linear_conv_kernel_dim: usize,
    #[serde(default)]
    linear_num_value_heads: usize,
    #[serde(default)]
    linear_num_key_heads: usize,
    #[serde(default)]
    linear_key_head_dim: usize,
    #[serde(default)]
    linear_value_head_dim: usize,
    #[serde(default)]
    rms_norm_eps: f32,
    #[serde(default)]
    num_experts: Option<usize>,
    #[serde(default)]
    num_experts_per_tok: Option<usize>,
    #[serde(default)]
    moe_intermediate_size: usize,
    #[serde(default)]
    shared_expert_intermediate_size: usize,
    #[serde(default = "default_true")]
    norm_topk_prob: bool,
    #[serde(default)]
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: usize,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
}

#[derive(Debug, Deserialize)]
struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    partial_rotary_factor: f32,
    #[serde(default, alias = "type")]
    rope_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Quantization {
    bits: usize,
    group_size: usize,
    mode: String,
    #[serde(flatten)]
    overrides: BTreeMap<String, QuantizationOverride>,
}

#[derive(Debug, Deserialize)]
struct QuantizationOverride {
    bits: usize,
    group_size: usize,
    #[serde(default)]
    mode: Option<String>,
}

fn default_true() -> bool {
    true
}

fn default_rope_theta() -> f32 {
    100_000.0
}

fn default_partial_rotary_factor() -> f32 {
    0.25
}

impl Quantization {
    fn module_parameters(&self, module: &str) -> (usize, usize, String) {
        self.overrides.get(module).map_or_else(
            || (self.bits, self.group_size, self.mode.clone()),
            |overridden| {
                (
                    overridden.bits,
                    overridden.group_size,
                    overridden.mode.clone().unwrap_or_else(|| self.mode.clone()),
                )
            },
        )
    }
}

impl ModelPlan {
    /// Resolves and validates the intentionally narrow Qwen3.5 execution plan.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] when the model is outside the currently
    /// admitted family or uses an unsupported quantization/layout.
    #[allow(clippy::too_many_lines)]
    pub fn from_directory(model_directory: &Path) -> Result<Self, EngineError> {
        let config_path = model_directory.join("config.json");
        let config_bytes = fs::read(&config_path).map_err(|source| EngineError::Io {
            path: config_path.clone(),
            source,
        })?;
        let config: ModelConfig =
            serde_json::from_slice(&config_bytes).map_err(|source| EngineError::Json {
                path: config_path,
                source,
            })?;

        if !matches!(config.model_type.as_str(), "qwen3_5" | "qwen3_5_moe") {
            return Err(EngineError::Unsupported(format!(
                "model_type must be qwen3_5 or qwen3_5_moe, observed {}",
                config.model_type
            )));
        }
        if config.text_config.full_attention_interval == 0 {
            return Err(EngineError::Unsupported(
                "full_attention_interval must be greater than zero".into(),
            ));
        }
        if config.text_config.num_hidden_layers == 0
            || !config
                .text_config
                .num_hidden_layers
                .is_multiple_of(config.text_config.full_attention_interval)
        {
            return Err(EngineError::Unsupported(format!(
                "layer count must be a non-zero multiple of full_attention_interval, observed {} and {}",
                config.text_config.num_hidden_layers, config.text_config.full_attention_interval
            )));
        }
        if config.text_config.vocab_size == 0 {
            return Err(EngineError::Unsupported(format!(
                "the specialized engine requires a non-empty vocabulary, observed {}",
                config.text_config.vocab_size
            )));
        }
        if config.text_config.linear_conv_kernel_dim != 4 {
            return Err(EngineError::Unsupported(format!(
                "linear_conv_kernel_dim must be 4, observed {}",
                config.text_config.linear_conv_kernel_dim
            )));
        }
        if config.text_config.linear_num_value_heads == 0
            || config.text_config.linear_num_key_heads == 0
            || config.text_config.linear_key_head_dim == 0
            || config.text_config.linear_value_head_dim == 0
        {
            return Err(EngineError::Unsupported(
                "Qwen3.5 GDN head dimensions must all be non-zero".into(),
            ));
        }
        if !config
            .text_config
            .linear_num_value_heads
            .is_multiple_of(config.text_config.linear_num_key_heads)
        {
            return Err(EngineError::Unsupported(
                "Qwen3.5 value-head count must be divisible by key-head count".into(),
            ));
        }
        if !config.text_config.linear_key_head_dim.is_multiple_of(32) {
            return Err(EngineError::Unsupported(
                "Qwen3.5 key-head dimension must be divisible by one Metal SIMD group".into(),
            ));
        }
        if config.text_config.rms_norm_eps <= 0.0 {
            return Err(EngineError::Unsupported(
                "Qwen3.5 rms_norm_eps must be positive".into(),
            ));
        }
        if config.text_config.num_attention_heads == 0
            || config.text_config.num_key_value_heads == 0
            || config.text_config.head_dim == 0
            || !config
                .text_config
                .num_attention_heads
                .is_multiple_of(config.text_config.num_key_value_heads)
        {
            return Err(EngineError::Unsupported(
                "Qwen3.5 full-attention dimensions must be non-zero with query heads divisible by key/value heads".into(),
            ));
        }
        if config.text_config.attention_bias {
            return Err(EngineError::Unsupported(
                "the specialized engine does not admit attention projection biases".into(),
            ));
        }
        let rope = config.text_config.rope_parameters.as_ref().ok_or_else(|| {
            EngineError::Unsupported("Qwen3.5 rope_parameters are required".into())
        })?;
        if rope
            .rope_type
            .as_deref()
            .is_some_and(|rope_type| rope_type != "default")
            || rope.rope_theta <= 0.0
            || rope.partial_rotary_factor.to_bits() != 0.25_f32.to_bits()
        {
            return Err(EngineError::Unsupported(format!(
                "the specialized engine requires default positive-base RoPE with a 0.25 partial factor, observed type {:?}, base {}, factor {}",
                rope.rope_type, rope.rope_theta, rope.partial_rotary_factor
            )));
        }
        if !config.text_config.head_dim.is_multiple_of(4) {
            return Err(EngineError::Unsupported(format!(
                "attention head dimension must be divisible by four for partial RoPE, observed {}",
                config.text_config.head_dim
            )));
        }
        let rotary_dimension = config.text_config.head_dim / 4;
        if rotary_dimension == 0 || !rotary_dimension.is_multiple_of(2) {
            return Err(EngineError::Unsupported(format!(
                "partial rotary dimension must be a positive even integer, observed {rotary_dimension}"
            )));
        }
        if config.quantization.bits != 4 || config.quantization.mode != "affine" {
            return Err(EngineError::Unsupported(format!(
                "bootstrap requires affine 4-bit weights, observed {}-bit {}",
                config.quantization.bits, config.quantization.mode
            )));
        }
        if config.quantization.group_size == 0 {
            return Err(EngineError::Unsupported(
                "quantization group size must be greater than zero".into(),
            ));
        }

        let full_attention_layer_count =
            config.text_config.num_hidden_layers / config.text_config.full_attention_interval;
        let recurrent_layer_count =
            config.text_config.num_hidden_layers - full_attention_layer_count;
        let expert_count = config.text_config.num_experts.unwrap_or_default();
        let experts_per_token = config.text_config.num_experts_per_tok.unwrap_or_default();
        if config.model_type == "qwen3_5_moe" {
            if expert_count == 0
                || experts_per_token == 0
                || config.text_config.moe_intermediate_size == 0
                || config.text_config.shared_expert_intermediate_size == 0
            {
                return Err(EngineError::Unsupported(
                    "qwen3_5_moe requires non-zero expert routing and intermediate dimensions"
                        .into(),
                ));
            }
            if experts_per_token > expert_count {
                return Err(EngineError::Unsupported(format!(
                    "experts_per_token {experts_per_token} exceeds expert_count {expert_count}"
                )));
            }
        }
        let router_module = "language_model.model.layers.0.mlp.gate";
        let shared_gate_module = "language_model.model.layers.0.mlp.shared_expert_gate";
        let (router_quantization_bits, router_quantization_group_size, router_quantization_mode) =
            config.quantization.module_parameters(router_module);
        let (
            shared_gate_quantization_bits,
            shared_gate_quantization_group_size,
            shared_gate_quantization_mode,
        ) = config.quantization.module_parameters(shared_gate_module);
        for (name, bits, group_size, mode) in [
            (
                router_module,
                router_quantization_bits,
                router_quantization_group_size,
                router_quantization_mode.as_str(),
            ),
            (
                shared_gate_module,
                shared_gate_quantization_bits,
                shared_gate_quantization_group_size,
                shared_gate_quantization_mode.as_str(),
            ),
        ] {
            if !(2..=8).contains(&bits) || group_size == 0 || mode != "affine" {
                return Err(EngineError::Unsupported(format!(
                    "{name} requires affine 2-8 bit quantization with a non-zero group size, observed {bits}-bit {mode} group {group_size}"
                )));
            }
        }
        for layer_index in 0..config.text_config.num_hidden_layers {
            for (module_suffix, expected_bits, expected_group_size, expected_mode) in [
                (
                    "gate",
                    router_quantization_bits,
                    router_quantization_group_size,
                    router_quantization_mode.as_str(),
                ),
                (
                    "shared_expert_gate",
                    shared_gate_quantization_bits,
                    shared_gate_quantization_group_size,
                    shared_gate_quantization_mode.as_str(),
                ),
            ] {
                let module =
                    format!("language_model.model.layers.{layer_index}.mlp.{module_suffix}");
                let (bits, group_size, mode) = config.quantization.module_parameters(&module);
                if bits != expected_bits
                    || group_size != expected_group_size
                    || mode != expected_mode
                {
                    return Err(EngineError::Unsupported(format!(
                        "{module} quantization differs across decoder layers: expected {expected_bits}-bit {expected_mode} group {expected_group_size}, observed {bits}-bit {mode} group {group_size}"
                    )));
                }
            }
        }
        let full_model_quantized_modules = if config.text_config.tie_word_embeddings {
            &["language_model.model.embed_tokens"][..]
        } else {
            &[
                "language_model.model.embed_tokens",
                "language_model.lm_head",
            ][..]
        };
        for module in full_model_quantized_modules {
            let (bits, group_size, mode) = config.quantization.module_parameters(module);
            if bits != config.quantization.bits
                || group_size != config.quantization.group_size
                || mode != config.quantization.mode
            {
                return Err(EngineError::Unsupported(format!(
                    "{module} must use the model-default quantization for the specialized full-model path"
                )));
            }
        }

        Ok(Self {
            architecture: config.model_type,
            hidden_size: config.text_config.hidden_size,
            vocabulary_size: config.text_config.vocab_size,
            layer_count: config.text_config.num_hidden_layers,
            tie_word_embeddings: config.text_config.tie_word_embeddings,
            full_attention_interval: config.text_config.full_attention_interval,
            key_head_count: config.text_config.linear_num_key_heads,
            value_head_count: config.text_config.linear_num_value_heads,
            key_head_dimension: config.text_config.linear_key_head_dim,
            value_head_dimension: config.text_config.linear_value_head_dim,
            convolution_kernel_size: config.text_config.linear_conv_kernel_dim,
            rms_norm_epsilon: config.text_config.rms_norm_eps,
            recurrent_layer_count,
            full_attention_layer_count,
            expert_count,
            experts_per_token,
            moe_intermediate_size: config.text_config.moe_intermediate_size,
            shared_expert_intermediate_size: config.text_config.shared_expert_intermediate_size,
            norm_topk_prob: config.text_config.norm_topk_prob,
            attention_head_count: config.text_config.num_attention_heads,
            key_value_head_count: config.text_config.num_key_value_heads,
            attention_head_dimension: config.text_config.head_dim,
            rotary_dimension,
            rope_base: rope.rope_theta,
            quantization_bits: config.quantization.bits,
            quantization_group_size: config.quantization.group_size,
            quantization_mode: config.quantization.mode,
            router_quantization_bits,
            router_quantization_group_size,
            router_quantization_mode,
            shared_gate_quantization_bits,
            shared_gate_quantization_group_size,
            shared_gate_quantization_mode,
        })
    }
}

/// Lightweight model-weight inspection through the production Rust/MLX path.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInspection {
    pub plan: ModelPlan,
    pub shard_count: usize,
    pub tensor_count: usize,
    pub logical_nbytes: usize,
    pub has_first_gdn_projection: bool,
}

/// Loads each local model shard through MLX C and validates a first
/// Qwen3.5 GDN projection.
///
/// # Errors
///
/// Returns [`EngineError`] for unsupported configs, unreadable directories,
/// MLX load failures, or missing load-bearing tensors.
pub fn inspect_model(model_directory: &Path) -> Result<ModelInspection, EngineError> {
    let plan = ModelPlan::from_directory(model_directory)?;
    let shards = safetensor_files(model_directory)?;
    if shards.is_empty() {
        return Err(EngineError::Unsupported(
            "model directory contains no .safetensors files".into(),
        ));
    }

    let mut tensor_count = 0usize;
    let mut logical_nbytes = 0usize;
    let mut has_first_gdn_projection = false;
    for shard in &shards {
        let tensors = SafeTensors::load(shard).map_err(EngineError::Mlx)?;
        tensor_count = tensor_count
            .checked_add(tensors.len())
            .ok_or_else(|| EngineError::Unsupported("tensor count overflow".into()))?;
        logical_nbytes = logical_nbytes
            .checked_add(tensors.total_nbytes())
            .ok_or_else(|| EngineError::Unsupported("logical byte count overflow".into()))?;
        has_first_gdn_projection |= tensors
            .tensor("language_model.model.layers.0.linear_attn.in_proj_qkv.weight")
            .is_some();
    }

    if !has_first_gdn_projection {
        return Err(EngineError::Unsupported(
            "missing first Qwen3.5 GDN in_proj_qkv.weight tensor".into(),
        ));
    }

    Ok(ModelInspection {
        plan,
        shard_count: shards.len(),
        tensor_count,
        logical_nbytes,
        has_first_gdn_projection,
    })
}

/// Computes the exact model identity used to admit or restore inference state.
///
/// The combined weights digest follows the retained Python oracle: sorted
/// shard file names and each shard's raw SHA-256 digest are fed into one
/// outer SHA-256 digest. Tokenizer and template identity are the raw digests
/// of `tokenizer.json` and `tokenizer_config.json`.
///
/// # Errors
///
/// Returns [`EngineError`] when the model is unsupported, an identity file is
/// missing or unreadable, no weight shards exist, or a file name is not UTF-8.
pub fn identify_model(model_directory: &Path) -> Result<ModelIdentity, EngineError> {
    let plan = ModelPlan::from_directory(model_directory)?;
    let shards = safetensor_files(model_directory)?;
    if shards.is_empty() {
        return Err(EngineError::Unsupported(
            "model directory contains no .safetensors files".into(),
        ));
    }
    Ok(ModelIdentity {
        architecture: plan.architecture,
        config_digest: sha256_file(&model_directory.join("config.json"))?,
        weights_digest: sha256_named_files(&shards)?,
        tokenizer_digest: sha256_file(&model_directory.join("tokenizer.json"))?,
        template_digest: sha256_file(&model_directory.join("tokenizer_config.json"))?,
    })
}

fn safetensor_files(model_directory: &Path) -> Result<Vec<PathBuf>, EngineError> {
    let mut result = Vec::new();
    let entries = fs::read_dir(model_directory).map_err(|source| EngineError::Io {
        path: model_directory.to_path_buf(),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| EngineError::Io {
            path: model_directory.to_path_buf(),
            source,
        })?;
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "safetensors")
            && path
                .file_name()
                .is_some_and(|name| !name.to_string_lossy().contains("checkpoint"))
        {
            result.push(path);
        }
    }
    result.sort();
    Ok(result)
}

fn sha256_file_bytes(path: &Path) -> Result<[u8; 32], EngineError> {
    let file = File::open(path).map_err(|source| EngineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer).map_err(|source| EngineError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(digest.finalize().into())
}

fn sha256_named_files(paths: &[PathBuf]) -> Result<String, EngineError> {
    let mut paths = paths.to_vec();
    paths.sort();
    let mut digest = Sha256::new();
    for path in paths {
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                EngineError::Unsupported(format!(
                    "model shard file name is not valid UTF-8: {}",
                    path.display()
                ))
            })?;
        digest.update(file_name.as_bytes());
        digest.update(sha256_file_bytes(&path)?);
    }
    Ok(hex_sha256(digest.finalize().into()))
}

fn sha256_file(path: &Path) -> Result<String, EngineError> {
    Ok(hex_sha256(sha256_file_bytes(path)?))
}

fn hex_sha256(bytes: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(64);
    for byte in bytes {
        result.push(char::from(HEX[usize::from(byte >> 4)]));
        result.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    result
}

/// Bootstrap admission or loading error.
#[derive(Debug)]
pub enum EngineError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    Mlx(MlxError),
    Unsupported(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(formatter, "{}: {}", path.display(), source)
            }
            Self::Json { path, source } => {
                write!(formatter, "{}: {}", path.display(), source)
            }
            Self::Mlx(error) => error.fmt(formatter),
            Self::Unsupported(detail) => formatter.write_str(detail),
        }
    }
}

impl Error for EngineError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::Mlx(error) => Some(error),
            Self::Unsupported(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_qwen_family_before_loading_weights() {
        let temporary =
            std::env::temp_dir().join(format!("echo-inference-config-test-{}", std::process::id()));
        fs::create_dir_all(&temporary).expect("temporary directory");
        fs::write(
            temporary.join("config.json"),
            br#"{
              "model_type": "llama",
              "text_config": {
                "hidden_size": 2048,
                "num_hidden_layers": 4,
                "full_attention_interval": 4,
                "linear_conv_kernel_dim": 4
              },
              "quantization": {"bits": 4, "group_size": 64, "mode": "affine"}
            }"#,
        )
        .expect("config write");

        let error = ModelPlan::from_directory(&temporary).expect_err("must reject");
        assert!(matches!(error, EngineError::Unsupported(_)));
        fs::remove_dir_all(temporary).expect("temporary cleanup");
    }

    #[test]
    fn resolves_moe_dimensions_and_per_module_quantization() {
        let temporary = std::env::temp_dir().join(format!(
            "echo-inference-moe-config-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&temporary).expect("temporary directory");
        fs::write(
            temporary.join("config.json"),
            br#"{
              "model_type": "qwen3_5_moe",
              "text_config": {
                "hidden_size": 2048,
                "vocab_size": 248320,
                "num_hidden_layers": 4,
                "tie_word_embeddings": true,
                "full_attention_interval": 4,
                "linear_conv_kernel_dim": 4,
                "linear_num_value_heads": 32,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "rms_norm_eps": 0.000001,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "attention_bias": false,
                "rope_parameters": {
                  "rope_theta": 10000000.0,
                  "partial_rotary_factor": 0.25,
                  "rope_type": "default"
                },
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512
              },
              "quantization": {
                "bits": 4,
                "group_size": 64,
                "mode": "affine",
                "language_model.model.layers.0.mlp.gate": {
                  "bits": 8,
                  "group_size": 64
                },
                "language_model.model.layers.0.mlp.shared_expert_gate": {
                  "bits": 6,
                  "group_size": 32,
                  "mode": "affine"
                },
                "language_model.model.layers.1.mlp.gate": {
                  "bits": 8,
                  "group_size": 64
                },
                "language_model.model.layers.1.mlp.shared_expert_gate": {
                  "bits": 6,
                  "group_size": 32,
                  "mode": "affine"
                },
                "language_model.model.layers.2.mlp.gate": {
                  "bits": 8,
                  "group_size": 64
                },
                "language_model.model.layers.2.mlp.shared_expert_gate": {
                  "bits": 6,
                  "group_size": 32,
                  "mode": "affine"
                },
                "language_model.model.layers.3.mlp.gate": {
                  "bits": 8,
                  "group_size": 64
                },
                "language_model.model.layers.3.mlp.shared_expert_gate": {
                  "bits": 6,
                  "group_size": 32,
                  "mode": "affine"
                }
              }
            }"#,
        )
        .expect("config write");

        let plan = ModelPlan::from_directory(&temporary).expect("admitted MoE plan");
        assert_eq!(plan.expert_count, 256);
        assert_eq!(plan.experts_per_token, 8);
        assert_eq!(plan.moe_intermediate_size, 512);
        assert_eq!(plan.shared_expert_intermediate_size, 512);
        assert!(plan.norm_topk_prob);
        assert_eq!(plan.router_quantization_bits, 8);
        assert_eq!(plan.router_quantization_group_size, 64);
        assert_eq!(plan.router_quantization_mode, "affine");
        assert_eq!(plan.shared_gate_quantization_bits, 6);
        assert_eq!(plan.shared_gate_quantization_group_size, 32);
        assert_eq!(plan.shared_gate_quantization_mode, "affine");
        assert!(plan.tie_word_embeddings);
        assert_eq!(plan.attention_head_count, 16);
        assert_eq!(plan.key_value_head_count, 2);
        assert_eq!(plan.attention_head_dimension, 256);
        assert_eq!(plan.rotary_dimension, 64);
        assert_eq!(plan.rope_base.to_bits(), 10_000_000.0_f32.to_bits());
        fs::remove_dir_all(temporary).expect("temporary cleanup");
    }

    #[test]
    fn rejects_more_selected_experts_than_available() {
        let temporary = std::env::temp_dir().join(format!(
            "echo-inference-invalid-topk-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&temporary).expect("temporary directory");
        fs::write(
            temporary.join("config.json"),
            br#"{
              "model_type": "qwen3_5_moe",
              "text_config": {
                "hidden_size": 2048,
                "vocab_size": 248320,
                "num_hidden_layers": 4,
                "tie_word_embeddings": false,
                "full_attention_interval": 4,
                "linear_conv_kernel_dim": 4,
                "linear_num_value_heads": 32,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "rms_norm_eps": 0.000001,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "attention_bias": false,
                "rope_parameters": {
                  "rope_theta": 10000000.0,
                  "partial_rotary_factor": 0.25,
                  "rope_type": "default"
                },
                "num_experts": 4,
                "num_experts_per_tok": 8,
                "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512
              },
              "quantization": {
                "bits": 4,
                "group_size": 64,
                "mode": "affine"
              }
            }"#,
        )
        .expect("config write");

        let error = ModelPlan::from_directory(&temporary).expect_err("must reject");
        assert!(
            error
                .to_string()
                .contains("experts_per_token 8 exceeds expert_count 4")
        );
        fs::remove_dir_all(temporary).expect("temporary cleanup");
    }

    #[test]
    fn hashes_sorted_shard_names_and_raw_file_digests() {
        let temporary = std::env::temp_dir().join(format!(
            "echo-inference-weight-identity-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&temporary).expect("temporary directory");
        let first = temporary.join("a.safetensors");
        let second = temporary.join("b.safetensors");
        fs::write(&first, b"alpha").expect("first shard write");
        fs::write(&second, b"beta").expect("second shard write");

        let digest = sha256_named_files(&[second, first]).expect("combined shard identity digest");
        assert_eq!(
            digest,
            "30628e08a12eac2d9186dfa610487a01aeb97f20ec36c3011a5bde743fc67204"
        );
        fs::remove_dir_all(temporary).expect("temporary cleanup");
    }
}
