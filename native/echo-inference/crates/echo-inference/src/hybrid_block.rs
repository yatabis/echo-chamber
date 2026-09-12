use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use echo_mlx::{Array, DType, Gpu, SafeTensors};
use serde::{Deserialize, Serialize};

use super::decoder::MoeKernel;
use super::gdn::{load_weight_shard_containing, require_tensor, validate_array};
use super::layer::{execute_attention_decoder_layer, execute_gdn_decoder_layer};
use super::{EngineError, ModelPlan, sha256_file};

const BLOCK_START_LAYER: usize = 0;
const BLOCK_LAYER_COUNT: usize = 4;
const EXPECTED_LAYER_CLASSES: [&str; BLOCK_LAYER_COUNT] = ["gdn", "gdn", "gdn", "full_attention"];

#[derive(Debug, Deserialize)]
struct FixtureManifest {
    schema_version: u32,
    model_type: String,
    config_sha256: String,
    block_start_layer: usize,
    block_layer_count: usize,
    layer_classes: Vec<String>,
    prefix_length: usize,
    continuation_length: usize,
    input_dtype: String,
    gdn_mask_mode: String,
    attention_mask_mode: String,
    sorted_expert_path: bool,
    dimensions: FixtureDimensions,
    rope: FixtureRope,
    quantization: FixtureQuantization,
    norm_topk_prob: bool,
    fixture_sha256: String,
    fixture_tensor_count: usize,
}

#[derive(Debug, Deserialize, Eq, PartialEq)]
struct FixtureDimensions {
    hidden_size: usize,
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

/// Direct comparison between the first Qwen3.5 hybrid block executed from
/// Rust and the official Python/MLX calls.
#[derive(Clone, Debug, Serialize)]
pub struct HybridBlockParity {
    pub architecture: String,
    pub block_start_layer: usize,
    pub block_layer_count: usize,
    pub batch_size: usize,
    pub prefix_length: usize,
    pub continuation_length: usize,
    pub fixture_sha256: String,
    pub output_max_absolute_difference: f32,
    pub layer_input_max_absolute_difference: f32,
    pub layer_output_max_absolute_difference: f32,
    pub state_max_absolute_difference: f32,
    pub differences: BTreeMap<String, f32>,
    pub exact: bool,
}

/// Executes the first Qwen3.5 hybrid block through Rust and MLX C.
///
/// The block contains three GDN decoder layers followed by one full-attention
/// decoder layer. Every layer includes its sparse `MoE`, while all six GDN
/// state tensors and the full-attention KV pair are carried from a non-empty
/// prefix.
///
/// # Errors
///
/// Returns [`EngineError`] when the model, oracle, weights, shapes, dtypes, or
/// an MLX operation do not match the admitted execution plan.
#[allow(clippy::too_many_lines)]
pub fn run_hybrid_block_parity(
    model_directory: &Path,
    fixture_path: &Path,
    manifest_path: &Path,
) -> Result<HybridBlockParity, EngineError> {
    let plan = ModelPlan::from_directory(model_directory)?;
    let manifest = load_manifest(manifest_path)?;
    validate_manifest(&plan, model_directory, fixture_path, &manifest)?;

    let fixture = SafeTensors::load(fixture_path).map_err(EngineError::Mlx)?;
    if fixture.len() != manifest.fixture_tensor_count {
        return Err(EngineError::Unsupported(format!(
            "hybrid-block fixture tensor count drift: manifest {}, payload {}",
            manifest.fixture_tensor_count,
            fixture.len()
        )));
    }
    let weights = load_weight_shard_containing(
        model_directory,
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
    )?;
    let gpu = Gpu::new();
    let moe_kernel = MoeKernel::new(&plan)?;

    let input = require_tensor(&fixture, "continuation_input")?;
    let shape = input.shape();
    let [batch_size, sequence_length, hidden_size] = <[usize; 3]>::try_from(shape.clone())
        .map_err(|shape| {
            EngineError::Unsupported(format!(
                "continuation_input must be rank 3, observed {shape:?}"
            ))
        })?;
    validate_array(
        input,
        &[1, manifest.continuation_length, plan.hidden_size],
        DType::BFloat16,
        "continuation_input",
    )?;
    if sequence_length != manifest.continuation_length || hidden_size != plan.hidden_size {
        return Err(EngineError::Unsupported(
            "continuation input dimensions changed after validation".into(),
        ));
    }

    let mut hidden = input.try_clone().map_err(EngineError::Mlx)?;
    let mut differences = BTreeMap::new();
    for relative_index in 0..manifest.block_layer_count {
        let layer_index = manifest.block_start_layer + relative_index;
        compare_fixture(
            &gpu,
            &fixture,
            &format!("layer.{relative_index}.input"),
            &hidden,
            &mut differences,
        )?;

        let execution = if manifest.layer_classes[relative_index] == "gdn" {
            execute_gdn_decoder_layer(
                &gpu,
                &hidden,
                require_tensor(
                    &fixture,
                    &format!("layer.{relative_index}.initial_conv_state"),
                )?,
                require_tensor(
                    &fixture,
                    &format!("layer.{relative_index}.initial_recurrent_state"),
                )?,
                &weights,
                layer_index,
                &plan,
            )?
        } else {
            execute_attention_decoder_layer(
                &gpu,
                &hidden,
                require_tensor(&fixture, &format!("layer.{relative_index}.initial_keys"))?,
                require_tensor(&fixture, &format!("layer.{relative_index}.initial_values"))?,
                &weights,
                layer_index,
                &plan,
                manifest.prefix_length,
                true,
                &moe_kernel,
            )?
        };

        compare_fixture(
            &gpu,
            &fixture,
            &format!("layer.{relative_index}.expected_output"),
            &execution.output,
            &mut differences,
        )?;
        let (first_state_name, second_state_name) =
            if manifest.layer_classes[relative_index] == "gdn" {
                ("expected_conv_state", "expected_recurrent_state")
            } else {
                ("expected_keys", "expected_values")
            };
        compare_fixture(
            &gpu,
            &fixture,
            &format!("layer.{relative_index}.{first_state_name}"),
            &execution.first_state,
            &mut differences,
        )?;
        compare_fixture(
            &gpu,
            &fixture,
            &format!("layer.{relative_index}.{second_state_name}"),
            &execution.second_state,
            &mut differences,
        )?;
        hidden = execution.output;
    }

    let expected_output = require_tensor(&fixture, "expected_output")?;
    let output_max_absolute_difference = gpu
        .max_abs_difference(&hidden, expected_output)
        .map_err(EngineError::Mlx)?;
    differences.insert(
        "block.expected_output".into(),
        output_max_absolute_difference,
    );

    let layer_input_max_absolute_difference = maximum_layer_named(&differences, ".input");
    let layer_output_max_absolute_difference =
        maximum_layer_named(&differences, ".expected_output");
    let state_max_absolute_difference = differences
        .iter()
        .filter(|(name, _)| {
            name.contains("state") || name.ends_with("keys") || name.ends_with("values")
        })
        .map(|(_, difference)| *difference)
        .fold(0.0_f32, f32::max);
    let exact = differences.values().all(|difference| *difference == 0.0);

    Ok(HybridBlockParity {
        architecture: plan.architecture,
        block_start_layer: manifest.block_start_layer,
        block_layer_count: manifest.block_layer_count,
        batch_size,
        prefix_length: manifest.prefix_length,
        continuation_length: manifest.continuation_length,
        fixture_sha256: manifest.fixture_sha256,
        output_max_absolute_difference,
        layer_input_max_absolute_difference,
        layer_output_max_absolute_difference,
        state_max_absolute_difference,
        differences,
        exact,
    })
}

fn compare_fixture(
    gpu: &Gpu,
    fixture: &SafeTensors,
    name: &str,
    actual: &Array,
    differences: &mut BTreeMap<String, f32>,
) -> Result<(), EngineError> {
    let expected = require_tensor(fixture, name)?;
    let difference = gpu
        .max_abs_difference(actual, expected)
        .map_err(EngineError::Mlx)?;
    differences.insert(name.to_owned(), difference);
    Ok(())
}

fn maximum_layer_named(differences: &BTreeMap<String, f32>, suffix: &str) -> f32 {
    differences
        .iter()
        .filter(|(name, _)| name.starts_with("layer.") && name.ends_with(suffix))
        .map(|(_, difference)| *difference)
        .fold(0.0_f32, f32::max)
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
            "hybrid-block fixture schema must be 1, observed {}",
            manifest.schema_version
        )));
    }
    if plan.architecture != "qwen3_5_moe" || manifest.model_type != plan.architecture {
        return Err(EngineError::Unsupported(format!(
            "hybrid-block fixture requires qwen3_5_moe, observed fixture {} and model {}",
            manifest.model_type, plan.architecture
        )));
    }
    if manifest.block_start_layer != BLOCK_START_LAYER
        || manifest.block_layer_count != BLOCK_LAYER_COUNT
        || plan.full_attention_interval != BLOCK_LAYER_COUNT
        || manifest.layer_classes != EXPECTED_LAYER_CLASSES.map(str::to_owned).to_vec()
    {
        return Err(EngineError::Unsupported(format!(
            "the current hybrid milestone admits layers 0-3 as {:?}, observed start {}, count {}, interval {}, classes {:?}",
            EXPECTED_LAYER_CLASSES,
            manifest.block_start_layer,
            manifest.block_layer_count,
            plan.full_attention_interval,
            manifest.layer_classes
        )));
    }
    if manifest.prefix_length == 0 || manifest.continuation_length <= 1 {
        return Err(EngineError::Unsupported(
            "hybrid-block fixture requires a non-empty prefix and multi-token continuation".into(),
        ));
    }
    if manifest.gdn_mask_mode != "None" || manifest.attention_mask_mode != "causal" {
        return Err(EngineError::Unsupported(format!(
            "hybrid-block mask modes must be GDN None and full-attention causal, observed {} and {}",
            manifest.gdn_mask_mode, manifest.attention_mask_mode
        )));
    }
    let selected_experts = manifest
        .continuation_length
        .checked_mul(plan.experts_per_token)
        .ok_or_else(|| {
            EngineError::Unsupported("hybrid-block expert-selection count overflow".into())
        })?;
    if manifest.sorted_expert_path != (selected_experts >= 64) {
        return Err(EngineError::Unsupported(format!(
            "hybrid-block sorted-expert path drift: fixture {}, selection count {selected_experts}",
            manifest.sorted_expert_path
        )));
    }
    if manifest.input_dtype != "mlx.core.bfloat16" {
        return Err(EngineError::Unsupported(format!(
            "hybrid-block fixture must use bfloat16 input, observed {}",
            manifest.input_dtype
        )));
    }
    let dimensions = FixtureDimensions {
        hidden_size: plan.hidden_size,
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
    };
    if manifest.dimensions != dimensions {
        return Err(EngineError::Unsupported(format!(
            "hybrid-block fixture dimension drift: fixture {:?}, model {:?}",
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
            "hybrid-block fixture RoPE drift: fixture {:?}, model {:?}",
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
    if manifest.quantization != quantization {
        return Err(EngineError::Unsupported(format!(
            "hybrid-block fixture quantization drift: fixture {:?}, model {:?}",
            manifest.quantization, quantization
        )));
    }
    if manifest.norm_topk_prob != plan.norm_topk_prob {
        return Err(EngineError::Unsupported(format!(
            "hybrid-block fixture norm_topk_prob drift: fixture {}, model {}",
            manifest.norm_topk_prob, plan.norm_topk_prob
        )));
    }
    let config_path = model_directory.join("config.json");
    let config_digest = sha256_file(&config_path)?;
    if config_digest != manifest.config_sha256 {
        return Err(EngineError::Unsupported(format!(
            "hybrid-block fixture config digest drift: fixture {}, model {}",
            manifest.config_sha256, config_digest
        )));
    }
    let fixture_digest = sha256_file(fixture_path)?;
    if fixture_digest != manifest.fixture_sha256 {
        return Err(EngineError::Unsupported(format!(
            "hybrid-block fixture payload digest drift: manifest {}, payload {}",
            manifest.fixture_sha256, fixture_digest
        )));
    }
    Ok(())
}
