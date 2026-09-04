use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use echo_mlx::{Array, DType, DynamicRopeConfig, Gpu, RopeConfig, SafeTensors};
use serde::{Deserialize, Serialize};

use super::decoder::{MoeKernel, execute_sparse_moe};
use super::gdn::{
    apply_bound_quantized_linear, dimension_f32, dimension_i32, load_weight_shard_containing,
    quantized_linear_with_config, require_tensor, validate_array,
};
use super::weights::{BoundAttentionWeights, TensorLookup};
use super::{EngineError, ModelPlan, sha256_file};

#[derive(Debug, Deserialize)]
struct FixtureManifest {
    schema_version: u32,
    model_type: String,
    config_sha256: String,
    layer_index: usize,
    prefix_length: usize,
    continuation_length: usize,
    input_dtype: String,
    mask_mode: String,
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
    attention_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
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

pub(crate) struct AttentionExecution {
    output: Array,
    keys: Array,
    values: Array,
    trace: Option<BTreeMap<&'static str, Array>>,
}

enum AttentionWeightSource<'a> {
    Named {
        weights: &'a dyn TensorLookup,
        prefix: &'a str,
    },
    Bound(&'a BoundAttentionWeights),
}

#[derive(Clone, Copy)]
enum AttentionLinear {
    Query,
    Key,
    Value,
    Output,
}

impl AttentionWeightSource<'_> {
    fn apply_linear(
        &self,
        gpu: &Gpu,
        input: &Array,
        projection: AttentionLinear,
        plan: &ModelPlan,
    ) -> Result<Array, EngineError> {
        match self {
            Self::Named { weights, prefix } => {
                let suffix = match projection {
                    AttentionLinear::Query => "q_proj",
                    AttentionLinear::Key => "k_proj",
                    AttentionLinear::Value => "v_proj",
                    AttentionLinear::Output => "o_proj",
                };
                quantized_linear_with_config(
                    gpu,
                    input,
                    *weights,
                    &format!("{prefix}.{suffix}"),
                    plan.quantization_group_size,
                    plan.quantization_bits,
                    &plan.quantization_mode,
                )
            }
            Self::Bound(weights) => {
                let weights = match projection {
                    AttentionLinear::Query => &weights.q_proj,
                    AttentionLinear::Key => &weights.k_proj,
                    AttentionLinear::Value => &weights.v_proj,
                    AttentionLinear::Output => &weights.o_proj,
                };
                apply_bound_quantized_linear(gpu, input, weights)
            }
        }
    }

    fn q_norm_weight(&self) -> Result<&Array, EngineError> {
        match self {
            Self::Named { weights, prefix } => {
                require_tensor(*weights, &format!("{prefix}.q_norm.weight"))
            }
            Self::Bound(weights) => Ok(&weights.q_norm_weight),
        }
    }

    fn k_norm_weight(&self) -> Result<&Array, EngineError> {
        match self {
            Self::Named { weights, prefix } => {
                require_tensor(*weights, &format!("{prefix}.k_norm.weight"))
            }
            Self::Bound(weights) => Ok(&weights.k_norm_weight),
        }
    }
}

impl AttentionExecution {
    pub(crate) fn require(&self, name: &'static str) -> Result<&Array, EngineError> {
        match name {
            "output" => Ok(&self.output),
            "keys" => Ok(&self.keys),
            "values" => Ok(&self.values),
            _ => self
                .trace
                .as_ref()
                .and_then(|trace| trace.get(name))
                .ok_or_else(|| {
                    EngineError::Unsupported(format!("internal attention trace omitted {name}"))
                }),
        }
    }
}

/// Direct comparison between Qwen3.5 full-attention decoder layer 3 executed
/// from Rust and the official Python/MLX call.
#[derive(Clone, Debug, Serialize)]
pub struct AttentionLayerParity {
    pub architecture: String,
    pub layer_index: usize,
    pub batch_size: usize,
    pub prefix_length: usize,
    pub continuation_length: usize,
    pub fixture_sha256: String,
    pub output_max_absolute_difference: f32,
    pub keys_max_absolute_difference: f32,
    pub values_max_absolute_difference: f32,
    pub trace_max_absolute_difference: f32,
    pub trace_differences: BTreeMap<String, f32>,
    pub exact: bool,
}

/// Executes Qwen3.5 full-attention decoder layer 3 through Rust and MLX C,
/// including partial `RoPE`, grouped-query causal attention, KV state carry, both
/// residuals, and the sparse `MoE`.
///
/// # Errors
///
/// Returns [`EngineError`] when the model, oracle, weights, shapes, dtypes, or
/// an MLX operation do not match the admitted execution plan.
#[allow(clippy::too_many_lines)]
pub fn run_attention_layer_parity(
    model_directory: &Path,
    fixture_path: &Path,
    manifest_path: &Path,
) -> Result<AttentionLayerParity, EngineError> {
    let plan = ModelPlan::from_directory(model_directory)?;
    let manifest = load_manifest(manifest_path)?;
    validate_manifest(&plan, model_directory, fixture_path, &manifest)?;

    let fixture = SafeTensors::load(fixture_path).map_err(EngineError::Mlx)?;
    if fixture.len() != manifest.fixture_tensor_count {
        return Err(EngineError::Unsupported(format!(
            "attention fixture tensor count drift: manifest {}, payload {}",
            manifest.fixture_tensor_count,
            fixture.len()
        )));
    }

    let layer_prefix = format!("language_model.model.layers.{}", manifest.layer_index);
    let attention_prefix = format!("{layer_prefix}.self_attn");
    let mlp_prefix = format!("{layer_prefix}.mlp");
    let marker = format!("{attention_prefix}.q_proj.weight");
    let weights = load_weight_shard_containing(model_directory, &marker)?;
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
        &[batch_size, manifest.continuation_length, plan.hidden_size],
        DType::BFloat16,
        "continuation_input",
    )?;
    if sequence_length != manifest.continuation_length || hidden_size != plan.hidden_size {
        return Err(EngineError::Unsupported(
            "continuation input dimensions changed after validation".into(),
        ));
    }

    let input_norm_weight =
        require_tensor(&weights, &format!("{layer_prefix}.input_layernorm.weight"))?;
    validate_array(
        input_norm_weight,
        &[plan.hidden_size],
        input.dtype(),
        "input layer normalization weight",
    )?;
    let normalized_input = gpu
        .rms_norm(input, Some(input_norm_weight), plan.rms_norm_epsilon)
        .map_err(EngineError::Mlx)?;
    let initial_keys = require_tensor(&fixture, "initial_keys")?;
    let initial_values = require_tensor(&fixture, "initial_values")?;
    let attention = execute_attention(
        &gpu,
        &normalized_input,
        initial_keys,
        initial_values,
        &weights,
        &attention_prefix,
        &plan,
        manifest.prefix_length,
        true,
    )?;
    let post_attention_hidden = gpu
        .add(input, attention.require("output")?)
        .map_err(EngineError::Mlx)?;

    let post_attention_norm_weight = require_tensor(
        &weights,
        &format!("{layer_prefix}.post_attention_layernorm.weight"),
    )?;
    validate_array(
        post_attention_norm_weight,
        &[plan.hidden_size],
        input.dtype(),
        "post-attention layer normalization weight",
    )?;
    let normalized_hidden = gpu
        .rms_norm(
            &post_attention_hidden,
            Some(post_attention_norm_weight),
            plan.rms_norm_epsilon,
        )
        .map_err(EngineError::Mlx)?;
    let moe = execute_sparse_moe(
        &gpu,
        &normalized_hidden,
        &weights,
        &mlp_prefix,
        &plan,
        &moe_kernel,
    )?;
    let output = gpu
        .add(&post_attention_hidden, moe.require("moe_output")?)
        .map_err(EngineError::Mlx)?;

    let expected_output = require_tensor(&fixture, "expected_output")?;
    let expected_keys = require_tensor(&fixture, "expected_keys")?;
    let expected_values = require_tensor(&fixture, "expected_values")?;
    let output_difference = gpu
        .max_abs_difference(&output, expected_output)
        .map_err(EngineError::Mlx)?;
    let keys_difference = gpu
        .max_abs_difference(attention.require("keys")?, expected_keys)
        .map_err(EngineError::Mlx)?;
    let values_difference = gpu
        .max_abs_difference(attention.require("values")?, expected_values)
        .map_err(EngineError::Mlx)?;

    let mut trace_differences = BTreeMap::new();
    for (name, actual) in [
        ("attention_output", attention.require("output")?),
        ("decoder_output", &output),
        ("normalized_hidden", &normalized_hidden),
        ("normalized_input", &normalized_input),
        ("post_attention_hidden", &post_attention_hidden),
    ] {
        compare_trace(&gpu, &fixture, name, actual, &mut trace_differences)?;
    }
    for name in [
        "attention_flat",
        "attention_heads",
        "gate",
        "gated_attention",
        "key_projection",
        "keys",
        "new_keys",
        "new_values",
        "normalized_keys",
        "normalized_queries",
        "output",
        "q_projection",
        "queries",
        "rotated_new_keys",
        "rotated_queries",
        "value_projection",
        "values",
    ] {
        compare_named_trace(
            &gpu,
            &fixture,
            &format!("attention.{name}"),
            attention.require(name)?,
            &mut trace_differences,
        )?;
    }
    for name in [
        "expert_activated",
        "expert_gate",
        "expert_indices",
        "expert_outputs",
        "expert_scores",
        "expert_up",
        "moe_output",
        "routed_output",
        "router_logits",
        "router_probabilities",
        "shared_activated",
        "shared_expert_output",
        "shared_gate",
        "shared_gate_logits",
        "shared_gate_projection",
        "shared_output",
        "shared_up_projection",
    ] {
        compare_trace(
            &gpu,
            &fixture,
            name,
            moe.require(name)?,
            &mut trace_differences,
        )?;
    }
    if manifest.sorted_expert_path {
        for name in [
            "expert_inverse_order",
            "expert_sort_order",
            "expert_sorted_indices",
            "expert_sorted_inputs",
            "expert_sorted_outputs",
        ] {
            compare_trace(
                &gpu,
                &fixture,
                name,
                moe.require(name)?,
                &mut trace_differences,
            )?;
        }
    }

    let trace_max_absolute_difference = trace_differences.values().copied().fold(0.0_f32, f32::max);
    let exact = output_difference == 0.0
        && keys_difference == 0.0
        && values_difference == 0.0
        && trace_max_absolute_difference == 0.0;

    Ok(AttentionLayerParity {
        architecture: plan.architecture,
        layer_index: manifest.layer_index,
        batch_size,
        prefix_length: manifest.prefix_length,
        continuation_length: manifest.continuation_length,
        fixture_sha256: manifest.fixture_sha256,
        output_max_absolute_difference: output_difference,
        keys_max_absolute_difference: keys_difference,
        values_max_absolute_difference: values_difference,
        trace_max_absolute_difference,
        trace_differences,
        exact,
    })
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn execute_attention(
    gpu: &Gpu,
    input: &Array,
    initial_keys: &Array,
    initial_values: &Array,
    weights: &dyn TensorLookup,
    prefix: &str,
    plan: &ModelPlan,
    offset: usize,
    causal: bool,
) -> Result<AttentionExecution, EngineError> {
    execute_attention_impl(
        gpu,
        input,
        initial_keys,
        initial_values,
        &AttentionWeightSource::Named { weights, prefix },
        plan,
        offset,
        causal,
        None,
        None,
        None,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_attention_runtime(
    gpu: &Gpu,
    input: &Array,
    initial_keys: &Array,
    initial_values: &Array,
    weights: &dyn TensorLookup,
    prefix: &str,
    plan: &ModelPlan,
    offset: usize,
    causal: bool,
) -> Result<AttentionExecution, EngineError> {
    execute_attention_impl(
        gpu,
        input,
        initial_keys,
        initial_values,
        &AttentionWeightSource::Named { weights, prefix },
        plan,
        offset,
        causal,
        None,
        None,
        None,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_attention_runtime_buffered_with_bound_weights(
    gpu: &Gpu,
    input: &Array,
    key_buffer: &Array,
    value_buffer: &Array,
    weights: &BoundAttentionWeights,
    plan: &ModelPlan,
    offset: usize,
    causal: bool,
    rope_offsets: Option<&Array>,
    attention_mask: Option<&Array>,
) -> Result<AttentionExecution, EngineError> {
    let capacity =
        key_buffer.shape().get(2).copied().ok_or_else(|| {
            EngineError::Unsupported("attention key buffer must have rank 4".into())
        })?;
    execute_attention_impl(
        gpu,
        input,
        key_buffer,
        value_buffer,
        &AttentionWeightSource::Bound(weights),
        plan,
        offset,
        causal,
        rope_offsets,
        attention_mask,
        Some(capacity),
        false,
    )
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn execute_attention_impl(
    gpu: &Gpu,
    input: &Array,
    initial_keys: &Array,
    initial_values: &Array,
    weights: &AttentionWeightSource<'_>,
    plan: &ModelPlan,
    offset: usize,
    causal: bool,
    rope_offsets: Option<&Array>,
    attention_mask: Option<&Array>,
    cache_capacity: Option<usize>,
    capture_trace: bool,
) -> Result<AttentionExecution, EngineError> {
    let shape = input.shape();
    let [batch_size, sequence_length, hidden_size] = <[usize; 3]>::try_from(shape.clone())
        .map_err(|shape| {
            EngineError::Unsupported(format!(
                "attention input must be rank 3, observed {shape:?}"
            ))
        })?;
    if hidden_size != plan.hidden_size {
        return Err(EngineError::Unsupported(format!(
            "attention hidden size mismatch: expected {}, observed {hidden_size}",
            plan.hidden_size
        )));
    }
    let q_projection = weights.apply_linear(gpu, input, AttentionLinear::Query, plan)?;
    let q_projection_heads = gpu
        .reshape(
            &q_projection,
            &[
                batch_size,
                sequence_length,
                plan.attention_head_count,
                2 * plan.attention_head_dimension,
            ],
        )
        .map_err(EngineError::Mlx)?;
    let queries = last_axis_slice_4d(
        gpu,
        &q_projection_heads,
        0,
        plan.attention_head_dimension,
        "attention queries",
    )?;
    let gate_heads = last_axis_slice_4d(
        gpu,
        &q_projection_heads,
        plan.attention_head_dimension,
        2 * plan.attention_head_dimension,
        "attention output gate",
    )?;
    let gate = gpu
        .reshape(
            &gate_heads,
            &[
                batch_size,
                sequence_length,
                plan.attention_head_count * plan.attention_head_dimension,
            ],
        )
        .map_err(EngineError::Mlx)?;
    let key_projection = weights.apply_linear(gpu, input, AttentionLinear::Key, plan)?;
    let value_projection = weights.apply_linear(gpu, input, AttentionLinear::Value, plan)?;

    let q_norm_weight = weights.q_norm_weight()?;
    let k_norm_weight = weights.k_norm_weight()?;
    if matches!(weights, AttentionWeightSource::Named { .. }) {
        for (name, weight) in [
            ("query normalization weight", q_norm_weight),
            ("key normalization weight", k_norm_weight),
        ] {
            validate_array(
                weight,
                &[plan.attention_head_dimension],
                input.dtype(),
                name,
            )?;
        }
    }
    let normalized_queries = gpu
        .rms_norm(&queries, Some(q_norm_weight), plan.rms_norm_epsilon)
        .map_err(EngineError::Mlx)?;
    let key_projection_heads = gpu
        .reshape(
            &key_projection,
            &[
                batch_size,
                sequence_length,
                plan.key_value_head_count,
                plan.attention_head_dimension,
            ],
        )
        .map_err(EngineError::Mlx)?;
    let normalized_keys = gpu
        .rms_norm(
            &key_projection_heads,
            Some(k_norm_weight),
            plan.rms_norm_epsilon,
        )
        .map_err(EngineError::Mlx)?;
    let queries = gpu
        .transpose(&normalized_queries, &[0, 2, 1, 3])
        .map_err(EngineError::Mlx)?;
    let new_keys = gpu
        .transpose(&normalized_keys, &[0, 2, 1, 3])
        .map_err(EngineError::Mlx)?;
    let new_values = gpu
        .reshape(
            &value_projection,
            &[
                batch_size,
                sequence_length,
                plan.key_value_head_count,
                plan.attention_head_dimension,
            ],
        )
        .and_then(|value| gpu.transpose(&value, &[0, 2, 1, 3]))
        .map_err(EngineError::Mlx)?;

    let logical_length = offset
        .checked_add(sequence_length)
        .ok_or_else(|| EngineError::Unsupported("attention cache length overflow".into()))?;
    let state_length = cache_capacity.unwrap_or(offset);
    if cache_capacity.is_some() && state_length < logical_length {
        return Err(EngineError::Unsupported(format!(
            "attention cache capacity {state_length} cannot hold offset {offset} plus {sequence_length} tokens"
        )));
    }
    validate_array(
        initial_keys,
        &[
            batch_size,
            plan.key_value_head_count,
            state_length,
            plan.attention_head_dimension,
        ],
        input.dtype(),
        "initial attention keys",
    )?;
    validate_array(
        initial_values,
        &[
            batch_size,
            plan.key_value_head_count,
            state_length,
            plan.attention_head_dimension,
        ],
        input.dtype(),
        "initial attention values",
    )?;
    let rotary_dimension = dimension_i32(plan.rotary_dimension, "rotary dimension")?;
    let (rotated_queries, rotated_new_keys) = if let Some(rope_offsets) = rope_offsets {
        validate_array(
            rope_offsets,
            &[batch_size],
            DType::Int32,
            "attention per-row RoPE offsets",
        )?;
        let rope = DynamicRopeConfig {
            dimensions: rotary_dimension,
            traditional: false,
            base: Some(plan.rope_base),
            scale: 1.0,
            offsets: rope_offsets,
            frequencies: None,
        };
        (
            gpu.rope_dynamic(&queries, rope).map_err(EngineError::Mlx)?,
            gpu.rope_dynamic(&new_keys, rope)
                .map_err(EngineError::Mlx)?,
        )
    } else {
        let rope = RopeConfig {
            dimensions: rotary_dimension,
            traditional: false,
            base: Some(plan.rope_base),
            scale: 1.0,
            offset: dimension_i32(offset, "attention offset")?,
            frequencies: None,
        };
        (
            gpu.rope(&queries, rope).map_err(EngineError::Mlx)?,
            gpu.rope(&new_keys, rope).map_err(EngineError::Mlx)?,
        )
    };
    let (keys, values) = if cache_capacity.is_some() {
        let start = [0, 0, dimension_i32(offset, "attention cache offset")?, 0];
        let stop = [
            dimension_i32(batch_size, "attention cache batch size")?,
            dimension_i32(plan.key_value_head_count, "attention cache heads")?,
            dimension_i32(logical_length, "attention cache logical length")?,
            dimension_i32(
                plan.attention_head_dimension,
                "attention cache head dimension",
            )?,
        ];
        (
            gpu.slice_update(
                initial_keys,
                &rotated_new_keys,
                &start,
                &stop,
                &[1, 1, 1, 1],
            )
            .map_err(EngineError::Mlx)?,
            gpu.slice_update(initial_values, &new_values, &start, &stop, &[1, 1, 1, 1])
                .map_err(EngineError::Mlx)?,
        )
    } else {
        (
            gpu.concatenate(&[initial_keys, &rotated_new_keys], 2)
                .map_err(EngineError::Mlx)?,
            gpu.concatenate(&[initial_values, &new_values], 2)
                .map_err(EngineError::Mlx)?,
        )
    };
    let (attention_keys, attention_values) = if cache_capacity.is_some() {
        let stop = [
            dimension_i32(batch_size, "attention cache batch size")?,
            dimension_i32(plan.key_value_head_count, "attention cache heads")?,
            dimension_i32(logical_length, "attention cache logical length")?,
            dimension_i32(
                plan.attention_head_dimension,
                "attention cache head dimension",
            )?,
        ];
        (
            gpu.slice(&keys, &[0, 0, 0, 0], &stop, &[1, 1, 1, 1])
                .map_err(EngineError::Mlx)?,
            gpu.slice(&values, &[0, 0, 0, 0], &stop, &[1, 1, 1, 1])
                .map_err(EngineError::Mlx)?,
        )
    } else {
        (
            keys.try_clone().map_err(EngineError::Mlx)?,
            values.try_clone().map_err(EngineError::Mlx)?,
        )
    };

    let head_dimension = dimension_f32(plan.attention_head_dimension, "attention head dimension")?;
    let attention_scale = 1.0 / head_dimension.sqrt();
    let attention_heads = if let Some(attention_mask) = attention_mask {
        validate_array(
            attention_mask,
            &[batch_size, 1, sequence_length, logical_length],
            DType::Bool,
            "attention per-row mask",
        )?;
        gpu.scaled_dot_product_attention_with_mask(
            &rotated_queries,
            &attention_keys,
            &attention_values,
            attention_scale,
            attention_mask,
        )
        .map_err(EngineError::Mlx)?
    } else {
        gpu.scaled_dot_product_attention(
            &rotated_queries,
            &attention_keys,
            &attention_values,
            attention_scale,
            causal,
        )
        .map_err(EngineError::Mlx)?
    };
    let attention_flat = gpu
        .transpose(&attention_heads, &[0, 2, 1, 3])
        .and_then(|value| {
            gpu.reshape(
                &value,
                &[
                    batch_size,
                    sequence_length,
                    plan.attention_head_count * plan.attention_head_dimension,
                ],
            )
        })
        .map_err(EngineError::Mlx)?;
    let gated_attention = gpu
        .sigmoid(&gate)
        .and_then(|gate_value| gpu.multiply(&attention_flat, &gate_value))
        .map_err(EngineError::Mlx)?;
    let output = weights.apply_linear(gpu, &gated_attention, AttentionLinear::Output, plan)?;

    let trace = capture_trace.then(|| {
        BTreeMap::from([
            ("attention_flat", attention_flat),
            ("attention_heads", attention_heads),
            ("gate", gate),
            ("gated_attention", gated_attention),
            ("key_projection", key_projection),
            ("new_keys", new_keys),
            ("new_values", new_values),
            ("normalized_keys", normalized_keys),
            ("normalized_queries", normalized_queries),
            ("q_projection", q_projection),
            ("queries", queries),
            ("rotated_new_keys", rotated_new_keys),
            ("rotated_queries", rotated_queries),
            ("value_projection", value_projection),
        ])
    });
    Ok(AttentionExecution {
        output,
        keys,
        values,
        trace,
    })
}

fn last_axis_slice_4d(
    gpu: &Gpu,
    array: &Array,
    feature_start: usize,
    feature_stop: usize,
    name: &str,
) -> Result<Array, EngineError> {
    let shape = array.shape();
    let [batch_size, sequence_length, heads, features] = <[usize; 4]>::try_from(shape.clone())
        .map_err(|shape| {
            EngineError::Unsupported(format!("{name} source must be rank 4: {shape:?}"))
        })?;
    if feature_start > feature_stop || feature_stop > features {
        return Err(EngineError::Unsupported(format!(
            "{name} cannot take [{feature_start}, {feature_stop}) from {shape:?}"
        )));
    }
    gpu.slice(
        array,
        &[0, 0, 0, dimension_i32(feature_start, name)?],
        &[
            dimension_i32(batch_size, name)?,
            dimension_i32(sequence_length, name)?,
            dimension_i32(heads, name)?,
            dimension_i32(feature_stop, name)?,
        ],
        &[1, 1, 1, 1],
    )
    .map_err(EngineError::Mlx)
}

fn compare_trace(
    gpu: &Gpu,
    fixture: &SafeTensors,
    name: &str,
    actual: &Array,
    differences: &mut BTreeMap<String, f32>,
) -> Result<(), EngineError> {
    compare_named_trace(gpu, fixture, name, actual, differences)
}

fn compare_named_trace(
    gpu: &Gpu,
    fixture: &SafeTensors,
    fixture_name: &str,
    actual: &Array,
    differences: &mut BTreeMap<String, f32>,
) -> Result<(), EngineError> {
    let expected = require_tensor(fixture, &format!("trace.{fixture_name}"))?;
    let difference = gpu
        .max_abs_difference(actual, expected)
        .map_err(EngineError::Mlx)?;
    differences.insert(fixture_name.to_owned(), difference);
    Ok(())
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
            "attention fixture schema must be 1, observed {}",
            manifest.schema_version
        )));
    }
    if plan.architecture != "qwen3_5_moe" || manifest.model_type != plan.architecture {
        return Err(EngineError::Unsupported(format!(
            "attention fixture requires qwen3_5_moe, observed fixture {} and model {}",
            manifest.model_type, plan.architecture
        )));
    }
    if manifest.layer_index + 1 != plan.full_attention_interval {
        return Err(EngineError::Unsupported(format!(
            "the current attention milestone admits layer {}, observed {}",
            plan.full_attention_interval - 1,
            manifest.layer_index
        )));
    }
    if manifest.prefix_length == 0 || manifest.continuation_length <= 1 {
        return Err(EngineError::Unsupported(
            "attention fixture requires a non-empty prefix and multi-token continuation".into(),
        ));
    }
    if manifest.mask_mode != "causal" {
        return Err(EngineError::Unsupported(format!(
            "attention fixture must use the causal mask, observed {}",
            manifest.mask_mode
        )));
    }
    let selected_experts = manifest
        .continuation_length
        .checked_mul(plan.experts_per_token)
        .ok_or_else(|| {
            EngineError::Unsupported("attention fixture expert-selection count overflow".into())
        })?;
    if manifest.sorted_expert_path != (selected_experts >= 64) {
        return Err(EngineError::Unsupported(format!(
            "attention fixture sorted-expert path drift: fixture {}, selection count {selected_experts}",
            manifest.sorted_expert_path
        )));
    }
    if manifest.input_dtype != "mlx.core.bfloat16" {
        return Err(EngineError::Unsupported(format!(
            "attention fixture must use bfloat16 input, observed {}",
            manifest.input_dtype
        )));
    }
    let dimensions = FixtureDimensions {
        hidden_size: plan.hidden_size,
        attention_heads: plan.attention_head_count,
        key_value_heads: plan.key_value_head_count,
        head_dim: plan.attention_head_dimension,
        rotary_dim: plan.rotary_dimension,
        expert_count: plan.expert_count,
        experts_per_token: plan.experts_per_token,
        moe_intermediate_size: plan.moe_intermediate_size,
        shared_expert_intermediate_size: plan.shared_expert_intermediate_size,
    };
    if manifest.dimensions != dimensions {
        return Err(EngineError::Unsupported(format!(
            "attention fixture dimension drift: fixture {:?}, model {:?}",
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
            "attention fixture RoPE drift: fixture {:?}, model {:?}",
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
            "attention fixture quantization drift: fixture {:?}, model {:?}",
            manifest.quantization, quantization
        )));
    }
    if manifest.norm_topk_prob != plan.norm_topk_prob {
        return Err(EngineError::Unsupported(format!(
            "attention fixture norm_topk_prob drift: fixture {}, model {}",
            manifest.norm_topk_prob, plan.norm_topk_prob
        )));
    }
    let config_path = model_directory.join("config.json");
    let config_digest = sha256_file(&config_path)?;
    if config_digest != manifest.config_sha256 {
        return Err(EngineError::Unsupported(format!(
            "attention fixture config digest drift: fixture {}, model {}",
            manifest.config_sha256, config_digest
        )));
    }
    let fixture_digest = sha256_file(fixture_path)?;
    if fixture_digest != manifest.fixture_sha256 {
        return Err(EngineError::Unsupported(format!(
            "attention fixture payload digest drift: manifest {}, payload {}",
            manifest.fixture_sha256, fixture_digest
        )));
    }
    Ok(())
}
