use echo_mlx::{Array, Gpu};

use super::attention::{
    execute_attention_runtime, execute_attention_runtime_buffered_with_bound_weights,
};
use super::decoder::{
    MoeKernel, execute_sparse_moe_output, execute_sparse_moe_output_with_bound_weights,
};
use super::gdn::{
    GdnKernel, execute_gdn_layer_with_bound_weights, execute_gdn_layer_with_kernel, require_tensor,
    validate_array,
};
use super::weights::{BoundDecoderLayerWeights, BoundMixerWeights, TensorLookup};
use super::{EngineError, ModelPlan};

pub(crate) struct LayerExecution {
    pub(crate) output: Array,
    pub(crate) first_state: Array,
    pub(crate) second_state: Array,
}

pub(crate) fn execute_gdn_decoder_layer(
    gpu: &Gpu,
    input: &Array,
    initial_conv_state: &Array,
    initial_recurrent_state: &Array,
    weights: &dyn TensorLookup,
    layer_index: usize,
    plan: &ModelPlan,
) -> Result<LayerExecution, EngineError> {
    let gdn_kernel = GdnKernel::new(gpu, plan)?;
    let moe_kernel = MoeKernel::new(plan)?;
    execute_gdn_decoder_layer_with_kernel(
        gpu,
        input,
        initial_conv_state,
        initial_recurrent_state,
        weights,
        layer_index,
        plan,
        &gdn_kernel,
        &moe_kernel,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_gdn_decoder_layer_with_kernel(
    gpu: &Gpu,
    input: &Array,
    initial_conv_state: &Array,
    initial_recurrent_state: &Array,
    weights: &dyn TensorLookup,
    layer_index: usize,
    plan: &ModelPlan,
    gdn_kernel: &GdnKernel,
    moe_kernel: &MoeKernel,
) -> Result<LayerExecution, EngineError> {
    let layer_prefix = format!("language_model.model.layers.{layer_index}");
    let normalized_input = normalize_layer_input(gpu, input, weights, &layer_prefix, plan)?;
    let gdn = execute_gdn_layer_with_kernel(
        gpu,
        &normalized_input,
        initial_conv_state,
        initial_recurrent_state,
        weights,
        &format!("{layer_prefix}.linear_attn"),
        plan,
        gdn_kernel,
    )?;
    let first_state = gdn
        .require("conv_state")?
        .try_clone()
        .map_err(EngineError::Mlx)?;
    let second_state = gdn
        .require("recurrent_state")?
        .try_clone()
        .map_err(EngineError::Mlx)?;
    let output = finish_decoder_layer(
        gpu,
        input,
        gdn.require("output")?,
        weights,
        &layer_prefix,
        plan,
        moe_kernel,
    )?;
    Ok(LayerExecution {
        output,
        first_state,
        second_state,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_gdn_decoder_layer_with_bound_weights(
    gpu: &Gpu,
    input: &Array,
    initial_conv_state: &Array,
    initial_recurrent_state: &Array,
    weights: &BoundDecoderLayerWeights,
    plan: &ModelPlan,
    gdn_kernel: &GdnKernel,
    moe_kernel: &MoeKernel,
) -> Result<LayerExecution, EngineError> {
    let BoundMixerWeights::Gdn(gdn_weights) = &weights.mixer else {
        return Err(EngineError::Unsupported(
            "runtime GDN state reached an attention-weight layer".into(),
        ));
    };
    let normalized_input = gpu
        .rms_norm(input, Some(&weights.input_norm), plan.rms_norm_epsilon)
        .map_err(EngineError::Mlx)?;
    let gdn = execute_gdn_layer_with_bound_weights(
        gpu,
        &normalized_input,
        initial_conv_state,
        initial_recurrent_state,
        gdn_weights,
        plan,
        gdn_kernel,
    )?;
    let first_state = gdn
        .require("conv_state")?
        .try_clone()
        .map_err(EngineError::Mlx)?;
    let second_state = gdn
        .require("recurrent_state")?
        .try_clone()
        .map_err(EngineError::Mlx)?;
    let output = finish_bound_decoder_layer(
        gpu,
        input,
        gdn.require("output")?,
        weights,
        plan,
        moe_kernel,
    )?;
    Ok(LayerExecution {
        output,
        first_state,
        second_state,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_attention_decoder_layer(
    gpu: &Gpu,
    input: &Array,
    initial_keys: &Array,
    initial_values: &Array,
    weights: &dyn TensorLookup,
    layer_index: usize,
    plan: &ModelPlan,
    offset: usize,
    causal: bool,
    moe_kernel: &MoeKernel,
) -> Result<LayerExecution, EngineError> {
    let layer_prefix = format!("language_model.model.layers.{layer_index}");
    let normalized_input = normalize_layer_input(gpu, input, weights, &layer_prefix, plan)?;
    let attention = execute_attention_runtime(
        gpu,
        &normalized_input,
        initial_keys,
        initial_values,
        weights,
        &format!("{layer_prefix}.self_attn"),
        plan,
        offset,
        causal,
    )?;
    let first_state = attention
        .require("keys")?
        .try_clone()
        .map_err(EngineError::Mlx)?;
    let second_state = attention
        .require("values")?
        .try_clone()
        .map_err(EngineError::Mlx)?;
    let output = finish_decoder_layer(
        gpu,
        input,
        attention.require("output")?,
        weights,
        &layer_prefix,
        plan,
        moe_kernel,
    )?;
    Ok(LayerExecution {
        output,
        first_state,
        second_state,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_buffered_attention_decoder_layer_with_bound_weights(
    gpu: &Gpu,
    input: &Array,
    key_buffer: &Array,
    value_buffer: &Array,
    weights: &BoundDecoderLayerWeights,
    plan: &ModelPlan,
    offset: usize,
    causal: bool,
    moe_kernel: &MoeKernel,
) -> Result<LayerExecution, EngineError> {
    let BoundMixerWeights::Attention(attention_weights) = &weights.mixer else {
        return Err(EngineError::Unsupported(
            "runtime attention state reached a GDN-weight layer".into(),
        ));
    };
    let normalized_input = gpu
        .rms_norm(input, Some(&weights.input_norm), plan.rms_norm_epsilon)
        .map_err(EngineError::Mlx)?;
    let attention = execute_attention_runtime_buffered_with_bound_weights(
        gpu,
        &normalized_input,
        key_buffer,
        value_buffer,
        attention_weights,
        plan,
        offset,
        causal,
    )?;
    let first_state = attention
        .require("keys")?
        .try_clone()
        .map_err(EngineError::Mlx)?;
    let second_state = attention
        .require("values")?
        .try_clone()
        .map_err(EngineError::Mlx)?;
    let output = finish_bound_decoder_layer(
        gpu,
        input,
        attention.require("output")?,
        weights,
        plan,
        moe_kernel,
    )?;
    Ok(LayerExecution {
        output,
        first_state,
        second_state,
    })
}

fn normalize_layer_input(
    gpu: &Gpu,
    input: &Array,
    weights: &dyn TensorLookup,
    layer_prefix: &str,
    plan: &ModelPlan,
) -> Result<Array, EngineError> {
    let weight = require_tensor(weights, &format!("{layer_prefix}.input_layernorm.weight"))?;
    validate_array(
        weight,
        &[plan.hidden_size],
        input.dtype(),
        "input layer normalization weight",
    )?;
    gpu.rms_norm(input, Some(weight), plan.rms_norm_epsilon)
        .map_err(EngineError::Mlx)
}

fn finish_decoder_layer(
    gpu: &Gpu,
    input: &Array,
    attention_output: &Array,
    weights: &dyn TensorLookup,
    layer_prefix: &str,
    plan: &ModelPlan,
    moe_kernel: &MoeKernel,
) -> Result<Array, EngineError> {
    let post_attention_hidden = gpu.add(input, attention_output).map_err(EngineError::Mlx)?;
    let norm_weight = require_tensor(
        weights,
        &format!("{layer_prefix}.post_attention_layernorm.weight"),
    )?;
    validate_array(
        norm_weight,
        &[plan.hidden_size],
        input.dtype(),
        "post-attention layer normalization weight",
    )?;
    let normalized_hidden = gpu
        .rms_norm(
            &post_attention_hidden,
            Some(norm_weight),
            plan.rms_norm_epsilon,
        )
        .map_err(EngineError::Mlx)?;
    let moe_output = execute_sparse_moe_output(
        gpu,
        &normalized_hidden,
        weights,
        &format!("{layer_prefix}.mlp"),
        plan,
        moe_kernel,
    )?;
    gpu.add(&post_attention_hidden, &moe_output)
        .map_err(EngineError::Mlx)
}

fn finish_bound_decoder_layer(
    gpu: &Gpu,
    input: &Array,
    attention_output: &Array,
    weights: &BoundDecoderLayerWeights,
    plan: &ModelPlan,
    moe_kernel: &MoeKernel,
) -> Result<Array, EngineError> {
    let post_attention_hidden = gpu.add(input, attention_output).map_err(EngineError::Mlx)?;
    let normalized_hidden = gpu
        .rms_norm(
            &post_attention_hidden,
            Some(&weights.post_attention_norm),
            plan.rms_norm_epsilon,
        )
        .map_err(EngineError::Mlx)?;
    let moe_output = execute_sparse_moe_output_with_bound_weights(
        gpu,
        &normalized_hidden,
        &weights.moe,
        plan,
        moe_kernel,
    )?;
    gpu.add(&post_attention_hidden, &moe_output)
        .map_err(EngineError::Mlx)
}
