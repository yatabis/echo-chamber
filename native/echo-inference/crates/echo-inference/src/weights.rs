use std::collections::BTreeMap;
use std::path::Path;

use echo_mlx::{Array, DType, SafeTensors};

use super::{EngineError, ModelPlan, safetensor_files};

pub(crate) struct QuantizedTensorRefs<'a> {
    pub(crate) weight: &'a Array,
    pub(crate) scales: &'a Array,
    pub(crate) biases: &'a Array,
}

pub(crate) struct BoundQuantizedWeights {
    pub(crate) weight: Array,
    pub(crate) scales: Array,
    pub(crate) biases: Array,
    pub(crate) group_size: i32,
    pub(crate) bits: i32,
    pub(crate) mode: String,
}

pub(crate) struct BoundGdnWeights {
    pub(crate) in_proj_qkv: BoundQuantizedWeights,
    pub(crate) in_proj_z: BoundQuantizedWeights,
    pub(crate) in_proj_a: BoundQuantizedWeights,
    pub(crate) in_proj_b: BoundQuantizedWeights,
    pub(crate) conv_weight: Array,
    pub(crate) dt_bias: Array,
    pub(crate) a_log: Array,
    pub(crate) norm_weight: Array,
    pub(crate) out_proj: BoundQuantizedWeights,
}

pub(crate) struct BoundAttentionWeights {
    pub(crate) q_proj: BoundQuantizedWeights,
    pub(crate) k_proj: BoundQuantizedWeights,
    pub(crate) v_proj: BoundQuantizedWeights,
    pub(crate) q_norm_weight: Array,
    pub(crate) k_norm_weight: Array,
    pub(crate) o_proj: BoundQuantizedWeights,
}

pub(crate) enum BoundMixerWeights {
    Gdn(BoundGdnWeights),
    Attention(BoundAttentionWeights),
}

pub(crate) struct BoundMoeWeights {
    pub(crate) router: BoundQuantizedWeights,
    pub(crate) expert_gate: BoundQuantizedWeights,
    pub(crate) expert_up: BoundQuantizedWeights,
    pub(crate) expert_down: BoundQuantizedWeights,
    pub(crate) shared_gate: BoundQuantizedWeights,
    pub(crate) shared_up: BoundQuantizedWeights,
    pub(crate) shared_down: BoundQuantizedWeights,
    pub(crate) shared_expert_gate: BoundQuantizedWeights,
}

pub(crate) struct BoundDecoderLayerWeights {
    pub(crate) input_norm: Array,
    pub(crate) post_attention_norm: Array,
    pub(crate) mixer: BoundMixerWeights,
    pub(crate) moe: BoundMoeWeights,
}

/// Model weights admitted once and directly addressable by the decode loop.
///
/// The fixture path intentionally retains string-based lookup so it can audit
/// arbitrary named tensors. The resident runtime instead pays that cost during
/// model admission, validates every static tensor shape and dtype, and retains
/// only typed handles required by Qwen execution.
pub(crate) struct BoundModelWeights {
    pub(crate) embedding: BoundQuantizedWeights,
    pub(crate) layers: Vec<BoundDecoderLayerWeights>,
    pub(crate) final_norm: Array,
    pub(crate) lm_head: BoundQuantizedWeights,
}

pub(crate) trait TensorLookup {
    fn tensor(&self, name: &str) -> Option<&Array>;

    fn quantized_tensors(&self, prefix: &str) -> Option<QuantizedTensorRefs<'_>> {
        let weight = self.tensor(&format!("{prefix}.weight"))?;
        let scales = self.tensor(&format!("{prefix}.scales"))?;
        let biases = self.tensor(&format!("{prefix}.biases"))?;
        Some(QuantizedTensorRefs {
            weight,
            scales,
            biases,
        })
    }
}

impl TensorLookup for SafeTensors {
    fn tensor(&self, name: &str) -> Option<&Array> {
        self.tensor(name)
    }
}

pub(crate) struct ShardedWeights {
    tensors: BTreeMap<String, Array>,
    quantized_tensors: BTreeMap<String, QuantizedTensorSet>,
    shard_count: usize,
}

struct QuantizedTensorSet {
    weight: Array,
    scales: Array,
    biases: Array,
}

impl ShardedWeights {
    pub(crate) fn load(model_directory: &Path) -> Result<Self, EngineError> {
        let paths = safetensor_files(model_directory)?;
        if paths.is_empty() {
            return Err(EngineError::Unsupported(
                "model directory contains no .safetensors files".into(),
            ));
        }

        let shard_count = paths.len();
        let mut tensors = BTreeMap::new();
        for path in paths {
            let shard = SafeTensors::load(&path).map_err(EngineError::Mlx)?;
            for (name, array) in shard.tensors() {
                let array = array.try_clone().map_err(EngineError::Mlx)?;
                if tensors.insert(name.to_owned(), array).is_some() {
                    return Err(EngineError::Unsupported(format!(
                        "model tensor {name} occurs in more than one shard"
                    )));
                }
            }
        }
        let quantized_prefixes = tensors
            .keys()
            .filter_map(|name| name.strip_suffix(".weight"))
            .filter(|prefix| {
                tensors.contains_key(&format!("{prefix}.scales"))
                    && tensors.contains_key(&format!("{prefix}.biases"))
            })
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let mut quantized_tensors = BTreeMap::new();
        for prefix in quantized_prefixes {
            let weight = tensors
                .get(&format!("{prefix}.weight"))
                .ok_or_else(|| {
                    EngineError::Unsupported(format!(
                        "quantized tensor set {prefix} lost its weight"
                    ))
                })?
                .try_clone()
                .map_err(EngineError::Mlx)?;
            let scales = tensors
                .get(&format!("{prefix}.scales"))
                .ok_or_else(|| {
                    EngineError::Unsupported(format!(
                        "quantized tensor set {prefix} lost its scales"
                    ))
                })?
                .try_clone()
                .map_err(EngineError::Mlx)?;
            let biases = tensors
                .get(&format!("{prefix}.biases"))
                .ok_or_else(|| {
                    EngineError::Unsupported(format!(
                        "quantized tensor set {prefix} lost its biases"
                    ))
                })?
                .try_clone()
                .map_err(EngineError::Mlx)?;
            quantized_tensors.insert(
                prefix,
                QuantizedTensorSet {
                    weight,
                    scales,
                    biases,
                },
            );
        }
        Ok(Self {
            tensors,
            quantized_tensors,
            shard_count,
        })
    }

    pub(crate) fn shard_count(&self) -> usize {
        self.shard_count
    }

    pub(crate) fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub(crate) fn bind_model(&self, plan: &ModelPlan) -> Result<BoundModelWeights, EngineError> {
        let embedding = self.bind_quantized_linear(
            "language_model.model.embed_tokens",
            plan.vocabulary_size,
            plan.hidden_size,
            plan.quantization_group_size,
            plan.quantization_bits,
            &plan.quantization_mode,
        )?;
        let mut layers = Vec::with_capacity(plan.layer_count);
        for layer_index in 0..plan.layer_count {
            layers.push(self.bind_decoder_layer(layer_index, plan)?);
        }
        let final_norm = self.bind_array(
            "language_model.model.norm.weight",
            &[plan.hidden_size],
            DType::BFloat16,
        )?;
        let lm_head = self.bind_quantized_linear(
            "language_model.lm_head",
            plan.vocabulary_size,
            plan.hidden_size,
            plan.quantization_group_size,
            plan.quantization_bits,
            &plan.quantization_mode,
        )?;
        Ok(BoundModelWeights {
            embedding,
            layers,
            final_norm,
            lm_head,
        })
    }

    fn bind_decoder_layer(
        &self,
        layer_index: usize,
        plan: &ModelPlan,
    ) -> Result<BoundDecoderLayerWeights, EngineError> {
        let prefix = format!("language_model.model.layers.{layer_index}");
        let input_norm = self.bind_array(
            &format!("{prefix}.input_layernorm.weight"),
            &[plan.hidden_size],
            DType::BFloat16,
        )?;
        let post_attention_norm = self.bind_array(
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[plan.hidden_size],
            DType::BFloat16,
        )?;
        let mixer = if (layer_index + 1).is_multiple_of(plan.full_attention_interval) {
            BoundMixerWeights::Attention(self.bind_attention(&prefix, plan)?)
        } else {
            BoundMixerWeights::Gdn(self.bind_gdn(&prefix, plan)?)
        };
        let moe = self.bind_moe(&prefix, plan)?;
        Ok(BoundDecoderLayerWeights {
            input_norm,
            post_attention_norm,
            mixer,
            moe,
        })
    }

    fn bind_gdn(
        &self,
        layer_prefix: &str,
        plan: &ModelPlan,
    ) -> Result<BoundGdnWeights, EngineError> {
        let prefix = format!("{layer_prefix}.linear_attn");
        let default_linear = |suffix: &str, output_dimension: usize, input_dimension: usize| {
            self.bind_quantized_linear(
                &format!("{prefix}.{suffix}"),
                output_dimension,
                input_dimension,
                plan.quantization_group_size,
                plan.quantization_bits,
                &plan.quantization_mode,
            )
        };
        Ok(BoundGdnWeights {
            in_proj_qkv: default_linear(
                "in_proj_qkv",
                plan.convolution_dimension(),
                plan.hidden_size,
            )?,
            in_proj_z: default_linear("in_proj_z", plan.value_dimension(), plan.hidden_size)?,
            in_proj_a: default_linear("in_proj_a", plan.value_head_count, plan.hidden_size)?,
            in_proj_b: default_linear("in_proj_b", plan.value_head_count, plan.hidden_size)?,
            conv_weight: self.bind_array(
                &format!("{prefix}.conv1d.weight"),
                &[
                    plan.convolution_dimension(),
                    plan.convolution_kernel_size,
                    1,
                ],
                DType::BFloat16,
            )?,
            dt_bias: self.bind_array(
                &format!("{prefix}.dt_bias"),
                &[plan.value_head_count],
                DType::BFloat16,
            )?,
            a_log: self.bind_array(
                &format!("{prefix}.A_log"),
                &[plan.value_head_count],
                DType::BFloat16,
            )?,
            norm_weight: self.bind_array(
                &format!("{prefix}.norm.weight"),
                &[plan.value_head_dimension],
                DType::BFloat16,
            )?,
            out_proj: default_linear("out_proj", plan.hidden_size, plan.value_dimension())?,
        })
    }

    fn bind_attention(
        &self,
        layer_prefix: &str,
        plan: &ModelPlan,
    ) -> Result<BoundAttentionWeights, EngineError> {
        let prefix = format!("{layer_prefix}.self_attn");
        let default_linear = |suffix: &str, output_dimension: usize, input_dimension: usize| {
            self.bind_quantized_linear(
                &format!("{prefix}.{suffix}"),
                output_dimension,
                input_dimension,
                plan.quantization_group_size,
                plan.quantization_bits,
                &plan.quantization_mode,
            )
        };
        let attention_dimension = plan.attention_head_count * plan.attention_head_dimension;
        let key_value_dimension = plan.key_value_head_count * plan.attention_head_dimension;
        Ok(BoundAttentionWeights {
            q_proj: default_linear("q_proj", 2 * attention_dimension, plan.hidden_size)?,
            k_proj: default_linear("k_proj", key_value_dimension, plan.hidden_size)?,
            v_proj: default_linear("v_proj", key_value_dimension, plan.hidden_size)?,
            q_norm_weight: self.bind_array(
                &format!("{prefix}.q_norm.weight"),
                &[plan.attention_head_dimension],
                DType::BFloat16,
            )?,
            k_norm_weight: self.bind_array(
                &format!("{prefix}.k_norm.weight"),
                &[plan.attention_head_dimension],
                DType::BFloat16,
            )?,
            o_proj: default_linear("o_proj", plan.hidden_size, attention_dimension)?,
        })
    }

    fn bind_moe(
        &self,
        layer_prefix: &str,
        plan: &ModelPlan,
    ) -> Result<BoundMoeWeights, EngineError> {
        let prefix = format!("{layer_prefix}.mlp");
        let default_linear = |suffix: &str, output_dimension: usize, input_dimension: usize| {
            self.bind_quantized_linear(
                &format!("{prefix}.{suffix}"),
                output_dimension,
                input_dimension,
                plan.quantization_group_size,
                plan.quantization_bits,
                &plan.quantization_mode,
            )
        };
        Ok(BoundMoeWeights {
            router: self.bind_quantized_linear(
                &format!("{prefix}.gate"),
                plan.expert_count,
                plan.hidden_size,
                plan.router_quantization_group_size,
                plan.router_quantization_bits,
                &plan.router_quantization_mode,
            )?,
            expert_gate: self.bind_quantized_experts(
                &format!("{prefix}.switch_mlp.gate_proj"),
                plan.expert_count,
                plan.moe_intermediate_size,
                plan.hidden_size,
                plan,
            )?,
            expert_up: self.bind_quantized_experts(
                &format!("{prefix}.switch_mlp.up_proj"),
                plan.expert_count,
                plan.moe_intermediate_size,
                plan.hidden_size,
                plan,
            )?,
            expert_down: self.bind_quantized_experts(
                &format!("{prefix}.switch_mlp.down_proj"),
                plan.expert_count,
                plan.hidden_size,
                plan.moe_intermediate_size,
                plan,
            )?,
            shared_gate: default_linear(
                "shared_expert.gate_proj",
                plan.shared_expert_intermediate_size,
                plan.hidden_size,
            )?,
            shared_up: default_linear(
                "shared_expert.up_proj",
                plan.shared_expert_intermediate_size,
                plan.hidden_size,
            )?,
            shared_down: default_linear(
                "shared_expert.down_proj",
                plan.hidden_size,
                plan.shared_expert_intermediate_size,
            )?,
            shared_expert_gate: self.bind_quantized_linear(
                &format!("{prefix}.shared_expert_gate"),
                1,
                plan.hidden_size,
                plan.shared_gate_quantization_group_size,
                plan.shared_gate_quantization_bits,
                &plan.shared_gate_quantization_mode,
            )?,
        })
    }

    fn bind_quantized_experts(
        &self,
        prefix: &str,
        expert_count: usize,
        output_dimension: usize,
        input_dimension: usize,
        plan: &ModelPlan,
    ) -> Result<BoundQuantizedWeights, EngineError> {
        let packed_input_dimension =
            packed_dimension(input_dimension, plan.quantization_bits, prefix)?;
        let parameter_dimension =
            parameter_dimension(input_dimension, plan.quantization_group_size, prefix)?;
        self.bind_quantized(
            prefix,
            &[expert_count, output_dimension, packed_input_dimension],
            &[expert_count, output_dimension, parameter_dimension],
            plan.quantization_group_size,
            plan.quantization_bits,
            &plan.quantization_mode,
        )
    }

    fn bind_quantized_linear(
        &self,
        prefix: &str,
        output_dimension: usize,
        input_dimension: usize,
        group_size: usize,
        bits: usize,
        mode: &str,
    ) -> Result<BoundQuantizedWeights, EngineError> {
        let packed_input_dimension = packed_dimension(input_dimension, bits, prefix)?;
        let parameter_dimension = parameter_dimension(input_dimension, group_size, prefix)?;
        self.bind_quantized(
            prefix,
            &[output_dimension, packed_input_dimension],
            &[output_dimension, parameter_dimension],
            group_size,
            bits,
            mode,
        )
    }

    fn bind_quantized(
        &self,
        prefix: &str,
        weight_shape: &[usize],
        parameter_shape: &[usize],
        group_size: usize,
        bits: usize,
        mode: &str,
    ) -> Result<BoundQuantizedWeights, EngineError> {
        let tensors = self.quantized_tensors.get(prefix).ok_or_else(|| {
            EngineError::Unsupported(format!("missing quantized tensor set {prefix}"))
        })?;
        validate_static_array(&tensors.weight, weight_shape, DType::Uint32, prefix)?;
        validate_static_array(&tensors.scales, parameter_shape, DType::BFloat16, prefix)?;
        validate_static_array(&tensors.biases, parameter_shape, DType::BFloat16, prefix)?;
        Ok(BoundQuantizedWeights {
            weight: tensors.weight.try_clone().map_err(EngineError::Mlx)?,
            scales: tensors.scales.try_clone().map_err(EngineError::Mlx)?,
            biases: tensors.biases.try_clone().map_err(EngineError::Mlx)?,
            group_size: dimension_i32(group_size, "quantization group size")?,
            bits: dimension_i32(bits, "quantization bits")?,
            mode: mode.to_owned(),
        })
    }

    fn bind_array(&self, name: &str, shape: &[usize], dtype: DType) -> Result<Array, EngineError> {
        let array = self
            .tensors
            .get(name)
            .ok_or_else(|| EngineError::Unsupported(format!("missing tensor {name}")))?;
        validate_static_array(array, shape, dtype, name)?;
        array.try_clone().map_err(EngineError::Mlx)
    }
}

impl TensorLookup for ShardedWeights {
    fn tensor(&self, name: &str) -> Option<&Array> {
        self.tensors.get(name)
    }

    fn quantized_tensors(&self, prefix: &str) -> Option<QuantizedTensorRefs<'_>> {
        let tensors = self.quantized_tensors.get(prefix)?;
        Some(QuantizedTensorRefs {
            weight: &tensors.weight,
            scales: &tensors.scales,
            biases: &tensors.biases,
        })
    }
}

fn packed_dimension(input_dimension: usize, bits: usize, name: &str) -> Result<usize, EngineError> {
    input_dimension
        .checked_mul(bits)
        .filter(|packed_bits| packed_bits.is_multiple_of(32))
        .map(|packed_bits| packed_bits / 32)
        .ok_or_else(|| {
            EngineError::Unsupported(format!(
                "{name} input dimension {input_dimension} is incompatible with {bits}-bit packing"
            ))
        })
}

fn parameter_dimension(
    input_dimension: usize,
    group_size: usize,
    name: &str,
) -> Result<usize, EngineError> {
    if group_size == 0 || !input_dimension.is_multiple_of(group_size) {
        return Err(EngineError::Unsupported(format!(
            "{name} input dimension {input_dimension} is not divisible by group size {group_size}"
        )));
    }
    Ok(input_dimension / group_size)
}

fn dimension_i32(value: usize, name: &str) -> Result<i32, EngineError> {
    i32::try_from(value).map_err(|error| {
        EngineError::Unsupported(format!("{name} does not fit MLX int32: {error}"))
    })
}

fn validate_static_array(
    array: &Array,
    shape: &[usize],
    dtype: DType,
    name: &str,
) -> Result<(), EngineError> {
    let observed_shape = array.shape();
    if observed_shape != shape {
        return Err(EngineError::Unsupported(format!(
            "{name} shape mismatch: expected {shape:?}, observed {observed_shape:?}"
        )));
    }
    if array.dtype() != dtype {
        return Err(EngineError::Unsupported(format!(
            "{name} dtype mismatch: expected {}, observed {}",
            dtype.name(),
            array.dtype_name()
        )));
    }
    Ok(())
}
