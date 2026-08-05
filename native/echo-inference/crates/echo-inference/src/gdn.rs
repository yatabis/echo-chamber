use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use echo_mlx::{
    Array, DType, Gpu, MetalKernel, MetalKernelDispatch, MetalOutput, MetalTemplate,
    QuantizedMatmulConfig, SafeTensors,
};
use serde::{Deserialize, Serialize};

use super::weights::{BoundGdnWeights, BoundQuantizedWeights, TensorLookup};
use super::{EngineError, MAX_ACTIVE_BATCH_SIZE, ModelPlan, safetensor_files, sha256_file};

const GATED_DELTA_SOURCE: &str = r"
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          if (true) {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
              y[dv_idx] = static_cast<InT>(out);
            }
          } else {
            y[dv_idx] = static_cast<InT>(0);
          }
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
    ";

const GDN_DECODE_PREPROCESS_SOURCE: &str = r"
        constexpr uint values_per_lane = 4;
        constexpr uint head_dimension = __HEAD_DIM__;
        constexpr uint key_blocks = __KEY_BLOCKS__;
        constexpr uint value_blocks = __VALUE_BLOCKS__;
        constexpr uint conv_dimension = __CONV_DIM__;

        auto block = threadgroup_position_in_grid.y;
        auto batch_index = threadgroup_position_in_grid.z;
        auto lane = thread_index_in_simdgroup;
        auto local_base = lane * values_per_lane;
        auto channel_base = block * head_dimension + local_base;
        auto row_conv_offset = batch_index * conv_dimension;
        auto row_state_offset = batch_index * 3 * conv_dimension;

        bfloat16_t activated[values_per_lane];
        float squared_sum = 0.0f;
        for (uint i = 0; i < values_per_lane; ++i) {
          auto channel = channel_base + i;
          float convolution = 0.0f;
          convolution += static_cast<float>(
                             conv_state_in[row_state_offset + channel]) *
              conv_weight[channel * 4];
          convolution += static_cast<float>(
                             conv_state_in[row_state_offset + conv_dimension +
                                           channel]) *
              conv_weight[channel * 4 + 1];
          convolution += static_cast<float>(
                             conv_state_in[row_state_offset +
                                           2 * conv_dimension + channel]) *
              conv_weight[channel * 4 + 2];
          convolution += static_cast<float>(qkv[row_conv_offset + channel]) *
              conv_weight[channel * 4 + 3];

          auto convolution_bf16 = static_cast<bfloat16_t>(convolution);
          bfloat16_t sigmoid_tail =
              1 / (1 + metal::exp(metal::abs(convolution_bf16)));
          bfloat16_t sigmoid_value = convolution_bf16 < 0
              ? sigmoid_tail
              : static_cast<bfloat16_t>(1 - sigmoid_tail);
          activated[i] =
              static_cast<bfloat16_t>(convolution_bf16 * sigmoid_value);
          squared_sum += static_cast<float>(activated[i]) * activated[i];

          conv_state_out[row_state_offset + channel] =
              conv_state_in[row_state_offset + conv_dimension + channel];
          conv_state_out[row_state_offset + conv_dimension + channel] =
              conv_state_in[row_state_offset + 2 * conv_dimension + channel];
          conv_state_out[row_state_offset + 2 * conv_dimension + channel] =
              qkv[row_conv_offset + channel];
        }

        if (block < 2 * key_blocks) {
          threadgroup float inverse_rms[1];
          squared_sum = simd_sum(squared_sum);
          if (lane == 0) {
            inverse_rms[0] = metal::precise::rsqrt(
                squared_sum / static_cast<float>(head_dimension) + 1e-6f);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          auto output_offset =
              batch_index * key_blocks * head_dimension +
              (block % key_blocks) * head_dimension + local_base;
          auto scale = block < key_blocks ? q_scale : k_scale;
          for (uint i = 0; i < values_per_lane; ++i) {
            auto normalized = static_cast<bfloat16_t>(
                static_cast<float>(activated[i]) * inverse_rms[0]);
            auto scaled = static_cast<bfloat16_t>(normalized * scale);
            if (block < key_blocks) {
              q[output_offset + i] = scaled;
            } else {
              k[output_offset + i] = scaled;
            }
          }
        } else {
          auto value_head = block - 2 * key_blocks;
          auto value_output_offset =
              batch_index * value_blocks * head_dimension;
          auto output_offset =
              value_output_offset + value_head * head_dimension + local_base;
          for (uint i = 0; i < values_per_lane; ++i) {
            v[output_offset + i] = activated[i];
          }

          if (lane == 0) {
            auto scalar_offset = batch_index * value_blocks + value_head;
            auto beta_input = b[scalar_offset];
            bfloat16_t beta_tail =
                1 / (1 + metal::exp(metal::abs(beta_input)));
            beta[scalar_offset] = beta_input < 0
                ? beta_tail
                : static_cast<bfloat16_t>(1 - beta_tail);

            auto biased_a =
                static_cast<bfloat16_t>(a[scalar_offset] + dt_bias[value_head]);
            constexpr bfloat16_t zero = static_cast<bfloat16_t>(0);
            auto maximum = metal::max(biased_a, zero);
            auto minimum = metal::min(biased_a, zero);
            auto softplus = static_cast<bfloat16_t>(
                maximum + log1p(metal::exp(minimum - maximum)));
            auto exponentiated_a_log =
                metal::precise::exp(static_cast<float>(a_log[value_head]));
            decay[scalar_offset] = metal::precise::exp(
                -exponentiated_a_log * static_cast<float>(softplus));
          }
        }
    ";

const GDN_DECODE_POSTPROCESS_SOURCE: &str = r"
        constexpr uint values_per_lane = 4;
        constexpr uint head_dimension = __HEAD_DIM__;
        constexpr uint value_heads = __VALUE_HEADS__;

        auto value_head = threadgroup_position_in_grid.y;
        auto batch_index = threadgroup_position_in_grid.z;
        auto lane = thread_index_in_simdgroup;
        auto local_base = lane * values_per_lane;
        auto offset =
            (batch_index * value_heads + value_head) * head_dimension +
            local_base;

        float hidden[values_per_lane];
        float squared_sum = 0.0f;
        for (uint i = 0; i < values_per_lane; ++i) {
          hidden[i] = static_cast<float>(recurrent_output[offset + i]);
          squared_sum += hidden[i] * hidden[i];
        }
        squared_sum = simd_sum(squared_sum);
        auto inverse_rms = metal::precise::rsqrt(
            squared_sum / static_cast<float>(head_dimension) + epsilon);

        for (uint i = 0; i < values_per_lane; ++i) {
          auto feature = local_base + i;
          auto unweighted =
              static_cast<bfloat16_t>(hidden[i] * inverse_rms);
          auto normalized =
              static_cast<bfloat16_t>(norm_weight[feature] * unweighted);

          auto gate = static_cast<float>(z[offset + i]);
          auto sigmoid_tail = 1.0f / (1.0f + metal::exp(metal::abs(gate)));
          auto sigmoid_value = gate < 0.0f ? sigmoid_tail : 1.0f - sigmoid_tail;
          auto activated_gate = gate * sigmoid_value;
          normalized_output[offset + i] = static_cast<bfloat16_t>(
              activated_gate * static_cast<float>(normalized));
        }
    ";

#[derive(Debug, Deserialize)]
struct FixtureManifest {
    schema_version: u32,
    model_type: String,
    config_sha256: String,
    layer_index: usize,
    prefix_length: usize,
    continuation_length: usize,
    input_dtype: String,
    dimensions: FixtureDimensions,
    quantization: FixtureQuantization,
    fixture_sha256: String,
    fixture_tensor_count: usize,
}

#[derive(Debug, Deserialize, Eq, PartialEq)]
struct FixtureDimensions {
    hidden_size: usize,
    key_heads: usize,
    value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    conv_kernel_size: usize,
    conv_dim: usize,
}

#[derive(Debug, Deserialize, Eq, PartialEq)]
struct FixtureQuantization {
    group_size: usize,
    bits: usize,
    mode: String,
}

/// Direct comparison between one real Qwen3.5 GDN layer executed from Rust
/// and the official Python/MLX call.
#[derive(Clone, Debug, Serialize)]
pub struct GdnLayerParity {
    pub architecture: String,
    pub layer_index: usize,
    pub batch_size: usize,
    pub prefix_length: usize,
    pub continuation_length: usize,
    pub fixture_sha256: String,
    pub output_max_absolute_difference: f32,
    pub conv_state_max_absolute_difference: f32,
    pub recurrent_state_max_absolute_difference: f32,
    pub trace_max_absolute_difference: f32,
    pub trace_differences: BTreeMap<String, f32>,
    pub exact: bool,
}

pub(crate) struct GdnExecution {
    output: Array,
    conv_state: Array,
    recurrent_state: Array,
    trace: Option<BTreeMap<&'static str, Array>>,
}

enum GdnWeightSource<'a> {
    Named {
        weights: &'a dyn TensorLookup,
        prefix: &'a str,
    },
    Bound(&'a BoundGdnWeights),
}

#[derive(Clone, Copy)]
enum GdnLinear {
    InProjQkv,
    InProjZ,
    InProjA,
    InProjB,
    OutProj,
}

impl GdnWeightSource<'_> {
    fn apply_linear(
        &self,
        gpu: &Gpu,
        input: &Array,
        projection: GdnLinear,
        plan: &ModelPlan,
    ) -> Result<Array, EngineError> {
        match self {
            Self::Named { weights, prefix } => {
                let suffix = match projection {
                    GdnLinear::InProjQkv => "in_proj_qkv",
                    GdnLinear::InProjZ => "in_proj_z",
                    GdnLinear::InProjA => "in_proj_a",
                    GdnLinear::InProjB => "in_proj_b",
                    GdnLinear::OutProj => "out_proj",
                };
                quantized_linear(gpu, input, *weights, &format!("{prefix}.{suffix}"), plan)
            }
            Self::Bound(weights) => {
                let weights = match projection {
                    GdnLinear::InProjQkv => &weights.in_proj_qkv,
                    GdnLinear::InProjZ => &weights.in_proj_z,
                    GdnLinear::InProjA => &weights.in_proj_a,
                    GdnLinear::InProjB => &weights.in_proj_b,
                    GdnLinear::OutProj => &weights.out_proj,
                };
                apply_bound_quantized_linear(gpu, input, weights)
            }
        }
    }

    fn conv_weight(&self) -> Result<&Array, EngineError> {
        match self {
            Self::Named { weights, prefix } => {
                require_tensor(*weights, &format!("{prefix}.conv1d.weight"))
            }
            Self::Bound(weights) => Ok(&weights.conv_weight),
        }
    }

    fn dt_bias(&self) -> Result<&Array, EngineError> {
        match self {
            Self::Named { weights, prefix } => {
                require_tensor(*weights, &format!("{prefix}.dt_bias"))
            }
            Self::Bound(weights) => Ok(&weights.dt_bias),
        }
    }

    fn a_log(&self) -> Result<&Array, EngineError> {
        match self {
            Self::Named { weights, prefix } => require_tensor(*weights, &format!("{prefix}.A_log")),
            Self::Bound(weights) => Ok(&weights.a_log),
        }
    }

    fn norm_weight(&self) -> Result<&Array, EngineError> {
        match self {
            Self::Named { weights, prefix } => {
                require_tensor(*weights, &format!("{prefix}.norm.weight"))
            }
            Self::Bound(weights) => Ok(&weights.norm_weight),
        }
    }
}

/// Reusable generic and fixed-shape decode GDN resources.
///
/// MLX caches compiled template variants on each kernel handle. Keeping the
/// decode source, dispatch metadata, normalization scalars, and time scalar
/// resident avoids rebuilding identical graph inputs for every GDN layer and
/// generated token.
pub(crate) struct GdnKernel {
    generic_kernel: MetalKernel,
    decode_kernel: MetalKernel,
    decode_dispatches: Vec<MetalKernelDispatch>,
    decode_preprocess_kernel: MetalKernel,
    decode_preprocess_dispatches: Vec<MetalKernelDispatch>,
    decode_postprocess_kernel: MetalKernel,
    decode_postprocess_dispatches: Vec<MetalKernelDispatch>,
    decode_key_dimension: usize,
    decode_value_dimension: usize,
    decode_key_heads: usize,
    decode_value_heads: usize,
    q_scale: Array,
    k_scale: Array,
    decode_time: Array,
    decode_norm_epsilon: Array,
}

impl GdnKernel {
    pub(crate) fn new(gpu: &Gpu, plan: &ModelPlan) -> Result<Self, EngineError> {
        let generic_kernel = MetalKernel::new(
            "gated_delta_step",
            &["q", "k", "v", "g", "beta", "state_in", "T"],
            &["y", "state_out"],
            GATED_DELTA_SOURCE,
        )
        .map_err(EngineError::Mlx)?;
        let (decode_kernel, decode_dispatches) = prepare_decode_recurrent_kernel(plan)?;
        let (decode_preprocess_kernel, decode_preprocess_dispatches) =
            prepare_decode_preprocess_kernel(plan)?;
        let (decode_postprocess_kernel, decode_postprocess_dispatches) =
            prepare_decode_postprocess_kernel(plan)?;
        let key_head_dimension = dimension_f32(plan.key_head_dimension, "key head dimension")?;
        let q_scale = gpu
            .scalar_like(1.0 / key_head_dimension, DType::BFloat16)
            .map_err(EngineError::Mlx)?;
        let k_scale = gpu
            .scalar_like(1.0 / key_head_dimension.sqrt(), DType::BFloat16)
            .map_err(EngineError::Mlx)?;
        let decode_time = gpu.scalar_i32(1);
        let decode_norm_epsilon = gpu
            .scalar_like(plan.rms_norm_epsilon, DType::Float32)
            .map_err(EngineError::Mlx)?;
        Ok(Self {
            generic_kernel,
            decode_kernel,
            decode_dispatches,
            decode_preprocess_kernel,
            decode_preprocess_dispatches,
            decode_postprocess_kernel,
            decode_postprocess_dispatches,
            decode_key_dimension: plan.key_head_dimension,
            decode_value_dimension: plan.value_head_dimension,
            decode_key_heads: plan.key_head_count,
            decode_value_heads: plan.value_head_count,
            q_scale,
            k_scale,
            decode_time,
            decode_norm_epsilon,
        })
    }
}

fn prepare_decode_recurrent_kernel(
    plan: &ModelPlan,
) -> Result<(MetalKernel, Vec<MetalKernelDispatch>), EngineError> {
    create_decode_recurrent_kernel(
        plan.key_head_dimension,
        plan.value_head_dimension,
        plan.key_head_count,
        plan.value_head_count,
    )
}

fn create_decode_recurrent_kernel(
    key_head_dimension: usize,
    value_head_dimension: usize,
    key_head_count: usize,
    value_head_count: usize,
) -> Result<(MetalKernel, Vec<MetalKernelDispatch>), EngineError> {
    let source = specialize_decode_source(
        key_head_dimension,
        value_head_dimension,
        key_head_count,
        value_head_count,
    );
    let name = format!(
        "gated_delta_decode_bf16_f32_dk{key_head_dimension}_dv{value_head_dimension}_hk{key_head_count}_hv{value_head_count}"
    );
    let kernel = MetalKernel::new(
        &name,
        &["q", "k", "v", "g", "beta", "state_in", "T"],
        &["y", "state_out"],
        &source,
    )
    .map_err(EngineError::Mlx)?;
    let dispatches = (1..=MAX_ACTIVE_BATCH_SIZE)
        .map(|batch_size| {
            MetalKernel::prepare_dispatch(
                &[
                    MetalOutput {
                        shape: vec![batch_size, 1, value_head_count, value_head_dimension],
                        dtype: DType::BFloat16,
                    },
                    MetalOutput {
                        shape: vec![
                            batch_size,
                            value_head_count,
                            value_head_dimension,
                            key_head_dimension,
                        ],
                        dtype: DType::Float32,
                    },
                ],
                &[],
                [32, value_head_dimension, batch_size * value_head_count],
                [32, 4, 1],
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(EngineError::Mlx)?;
    Ok((kernel, dispatches))
}

fn prepare_decode_preprocess_kernel(
    plan: &ModelPlan,
) -> Result<(MetalKernel, Vec<MetalKernelDispatch>), EngineError> {
    create_decode_preprocess_kernel(
        plan.key_head_dimension,
        plan.value_head_dimension,
        plan.key_head_count,
        plan.value_head_count,
        plan.convolution_kernel_size,
    )
}

fn create_decode_preprocess_kernel(
    key_head_dimension: usize,
    value_head_dimension: usize,
    key_head_count: usize,
    value_head_count: usize,
    convolution_kernel_size: usize,
) -> Result<(MetalKernel, Vec<MetalKernelDispatch>), EngineError> {
    let convolution_dimension =
        2 * key_head_count * key_head_dimension + value_head_count * value_head_dimension;
    let source = specialize_decode_preprocess_source(
        key_head_dimension,
        key_head_count,
        value_head_count,
        convolution_dimension,
    );
    let name = format!(
        "gdn_decode_preprocess_bf16_dk{key_head_dimension}_hk{key_head_count}_hv{value_head_count}"
    );
    let kernel = MetalKernel::new(
        &name,
        &[
            "qkv",
            "a",
            "b",
            "conv_state_in",
            "conv_weight",
            "a_log",
            "dt_bias",
            "q_scale",
            "k_scale",
        ],
        &["q", "k", "v", "decay", "beta", "conv_state_out"],
        &source,
    )
    .map_err(EngineError::Mlx)?;
    let dispatches = (1..=MAX_ACTIVE_BATCH_SIZE)
        .map(|batch_size| {
            MetalKernel::prepare_dispatch(
                &[
                    MetalOutput {
                        shape: vec![batch_size, 1, key_head_count, key_head_dimension],
                        dtype: DType::BFloat16,
                    },
                    MetalOutput {
                        shape: vec![batch_size, 1, key_head_count, key_head_dimension],
                        dtype: DType::BFloat16,
                    },
                    MetalOutput {
                        shape: vec![batch_size, 1, value_head_count, value_head_dimension],
                        dtype: DType::BFloat16,
                    },
                    MetalOutput {
                        shape: vec![batch_size, 1, value_head_count],
                        dtype: DType::Float32,
                    },
                    MetalOutput {
                        shape: vec![batch_size, 1, value_head_count],
                        dtype: DType::BFloat16,
                    },
                    MetalOutput {
                        shape: vec![
                            batch_size,
                            convolution_kernel_size - 1,
                            convolution_dimension,
                        ],
                        dtype: DType::BFloat16,
                    },
                ],
                &[],
                [32, 2 * key_head_count + value_head_count, batch_size],
                [32, 1, 1],
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(EngineError::Mlx)?;
    Ok((kernel, dispatches))
}

fn prepare_decode_postprocess_kernel(
    plan: &ModelPlan,
) -> Result<(MetalKernel, Vec<MetalKernelDispatch>), EngineError> {
    create_decode_postprocess_kernel(plan.value_head_dimension, plan.value_head_count)
}

fn create_decode_postprocess_kernel(
    value_head_dimension: usize,
    value_head_count: usize,
) -> Result<(MetalKernel, Vec<MetalKernelDispatch>), EngineError> {
    let source = specialize_decode_postprocess_source(value_head_dimension, value_head_count);
    let name = format!("gdn_decode_postprocess_bf16_dv{value_head_dimension}_hv{value_head_count}");
    let kernel = MetalKernel::new(
        &name,
        &["recurrent_output", "z", "norm_weight", "epsilon"],
        &["normalized_output"],
        &source,
    )
    .map_err(EngineError::Mlx)?;
    let dispatches = (1..=MAX_ACTIVE_BATCH_SIZE)
        .map(|batch_size| {
            MetalKernel::prepare_dispatch(
                &[MetalOutput {
                    shape: vec![batch_size, 1, value_head_count, value_head_dimension],
                    dtype: DType::BFloat16,
                }],
                &[],
                [32, value_head_count, batch_size],
                [32, 1, 1],
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(EngineError::Mlx)?;
    Ok((kernel, dispatches))
}

fn specialize_decode_source(
    key_head_dimension: usize,
    value_head_dimension: usize,
    key_head_count: usize,
    value_head_count: usize,
) -> String {
    GATED_DELTA_SOURCE
        .replace("InT", "bfloat16_t")
        .replace("StT", "float")
        .replace("Dk", &key_head_dimension.to_string())
        .replace("Dv", &value_head_dimension.to_string())
        .replace("Hk", &key_head_count.to_string())
        .replace("Hv", &value_head_count.to_string())
}

fn specialize_decode_preprocess_source(
    head_dimension: usize,
    key_head_count: usize,
    value_head_count: usize,
    convolution_dimension: usize,
) -> String {
    GDN_DECODE_PREPROCESS_SOURCE
        .replace("__HEAD_DIM__", &head_dimension.to_string())
        .replace("__KEY_BLOCKS__", &key_head_count.to_string())
        .replace("__VALUE_BLOCKS__", &value_head_count.to_string())
        .replace("__CONV_DIM__", &convolution_dimension.to_string())
}

fn specialize_decode_postprocess_source(
    value_head_dimension: usize,
    value_head_count: usize,
) -> String {
    GDN_DECODE_POSTPROCESS_SOURCE
        .replace("__HEAD_DIM__", &value_head_dimension.to_string())
        .replace("__VALUE_HEADS__", &value_head_count.to_string())
}

fn decode_dispatch_for_batch<'a>(
    dispatches: &'a [MetalKernelDispatch],
    batch_size: usize,
    label: &str,
) -> Result<&'a MetalKernelDispatch, EngineError> {
    batch_size
        .checked_sub(1)
        .and_then(|index| dispatches.get(index))
        .ok_or_else(|| {
            EngineError::Unsupported(format!(
                "{label} supports decode batch sizes 1..={}, observed {batch_size}",
                dispatches.len()
            ))
        })
}

impl GdnExecution {
    pub(crate) fn require(&self, name: &'static str) -> Result<&Array, EngineError> {
        match name {
            "output" => Ok(&self.output),
            "conv_state" => Ok(&self.conv_state),
            "recurrent_state" => Ok(&self.recurrent_state),
            _ => self
                .trace
                .as_ref()
                .and_then(|trace| trace.get(name))
                .ok_or_else(|| {
                    EngineError::Unsupported(format!("internal GDN trace omitted {name}"))
                }),
        }
    }
}

/// Executes one actual Qwen3.5 GDN layer through Rust and MLX C, including
/// affine Q4 projections, depthwise convolution, the official recurrent Metal
/// kernel, gated RMS normalization, and the output projection.
///
/// # Errors
///
/// Returns [`EngineError`] when the model, oracle, weights, shapes, dtypes, or
/// an MLX operation do not match the admitted execution plan.
#[allow(clippy::too_many_lines)]
pub fn run_gdn_layer_parity(
    model_directory: &Path,
    fixture_path: &Path,
    manifest_path: &Path,
) -> Result<GdnLayerParity, EngineError> {
    let plan = ModelPlan::from_directory(model_directory)?;
    let manifest = load_manifest(manifest_path)?;
    validate_manifest(&plan, model_directory, fixture_path, &manifest)?;

    let fixture = SafeTensors::load(fixture_path).map_err(EngineError::Mlx)?;
    if fixture.len() != manifest.fixture_tensor_count {
        return Err(EngineError::Unsupported(format!(
            "GDN fixture tensor count drift: manifest {}, payload {}",
            manifest.fixture_tensor_count,
            fixture.len()
        )));
    }
    let weight_prefix = format!(
        "language_model.model.layers.{}.linear_attn",
        manifest.layer_index
    );
    let weights = load_weight_shard(model_directory, &weight_prefix)?;
    let gpu = Gpu::new();

    let input = require_tensor(&fixture, "continuation_input")?;
    let input_shape = input.shape();
    if input_shape.len() != 3 {
        return Err(EngineError::Unsupported(format!(
            "continuation_input must be rank 3, observed {input_shape:?}"
        )));
    }
    let batch_size = input_shape[0];
    let sequence_length = input_shape[1];
    validate_array(
        input,
        &[batch_size, manifest.continuation_length, plan.hidden_size],
        DType::BFloat16,
        "continuation_input",
    )?;
    if sequence_length != manifest.continuation_length {
        return Err(EngineError::Unsupported(
            "continuation sequence length changed after validation".into(),
        ));
    }

    let initial_conv_state = require_tensor(&fixture, "initial_conv_state")?;
    let initial_recurrent_state = require_tensor(&fixture, "initial_recurrent_state")?;
    let execution = execute_gdn_layer(
        &gpu,
        input,
        initial_conv_state,
        initial_recurrent_state,
        &weights,
        &weight_prefix,
        &plan,
    )?;

    let expected_output = require_tensor(&fixture, "expected_output")?;
    let expected_conv_state = require_tensor(&fixture, "expected_conv_state")?;
    let expected_recurrent_state = require_tensor(&fixture, "expected_recurrent_state")?;
    let output_difference = gpu
        .max_abs_difference(execution.require("output")?, expected_output)
        .map_err(EngineError::Mlx)?;
    let conv_state_difference = gpu
        .max_abs_difference(execution.require("conv_state")?, expected_conv_state)
        .map_err(EngineError::Mlx)?;
    let recurrent_state_difference = gpu
        .max_abs_difference(
            execution.require("recurrent_state")?,
            expected_recurrent_state,
        )
        .map_err(EngineError::Mlx)?;

    let mut trace_differences = BTreeMap::new();
    for name in [
        "a",
        "b",
        "beta",
        "conv_input",
        "conv_output",
        "conv_state",
        "decay",
        "k",
        "normalized_output",
        "output",
        "q",
        "qkv",
        "recurrent_output",
        "recurrent_state",
        "v",
        "z",
    ] {
        let expected = require_tensor(&fixture, &format!("trace.{name}"))?;
        let difference = gpu
            .max_abs_difference(execution.require(name)?, expected)
            .map_err(EngineError::Mlx)?;
        trace_differences.insert(name.to_owned(), difference);
    }
    let trace_max_absolute_difference = trace_differences.values().copied().fold(0.0_f32, f32::max);
    let exact = output_difference == 0.0
        && conv_state_difference == 0.0
        && recurrent_state_difference == 0.0
        && trace_max_absolute_difference == 0.0;

    Ok(GdnLayerParity {
        architecture: plan.architecture,
        layer_index: manifest.layer_index,
        batch_size,
        prefix_length: manifest.prefix_length,
        continuation_length: manifest.continuation_length,
        fixture_sha256: manifest.fixture_sha256,
        output_max_absolute_difference: output_difference,
        conv_state_max_absolute_difference: conv_state_difference,
        recurrent_state_max_absolute_difference: recurrent_state_difference,
        trace_max_absolute_difference,
        trace_differences,
        exact,
    })
}

// This intentionally stays as one auditable, linear transcription of the
// official GatedDeltaNet call; q/k/v/a/b/z names match the reference equations.
#[allow(clippy::many_single_char_names, clippy::too_many_lines)]
pub(crate) fn execute_gdn_layer(
    gpu: &Gpu,
    input: &Array,
    initial_conv_state: &Array,
    initial_recurrent_state: &Array,
    weights: &dyn TensorLookup,
    weight_prefix: &str,
    plan: &ModelPlan,
) -> Result<GdnExecution, EngineError> {
    let kernel = GdnKernel::new(gpu, plan)?;
    execute_gdn_layer_impl(
        gpu,
        input,
        initial_conv_state,
        initial_recurrent_state,
        &GdnWeightSource::Named {
            weights,
            prefix: weight_prefix,
        },
        plan,
        &kernel,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_gdn_layer_with_kernel(
    gpu: &Gpu,
    input: &Array,
    initial_conv_state: &Array,
    initial_recurrent_state: &Array,
    weights: &dyn TensorLookup,
    weight_prefix: &str,
    plan: &ModelPlan,
    kernel: &GdnKernel,
) -> Result<GdnExecution, EngineError> {
    execute_gdn_layer_impl(
        gpu,
        input,
        initial_conv_state,
        initial_recurrent_state,
        &GdnWeightSource::Named {
            weights,
            prefix: weight_prefix,
        },
        plan,
        kernel,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_gdn_layer_with_bound_weights(
    gpu: &Gpu,
    input: &Array,
    initial_conv_state: &Array,
    initial_recurrent_state: &Array,
    weights: &BoundGdnWeights,
    plan: &ModelPlan,
    kernel: &GdnKernel,
) -> Result<GdnExecution, EngineError> {
    execute_gdn_layer_impl(
        gpu,
        input,
        initial_conv_state,
        initial_recurrent_state,
        &GdnWeightSource::Bound(weights),
        plan,
        kernel,
        false,
    )
}

// This intentionally stays as one auditable, linear transcription of the
// official GatedDeltaNet call; q/k/v/a/b/z names match the reference equations.
#[allow(
    clippy::many_single_char_names,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
fn execute_gdn_layer_impl(
    gpu: &Gpu,
    input: &Array,
    initial_conv_state: &Array,
    initial_recurrent_state: &Array,
    weights: &GdnWeightSource<'_>,
    plan: &ModelPlan,
    kernel: &GdnKernel,
    capture_trace: bool,
) -> Result<GdnExecution, EngineError> {
    let input_shape = input.shape();
    let [batch_size, sequence_length, hidden_size] = <[usize; 3]>::try_from(input_shape.clone())
        .map_err(|shape| {
            EngineError::Unsupported(format!("GDN input must be rank 3, observed {shape:?}"))
        })?;
    if hidden_size != plan.hidden_size {
        return Err(EngineError::Unsupported(format!(
            "GDN input hidden size mismatch: expected {}, observed {hidden_size}",
            plan.hidden_size
        )));
    }
    let qkv = weights.apply_linear(gpu, input, GdnLinear::InProjQkv, plan)?;
    let z_flat = weights.apply_linear(gpu, input, GdnLinear::InProjZ, plan)?;
    let z = gpu
        .reshape(
            &z_flat,
            &[
                batch_size,
                sequence_length,
                plan.value_head_count,
                plan.value_head_dimension,
            ],
        )
        .map_err(EngineError::Mlx)?;
    let a = weights.apply_linear(gpu, input, GdnLinear::InProjA, plan)?;
    let b = weights.apply_linear(gpu, input, GdnLinear::InProjB, plan)?;
    validate_array(
        initial_conv_state,
        &[
            batch_size,
            plan.convolution_kernel_size - 1,
            plan.convolution_dimension(),
        ],
        input.dtype(),
        "initial_conv_state",
    )?;
    let conv_weight = weights.conv_weight()?;
    let dt_bias = weights.dt_bias()?;
    let a_log = weights.a_log()?;
    if matches!(weights, GdnWeightSource::Named { .. }) {
        validate_array(
            conv_weight,
            &[
                plan.convolution_dimension(),
                plan.convolution_kernel_size,
                1,
            ],
            input.dtype(),
            "GDN convolution weight",
        )?;
    }
    let use_decode_preprocess = !capture_trace
        && (1..=MAX_ACTIVE_BATCH_SIZE).contains(&batch_size)
        && sequence_length == 1
        && input.dtype() == DType::BFloat16
        && plan.convolution_kernel_size == 4
        && plan.key_head_dimension == plan.value_head_dimension
        && plan.convolution_dimension()
            == (2 * plan.key_head_count + plan.value_head_count) * plan.key_head_dimension;
    let (q, k, v, decay, beta, conv_state, conv_input, conv_output) = if use_decode_preprocess {
        let dispatch = decode_dispatch_for_batch(
            &kernel.decode_preprocess_dispatches,
            batch_size,
            "decode GDN preprocess fusion",
        )?;
        let outputs = kernel
            .decode_preprocess_kernel
            .apply_prepared(
                gpu,
                &[
                    &qkv,
                    &a,
                    &b,
                    initial_conv_state,
                    conv_weight,
                    a_log,
                    dt_bias,
                    &kernel.q_scale,
                    &kernel.k_scale,
                ],
                dispatch,
            )
            .map_err(EngineError::Mlx)?;
        let output_count = outputs.len();
        let [q, k, v, decay, beta, conv_state] = <[Array; 6]>::try_from(outputs).map_err(|_| {
            EngineError::Unsupported(format!(
                "decode GDN preprocess returned {output_count} outputs instead of 6"
            ))
        })?;
        (q, k, v, decay, beta, conv_state, None, None)
    } else {
        let conv_input = gpu
            .concatenate(&[initial_conv_state, &qkv], 1)
            .map_err(EngineError::Mlx)?;
        let conv_state = tail_sequence(
            gpu,
            &conv_input,
            plan.convolution_kernel_size - 1,
            "GDN conv state",
        )?;
        let conv_output = gpu
            .conv1d(
                &conv_input,
                conv_weight,
                1,
                0,
                1,
                dimension_i32(plan.convolution_dimension(), "conv groups")?,
            )
            .and_then(|value| gpu.silu(&value))
            .map_err(EngineError::Mlx)?;

        let key_dimension = plan.key_dimension();
        let q = last_axis_slice(gpu, &conv_output, 0, key_dimension, "GDN q projection")?;
        let q = gpu
            .reshape(
                &q,
                &[
                    batch_size,
                    sequence_length,
                    plan.key_head_count,
                    plan.key_head_dimension,
                ],
            )
            .map_err(EngineError::Mlx)?;
        let k = last_axis_slice(
            gpu,
            &conv_output,
            key_dimension,
            2 * key_dimension,
            "GDN k projection",
        )?;
        let k = gpu
            .reshape(
                &k,
                &[
                    batch_size,
                    sequence_length,
                    plan.key_head_count,
                    plan.key_head_dimension,
                ],
            )
            .map_err(EngineError::Mlx)?;
        let v = last_axis_slice(
            gpu,
            &conv_output,
            2 * key_dimension,
            plan.convolution_dimension(),
            "GDN v projection",
        )?;
        let v = gpu
            .reshape(
                &v,
                &[
                    batch_size,
                    sequence_length,
                    plan.value_head_count,
                    plan.value_head_dimension,
                ],
            )
            .map_err(EngineError::Mlx)?;

        let q = scale_normalized(gpu, &q, &kernel.q_scale)?;
        let k = scale_normalized(gpu, &k, &kernel.k_scale)?;
        let beta = gpu.sigmoid(&b).map_err(EngineError::Mlx)?;
        let decay = gpu
            .compute_g(a_log, &a, dt_bias)
            .map_err(EngineError::Mlx)?;
        (
            q,
            k,
            v,
            decay,
            beta,
            conv_state,
            Some(conv_input),
            Some(conv_output),
        )
    };

    validate_array(
        initial_recurrent_state,
        &[
            batch_size,
            plan.value_head_count,
            plan.value_head_dimension,
            plan.key_head_dimension,
        ],
        DType::Float32,
        "initial_recurrent_state",
    )?;
    let (recurrent_output, recurrent_state) = gated_delta(
        gpu,
        kernel,
        &q,
        &k,
        &v,
        &decay,
        &beta,
        initial_recurrent_state,
    )?;

    let norm_weight = weights.norm_weight()?;
    let normalized_output = if use_decode_preprocess {
        let dispatch = decode_dispatch_for_batch(
            &kernel.decode_postprocess_dispatches,
            batch_size,
            "decode GDN postprocess fusion",
        )?;
        let outputs = kernel
            .decode_postprocess_kernel
            .apply_prepared(
                gpu,
                &[
                    &recurrent_output,
                    &z,
                    norm_weight,
                    &kernel.decode_norm_epsilon,
                ],
                dispatch,
            )
            .map_err(EngineError::Mlx)?;
        let output_count = outputs.len();
        let [normalized_output] = <[Array; 1]>::try_from(outputs).map_err(|_| {
            EngineError::Unsupported(format!(
                "decode GDN postprocess returned {output_count} outputs instead of 1"
            ))
        })?;
        normalized_output
    } else {
        let normalized = gpu
            .rms_norm(&recurrent_output, Some(norm_weight), plan.rms_norm_epsilon)
            .map_err(EngineError::Mlx)?;
        gpu.precise_swiglu(&recurrent_output, &z, &normalized)
            .map_err(EngineError::Mlx)?
    };
    let flattened = gpu
        .reshape(
            &normalized_output,
            &[batch_size, sequence_length, plan.value_dimension()],
        )
        .map_err(EngineError::Mlx)?;
    let output = weights.apply_linear(gpu, &flattened, GdnLinear::OutProj, plan)?;

    let trace = capture_trace.then(|| {
        BTreeMap::from([
            ("a", a),
            ("b", b),
            ("beta", beta),
            (
                "conv_input",
                conv_input.expect("captured GDN execution must retain conv_input"),
            ),
            (
                "conv_output",
                conv_output.expect("captured GDN execution must retain conv_output"),
            ),
            ("decay", decay),
            ("k", k),
            ("normalized_output", normalized_output),
            ("q", q),
            ("qkv", qkv),
            ("recurrent_output", recurrent_output),
            ("v", v),
            ("z", z),
        ])
    });
    Ok(GdnExecution {
        output,
        conv_state,
        recurrent_state,
        trace,
    })
}

impl ModelPlan {
    pub(crate) fn key_dimension(&self) -> usize {
        self.key_head_count * self.key_head_dimension
    }

    pub(crate) fn value_dimension(&self) -> usize {
        self.value_head_count * self.value_head_dimension
    }

    pub(crate) fn convolution_dimension(&self) -> usize {
        2 * self.key_dimension() + self.value_dimension()
    }
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

fn validate_manifest(
    plan: &ModelPlan,
    model_directory: &Path,
    fixture_path: &Path,
    manifest: &FixtureManifest,
) -> Result<(), EngineError> {
    if manifest.schema_version != 1 {
        return Err(EngineError::Unsupported(format!(
            "GDN fixture schema must be 1, observed {}",
            manifest.schema_version
        )));
    }
    if manifest.model_type != plan.architecture {
        return Err(EngineError::Unsupported(format!(
            "GDN fixture architecture drift: fixture {}, model {}",
            manifest.model_type, plan.architecture
        )));
    }
    if manifest.layer_index >= plan.layer_count
        || (manifest.layer_index + 1).is_multiple_of(plan.full_attention_interval)
    {
        return Err(EngineError::Unsupported(format!(
            "layer {} is not an admitted Qwen3.5 GDN layer",
            manifest.layer_index
        )));
    }
    if manifest.prefix_length == 0 || manifest.continuation_length == 0 {
        return Err(EngineError::Unsupported(
            "GDN fixture segments must both be non-empty".into(),
        ));
    }
    if manifest.input_dtype != "mlx.core.bfloat16" {
        return Err(EngineError::Unsupported(format!(
            "GDN fixture must use bfloat16 input, observed {}",
            manifest.input_dtype
        )));
    }
    let dimensions = FixtureDimensions {
        hidden_size: plan.hidden_size,
        key_heads: plan.key_head_count,
        value_heads: plan.value_head_count,
        key_head_dim: plan.key_head_dimension,
        value_head_dim: plan.value_head_dimension,
        conv_kernel_size: plan.convolution_kernel_size,
        conv_dim: plan.convolution_dimension(),
    };
    if manifest.dimensions != dimensions {
        return Err(EngineError::Unsupported(format!(
            "GDN fixture dimension drift: fixture {:?}, model {:?}",
            manifest.dimensions, dimensions
        )));
    }
    let quantization = FixtureQuantization {
        group_size: plan.quantization_group_size,
        bits: plan.quantization_bits,
        mode: plan.quantization_mode.clone(),
    };
    if manifest.quantization != quantization {
        return Err(EngineError::Unsupported(format!(
            "GDN fixture quantization drift: fixture {:?}, model {:?}",
            manifest.quantization, quantization
        )));
    }
    let config_path = model_directory.join("config.json");
    let config_digest = sha256_file(&config_path)?;
    if config_digest != manifest.config_sha256 {
        return Err(EngineError::Unsupported(format!(
            "GDN fixture config digest drift: fixture {}, model {}",
            manifest.config_sha256, config_digest
        )));
    }
    let fixture_digest = sha256_file(fixture_path)?;
    if fixture_digest != manifest.fixture_sha256 {
        return Err(EngineError::Unsupported(format!(
            "GDN fixture payload digest drift: manifest {}, payload {}",
            manifest.fixture_sha256, fixture_digest
        )));
    }
    Ok(())
}

pub(crate) fn load_weight_shard(
    model_directory: &Path,
    weight_prefix: &str,
) -> Result<SafeTensors, EngineError> {
    let marker = format!("{weight_prefix}.in_proj_qkv.weight");
    load_weight_shard_containing(model_directory, &marker)
}

pub(crate) fn load_weight_shard_containing(
    model_directory: &Path,
    marker: &str,
) -> Result<SafeTensors, EngineError> {
    for path in safetensor_files(model_directory)? {
        let tensors = SafeTensors::load(&path).map_err(EngineError::Mlx)?;
        if tensors.tensor(marker).is_some() {
            return Ok(tensors);
        }
    }
    Err(EngineError::Unsupported(format!(
        "no model shard contains {marker}"
    )))
}

pub(crate) fn require_tensor<'a>(
    tensors: &'a dyn TensorLookup,
    name: &str,
) -> Result<&'a Array, EngineError> {
    tensors
        .tensor(name)
        .ok_or_else(|| EngineError::Unsupported(format!("missing tensor {name}")))
}

pub(crate) fn validate_array(
    array: &Array,
    shape: &[usize],
    dtype: DType,
    name: &str,
) -> Result<(), EngineError> {
    if array.shape() != shape {
        return Err(EngineError::Unsupported(format!(
            "{name} shape mismatch: expected {shape:?}, observed {:?}",
            array.shape()
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

pub(crate) fn apply_bound_quantized_linear(
    gpu: &Gpu,
    input: &Array,
    weights: &BoundQuantizedWeights,
) -> Result<Array, EngineError> {
    gpu.quantized_matmul(
        input,
        &weights.weight,
        &weights.scales,
        QuantizedMatmulConfig {
            biases: Some(&weights.biases),
            transpose: true,
            group_size: weights.group_size,
            bits: weights.bits,
            mode: &weights.mode,
        },
    )
    .map_err(EngineError::Mlx)
}

fn quantized_linear(
    gpu: &Gpu,
    input: &Array,
    weights: &dyn TensorLookup,
    prefix: &str,
    plan: &ModelPlan,
) -> Result<Array, EngineError> {
    quantized_linear_with_config(
        gpu,
        input,
        weights,
        prefix,
        plan.quantization_group_size,
        plan.quantization_bits,
        &plan.quantization_mode,
    )
}

pub(crate) fn quantized_linear_with_config(
    gpu: &Gpu,
    input: &Array,
    weights: &dyn TensorLookup,
    prefix: &str,
    group_size: usize,
    bits: usize,
    mode: &str,
) -> Result<Array, EngineError> {
    let tensors = weights.quantized_tensors(prefix).ok_or_else(|| {
        EngineError::Unsupported(format!("missing quantized tensor set {prefix}"))
    })?;
    let weight = tensors.weight;
    let scales = tensors.scales;
    let biases = tensors.biases;
    let weight_shape = weight.shape();
    let input_dimension = input
        .shape()
        .last()
        .copied()
        .ok_or_else(|| EngineError::Unsupported(format!("{prefix} received a scalar input")))?;
    if group_size == 0
        || !input_dimension.is_multiple_of(group_size)
        || !(2..=8).contains(&bits)
        || input_dimension
            .checked_mul(bits)
            .is_none_or(|packed_bits| !packed_bits.is_multiple_of(32))
    {
        return Err(EngineError::Unsupported(format!(
            "{prefix} cannot quantize input dimension {input_dimension} with {bits} bits and group size {group_size}"
        )));
    }
    if weight_shape.len() != 2
        || weight_shape[1]
            .checked_mul(32)
            .and_then(|value| value.checked_div(bits))
            != Some(input_dimension)
    {
        return Err(EngineError::Unsupported(format!(
            "{prefix}.weight is incompatible with input dimension {input_dimension}: {weight_shape:?}"
        )));
    }
    let expected_scale_shape = [weight_shape[0], input_dimension / group_size];
    validate_array(weight, &weight_shape, DType::Uint32, prefix)?;
    validate_array(scales, &expected_scale_shape, input.dtype(), prefix)?;
    validate_array(biases, &expected_scale_shape, input.dtype(), prefix)?;
    gpu.quantized_matmul(
        input,
        weight,
        scales,
        QuantizedMatmulConfig {
            biases: Some(biases),
            transpose: true,
            group_size: dimension_i32(group_size, "quantization group size")?,
            bits: dimension_i32(bits, "quantization bits")?,
            mode,
        },
    )
    .map_err(EngineError::Mlx)
}

fn tail_sequence(gpu: &Gpu, array: &Array, count: usize, name: &str) -> Result<Array, EngineError> {
    let shape = array.shape();
    if shape.len() != 3 || count > shape[1] {
        return Err(EngineError::Unsupported(format!(
            "{name} cannot take {count} sequence entries from {shape:?}"
        )));
    }
    let result = gpu
        .slice(
            array,
            &[0, dimension_i32(shape[1] - count, name)?, 0],
            &[
                dimension_i32(shape[0], name)?,
                dimension_i32(shape[1], name)?,
                dimension_i32(shape[2], name)?,
            ],
            &[1, 1, 1],
        )
        .and_then(|value| gpu.contiguous(&value))
        .map_err(EngineError::Mlx)?;
    Ok(result)
}

fn last_axis_slice(
    gpu: &Gpu,
    array: &Array,
    feature_start: usize,
    feature_stop: usize,
    name: &str,
) -> Result<Array, EngineError> {
    let shape = array.shape();
    if shape.len() != 3 || feature_start > feature_stop || feature_stop > shape[2] {
        return Err(EngineError::Unsupported(format!(
            "{name} cannot take [{feature_start}, {feature_stop}) from {shape:?}"
        )));
    }
    gpu.slice(
        array,
        &[0, 0, dimension_i32(feature_start, name)?],
        &[
            dimension_i32(shape[0], name)?,
            dimension_i32(shape[1], name)?,
            dimension_i32(feature_stop, name)?,
        ],
        &[1, 1, 1],
    )
    .map_err(EngineError::Mlx)
}

fn scale_normalized(gpu: &Gpu, input: &Array, scale: &Array) -> Result<Array, EngineError> {
    let normalized = gpu.rms_norm(input, None, 1e-6).map_err(EngineError::Mlx)?;
    gpu.multiply(&normalized, scale).map_err(EngineError::Mlx)
}

#[allow(clippy::too_many_arguments)]
fn gated_delta(
    gpu: &Gpu,
    kernel: &GdnKernel,
    q: &Array,
    k: &Array,
    v: &Array,
    decay: &Array,
    beta: &Array,
    state: &Array,
) -> Result<(Array, Array), EngineError> {
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();
    if q_shape.len() != 4 || q_shape != k_shape || v_shape.len() != 4 {
        return Err(EngineError::Unsupported(format!(
            "invalid gated-delta q/k/v shapes: q={q_shape:?}, k={k_shape:?}, v={v_shape:?}"
        )));
    }
    let [batch_size, sequence_length, key_heads, key_dimension] = <[usize; 4]>::try_from(q_shape)
        .map_err(|shape| {
        EngineError::Unsupported(format!("invalid gated-delta q shape: {shape:?}"))
    })?;
    let value_heads = v_shape[2];
    let value_dimension = v_shape[3];
    validate_array(
        state,
        &[batch_size, value_heads, value_dimension, key_dimension],
        DType::Float32,
        "gated-delta state",
    )?;
    let dynamic_time;
    let time = if sequence_length == 1 {
        &kernel.decode_time
    } else {
        dynamic_time = gpu.scalar_i32(dimension_i32(sequence_length, "gated-delta time")?);
        &dynamic_time
    };
    let specialized_decode = (1..=MAX_ACTIVE_BATCH_SIZE).contains(&batch_size)
        && sequence_length == 1
        && q.dtype() == DType::BFloat16
        && state.dtype() == DType::Float32
        && key_dimension == kernel.decode_key_dimension
        && value_dimension == kernel.decode_value_dimension
        && key_heads == kernel.decode_key_heads
        && value_heads == kernel.decode_value_heads;
    let generic_templates = [
        MetalTemplate::DType("InT", q.dtype()),
        MetalTemplate::DType("StT", state.dtype()),
        MetalTemplate::Int("Dk", dimension_i32(key_dimension, "gated-delta Dk")?),
        MetalTemplate::Int("Dv", dimension_i32(value_dimension, "gated-delta Dv")?),
        MetalTemplate::Int("Hk", dimension_i32(key_heads, "gated-delta Hk")?),
        MetalTemplate::Int("Hv", dimension_i32(value_heads, "gated-delta Hv")?),
    ];
    let inputs = &[q, k, v, decay, beta, state, time];
    let outputs = if specialized_decode {
        let dispatch = decode_dispatch_for_batch(
            &kernel.decode_dispatches,
            batch_size,
            "decode GDN recurrent kernel",
        )?;
        kernel.decode_kernel.apply_prepared(gpu, inputs, dispatch)
    } else {
        kernel.generic_kernel.apply(
            gpu,
            inputs,
            &[
                MetalOutput {
                    shape: vec![batch_size, sequence_length, value_heads, value_dimension],
                    dtype: q.dtype(),
                },
                MetalOutput {
                    shape: state.shape(),
                    dtype: state.dtype(),
                },
            ],
            &generic_templates,
            [32, value_dimension, batch_size * value_heads],
            [32, 4, 1],
        )
    }
    .map_err(EngineError::Mlx)?;
    let mut outputs = outputs.into_iter();
    let output = outputs
        .next()
        .ok_or_else(|| EngineError::Unsupported("gated-delta kernel omitted its output".into()))?;
    let state = outputs
        .next()
        .ok_or_else(|| EngineError::Unsupported("gated-delta kernel omitted its state".into()))?;
    Ok((output, state))
}

pub(crate) fn dimension_i32(value: usize, name: &str) -> Result<i32, EngineError> {
    i32::try_from(value)
        .map_err(|error| EngineError::Unsupported(format!("{name} exceeds MLX int ABI: {error}")))
}

pub(crate) fn dimension_f32(value: usize, name: &str) -> Result<f32, EngineError> {
    let value = u16::try_from(value).map_err(|error| {
        EngineError::Unsupported(format!(
            "{name} cannot be represented exactly as f32: {error}"
        ))
    })?;
    Ok(f32::from(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD_DIMENSION: usize = 128;
    const KEY_HEADS: usize = 2;
    const VALUE_HEADS: usize = 4;
    const CONVOLUTION_DIMENSION: usize = (2 * KEY_HEADS + VALUE_HEADS) * HEAD_DIMENSION;

    fn deterministic_values(count: usize, seed: u32, amplitude: f32) -> Vec<f32> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let bytes = state.to_le_bytes();
                let sample = u16::from_le_bytes([bytes[2], bytes[3]]);
                (f32::from(sample) / 32_767.5 - 1.0) * amplitude
            })
            .collect()
    }

    fn bf16_array(gpu: &Gpu, values: &[f32], shape: &[usize], name: &str) -> Array {
        Array::from_f32_slice(values, shape)
            .and_then(|value| gpu.astype(&value, DType::BFloat16))
            .unwrap_or_else(|error| panic!("failed to create {name}: {error}"))
    }

    #[allow(clippy::float_cmp)]
    fn assert_exact(gpu: &Gpu, actual: &Array, expected: &Array, label: &str, batch_size: usize) {
        assert_eq!(
            actual.shape(),
            expected.shape(),
            "{label} shape changed at batch size {batch_size}"
        );
        assert_eq!(
            actual.dtype(),
            expected.dtype(),
            "{label} dtype changed at batch size {batch_size}"
        );
        assert_eq!(
            gpu.max_abs_difference(actual, expected)
                .unwrap_or_else(|error| panic!("failed to compare {label}: {error}")),
            0.0,
            "{label} values changed at batch size {batch_size}"
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn batched_decode_preprocess_matches_mlx_exactly() {
        let gpu = Gpu::new();
        let (kernel, dispatches) = create_decode_preprocess_kernel(
            HEAD_DIMENSION,
            HEAD_DIMENSION,
            KEY_HEADS,
            VALUE_HEADS,
            4,
        )
        .expect("decode preprocess kernel");
        let convolution_weight_values =
            deterministic_values(CONVOLUTION_DIMENSION * 4, 0x243f_6a88, 0.125);
        let convolution_weight = bf16_array(
            &gpu,
            &convolution_weight_values,
            &[CONVOLUTION_DIMENSION, 4, 1],
            "convolution weights",
        );
        let a_log = bf16_array(&gpu, &[-2.0, -1.75, -1.5, -1.25], &[VALUE_HEADS], "A_log");
        let dt_bias = bf16_array(
            &gpu,
            &[-0.25, -0.125, 0.125, 0.25],
            &[VALUE_HEADS],
            "dt bias",
        );
        let q_scale = gpu
            .scalar_like(1.0 / 128.0, DType::BFloat16)
            .expect("q scale");
        let k_scale = gpu
            .scalar_like(1.0 / 128.0_f32.sqrt(), DType::BFloat16)
            .expect("k scale");

        for batch_size in 1..=MAX_ACTIVE_BATCH_SIZE {
            let seed_offset = u32::try_from(batch_size).expect("batch seed");
            let qkv_values = deterministic_values(
                batch_size * CONVOLUTION_DIMENSION,
                0x85a3_08d3 ^ seed_offset,
                0.5,
            );
            let qkv = bf16_array(
                &gpu,
                &qkv_values,
                &[batch_size, 1, CONVOLUTION_DIMENSION],
                "qkv",
            );
            let a_values =
                deterministic_values(batch_size * VALUE_HEADS, 0x1319_8a2e ^ seed_offset, 0.75);
            let a = bf16_array(&gpu, &a_values, &[batch_size, 1, VALUE_HEADS], "a");
            let b_values =
                deterministic_values(batch_size * VALUE_HEADS, 0x0370_7344 ^ seed_offset, 1.5);
            let b = bf16_array(&gpu, &b_values, &[batch_size, 1, VALUE_HEADS], "b");
            let state_values = deterministic_values(
                batch_size * 3 * CONVOLUTION_DIMENSION,
                0xa409_3822 ^ seed_offset,
                0.5,
            );
            let initial_state = bf16_array(
                &gpu,
                &state_values,
                &[batch_size, 3, CONVOLUTION_DIMENSION],
                "convolution state",
            );

            let dispatch = decode_dispatch_for_batch(
                &dispatches,
                batch_size,
                "test decode GDN preprocess fusion",
            )
            .expect("preprocess dispatch");
            let outputs = kernel
                .apply_prepared(
                    &gpu,
                    &[
                        &qkv,
                        &a,
                        &b,
                        &initial_state,
                        &convolution_weight,
                        &a_log,
                        &dt_bias,
                        &q_scale,
                        &k_scale,
                    ],
                    dispatch,
                )
                .expect("batched decode preprocess");
            let [
                actual_q,
                actual_k,
                actual_v,
                actual_decay,
                actual_beta,
                actual_state,
            ] = <[Array; 6]>::try_from(outputs).expect("six preprocess outputs");

            let convolution_input = gpu
                .concatenate(&[&initial_state, &qkv], 1)
                .expect("reference convolution input");
            let expected_state =
                tail_sequence(&gpu, &convolution_input, 3, "reference convolution state")
                    .expect("reference convolution state");
            let convolution_output = gpu
                .conv1d(
                    &convolution_input,
                    &convolution_weight,
                    1,
                    0,
                    1,
                    i32::try_from(CONVOLUTION_DIMENSION).expect("convolution groups"),
                )
                .and_then(|value| gpu.silu(&value))
                .expect("reference convolution");
            let key_dimension = KEY_HEADS * HEAD_DIMENSION;
            let expected_q =
                last_axis_slice(&gpu, &convolution_output, 0, key_dimension, "reference q")
                    .and_then(|value| {
                        gpu.reshape(&value, &[batch_size, 1, KEY_HEADS, HEAD_DIMENSION])
                            .map_err(EngineError::Mlx)
                    })
                    .and_then(|value| scale_normalized(&gpu, &value, &q_scale))
                    .expect("reference normalized q");
            let expected_k = last_axis_slice(
                &gpu,
                &convolution_output,
                key_dimension,
                2 * key_dimension,
                "reference k",
            )
            .and_then(|value| {
                gpu.reshape(&value, &[batch_size, 1, KEY_HEADS, HEAD_DIMENSION])
                    .map_err(EngineError::Mlx)
            })
            .and_then(|value| scale_normalized(&gpu, &value, &k_scale))
            .expect("reference normalized k");
            let expected_v = last_axis_slice(
                &gpu,
                &convolution_output,
                2 * key_dimension,
                CONVOLUTION_DIMENSION,
                "reference v",
            )
            .and_then(|value| {
                gpu.reshape(&value, &[batch_size, 1, VALUE_HEADS, HEAD_DIMENSION])
                    .map_err(EngineError::Mlx)
            })
            .expect("reference v");
            let expected_decay = gpu
                .compute_g(&a_log, &a, &dt_bias)
                .expect("reference decay");
            let expected_beta = gpu.sigmoid(&b).expect("reference beta");

            for (label, actual, expected) in [
                ("q", &actual_q, &expected_q),
                ("k", &actual_k, &expected_k),
                ("v", &actual_v, &expected_v),
                ("decay", &actual_decay, &expected_decay),
                ("beta", &actual_beta, &expected_beta),
                ("convolution state", &actual_state, &expected_state),
            ] {
                assert_exact(&gpu, actual, expected, label, batch_size);
            }
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn prepared_batched_recurrent_dispatch_matches_dynamic_kernel_exactly() {
        let gpu = Gpu::new();
        let generic_kernel = MetalKernel::new(
            "gated_delta_step_batch_reference",
            &["q", "k", "v", "g", "beta", "state_in", "T"],
            &["y", "state_out"],
            GATED_DELTA_SOURCE,
        )
        .expect("generic recurrent kernel");
        let (decode_kernel, dispatches) =
            create_decode_recurrent_kernel(HEAD_DIMENSION, HEAD_DIMENSION, KEY_HEADS, VALUE_HEADS)
                .expect("prepared recurrent kernel");
        let time = gpu.scalar_i32(1);
        let templates = [
            MetalTemplate::DType("InT", DType::BFloat16),
            MetalTemplate::DType("StT", DType::Float32),
            MetalTemplate::Int("Dk", i32::try_from(HEAD_DIMENSION).expect("Dk")),
            MetalTemplate::Int("Dv", i32::try_from(HEAD_DIMENSION).expect("Dv")),
            MetalTemplate::Int("Hk", i32::try_from(KEY_HEADS).expect("Hk")),
            MetalTemplate::Int("Hv", i32::try_from(VALUE_HEADS).expect("Hv")),
        ];

        for batch_size in 1..=MAX_ACTIVE_BATCH_SIZE {
            let seed_offset = u32::try_from(batch_size).expect("batch seed");
            let q = bf16_array(
                &gpu,
                &deterministic_values(
                    batch_size * KEY_HEADS * HEAD_DIMENSION,
                    0x299f_31d0 ^ seed_offset,
                    0.125,
                ),
                &[batch_size, 1, KEY_HEADS, HEAD_DIMENSION],
                "q",
            );
            let k = bf16_array(
                &gpu,
                &deterministic_values(
                    batch_size * KEY_HEADS * HEAD_DIMENSION,
                    0x082e_fa98 ^ seed_offset,
                    0.125,
                ),
                &[batch_size, 1, KEY_HEADS, HEAD_DIMENSION],
                "k",
            );
            let v = bf16_array(
                &gpu,
                &deterministic_values(
                    batch_size * VALUE_HEADS * HEAD_DIMENSION,
                    0xec4e_6c89 ^ seed_offset,
                    0.5,
                ),
                &[batch_size, 1, VALUE_HEADS, HEAD_DIMENSION],
                "v",
            );
            let decay_values = (0..batch_size * VALUE_HEADS)
                .map(|index| {
                    let bucket = u8::try_from(index % 9).expect("decay bucket");
                    0.75 + f32::from(bucket) / 100.0
                })
                .collect::<Vec<_>>();
            let decay =
                Array::from_f32_slice(&decay_values, &[batch_size, 1, VALUE_HEADS]).expect("decay");
            let beta_values = (0..batch_size * VALUE_HEADS)
                .map(|index| {
                    let bucket = u8::try_from(index % 7).expect("beta bucket");
                    0.25 + f32::from(bucket) / 16.0
                })
                .collect::<Vec<_>>();
            let beta = bf16_array(&gpu, &beta_values, &[batch_size, 1, VALUE_HEADS], "beta");
            let state = Array::from_f32_slice(
                &deterministic_values(
                    batch_size * VALUE_HEADS * HEAD_DIMENSION * HEAD_DIMENSION,
                    0x4528_21e6 ^ seed_offset,
                    0.03125,
                ),
                &[batch_size, VALUE_HEADS, HEAD_DIMENSION, HEAD_DIMENSION],
            )
            .expect("recurrent state");
            let inputs = [&q, &k, &v, &decay, &beta, &state, &time];
            let expected = generic_kernel
                .apply(
                    &gpu,
                    &inputs,
                    &[
                        MetalOutput {
                            shape: vec![batch_size, 1, VALUE_HEADS, HEAD_DIMENSION],
                            dtype: DType::BFloat16,
                        },
                        MetalOutput {
                            shape: state.shape(),
                            dtype: DType::Float32,
                        },
                    ],
                    &templates,
                    [32, HEAD_DIMENSION, batch_size * VALUE_HEADS],
                    [32, 4, 1],
                )
                .expect("dynamic recurrent reference");
            let [expected_output, expected_state] =
                <[Array; 2]>::try_from(expected).expect("two reference outputs");
            let dispatch = decode_dispatch_for_batch(
                &dispatches,
                batch_size,
                "test decode GDN recurrent kernel",
            )
            .expect("recurrent dispatch");
            let actual = decode_kernel
                .apply_prepared(&gpu, &inputs, dispatch)
                .expect("prepared recurrent output");
            let [actual_output, actual_state] =
                <[Array; 2]>::try_from(actual).expect("two prepared outputs");

            assert_exact(
                &gpu,
                &actual_output,
                &expected_output,
                "recurrent output",
                batch_size,
            );
            assert_exact(
                &gpu,
                &actual_state,
                &expected_state,
                "recurrent state",
                batch_size,
            );
        }
    }

    #[test]
    fn batched_decode_postprocess_matches_mlx_exactly() {
        let gpu = Gpu::new();
        let (kernel, dispatches) = create_decode_postprocess_kernel(HEAD_DIMENSION, VALUE_HEADS)
            .expect("decode postprocess kernel");
        let norm_weight = bf16_array(
            &gpu,
            &deterministic_values(HEAD_DIMENSION, 0xbe54_66cf, 0.25)
                .into_iter()
                .map(|value| value + 1.0)
                .collect::<Vec<_>>(),
            &[HEAD_DIMENSION],
            "norm weight",
        );
        let epsilon = gpu.scalar_like(1e-6, DType::Float32).expect("norm epsilon");

        for batch_size in 1..=MAX_ACTIVE_BATCH_SIZE {
            let seed_offset = u32::try_from(batch_size).expect("batch seed");
            let recurrent_output = bf16_array(
                &gpu,
                &deterministic_values(
                    batch_size * VALUE_HEADS * HEAD_DIMENSION,
                    0xc0ac_29b7 ^ seed_offset,
                    2.0,
                ),
                &[batch_size, 1, VALUE_HEADS, HEAD_DIMENSION],
                "recurrent output",
            );
            let z = bf16_array(
                &gpu,
                &deterministic_values(
                    batch_size * VALUE_HEADS * HEAD_DIMENSION,
                    0xc97c_50dd ^ seed_offset,
                    3.0,
                ),
                &[batch_size, 1, VALUE_HEADS, HEAD_DIMENSION],
                "z",
            );
            let normalized = gpu
                .rms_norm(&recurrent_output, Some(&norm_weight), 1e-6)
                .expect("reference RMS norm");
            let expected = gpu
                .precise_swiglu(&recurrent_output, &z, &normalized)
                .expect("reference precise SwiGLU");
            let dispatch = decode_dispatch_for_batch(
                &dispatches,
                batch_size,
                "test decode GDN postprocess fusion",
            )
            .expect("postprocess dispatch");
            let outputs = kernel
                .apply_prepared(
                    &gpu,
                    &[&recurrent_output, &z, &norm_weight, &epsilon],
                    dispatch,
                )
                .expect("batched decode postprocess");
            let [actual] = <[Array; 1]>::try_from(outputs).expect("one postprocess output");
            assert_exact(&gpu, &actual, &expected, "postprocess", batch_size);
        }
    }
}
