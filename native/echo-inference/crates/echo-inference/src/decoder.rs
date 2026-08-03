use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use echo_mlx::{
    Array, DType, GatherQuantizedMatmulConfig, Gpu, MetalKernel, MetalKernelDispatch, MetalOutput,
    SafeTensors,
};
use serde::{Deserialize, Serialize};

use super::gdn::{
    apply_bound_quantized_linear, dimension_i32, execute_gdn_layer, load_weight_shard,
    quantized_linear_with_config, require_tensor, validate_array,
};
use super::weights::{BoundMoeWeights, BoundQuantizedWeights, TensorLookup};
use super::{EngineError, ModelPlan, sha256_file};

const MOE_DECODE_ROUTER_SOURCE: &str = r"
        constexpr uint expert_count = __EXPERT_COUNT__;
        constexpr uint selected_count = __SELECTED_COUNT__;
        constexpr uint values_per_thread = 4;
        constexpr uint selection_values_per_lane =
            __SELECTION_VALUES_PER_LANE__;

        auto lid = thread_position_in_threadgroup.x;
        auto lane = thread_index_in_simdgroup;
        auto simd_group = simdgroup_index_in_threadgroup;

        threadgroup float partial_maxima[32];
        threadgroup float partial_normalizers[32];
        threadgroup bfloat16_t probabilities[expert_count];
        threadgroup bfloat16_t selected_probabilities[selected_count];
        threadgroup uint selected_indices[selected_count];

        float logits[values_per_thread];
        float exponentials[values_per_thread];
        auto input_offset = lid * values_per_thread;
        for (uint i = 0; i < values_per_thread; ++i) {
          auto index = input_offset + i;
          logits[i] = index < expert_count
              ? static_cast<float>(router_logits[index])
              : Limits<float>::min;
        }

        // Reproduce MLX's precise BF16 block-softmax reduction exactly. The
        // output cast is load-bearing because routing is performed on the
        // rounded BF16 probabilities, including its stable tie behavior.
        if (simd_group == 0) {
          partial_maxima[lane] = Limits<float>::min;
          partial_normalizers[lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float maximum = Limits<float>::finite_min;
        for (uint i = 0; i < values_per_thread; ++i) {
          maximum = maximum < logits[i] ? logits[i] : maximum;
        }
        maximum = simd_max(maximum);
        if (lane == 0) {
          partial_maxima[simd_group] = maximum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
          maximum = simd_max(partial_maxima[lane]);
          if (lane == 0) {
            partial_maxima[0] = maximum;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        maximum = partial_maxima[0];

        float normalizer = 0.0f;
        for (uint i = 0; i < values_per_thread; ++i) {
          exponentials[i] = fast::exp(logits[i] - maximum);
          normalizer += exponentials[i];
        }
        normalizer = simd_sum(normalizer);
        if (lane == 0) {
          partial_normalizers[simd_group] = normalizer;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
          normalizer = simd_sum(partial_normalizers[lane]);
          if (lane == 0) {
            partial_normalizers[0] = normalizer;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto inverse_normalizer = 1.0f / partial_normalizers[0];

        for (uint i = 0; i < values_per_thread; ++i) {
          auto index = input_offset + i;
          if (index < expert_count) {
            probabilities[index] = static_cast<bfloat16_t>(
                exponentials[i] * inverse_normalizer);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MLX currently implements argpartition as a stable full sort. Its
        // final top-k is therefore the largest (probability, original index)
        // pairs. Select those pairs directly and write them in the same
        // ascending order as the tail of MLX's sorted output.
        if (simd_group == 0) {
          uint lane_keys[selection_values_per_lane];
          for (uint i = 0; i < selection_values_per_lane; ++i) {
            auto index = lane + i * 32;
            if (index < expert_count) {
              auto probability = probabilities[index];
              auto probability_bits = isnan(probability)
                  ? uint(0xffff)
                  : static_cast<uint>(bfloat16_to_uint16(probability));
              lane_keys[i] = (probability_bits << 16) | index;
            } else {
              lane_keys[i] = 0;
            }
          }

          for (uint rank = 0; rank < selected_count; ++rank) {
            uint best_key = 0;
            for (uint i = 0; i < selection_values_per_lane; ++i) {
              best_key = max(best_key, lane_keys[i]);
            }
            best_key = simd_max(best_key);
            auto best_index = best_key & 0xffff;

            if (lane == 0) {
              auto output_index = selected_count - rank - 1;
              selected_probabilities[output_index] = probabilities[best_index];
              selected_indices[output_index] = best_index;
            }
            for (uint i = 0; i < selection_values_per_lane; ++i) {
              if (lane_keys[i] == best_key) {
                lane_keys[i] = 0;
              }
            }
          }

          if (lane == 0) {
            // MLX's BF16 row reduction accumulates in selected-array order.
            bfloat16_t total = static_cast<bfloat16_t>(0.0f);
            for (uint i = 0; i < selected_count; ++i) {
              total = selected_probabilities[i] + total;
            }
            for (uint i = 0; i < selected_count; ++i) {
              expert_indices[i] = selected_indices[i];
              expert_scores[i] = static_cast<bfloat16_t>(
                  selected_probabilities[i] / total);
            }
          }
        }
    ";

// Fixed-shape specialization of MLX 0.32.0's affine `qmv_fast` arithmetic.
// In particular, the BF16 four-value sum below is load-bearing for exactness.
const MOE_DECODE_EXPERT_GATE_UP_SOURCE: &str = r"
        constexpr uint input_dimension = __INPUT_DIMENSION__;
        constexpr uint intermediate_dimension = __INTERMEDIATE_DIMENSION__;
        constexpr uint rows_per_threadgroup = 8;
        constexpr uint results_per_simdgroup = 4;
        constexpr uint values_per_thread = 16;
        constexpr uint block_size = 512;
        constexpr uint group_size = 64;
        constexpr uint groups_per_row = input_dimension / group_size;
        constexpr uint bytes_per_row = input_dimension / 2;

        auto expert_slot = threadgroup_position_in_grid.z;
        auto output_block = threadgroup_position_in_grid.y;
        auto simd_group = simdgroup_index_in_threadgroup;
        auto lane = thread_index_in_simdgroup;
        auto expert_index = expert_indices[expert_slot];
        auto output_row =
            output_block * rows_per_threadgroup
            + simd_group * results_per_simdgroup;

        auto expert_byte_offset =
            expert_index * intermediate_dimension * bytes_per_row;
        auto expert_group_offset =
            expert_index * intermediate_dimension * groups_per_row;
        const device uchar* gate_weights =
            (const device uchar*)expert_gate_weight
            + expert_byte_offset + output_row * bytes_per_row + lane * 8;
        const device uchar* up_weights =
            (const device uchar*)expert_up_weight
            + expert_byte_offset + output_row * bytes_per_row + lane * 8;
        auto gate_scales = expert_gate_scales
            + expert_group_offset + output_row * groups_per_row + lane / 4;
        auto gate_biases = expert_gate_biases
            + expert_group_offset + output_row * groups_per_row + lane / 4;
        auto up_scales = expert_up_scales
            + expert_group_offset + output_row * groups_per_row + lane / 4;
        auto up_biases = expert_up_biases
            + expert_group_offset + output_row * groups_per_row + lane / 4;
        auto input_values = expert_input + lane * values_per_thread;

        float gate_results[results_per_simdgroup] = {0.0f};
        float up_results[results_per_simdgroup] = {0.0f};
        float input_thread[values_per_thread];

        for (uint block = 0; block < input_dimension; block += block_size) {
          float input_sum = 0.0f;
          for (uint index = 0; index < values_per_thread; index += 4) {
            input_sum += input_values[index]
                + input_values[index + 1]
                + input_values[index + 2]
                + input_values[index + 3];
            input_thread[index] = input_values[index];
            input_thread[index + 1] = input_values[index + 1] / 16.0f;
            input_thread[index + 2] = input_values[index + 2] / 256.0f;
            input_thread[index + 3] = input_values[index + 3] / 4096.0f;
          }

          for (uint row = 0; row < results_per_simdgroup; ++row) {
            const device ushort* gate_words = (const device ushort*)(
                gate_weights + row * bytes_per_row);
            const device ushort* up_words = (const device ushort*)(
                up_weights + row * bytes_per_row);
            float gate_accumulator = 0.0f;
            float up_accumulator = 0.0f;
            for (uint index = 0; index < values_per_thread / 4; ++index) {
              auto x_offset = index * 4;
              auto gate_word = gate_words[index];
              auto up_word = up_words[index];
              gate_accumulator +=
                  input_thread[x_offset] * (gate_word & 0x000f)
                  + input_thread[x_offset + 1] * (gate_word & 0x00f0)
                  + input_thread[x_offset + 2] * (gate_word & 0x0f00)
                  + input_thread[x_offset + 3] * (gate_word & 0xf000);
              up_accumulator +=
                  input_thread[x_offset] * (up_word & 0x000f)
                  + input_thread[x_offset + 1] * (up_word & 0x00f0)
                  + input_thread[x_offset + 2] * (up_word & 0x0f00)
                  + input_thread[x_offset + 3] * (up_word & 0xf000);
            }

            auto row_group_offset = row * groups_per_row;
            auto gate_scale =
                static_cast<float>(gate_scales[row_group_offset]);
            auto gate_bias =
                static_cast<float>(gate_biases[row_group_offset]);
            auto up_scale = static_cast<float>(up_scales[row_group_offset]);
            auto up_bias = static_cast<float>(up_biases[row_group_offset]);
            gate_results[row] +=
                gate_scale * gate_accumulator + input_sum * gate_bias;
            up_results[row] +=
                up_scale * up_accumulator + input_sum * up_bias;
          }

          gate_weights += block_size / 2;
          up_weights += block_size / 2;
          gate_scales += block_size / group_size;
          gate_biases += block_size / group_size;
          up_scales += block_size / group_size;
          up_biases += block_size / group_size;
          input_values += block_size;
        }

        auto output_offset =
            expert_slot * intermediate_dimension + output_row;
        for (uint row = 0; row < results_per_simdgroup; ++row) {
          gate_results[row] = simd_sum(gate_results[row]);
          up_results[row] = simd_sum(up_results[row]);
          if (lane == 0) {
            expert_gate[output_offset + row] =
                static_cast<bfloat16_t>(gate_results[row]);
            expert_up[output_offset + row] =
                static_cast<bfloat16_t>(up_results[row]);
          }
        }
    ";

// Fixed-shape specialization of MLX 0.32.0's affine `qmv_fast`, followed by
// the existing BF16 score multiplication and selected-expert reduction.
const MOE_DECODE_ROUTED_DOWN_REDUCE_SOURCE: &str = r"
        constexpr uint input_dimension = __INPUT_DIMENSION__;
        constexpr uint output_dimension = __OUTPUT_DIMENSION__;
        constexpr uint selected_count = __SELECTED_COUNT__;
        constexpr uint results_per_simdgroup = 4;
        constexpr uint values_per_thread = 16;
        constexpr uint block_size = 512;
        constexpr uint group_size = 64;
        constexpr uint groups_per_row = input_dimension / group_size;
        constexpr uint bytes_per_row = input_dimension / 2;

        auto expert_slot = simdgroup_index_in_threadgroup;
        auto lane = thread_index_in_simdgroup;
        auto output_row =
            threadgroup_position_in_grid.y * results_per_simdgroup;
        auto expert_index = expert_indices[expert_slot];

        auto expert_byte_offset =
            expert_index * output_dimension * bytes_per_row;
        auto expert_group_offset =
            expert_index * output_dimension * groups_per_row;
        const device uchar* down_weights =
            (const device uchar*)expert_down_weight
            + expert_byte_offset + output_row * bytes_per_row + lane * 8;
        auto down_scales = expert_down_scales
            + expert_group_offset + output_row * groups_per_row + lane / 4;
        auto down_biases = expert_down_biases
            + expert_group_offset + output_row * groups_per_row + lane / 4;
        auto input_values = expert_activated
            + expert_slot * input_dimension + lane * values_per_thread;

        float results[results_per_simdgroup] = {0.0f};
        float input_thread[values_per_thread];

        for (uint block = 0; block < input_dimension; block += block_size) {
          float input_sum = 0.0f;
          for (uint index = 0; index < values_per_thread; index += 4) {
            input_sum += input_values[index]
                + input_values[index + 1]
                + input_values[index + 2]
                + input_values[index + 3];
            input_thread[index] = input_values[index];
            input_thread[index + 1] = input_values[index + 1] / 16.0f;
            input_thread[index + 2] = input_values[index + 2] / 256.0f;
            input_thread[index + 3] = input_values[index + 3] / 4096.0f;
          }

          for (uint row = 0; row < results_per_simdgroup; ++row) {
            const device ushort* down_words = (const device ushort*)(
                down_weights + row * bytes_per_row);
            float accumulator = 0.0f;
            for (uint index = 0; index < values_per_thread / 4; ++index) {
              auto x_offset = index * 4;
              auto word = down_words[index];
              accumulator +=
                  input_thread[x_offset] * (word & 0x000f)
                  + input_thread[x_offset + 1] * (word & 0x00f0)
                  + input_thread[x_offset + 2] * (word & 0x0f00)
                  + input_thread[x_offset + 3] * (word & 0xf000);
            }

            auto row_group_offset = row * groups_per_row;
            auto scale = static_cast<float>(
                down_scales[row_group_offset]);
            auto bias = static_cast<float>(
                down_biases[row_group_offset]);
            results[row] += scale * accumulator + input_sum * bias;
          }

          down_weights += block_size / 2;
          down_scales += block_size / group_size;
          down_biases += block_size / group_size;
          input_values += block_size;
        }

        threadgroup bfloat16_t
            weighted[selected_count][results_per_simdgroup];
        for (uint row = 0; row < results_per_simdgroup; ++row) {
          results[row] = simd_sum(results[row]);
          if (lane == 0) {
            auto projected = static_cast<bfloat16_t>(results[row]);
            weighted[expert_slot][row] = static_cast<bfloat16_t>(
                projected * expert_scores[expert_slot]);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (expert_slot == 0 && lane < results_per_simdgroup) {
          bfloat16_t total = static_cast<bfloat16_t>(0.0f);
          for (uint index = 0; index < selected_count; ++index) {
            total = weighted[index][lane] + total;
          }
          routed_output[output_row + lane] = total;
        }
    ";

#[cfg(feature = "moe-performance-diagnostics")]
const MOE_PERFORMANCE_MODE_ENVIRONMENT: &str = "ECHO_MOE_PERFORMANCE_MODE";

#[cfg(feature = "moe-performance-diagnostics")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MoePerformanceMode {
    Full,
    None,
    RouterOnly,
    RoutedOnly,
    SharedOnly,
}

#[cfg(feature = "moe-performance-diagnostics")]
impl MoePerformanceMode {
    fn from_environment() -> Result<Self, EngineError> {
        match std::env::var(MOE_PERFORMANCE_MODE_ENVIRONMENT)
            .unwrap_or_else(|_| "full".into())
            .as_str()
        {
            "full" => Ok(Self::Full),
            "none" => Ok(Self::None),
            "router_only" => Ok(Self::RouterOnly),
            "routed_only" => Ok(Self::RoutedOnly),
            "shared_only" => Ok(Self::SharedOnly),
            value => Err(EngineError::Unsupported(format!(
                "{MOE_PERFORMANCE_MODE_ENVIRONMENT} must be full, none, router_only, routed_only, or shared_only; observed {value}"
            ))),
        }
    }
}

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
    norm_topk_prob: bool,
    #[serde(default)]
    sorted_expert_path: bool,
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
    expert_count: usize,
    experts_per_token: usize,
    moe_intermediate_size: usize,
    shared_expert_intermediate_size: usize,
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

pub(crate) struct MoeExecution {
    output: Array,
    trace: Option<BTreeMap<&'static str, Array>>,
}

/// Reusable fixed-shape routing resources for one-token `MoE` decode.
///
/// The kernel reproduces MLX's precise softmax rounding and stable
/// `argpartition` tail before normalizing the selected scores. Keeping all
/// 256 probabilities in threadgroup memory removes the intermediate arrays and
/// small dispatches without changing the admitted BF16 behavior.
pub(crate) struct MoeKernel {
    decode_router_kernel: MetalKernel,
    decode_router_dispatch: MetalKernelDispatch,
    decode_expert_gate_up: Option<MoeExpertGateUpKernel>,
    decode_routed_down_reduce: Option<MoeRoutedDownReduceKernel>,
    expert_count: usize,
    experts_per_token: usize,
    #[cfg(feature = "moe-performance-diagnostics")]
    performance_mode: MoePerformanceMode,
}

struct MoeExpertGateUpKernel {
    kernel: MetalKernel,
    dispatch: MetalKernelDispatch,
    input_dimension: usize,
    intermediate_dimension: usize,
}

struct MoeRoutedDownReduceKernel {
    kernel: MetalKernel,
    dispatch: MetalKernelDispatch,
    input_dimension: usize,
    output_dimension: usize,
    selected_count: usize,
}

impl MoeKernel {
    pub(crate) fn new(plan: &ModelPlan) -> Result<Self, EngineError> {
        let mut kernel = Self::for_dimensions(plan.expert_count, plan.experts_per_token)?;
        kernel.decode_expert_gate_up = prepare_decode_expert_gate_up_kernel(plan)?;
        kernel.decode_routed_down_reduce = prepare_decode_routed_down_reduce_kernel(plan)?;
        #[cfg(feature = "moe-performance-diagnostics")]
        let kernel = Self {
            performance_mode: MoePerformanceMode::from_environment()?,
            ..kernel
        };
        Ok(kernel)
    }

    fn for_dimensions(expert_count: usize, experts_per_token: usize) -> Result<Self, EngineError> {
        if expert_count == 0 || expert_count > 4096 {
            return Err(EngineError::Unsupported(format!(
                "decode MoE router fusion requires 1..=4096 experts, observed {expert_count}"
            )));
        }
        if experts_per_token == 0 || experts_per_token > 32 || experts_per_token > expert_count {
            return Err(EngineError::Unsupported(format!(
                "decode MoE router fusion requires 1..=32 selected experts within {expert_count}, observed {experts_per_token}"
            )));
        }
        let values_per_thread = 4;
        let simd_width = 32;
        let thread_count = expert_count
            .div_ceil(values_per_thread)
            .div_ceil(simd_width)
            * simd_width;
        let selection_values_per_lane = expert_count.div_ceil(simd_width);
        let source = MOE_DECODE_ROUTER_SOURCE
            .replace("__EXPERT_COUNT__", &expert_count.to_string())
            .replace("__SELECTED_COUNT__", &experts_per_token.to_string())
            .replace(
                "__SELECTION_VALUES_PER_LANE__",
                &selection_values_per_lane.to_string(),
            );
        let name = format!("moe_decode_router_bf16_e{expert_count}_top{experts_per_token}");
        let decode_router_kernel = MetalKernel::new(
            &name,
            &["router_logits"],
            &["expert_indices", "expert_scores"],
            &source,
        )
        .map_err(EngineError::Mlx)?;
        let decode_router_dispatch = MetalKernel::prepare_dispatch(
            &[
                MetalOutput {
                    shape: vec![1, 1, experts_per_token],
                    dtype: DType::Uint32,
                },
                MetalOutput {
                    shape: vec![1, 1, experts_per_token],
                    dtype: DType::BFloat16,
                },
            ],
            &[],
            [thread_count, 1, 1],
            [thread_count, 1, 1],
        )
        .map_err(EngineError::Mlx)?;
        Ok(Self {
            decode_router_kernel,
            decode_router_dispatch,
            decode_expert_gate_up: None,
            decode_routed_down_reduce: None,
            expert_count,
            experts_per_token,
            #[cfg(feature = "moe-performance-diagnostics")]
            performance_mode: MoePerformanceMode::Full,
        })
    }

    fn supports_decode_route(
        &self,
        input: &Array,
        batch_size: usize,
        sequence_length: usize,
        plan: &ModelPlan,
    ) -> bool {
        batch_size == 1
            && sequence_length == 1
            && input.dtype() == DType::BFloat16
            && plan.norm_topk_prob
            && plan.expert_count == self.expert_count
            && plan.experts_per_token == self.experts_per_token
    }

    fn route_decode(
        &self,
        gpu: &Gpu,
        router_logits: &Array,
    ) -> Result<(Array, Array), EngineError> {
        let mut outputs = self
            .decode_router_kernel
            .apply_prepared(gpu, &[router_logits], &self.decode_router_dispatch)
            .map_err(EngineError::Mlx)?;
        let expert_scores = outputs.pop().ok_or_else(|| {
            EngineError::Unsupported("decode MoE router kernel omitted expert scores".into())
        })?;
        let expert_indices = outputs.pop().ok_or_else(|| {
            EngineError::Unsupported("decode MoE router kernel omitted expert indices".into())
        })?;
        Ok((expert_indices, expert_scores))
    }

    fn expert_gate_up_decode(
        &self,
        gpu: &Gpu,
        input: &Array,
        expert_indices: &Array,
        weights: &BoundMoeWeights,
    ) -> Result<Option<(Array, Array)>, EngineError> {
        let Some(kernel) = &self.decode_expert_gate_up else {
            return Ok(None);
        };
        let mut outputs = kernel
            .kernel
            .apply_prepared(
                gpu,
                &[
                    &weights.expert_gate.weight,
                    &weights.expert_gate.scales,
                    &weights.expert_gate.biases,
                    &weights.expert_up.weight,
                    &weights.expert_up.scales,
                    &weights.expert_up.biases,
                    input,
                    expert_indices,
                ],
                &kernel.dispatch,
            )
            .map_err(EngineError::Mlx)?;
        let expert_up = outputs.pop().ok_or_else(|| {
            EngineError::Unsupported("decode expert fusion omitted up projection".into())
        })?;
        let expert_gate = outputs.pop().ok_or_else(|| {
            EngineError::Unsupported("decode expert fusion omitted gate projection".into())
        })?;
        Ok(Some((expert_gate, expert_up)))
    }

    fn supports_expert_gate_up_decode(
        &self,
        input: &Array,
        batch_size: usize,
        sequence_length: usize,
        plan: &ModelPlan,
        capture_trace: bool,
    ) -> bool {
        self.decode_expert_gate_up.as_ref().is_some_and(|kernel| {
            !capture_trace
                && batch_size == 1
                && sequence_length == 1
                && input.dtype() == DType::BFloat16
                && plan.hidden_size == kernel.input_dimension
                && plan.moe_intermediate_size == kernel.intermediate_dimension
        })
    }

    fn routed_down_reduce_decode(
        &self,
        gpu: &Gpu,
        expert_activated: &Array,
        expert_indices: &Array,
        expert_scores: &Array,
        weights: &BoundMoeWeights,
    ) -> Result<Option<Array>, EngineError> {
        let Some(kernel) = &self.decode_routed_down_reduce else {
            return Ok(None);
        };
        let mut outputs = kernel
            .kernel
            .apply_prepared(
                gpu,
                &[
                    &weights.expert_down.weight,
                    &weights.expert_down.scales,
                    &weights.expert_down.biases,
                    expert_activated,
                    expert_indices,
                    expert_scores,
                ],
                &kernel.dispatch,
            )
            .map_err(EngineError::Mlx)?;
        let routed_output = outputs.pop().ok_or_else(|| {
            EngineError::Unsupported("decode routed down fusion omitted output".into())
        })?;
        Ok(Some(routed_output))
    }

    fn supports_routed_down_reduce_decode(
        &self,
        input: &Array,
        batch_size: usize,
        sequence_length: usize,
        plan: &ModelPlan,
        capture_trace: bool,
        use_sorted_dispatch: bool,
    ) -> bool {
        self.decode_routed_down_reduce
            .as_ref()
            .is_some_and(|kernel| {
                !capture_trace
                    && !use_sorted_dispatch
                    && batch_size == 1
                    && sequence_length == 1
                    && input.dtype() == DType::BFloat16
                    && plan.moe_intermediate_size == kernel.input_dimension
                    && plan.hidden_size == kernel.output_dimension
                    && plan.experts_per_token == kernel.selected_count
            })
    }
}

fn prepare_decode_expert_gate_up_kernel(
    plan: &ModelPlan,
) -> Result<Option<MoeExpertGateUpKernel>, EngineError> {
    let compatible = plan.quantization_bits == 4
        && plan.quantization_group_size == 64
        && plan.quantization_mode == "affine"
        && plan.hidden_size.is_multiple_of(512)
        && plan.moe_intermediate_size.is_multiple_of(8);
    if !compatible {
        return Ok(None);
    }
    Ok(Some(create_decode_expert_gate_up_kernel(
        plan.hidden_size,
        plan.moe_intermediate_size,
        plan.expert_count,
        plan.experts_per_token,
    )?))
}

fn create_decode_expert_gate_up_kernel(
    input_dimension: usize,
    intermediate_dimension: usize,
    expert_count: usize,
    experts_per_token: usize,
) -> Result<MoeExpertGateUpKernel, EngineError> {
    let source = MOE_DECODE_EXPERT_GATE_UP_SOURCE
        .replace("__INPUT_DIMENSION__", &input_dimension.to_string())
        .replace(
            "__INTERMEDIATE_DIMENSION__",
            &intermediate_dimension.to_string(),
        );
    let name = format!(
        "moe_decode_expert_gate_up_bf16_q4_h{input_dimension}_m{intermediate_dimension}_e{expert_count}_top{experts_per_token}"
    );
    let kernel = MetalKernel::new(
        &name,
        &[
            "expert_gate_weight",
            "expert_gate_scales",
            "expert_gate_biases",
            "expert_up_weight",
            "expert_up_scales",
            "expert_up_biases",
            "expert_input",
            "expert_indices",
        ],
        &["expert_gate", "expert_up"],
        &source,
    )
    .map_err(EngineError::Mlx)?;
    let output_shape = vec![1, 1, experts_per_token, 1, intermediate_dimension];
    let dispatch = MetalKernel::prepare_dispatch(
        &[
            MetalOutput {
                shape: output_shape.clone(),
                dtype: DType::BFloat16,
            },
            MetalOutput {
                shape: output_shape,
                dtype: DType::BFloat16,
            },
        ],
        &[],
        [64, intermediate_dimension / 8, experts_per_token],
        [64, 1, 1],
    )
    .map_err(EngineError::Mlx)?;
    Ok(MoeExpertGateUpKernel {
        kernel,
        dispatch,
        input_dimension,
        intermediate_dimension,
    })
}

fn prepare_decode_routed_down_reduce_kernel(
    plan: &ModelPlan,
) -> Result<Option<MoeRoutedDownReduceKernel>, EngineError> {
    let compatible = plan.quantization_bits == 4
        && plan.quantization_group_size == 64
        && plan.quantization_mode == "affine"
        && plan.moe_intermediate_size.is_multiple_of(512)
        && plan.hidden_size.is_multiple_of(4)
        && (1..=8).contains(&plan.experts_per_token);
    if !compatible {
        return Ok(None);
    }
    Ok(Some(create_decode_routed_down_reduce_kernel(
        plan.moe_intermediate_size,
        plan.hidden_size,
        plan.expert_count,
        plan.experts_per_token,
    )?))
}

fn create_decode_routed_down_reduce_kernel(
    input_dimension: usize,
    output_dimension: usize,
    expert_count: usize,
    selected_count: usize,
) -> Result<MoeRoutedDownReduceKernel, EngineError> {
    let source = MOE_DECODE_ROUTED_DOWN_REDUCE_SOURCE
        .replace("__INPUT_DIMENSION__", &input_dimension.to_string())
        .replace("__OUTPUT_DIMENSION__", &output_dimension.to_string())
        .replace("__SELECTED_COUNT__", &selected_count.to_string());
    let name = format!(
        "moe_decode_routed_down_reduce_bf16_q4_m{input_dimension}_h{output_dimension}_e{expert_count}_top{selected_count}"
    );
    let kernel = MetalKernel::new(
        &name,
        &[
            "expert_down_weight",
            "expert_down_scales",
            "expert_down_biases",
            "expert_activated",
            "expert_indices",
            "expert_scores",
        ],
        &["routed_output"],
        &source,
    )
    .map_err(EngineError::Mlx)?;
    let thread_count = selected_count * 32;
    let dispatch = MetalKernel::prepare_dispatch(
        &[MetalOutput {
            shape: vec![1, 1, output_dimension],
            dtype: DType::BFloat16,
        }],
        &[],
        [thread_count, output_dimension / 4, 1],
        [thread_count, 1, 1],
    )
    .map_err(EngineError::Mlx)?;
    Ok(MoeRoutedDownReduceKernel {
        kernel,
        dispatch,
        input_dimension,
        output_dimension,
        selected_count,
    })
}

struct SharedExpertTrace {
    gate_projection: Array,
    up_projection: Array,
    activated: Array,
    expert_output: Array,
    gate_logits: Array,
    gate: Array,
    output: Array,
}

struct SharedExpertExecution {
    output: Array,
    trace: Option<SharedExpertTrace>,
}

enum MoeWeightSource<'a> {
    Named {
        weights: &'a dyn TensorLookup,
        prefix: &'a str,
    },
    Bound(&'a BoundMoeWeights),
}

#[derive(Clone, Copy)]
enum MoeLinear {
    Router,
    SharedGate,
    SharedUp,
    SharedDown,
    SharedExpertGate,
}

#[derive(Clone, Copy)]
enum MoeExpertLinear {
    Up,
    Gate,
    Down,
}

impl MoeWeightSource<'_> {
    fn apply_linear(
        &self,
        gpu: &Gpu,
        input: &Array,
        projection: MoeLinear,
        plan: &ModelPlan,
    ) -> Result<Array, EngineError> {
        match self {
            Self::Named { weights, prefix } => {
                let (suffix, group_size, bits, mode) = match projection {
                    MoeLinear::Router => (
                        "gate",
                        plan.router_quantization_group_size,
                        plan.router_quantization_bits,
                        plan.router_quantization_mode.as_str(),
                    ),
                    MoeLinear::SharedGate => (
                        "shared_expert.gate_proj",
                        plan.quantization_group_size,
                        plan.quantization_bits,
                        plan.quantization_mode.as_str(),
                    ),
                    MoeLinear::SharedUp => (
                        "shared_expert.up_proj",
                        plan.quantization_group_size,
                        plan.quantization_bits,
                        plan.quantization_mode.as_str(),
                    ),
                    MoeLinear::SharedDown => (
                        "shared_expert.down_proj",
                        plan.quantization_group_size,
                        plan.quantization_bits,
                        plan.quantization_mode.as_str(),
                    ),
                    MoeLinear::SharedExpertGate => (
                        "shared_expert_gate",
                        plan.shared_gate_quantization_group_size,
                        plan.shared_gate_quantization_bits,
                        plan.shared_gate_quantization_mode.as_str(),
                    ),
                };
                quantized_linear_with_config(
                    gpu,
                    input,
                    *weights,
                    &format!("{prefix}.{suffix}"),
                    group_size,
                    bits,
                    mode,
                )
            }
            Self::Bound(weights) => {
                let weights = match projection {
                    MoeLinear::Router => &weights.router,
                    MoeLinear::SharedGate => &weights.shared_gate,
                    MoeLinear::SharedUp => &weights.shared_up,
                    MoeLinear::SharedDown => &weights.shared_down,
                    MoeLinear::SharedExpertGate => &weights.shared_expert_gate,
                };
                apply_bound_quantized_linear(gpu, input, weights)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_expert_linear(
        &self,
        gpu: &Gpu,
        input: &Array,
        expert_indices: &Array,
        projection: MoeExpertLinear,
        expert_count: usize,
        output_dimension: usize,
        input_dimension: usize,
        plan: &ModelPlan,
        sorted_indices: bool,
    ) -> Result<Array, EngineError> {
        match self {
            Self::Named { weights, prefix } => {
                let suffix = match projection {
                    MoeExpertLinear::Up => "switch_mlp.up_proj",
                    MoeExpertLinear::Gate => "switch_mlp.gate_proj",
                    MoeExpertLinear::Down => "switch_mlp.down_proj",
                };
                gather_quantized_linear(
                    gpu,
                    input,
                    expert_indices,
                    *weights,
                    &format!("{prefix}.{suffix}"),
                    expert_count,
                    output_dimension,
                    input_dimension,
                    plan,
                    sorted_indices,
                )
            }
            Self::Bound(weights) => {
                let weights = match projection {
                    MoeExpertLinear::Up => &weights.expert_up,
                    MoeExpertLinear::Gate => &weights.expert_gate,
                    MoeExpertLinear::Down => &weights.expert_down,
                };
                gather_bound_quantized_linear(gpu, input, expert_indices, weights, sorted_indices)
            }
        }
    }
}

impl MoeExecution {
    pub(crate) fn require(&self, name: &'static str) -> Result<&Array, EngineError> {
        if name == "moe_output" {
            return Ok(&self.output);
        }
        self.trace
            .as_ref()
            .and_then(|trace| trace.get(name))
            .ok_or_else(|| EngineError::Unsupported(format!("internal MoE trace omitted {name}")))
    }
}

/// Direct comparison between Qwen3.5 `MoE` decoder layer 0 executed from Rust
/// and the official Python/MLX call.
#[derive(Clone, Debug, Serialize)]
pub struct DecoderLayerParity {
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

/// Executes Qwen3.5 `MoE` decoder layer 0 through Rust and MLX C, including both
/// residual paths, RMS normalization, GDN state carry, top-k routing, routed
/// experts, and the gated shared expert.
///
/// # Errors
///
/// Returns [`EngineError`] when the model, oracle, weights, shapes, dtypes, or
/// an MLX operation do not match the admitted execution plan.
#[allow(clippy::too_many_lines)]
pub fn run_decoder_layer_parity(
    model_directory: &Path,
    fixture_path: &Path,
    manifest_path: &Path,
) -> Result<DecoderLayerParity, EngineError> {
    let plan = ModelPlan::from_directory(model_directory)?;
    let manifest = load_manifest(manifest_path)?;
    validate_manifest(&plan, model_directory, fixture_path, &manifest)?;

    let fixture = SafeTensors::load(fixture_path).map_err(EngineError::Mlx)?;
    if fixture.len() != manifest.fixture_tensor_count {
        return Err(EngineError::Unsupported(format!(
            "decoder fixture tensor count drift: manifest {}, payload {}",
            manifest.fixture_tensor_count,
            fixture.len()
        )));
    }

    let layer_prefix = format!("language_model.model.layers.{}", manifest.layer_index);
    let gdn_prefix = format!("{layer_prefix}.linear_attn");
    let mlp_prefix = format!("{layer_prefix}.mlp");
    let weights = load_weight_shard(model_directory, &gdn_prefix)?;
    let gpu = Gpu::new();
    let moe_kernel = MoeKernel::new(&plan)?;

    let input = require_tensor(&fixture, "continuation_input")?;
    let input_shape = input.shape();
    let [batch_size, sequence_length, hidden_size] = <[usize; 3]>::try_from(input_shape.clone())
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

    let initial_conv_state = require_tensor(&fixture, "initial_conv_state")?;
    let initial_recurrent_state = require_tensor(&fixture, "initial_recurrent_state")?;
    let gdn = execute_gdn_layer(
        &gpu,
        &normalized_input,
        initial_conv_state,
        initial_recurrent_state,
        &weights,
        &gdn_prefix,
        &plan,
    )?;
    let post_attention_hidden = gpu
        .add(input, gdn.require("output")?)
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
    let expected_conv_state = require_tensor(&fixture, "expected_conv_state")?;
    let expected_recurrent_state = require_tensor(&fixture, "expected_recurrent_state")?;
    let output_difference = gpu
        .max_abs_difference(&output, expected_output)
        .map_err(EngineError::Mlx)?;
    let conv_state_difference = gpu
        .max_abs_difference(gdn.require("conv_state")?, expected_conv_state)
        .map_err(EngineError::Mlx)?;
    let recurrent_state_difference = gpu
        .max_abs_difference(gdn.require("recurrent_state")?, expected_recurrent_state)
        .map_err(EngineError::Mlx)?;

    let mut trace_differences = BTreeMap::new();
    for (name, actual) in [
        ("attention_output", gdn.require("output")?),
        ("decoder_output", &output),
        ("normalized_hidden", &normalized_hidden),
        ("normalized_input", &normalized_input),
        ("post_attention_hidden", &post_attention_hidden),
    ] {
        compare_trace(&gpu, &fixture, name, actual, &mut trace_differences)?;
    }
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
        compare_named_trace(
            &gpu,
            &fixture,
            &format!("gdn.{name}"),
            gdn.require(name)?,
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
        && conv_state_difference == 0.0
        && recurrent_state_difference == 0.0
        && trace_max_absolute_difference == 0.0;

    Ok(DecoderLayerParity {
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

#[allow(clippy::too_many_lines)]
pub(crate) fn execute_sparse_moe(
    gpu: &Gpu,
    input: &Array,
    weights: &dyn TensorLookup,
    prefix: &str,
    plan: &ModelPlan,
    kernel: &MoeKernel,
) -> Result<MoeExecution, EngineError> {
    execute_sparse_moe_impl(
        gpu,
        input,
        &MoeWeightSource::Named { weights, prefix },
        plan,
        kernel,
        true,
    )
}

pub(crate) fn execute_sparse_moe_output(
    gpu: &Gpu,
    input: &Array,
    weights: &dyn TensorLookup,
    prefix: &str,
    plan: &ModelPlan,
    kernel: &MoeKernel,
) -> Result<Array, EngineError> {
    Ok(execute_sparse_moe_impl(
        gpu,
        input,
        &MoeWeightSource::Named { weights, prefix },
        plan,
        kernel,
        false,
    )?
    .output)
}

pub(crate) fn execute_sparse_moe_output_with_bound_weights(
    gpu: &Gpu,
    input: &Array,
    weights: &BoundMoeWeights,
    plan: &ModelPlan,
    kernel: &MoeKernel,
) -> Result<Array, EngineError> {
    Ok(execute_sparse_moe_impl(
        gpu,
        input,
        &MoeWeightSource::Bound(weights),
        plan,
        kernel,
        false,
    )?
    .output)
}

fn execute_shared_expert(
    gpu: &Gpu,
    input: &Array,
    weights: &MoeWeightSource<'_>,
    plan: &ModelPlan,
    batch_size: usize,
    sequence_length: usize,
    capture_trace: bool,
) -> Result<SharedExpertExecution, EngineError> {
    let gate_projection = weights.apply_linear(gpu, input, MoeLinear::SharedGate, plan)?;
    let up_projection = weights.apply_linear(gpu, input, MoeLinear::SharedUp, plan)?;
    for (name, value) in [
        ("shared gate projection", &gate_projection),
        ("shared up projection", &up_projection),
    ] {
        validate_array(
            value,
            &[
                batch_size,
                sequence_length,
                plan.shared_expert_intermediate_size,
            ],
            input.dtype(),
            name,
        )?;
    }
    let activated = gpu
        .swiglu(&gate_projection, &up_projection)
        .map_err(EngineError::Mlx)?;
    let expert_output = weights.apply_linear(gpu, &activated, MoeLinear::SharedDown, plan)?;
    validate_array(
        &expert_output,
        &[batch_size, sequence_length, plan.hidden_size],
        input.dtype(),
        "shared expert output",
    )?;
    let gate_logits = weights.apply_linear(gpu, input, MoeLinear::SharedExpertGate, plan)?;
    validate_array(
        &gate_logits,
        &[batch_size, sequence_length, 1],
        input.dtype(),
        "shared expert gate logits",
    )?;
    let gate = gpu.sigmoid(&gate_logits).map_err(EngineError::Mlx)?;
    let output = gpu
        .multiply(&gate, &expert_output)
        .map_err(EngineError::Mlx)?;
    let trace = if capture_trace {
        Some(SharedExpertTrace {
            gate_projection,
            up_projection,
            activated,
            expert_output,
            gate_logits,
            gate,
            output: output.try_clone().map_err(EngineError::Mlx)?,
        })
    } else {
        None
    };
    Ok(SharedExpertExecution { output, trace })
}

#[allow(clippy::too_many_lines)]
fn execute_sparse_moe_impl(
    gpu: &Gpu,
    input: &Array,
    weights: &MoeWeightSource<'_>,
    plan: &ModelPlan,
    kernel: &MoeKernel,
    capture_trace: bool,
) -> Result<MoeExecution, EngineError> {
    let shape = input.shape();
    let [batch_size, sequence_length, hidden_size] = <[usize; 3]>::try_from(shape.clone())
        .map_err(|shape| {
            EngineError::Unsupported(format!("MoE input must be rank 3, observed {shape:?}"))
        })?;
    if hidden_size != plan.hidden_size {
        return Err(EngineError::Unsupported(format!(
            "MoE input hidden size mismatch: expected {}, observed {hidden_size}",
            plan.hidden_size
        )));
    }
    #[cfg(feature = "moe-performance-diagnostics")]
    let performance_mode = if capture_trace {
        MoePerformanceMode::Full
    } else {
        kernel.performance_mode
    };
    #[cfg(feature = "moe-performance-diagnostics")]
    if performance_mode == MoePerformanceMode::None {
        return Ok(MoeExecution {
            output: gpu.zeros_like(input).map_err(EngineError::Mlx)?,
            trace: None,
        });
    }
    #[cfg(feature = "moe-performance-diagnostics")]
    if performance_mode == MoePerformanceMode::SharedOnly {
        return Ok(MoeExecution {
            output: execute_shared_expert(
                gpu,
                input,
                weights,
                plan,
                batch_size,
                sequence_length,
                false,
            )?
            .output,
            trace: None,
        });
    }
    let router_logits = weights.apply_linear(gpu, input, MoeLinear::Router, plan)?;
    validate_array(
        &router_logits,
        &[batch_size, sequence_length, plan.expert_count],
        input.dtype(),
        "router logits",
    )?;
    let use_fused_decode_route =
        !capture_trace && kernel.supports_decode_route(input, batch_size, sequence_length, plan);
    let (expert_indices, expert_scores, router_probabilities) = if use_fused_decode_route {
        let (expert_indices, expert_scores) = kernel.route_decode(gpu, &router_logits)?;
        (expert_indices, expert_scores, None)
    } else {
        let router_probabilities = gpu
            .softmax_axis(&router_logits, -1, true)
            .map_err(EngineError::Mlx)?;
        let partitioned = gpu
            .argpartition_axis(
                &router_probabilities,
                -dimension_i32(plan.experts_per_token, "experts per token")?,
                -1,
            )
            .map_err(EngineError::Mlx)?;
        let expert_indices =
            last_axis_tail(gpu, &partitioned, plan.experts_per_token, "expert indices")?;
        let selected = gpu
            .take_along_axis(&router_probabilities, &expert_indices, -1)
            .map_err(EngineError::Mlx)?;
        let expert_scores = if plan.norm_topk_prob {
            let total = gpu
                .sum_axis(&selected, -1, true)
                .map_err(EngineError::Mlx)?;
            gpu.divide(&selected, &total).map_err(EngineError::Mlx)?
        } else {
            selected
        };
        (expert_indices, expert_scores, Some(router_probabilities))
    };
    validate_array(
        &expert_indices,
        &[batch_size, sequence_length, plan.experts_per_token],
        DType::Uint32,
        "expert indices",
    )?;
    validate_array(
        &expert_scores,
        &[batch_size, sequence_length, plan.experts_per_token],
        input.dtype(),
        "expert scores",
    )?;
    #[cfg(feature = "moe-performance-diagnostics")]
    if performance_mode == MoePerformanceMode::RouterOnly {
        let score_total = gpu
            .sum_axis(&expert_scores, -1, true)
            .map_err(EngineError::Mlx)?;
        return Ok(MoeExecution {
            output: gpu
                .multiply(input, &score_total)
                .map_err(EngineError::Mlx)?,
            trace: None,
        });
    }
    let switch_inputs = gpu
        .reshape(input, &[batch_size, sequence_length, 1, 1, hidden_size])
        .map_err(EngineError::Mlx)?;
    let selection_count = batch_size
        .checked_mul(sequence_length)
        .and_then(|value| value.checked_mul(plan.experts_per_token))
        .ok_or_else(|| EngineError::Unsupported("expert-selection count overflow".into()))?;
    let use_sorted_dispatch = selection_count >= 64;
    let sorted_dispatch = if use_sorted_dispatch {
        let flattened_indices = gpu
            .reshape(&expert_indices, &[selection_count])
            .map_err(EngineError::Mlx)?;
        let sort_order = gpu.argsort(&flattened_indices).map_err(EngineError::Mlx)?;
        let inverse_order = gpu.argsort(&sort_order).map_err(EngineError::Mlx)?;
        let flattened_inputs = gpu
            .reshape(
                &switch_inputs,
                &[batch_size * sequence_length, 1, hidden_size],
            )
            .map_err(EngineError::Mlx)?;
        let experts_per_token =
            gpu.scalar_i32(dimension_i32(plan.experts_per_token, "experts per token")?);
        let input_indices = gpu
            .floor_divide(&sort_order, &experts_per_token)
            .map_err(EngineError::Mlx)?;
        let sorted_inputs = gpu
            .take_axis(&flattened_inputs, &input_indices, 0)
            .map_err(EngineError::Mlx)?;
        let sorted_indices = gpu
            .take(&flattened_indices, &sort_order)
            .map_err(EngineError::Mlx)?;
        Some((sort_order, inverse_order, sorted_inputs, sorted_indices))
    } else {
        None
    };
    let (gather_inputs, gather_indices) = sorted_dispatch.as_ref().map_or(
        (&switch_inputs, &expert_indices),
        |(_, _, sorted_inputs, sorted_indices)| (sorted_inputs, sorted_indices),
    );
    let fused_gate_up = if kernel.supports_expert_gate_up_decode(
        input,
        batch_size,
        sequence_length,
        plan,
        capture_trace,
    ) {
        match weights {
            MoeWeightSource::Bound(bound_weights) => {
                kernel.expert_gate_up_decode(gpu, gather_inputs, gather_indices, bound_weights)?
            }
            MoeWeightSource::Named { .. } => None,
        }
    } else {
        None
    };
    let (expert_gate, expert_up) = if let Some(projections) = fused_gate_up {
        projections
    } else {
        let expert_up = weights.apply_expert_linear(
            gpu,
            gather_inputs,
            gather_indices,
            MoeExpertLinear::Up,
            plan.expert_count,
            plan.moe_intermediate_size,
            plan.hidden_size,
            plan,
            use_sorted_dispatch,
        )?;
        let expert_gate = weights.apply_expert_linear(
            gpu,
            gather_inputs,
            gather_indices,
            MoeExpertLinear::Gate,
            plan.expert_count,
            plan.moe_intermediate_size,
            plan.hidden_size,
            plan,
            use_sorted_dispatch,
        )?;
        (expert_gate, expert_up)
    };
    let expert_projection_shape = if use_sorted_dispatch {
        vec![selection_count, 1, plan.moe_intermediate_size]
    } else {
        vec![
            batch_size,
            sequence_length,
            plan.experts_per_token,
            1,
            plan.moe_intermediate_size,
        ]
    };
    validate_array(
        &expert_up,
        &expert_projection_shape,
        input.dtype(),
        "expert up projection",
    )?;
    validate_array(
        &expert_gate,
        &expert_projection_shape,
        input.dtype(),
        "expert gate projection",
    )?;
    let expert_activated = gpu
        .swiglu(&expert_gate, &expert_up)
        .map_err(EngineError::Mlx)?;
    let fused_routed_output = if kernel.supports_routed_down_reduce_decode(
        input,
        batch_size,
        sequence_length,
        plan,
        capture_trace,
        use_sorted_dispatch,
    ) {
        match weights {
            MoeWeightSource::Bound(bound_weights) => kernel.routed_down_reduce_decode(
                gpu,
                &expert_activated,
                gather_indices,
                &expert_scores,
                bound_weights,
            )?,
            MoeWeightSource::Named { .. } => None,
        }
    } else {
        None
    };
    let (expert_sorted_outputs, expert_outputs, routed_output) =
        if let Some(routed_output) = fused_routed_output {
            validate_array(
                &routed_output,
                &[batch_size, sequence_length, plan.hidden_size],
                input.dtype(),
                "fused routed output",
            )?;
            (None, None, routed_output)
        } else {
            let expert_sorted_outputs = weights.apply_expert_linear(
                gpu,
                &expert_activated,
                gather_indices,
                MoeExpertLinear::Down,
                plan.expert_count,
                plan.hidden_size,
                plan.moe_intermediate_size,
                plan,
                use_sorted_dispatch,
            )?;
            let ordered_expert_outputs = if let Some((_, inverse_order, _, _)) = &sorted_dispatch {
                gpu.take_axis(&expert_sorted_outputs, inverse_order, 0)
                    .map_err(EngineError::Mlx)?
            } else {
                expert_sorted_outputs
                    .try_clone()
                    .map_err(EngineError::Mlx)?
            };
            let expert_outputs = gpu
                .reshape(
                    &ordered_expert_outputs,
                    &[
                        batch_size,
                        sequence_length,
                        plan.experts_per_token,
                        plan.hidden_size,
                    ],
                )
                .map_err(EngineError::Mlx)?;
            let expanded_scores = gpu
                .reshape(
                    &expert_scores,
                    &[batch_size, sequence_length, plan.experts_per_token, 1],
                )
                .map_err(EngineError::Mlx)?;
            let routed_output = gpu
                .multiply(&expert_outputs, &expanded_scores)
                .and_then(|scaled| gpu.sum_axis(&scaled, -2, false))
                .map_err(EngineError::Mlx)?;
            (
                Some(expert_sorted_outputs),
                Some(expert_outputs),
                routed_output,
            )
        };
    #[cfg(feature = "moe-performance-diagnostics")]
    if performance_mode == MoePerformanceMode::RoutedOnly {
        return Ok(MoeExecution {
            output: routed_output,
            trace: None,
        });
    }
    let shared = execute_shared_expert(
        gpu,
        input,
        weights,
        plan,
        batch_size,
        sequence_length,
        capture_trace,
    )?;
    let moe_output = gpu
        .add(&routed_output, &shared.output)
        .map_err(EngineError::Mlx)?;

    let trace = if capture_trace {
        let expert_outputs = expert_outputs.ok_or_else(|| {
            EngineError::Unsupported("captured MoE trace omitted expert outputs".into())
        })?;
        let expert_sorted_outputs = expert_sorted_outputs.ok_or_else(|| {
            EngineError::Unsupported("captured MoE trace omitted sorted expert outputs".into())
        })?;
        let router_probabilities = router_probabilities.ok_or_else(|| {
            EngineError::Unsupported("captured MoE trace omitted router probabilities".into())
        })?;
        let shared = shared.trace.ok_or_else(|| {
            EngineError::Unsupported("captured MoE trace omitted shared expert trace".into())
        })?;
        let mut values = BTreeMap::from([
            ("expert_activated", expert_activated),
            ("expert_gate", expert_gate),
            ("expert_indices", expert_indices),
            ("expert_outputs", expert_outputs),
            ("expert_scores", expert_scores),
            ("expert_sorted_outputs", expert_sorted_outputs),
            ("expert_up", expert_up),
            ("routed_output", routed_output),
            ("router_logits", router_logits),
            ("router_probabilities", router_probabilities),
            ("shared_activated", shared.activated),
            ("shared_expert_output", shared.expert_output),
            ("shared_gate", shared.gate),
            ("shared_gate_logits", shared.gate_logits),
            ("shared_gate_projection", shared.gate_projection),
            ("shared_output", shared.output),
            ("shared_up_projection", shared.up_projection),
        ]);
        if let Some((sort_order, inverse_order, sorted_inputs, sorted_indices)) = sorted_dispatch {
            values.insert("expert_inverse_order", inverse_order);
            values.insert("expert_sort_order", sort_order);
            values.insert("expert_sorted_indices", sorted_indices);
            values.insert("expert_sorted_inputs", sorted_inputs);
        }
        Some(values)
    } else {
        None
    };
    Ok(MoeExecution {
        output: moe_output,
        trace,
    })
}

#[allow(clippy::too_many_arguments)]
fn gather_quantized_linear(
    gpu: &Gpu,
    input: &Array,
    expert_indices: &Array,
    weights: &dyn TensorLookup,
    prefix: &str,
    expert_count: usize,
    output_dimension: usize,
    input_dimension: usize,
    plan: &ModelPlan,
    sorted_indices: bool,
) -> Result<Array, EngineError> {
    let tensors = weights.quantized_tensors(prefix).ok_or_else(|| {
        EngineError::Unsupported(format!("missing quantized tensor set {prefix}"))
    })?;
    let weight = tensors.weight;
    let scales = tensors.scales;
    let biases = tensors.biases;
    let packed_input_dimension = input_dimension
        .checked_mul(plan.quantization_bits)
        .filter(|value| value.is_multiple_of(32))
        .map(|value| value / 32)
        .ok_or_else(|| {
            EngineError::Unsupported(format!("{prefix} has an invalid packed input dimension"))
        })?;
    validate_array(
        weight,
        &[expert_count, output_dimension, packed_input_dimension],
        DType::Uint32,
        prefix,
    )?;
    if plan.quantization_group_size == 0
        || !input_dimension.is_multiple_of(plan.quantization_group_size)
    {
        return Err(EngineError::Unsupported(format!(
            "{prefix} input dimension {input_dimension} is not divisible by quantization group size {}",
            plan.quantization_group_size
        )));
    }
    let scale_shape = [
        expert_count,
        output_dimension,
        input_dimension / plan.quantization_group_size,
    ];
    validate_array(scales, &scale_shape, input.dtype(), prefix)?;
    validate_array(biases, &scale_shape, input.dtype(), prefix)?;

    gpu.gather_quantized_matmul(
        input,
        weight,
        scales,
        GatherQuantizedMatmulConfig {
            biases: Some(biases),
            rhs_indices: expert_indices,
            transpose: true,
            group_size: dimension_i32(
                plan.quantization_group_size,
                "expert quantization group size",
            )?,
            bits: dimension_i32(plan.quantization_bits, "expert quantization bits")?,
            mode: &plan.quantization_mode,
            sorted_indices,
        },
    )
    .map_err(EngineError::Mlx)
}

fn gather_bound_quantized_linear(
    gpu: &Gpu,
    input: &Array,
    expert_indices: &Array,
    weights: &BoundQuantizedWeights,
    sorted_indices: bool,
) -> Result<Array, EngineError> {
    gpu.gather_quantized_matmul(
        input,
        &weights.weight,
        &weights.scales,
        GatherQuantizedMatmulConfig {
            biases: Some(&weights.biases),
            rhs_indices: expert_indices,
            transpose: true,
            group_size: weights.group_size,
            bits: weights.bits,
            mode: &weights.mode,
            sorted_indices,
        },
    )
    .map_err(EngineError::Mlx)
}

fn last_axis_tail(
    gpu: &Gpu,
    array: &Array,
    count: usize,
    name: &str,
) -> Result<Array, EngineError> {
    let shape = array.shape();
    let [batch_size, sequence_length, feature_count] = <[usize; 3]>::try_from(shape.clone())
        .map_err(|shape| {
            EngineError::Unsupported(format!("{name} source must be rank 3: {shape:?}"))
        })?;
    if count == 0 || count > feature_count {
        return Err(EngineError::Unsupported(format!(
            "{name} cannot take {count} features from {shape:?}"
        )));
    }
    gpu.slice(
        array,
        &[0, 0, dimension_i32(feature_count - count, "tail start")?],
        &[
            dimension_i32(batch_size, name)?,
            dimension_i32(sequence_length, name)?,
            dimension_i32(feature_count, name)?,
        ],
        &[1, 1, 1],
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
            "decoder fixture schema must be 1, observed {}",
            manifest.schema_version
        )));
    }
    if plan.architecture != "qwen3_5_moe" || manifest.model_type != plan.architecture {
        return Err(EngineError::Unsupported(format!(
            "decoder fixture requires qwen3_5_moe, observed fixture {} and model {}",
            manifest.model_type, plan.architecture
        )));
    }
    if manifest.layer_index != 0 {
        return Err(EngineError::Unsupported(format!(
            "the current decoder milestone admits only layer 0, observed {}",
            manifest.layer_index
        )));
    }
    if (manifest.layer_index + 1).is_multiple_of(plan.full_attention_interval) {
        return Err(EngineError::Unsupported(
            "decoder fixture layer must use GDN attention".into(),
        ));
    }
    if manifest.prefix_length == 0 || manifest.continuation_length == 0 {
        return Err(EngineError::Unsupported(
            "decoder fixture segments must both be non-empty".into(),
        ));
    }
    let selected_experts = manifest
        .continuation_length
        .checked_mul(plan.experts_per_token)
        .ok_or_else(|| {
            EngineError::Unsupported("decoder fixture expert-selection count overflow".into())
        })?;
    if manifest.sorted_expert_path != (selected_experts >= 64) {
        return Err(EngineError::Unsupported(format!(
            "decoder fixture sorted-expert path drift: fixture {}, selection count {selected_experts}",
            manifest.sorted_expert_path
        )));
    }
    if manifest.input_dtype != "mlx.core.bfloat16" {
        return Err(EngineError::Unsupported(format!(
            "decoder fixture must use bfloat16 input, observed {}",
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
        expert_count: plan.expert_count,
        experts_per_token: plan.experts_per_token,
        moe_intermediate_size: plan.moe_intermediate_size,
        shared_expert_intermediate_size: plan.shared_expert_intermediate_size,
    };
    if manifest.dimensions != dimensions {
        return Err(EngineError::Unsupported(format!(
            "decoder fixture dimension drift: fixture {:?}, model {:?}",
            manifest.dimensions, dimensions
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
            "decoder fixture quantization drift: fixture {:?}, model {:?}",
            manifest.quantization, quantization
        )));
    }
    if manifest.norm_topk_prob != plan.norm_topk_prob {
        return Err(EngineError::Unsupported(format!(
            "decoder fixture norm_topk_prob drift: fixture {}, model {}",
            manifest.norm_topk_prob, plan.norm_topk_prob
        )));
    }
    let config_path = model_directory.join("config.json");
    let config_digest = sha256_file(&config_path)?;
    if config_digest != manifest.config_sha256 {
        return Err(EngineError::Unsupported(format!(
            "decoder fixture config digest drift: fixture {}, model {}",
            manifest.config_sha256, config_digest
        )));
    }
    let fixture_digest = sha256_file(fixture_path)?;
    if fixture_digest != manifest.fixture_sha256 {
        return Err(EngineError::Unsupported(format!(
            "decoder fixture payload digest drift: manifest {}, payload {}",
            manifest.fixture_sha256, fixture_digest
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_decode_route(gpu: &Gpu, logits: &Array) -> (Array, Array) {
        let probabilities = gpu
            .softmax_axis(logits, -1, true)
            .expect("reference precise softmax");
        let partitioned = gpu
            .argpartition_axis(&probabilities, -8, -1)
            .expect("reference argpartition");
        let indices =
            last_axis_tail(gpu, &partitioned, 8, "reference indices").expect("reference top-8");
        let selected = gpu
            .take_along_axis(&probabilities, &indices, -1)
            .expect("reference selected probabilities");
        let total = gpu
            .sum_axis(&selected, -1, true)
            .expect("reference selected sum");
        let scores = gpu
            .divide(&selected, &total)
            .expect("reference normalized scores");
        (indices, scores)
    }

    fn router_logit_cases() -> Vec<Vec<f32>> {
        let mut cases = vec![vec![0.0; 256]];
        cases.push(
            (0..256)
                .map(|index| {
                    let bucket = u8::try_from(index % 17).expect("tie bucket");
                    f32::from(bucket) * 0.000_5
                })
                .collect(),
        );

        for case_index in 0..20 {
            let mut state = u32::try_from(case_index + 1).expect("case seed");
            let mut values = Vec::with_capacity(256);
            for _ in 0..256 {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let bytes = state.to_le_bytes();
                let sample = u16::from_le_bytes([bytes[2], bytes[3]]) % 4096;
                values.push(f32::from(sample) / 256.0 - 8.0);
            }
            cases.push(values);
        }
        cases
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn fused_decode_router_matches_mlx_bf16_routing_exactly() {
        let gpu = Gpu::new();
        let kernel = MoeKernel::for_dimensions(256, 8).expect("decode router kernel");

        for (case_index, values) in router_logit_cases().into_iter().enumerate() {
            let logits = Array::from_f32_slice(&values, &[1, 1, 256]).expect("float32 logits");
            let logits = gpu.astype(&logits, DType::BFloat16).expect("BF16 logits");
            let (expected_indices, expected_scores) = reference_decode_route(&gpu, &logits);
            let (actual_indices, actual_scores) =
                kernel.route_decode(&gpu, &logits).expect("fused route");

            let index_difference = gpu
                .max_abs_difference(&actual_indices, &expected_indices)
                .expect("index difference");
            let score_difference = gpu
                .max_abs_difference(&actual_scores, &expected_scores)
                .expect("score difference");
            assert_eq!(
                index_difference, 0.0,
                "router case {case_index} selected different experts"
            );
            assert_eq!(
                score_difference, 0.0,
                "router case {case_index} produced different scores"
            );
        }
    }

    #[test]
    #[allow(clippy::cast_possible_wrap, clippy::float_cmp, clippy::too_many_lines)]
    fn fused_decode_expert_gate_up_matches_mlx_qmv_exactly() {
        const INPUT_DIMENSION: usize = 2048;
        const INTERMEDIATE_DIMENSION: usize = 512;
        const EXPERT_COUNT: usize = 16;
        const EXPERTS_PER_TOKEN: usize = 8;
        const PACKED_INPUT_DIMENSION: usize = INPUT_DIMENSION / 8;
        const GROUPS_PER_ROW: usize = INPUT_DIMENSION / 64;

        let gpu = Gpu::new();
        let kernel = create_decode_expert_gate_up_kernel(
            INPUT_DIMENSION,
            INTERMEDIATE_DIMENSION,
            EXPERT_COUNT,
            EXPERTS_PER_TOKEN,
        )
        .expect("decode expert gate/up kernel");

        let weight_count = EXPERT_COUNT * INTERMEDIATE_DIMENSION * PACKED_INPUT_DIMENSION;
        let mut gate_state = 0x1234_5678_u32;
        let gate_words = (0..weight_count)
            .map(|_| {
                gate_state = gate_state
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223);
                (gate_state & 0x7fff_ffff) as i32
            })
            .collect::<Vec<_>>();
        let mut up_state = 0x9abc_def0_u32;
        let up_words = (0..weight_count)
            .map(|_| {
                up_state = up_state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (up_state & 0x7fff_ffff) as i32
            })
            .collect::<Vec<_>>();
        let weight_shape = [EXPERT_COUNT, INTERMEDIATE_DIMENSION, PACKED_INPUT_DIMENSION];
        let gate_weight = Array::from_i32_slice(&gate_words, &weight_shape)
            .and_then(|value| gpu.astype(&value, DType::Uint32))
            .expect("gate weights");
        let up_weight = Array::from_i32_slice(&up_words, &weight_shape)
            .and_then(|value| gpu.astype(&value, DType::Uint32))
            .expect("up weights");

        let group_value_count = EXPERT_COUNT * INTERMEDIATE_DIMENSION * GROUPS_PER_ROW;
        let scale_values = (0..group_value_count)
            .map(|index| f32::from(u8::try_from(index % 17 + 1).expect("scale")) / 128.0)
            .collect::<Vec<_>>();
        let bias_values = (0..group_value_count)
            .map(|index| {
                let bucket = i8::try_from(index % 11).expect("bias bucket") - 5;
                f32::from(bucket) / 64.0
            })
            .collect::<Vec<_>>();
        let group_shape = [EXPERT_COUNT, INTERMEDIATE_DIMENSION, GROUPS_PER_ROW];
        let gate_scales = Array::from_f32_slice(&scale_values, &group_shape)
            .and_then(|value| gpu.astype(&value, DType::BFloat16))
            .expect("gate scales");
        let gate_biases = Array::from_f32_slice(&bias_values, &group_shape)
            .and_then(|value| gpu.astype(&value, DType::BFloat16))
            .expect("gate biases");
        let up_scales = Array::from_f32_slice(
            &scale_values.iter().rev().copied().collect::<Vec<_>>(),
            &group_shape,
        )
        .and_then(|value| gpu.astype(&value, DType::BFloat16))
        .expect("up scales");
        let up_biases = Array::from_f32_slice(
            &bias_values.iter().rev().copied().collect::<Vec<_>>(),
            &group_shape,
        )
        .and_then(|value| gpu.astype(&value, DType::BFloat16))
        .expect("up biases");

        let mut input_state = 0x3141_5926_u32;
        let input_values = (0..INPUT_DIMENSION)
            .map(|_| {
                input_state = input_state
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223);
                let bytes = input_state.to_le_bytes();
                let sample = u16::from_le_bytes([bytes[2], bytes[3]]);
                f32::from(sample) / 257.0 - 128.0
            })
            .collect::<Vec<_>>();
        let input = Array::from_f32_slice(&input_values, &[1, 1, 1, 1, INPUT_DIMENSION])
            .and_then(|value| gpu.astype(&value, DType::BFloat16))
            .expect("expert input");
        let expert_indices =
            Array::from_i32_slice(&[15, 0, 7, 3, 11, 5, 14, 1], &[1, 1, EXPERTS_PER_TOKEN])
                .and_then(|value| gpu.astype(&value, DType::Uint32))
                .expect("expert indices");

        let expected_gate = gpu
            .gather_quantized_matmul(
                &input,
                &gate_weight,
                &gate_scales,
                GatherQuantizedMatmulConfig {
                    biases: Some(&gate_biases),
                    rhs_indices: &expert_indices,
                    transpose: true,
                    group_size: 64,
                    bits: 4,
                    mode: "affine",
                    sorted_indices: false,
                },
            )
            .expect("reference gate projection");
        let expected_up = gpu
            .gather_quantized_matmul(
                &input,
                &up_weight,
                &up_scales,
                GatherQuantizedMatmulConfig {
                    biases: Some(&up_biases),
                    rhs_indices: &expert_indices,
                    transpose: true,
                    group_size: 64,
                    bits: 4,
                    mode: "affine",
                    sorted_indices: false,
                },
            )
            .expect("reference up projection");
        let mut outputs = kernel
            .kernel
            .apply_prepared(
                &gpu,
                &[
                    &gate_weight,
                    &gate_scales,
                    &gate_biases,
                    &up_weight,
                    &up_scales,
                    &up_biases,
                    &input,
                    &expert_indices,
                ],
                &kernel.dispatch,
            )
            .expect("fused expert projections");
        let actual_up = outputs.pop().expect("fused up projection");
        let actual_gate = outputs.pop().expect("fused gate projection");

        assert_eq!(actual_gate.shape(), expected_gate.shape());
        assert_eq!(actual_up.shape(), expected_up.shape());
        assert_eq!(
            gpu.max_abs_difference(&actual_gate, &expected_gate)
                .expect("gate difference"),
            0.0
        );
        assert_eq!(
            gpu.max_abs_difference(&actual_up, &expected_up)
                .expect("up difference"),
            0.0
        );
    }

    #[test]
    #[allow(clippy::cast_possible_wrap, clippy::float_cmp, clippy::too_many_lines)]
    fn fused_decode_routed_down_reduce_matches_mlx_exactly() {
        const INPUT_DIMENSION: usize = 512;
        const OUTPUT_DIMENSION: usize = 2048;
        const EXPERT_COUNT: usize = 16;
        const SELECTED_COUNT: usize = 8;
        const PACKED_INPUT_DIMENSION: usize = INPUT_DIMENSION / 8;
        const GROUPS_PER_ROW: usize = INPUT_DIMENSION / 64;

        let gpu = Gpu::new();
        let kernel = create_decode_routed_down_reduce_kernel(
            INPUT_DIMENSION,
            OUTPUT_DIMENSION,
            EXPERT_COUNT,
            SELECTED_COUNT,
        )
        .expect("decode routed down/reduce kernel");

        let weight_count = EXPERT_COUNT * OUTPUT_DIMENSION * PACKED_INPUT_DIMENSION;
        let mut weight_state = 0x6a09_e667_u32;
        let weight_words = (0..weight_count)
            .map(|_| {
                weight_state = weight_state
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223);
                (weight_state & 0x7fff_ffff) as i32
            })
            .collect::<Vec<_>>();
        let weight_shape = [EXPERT_COUNT, OUTPUT_DIMENSION, PACKED_INPUT_DIMENSION];
        let down_weight = Array::from_i32_slice(&weight_words, &weight_shape)
            .and_then(|value| gpu.astype(&value, DType::Uint32))
            .expect("down weights");

        let group_value_count = EXPERT_COUNT * OUTPUT_DIMENSION * GROUPS_PER_ROW;
        let scale_values = (0..group_value_count)
            .map(|index| f32::from(u8::try_from(index % 19 + 1).expect("scale")) / 256.0)
            .collect::<Vec<_>>();
        let bias_values = (0..group_value_count)
            .map(|index| {
                let bucket = i8::try_from(index % 13).expect("bias bucket") - 6;
                f32::from(bucket) / 128.0
            })
            .collect::<Vec<_>>();
        let group_shape = [EXPERT_COUNT, OUTPUT_DIMENSION, GROUPS_PER_ROW];
        let down_scales = Array::from_f32_slice(&scale_values, &group_shape)
            .and_then(|value| gpu.astype(&value, DType::BFloat16))
            .expect("down scales");
        let down_biases = Array::from_f32_slice(&bias_values, &group_shape)
            .and_then(|value| gpu.astype(&value, DType::BFloat16))
            .expect("down biases");

        let activated_value_count = SELECTED_COUNT * INPUT_DIMENSION;
        let mut activated_state = 0xbb67_ae85_u32;
        let activated_values = (0..activated_value_count)
            .map(|_| {
                activated_state = activated_state
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223);
                let bytes = activated_state.to_le_bytes();
                let sample = u16::from_le_bytes([bytes[2], bytes[3]]);
                f32::from(sample) / 8192.0 - 4.0
            })
            .collect::<Vec<_>>();
        let activated = Array::from_f32_slice(
            &activated_values,
            &[1, 1, SELECTED_COUNT, 1, INPUT_DIMENSION],
        )
        .and_then(|value| gpu.astype(&value, DType::BFloat16))
        .expect("expert activations");
        let expert_indices =
            Array::from_i32_slice(&[15, 0, 7, 3, 11, 5, 14, 1], &[1, 1, SELECTED_COUNT])
                .and_then(|value| gpu.astype(&value, DType::Uint32))
                .expect("expert indices");
        let expert_scores = Array::from_f32_slice(
            &[0.031, 0.067, 0.109, 0.143, 0.157, 0.191, 0.127, 0.175],
            &[1, 1, SELECTED_COUNT],
        )
        .and_then(|value| gpu.astype(&value, DType::BFloat16))
        .expect("expert scores");

        let projected = gpu
            .gather_quantized_matmul(
                &activated,
                &down_weight,
                &down_scales,
                GatherQuantizedMatmulConfig {
                    biases: Some(&down_biases),
                    rhs_indices: &expert_indices,
                    transpose: true,
                    group_size: 64,
                    bits: 4,
                    mode: "affine",
                    sorted_indices: false,
                },
            )
            .expect("reference down projection");
        let projected = gpu
            .reshape(&projected, &[1, 1, SELECTED_COUNT, OUTPUT_DIMENSION])
            .expect("reference output shape");
        let expanded_scores = gpu
            .reshape(&expert_scores, &[1, 1, SELECTED_COUNT, 1])
            .expect("reference score shape");
        let expected = gpu
            .multiply(&projected, &expanded_scores)
            .and_then(|scaled| gpu.sum_axis(&scaled, -2, false))
            .expect("reference weighted reduction");

        let mut outputs = kernel
            .kernel
            .apply_prepared(
                &gpu,
                &[
                    &down_weight,
                    &down_scales,
                    &down_biases,
                    &activated,
                    &expert_indices,
                    &expert_scores,
                ],
                &kernel.dispatch,
            )
            .expect("fused routed down/reduce");
        let actual = outputs.pop().expect("fused routed output");

        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(
            gpu.max_abs_difference(&actual, &expected)
                .expect("routed output difference"),
            0.0
        );
    }
}
