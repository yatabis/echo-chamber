use echo_mlx::{Array, DType, Gpu};

use super::gdn::validate_array;
use super::{EngineError, ModelPlan};

#[derive(Debug)]
pub(crate) enum LayerState {
    Gdn {
        convolution: Array,
        recurrent: Array,
    },
    Attention {
        keys: Array,
        values: Array,
    },
}

/// Complete live MLX cache carried by one Qwen3.5-family model execution.
///
/// One value contains both GDN recurrent state and full-attention KV state for
/// every decoder layer. It is deliberately model-specific and is the payload
/// committed atomically by `echo-inference-state`.
#[derive(Debug)]
pub struct MlxInferenceState {
    layers: Vec<LayerState>,
}

impl MlxInferenceState {
    pub(crate) fn new(layers: Vec<LayerState>) -> Self {
        Self { layers }
    }

    pub(crate) fn layers(&self) -> &[LayerState] {
        &self.layers
    }

    /// Returns the number of decoder-layer cache entries.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Returns the number of independently owned MLX tensors.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.layers.len() * 2
    }

    /// Returns the sum of logical bytes represented by every state tensor.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] if the byte count overflows `usize`.
    pub fn logical_nbytes(&self) -> Result<usize, EngineError> {
        self.layers.iter().try_fold(0usize, |total, layer| {
            let [first, second] = layer.arrays();
            total
                .checked_add(first.nbytes())
                .and_then(|total| total.checked_add(second.nbytes()))
                .ok_or_else(|| EngineError::Unsupported("state logical byte count overflow".into()))
        })
    }

    /// Returns the sequence length encoded by every full-attention KV cache.
    ///
    /// The length is derived from the tensors themselves. E.C.H.O. does not
    /// persist a second token-count ledger that could drift from the cache it
    /// is intended to describe.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] when no attention layer exists or attention
    /// layers disagree about their retained sequence length.
    pub fn sequence_length(&self) -> Result<usize, EngineError> {
        let mut observed = None;
        for layer in &self.layers {
            let LayerState::Attention { keys, values } = layer else {
                continue;
            };
            let key_length = keys.shape().get(2).copied().ok_or_else(|| {
                EngineError::Unsupported(format!(
                    "attention key state must be rank 4, observed {:?}",
                    keys.shape()
                ))
            })?;
            let value_length = values.shape().get(2).copied().ok_or_else(|| {
                EngineError::Unsupported(format!(
                    "attention value state must be rank 4, observed {:?}",
                    values.shape()
                ))
            })?;
            if key_length != value_length {
                return Err(EngineError::Unsupported(format!(
                    "attention key/value sequence length mismatch: {key_length} != {value_length}"
                )));
            }
            if observed.is_some_and(|length| length != key_length) {
                return Err(EngineError::Unsupported(format!(
                    "attention cache sequence lengths disagree: {observed:?} and {key_length}"
                )));
            }
            observed = Some(key_length);
        }
        observed.ok_or_else(|| {
            EngineError::Unsupported("model state contains no full-attention cache".into())
        })
    }

    /// Validates the complete cache against one model.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] for a missing, misclassified, misshaped, or
    /// wrongly typed layer state, or inconsistent KV sequence lengths.
    pub fn validate(&self, plan: &ModelPlan, batch_size: usize) -> Result<(), EngineError> {
        if batch_size == 0 {
            return Err(EngineError::Unsupported(
                "state batch size must be greater than zero".into(),
            ));
        }
        if self.layers.len() != plan.layer_count {
            return Err(EngineError::Unsupported(format!(
                "model state layer count mismatch: expected {}, observed {}",
                plan.layer_count,
                self.layers.len()
            )));
        }
        let token_count = self.sequence_length()?;

        for (layer_index, state) in self.layers.iter().enumerate() {
            let full_attention = (layer_index + 1).is_multiple_of(plan.full_attention_interval);
            match (full_attention, state) {
                (true, LayerState::Attention { keys, values }) => {
                    let shape = [
                        batch_size,
                        plan.key_value_head_count,
                        token_count,
                        plan.attention_head_dimension,
                    ];
                    validate_array(
                        keys,
                        &shape,
                        DType::BFloat16,
                        &format!("layer {layer_index} attention key state"),
                    )?;
                    validate_array(
                        values,
                        &shape,
                        DType::BFloat16,
                        &format!("layer {layer_index} attention value state"),
                    )?;
                }
                (
                    false,
                    LayerState::Gdn {
                        convolution,
                        recurrent,
                    },
                ) => {
                    validate_array(
                        convolution,
                        &[
                            batch_size,
                            plan.convolution_kernel_size - 1,
                            plan.convolution_dimension(),
                        ],
                        DType::BFloat16,
                        &format!("layer {layer_index} GDN convolution state"),
                    )?;
                    validate_array(
                        recurrent,
                        &[
                            batch_size,
                            plan.value_head_count,
                            plan.value_head_dimension,
                            plan.key_head_dimension,
                        ],
                        DType::Float32,
                        &format!("layer {layer_index} GDN recurrent state"),
                    )?;
                }
                (true, LayerState::Gdn { .. }) => {
                    return Err(EngineError::Unsupported(format!(
                        "layer {layer_index} requires attention state, observed GDN state"
                    )));
                }
                (false, LayerState::Attention { .. }) => {
                    return Err(EngineError::Unsupported(format!(
                        "layer {layer_index} requires GDN state, observed attention state"
                    )));
                }
            }
        }
        Ok(())
    }

    pub(crate) fn empty(
        gpu: &Gpu,
        batch_size: usize,
        plan: &ModelPlan,
    ) -> Result<Self, EngineError> {
        let mut layers = Vec::with_capacity(plan.layer_count);
        for layer_index in 0..plan.layer_count {
            if (layer_index + 1).is_multiple_of(plan.full_attention_interval) {
                layers.push(empty_attention_state(gpu, batch_size, plan)?);
            } else {
                layers.push(LayerState::Gdn {
                    convolution: gpu
                        .zeros(
                            &[
                                batch_size,
                                plan.convolution_kernel_size - 1,
                                plan.convolution_dimension(),
                            ],
                            DType::BFloat16,
                        )
                        .map_err(EngineError::Mlx)?,
                    recurrent: gpu
                        .zeros(
                            &[
                                batch_size,
                                plan.value_head_count,
                                plan.value_head_dimension,
                                plan.key_head_dimension,
                            ],
                            DType::Float32,
                        )
                        .map_err(EngineError::Mlx)?,
                });
            }
        }
        Ok(Self::new(layers))
    }

    /// Derives the initial cache for a new E.C.H.O. thinking session.
    ///
    /// Both GDN tensors retain the preceding session's immutable MLX graph
    /// nodes. Every full-attention KV tensor is replaced with a zero-length
    /// cache so the fresh prompt cannot attend to the prior token lineage.
    pub(crate) fn begin_new_session(
        &self,
        gpu: &Gpu,
        batch_size: usize,
        plan: &ModelPlan,
    ) -> Result<Self, EngineError> {
        self.validate(plan, batch_size)?;
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            match layer {
                LayerState::Gdn {
                    convolution,
                    recurrent,
                } => layers.push(LayerState::Gdn {
                    convolution: convolution.try_clone().map_err(EngineError::Mlx)?,
                    recurrent: recurrent.try_clone().map_err(EngineError::Mlx)?,
                }),
                LayerState::Attention { .. } => {
                    layers.push(empty_attention_state(gpu, batch_size, plan)?);
                }
            }
        }
        Ok(Self::new(layers))
    }
}

fn empty_attention_state(
    gpu: &Gpu,
    batch_size: usize,
    plan: &ModelPlan,
) -> Result<LayerState, EngineError> {
    let shape = [
        batch_size,
        plan.key_value_head_count,
        0,
        plan.attention_head_dimension,
    ];
    Ok(LayerState::Attention {
        keys: gpu
            .zeros(&shape, DType::BFloat16)
            .map_err(EngineError::Mlx)?,
        values: gpu
            .zeros(&shape, DType::BFloat16)
            .map_err(EngineError::Mlx)?,
    })
}

impl LayerState {
    pub(crate) fn arrays(&self) -> [&Array; 2] {
        match self {
            Self::Gdn {
                convolution,
                recurrent,
            } => [convolution, recurrent],
            Self::Attention { keys, values } => [keys, values],
        }
    }
}
