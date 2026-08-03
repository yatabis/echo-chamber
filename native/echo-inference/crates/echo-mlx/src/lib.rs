//! Small ownership-safe MLX surface used by the specialized engine.

use std::cell::OnceCell;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::ffi::{CStr, CString, c_char, c_void};
use std::fmt;
use std::path::Path;
use std::ptr;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use echo_mlx_sys as sys;

const ITERATOR_END: i32 = 2;

/// One MLX C operation failed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MlxError {
    operation: &'static str,
    status: i32,
    detail: String,
}

impl MlxError {
    fn status(operation: &'static str, status: i32) -> Self {
        Self {
            operation,
            status,
            detail: String::new(),
        }
    }

    fn detail(operation: &'static str, detail: impl Into<String>) -> Self {
        Self {
            operation,
            status: -1,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for MlxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.detail.is_empty() {
            write!(
                formatter,
                "{} failed with MLX status {}",
                self.operation, self.status
            )
        } else {
            write!(formatter, "{} failed: {}", self.operation, self.detail)
        }
    }
}

impl Error for MlxError {}

fn check(status: i32, operation: &'static str) -> Result<(), MlxError> {
    if status == 0 {
        Ok(())
    } else {
        Err(MlxError::status(operation, status))
    }
}

struct MlxString(sys::mlx_string);

impl MlxString {
    fn new() -> Self {
        Self(unsafe { sys::mlx_string_new() })
    }

    fn to_string(&self) -> Result<String, MlxError> {
        let pointer = unsafe { sys::mlx_string_data(self.0) };
        if pointer.is_null() {
            return Err(MlxError::detail("mlx_string_data", "returned null"));
        }
        Ok(unsafe { CStr::from_ptr(pointer) }
            .to_str()
            .map_err(|error| MlxError::detail("mlx_string_data", error.to_string()))?
            .to_owned())
    }
}

impl Drop for MlxString {
    fn drop(&mut self) {
        unsafe { sys::mlx_string_free(self.0) };
    }
}

/// Matching MLX runtime information.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeInfo {
    pub version: String,
    pub metal_available: bool,
}

/// Process-wide Metal allocator observations reported by MLX.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetalMemoryStats {
    pub active_nbytes: usize,
    pub cache_nbytes: usize,
    pub peak_nbytes: usize,
}

/// Number of times MLX traced each lazily initialized compiled graph.
///
/// An uninitialized graph reports zero. A shape-polymorphic graph should stay
/// at one while it is reused across compatible activation shapes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompiledTraceCounts {
    pub silu: usize,
    pub swiglu: usize,
    pub precise_swiglu: usize,
    pub compute_g: usize,
}

/// Reads the linked MLX runtime identity without initializing a model.
///
/// # Errors
///
/// Returns [`MlxError`] if the linked runtime cannot report its version or
/// Metal availability.
pub fn runtime_info() -> Result<RuntimeInfo, MlxError> {
    let mut version = MlxString::new();
    check(
        unsafe { sys::mlx_version(&raw mut version.0) },
        "mlx_version",
    )?;
    let mut metal_available = false;
    check(
        unsafe { sys::mlx_metal_is_available(&raw mut metal_available) },
        "mlx_metal_is_available",
    )?;
    Ok(RuntimeInfo {
        version: version.to_string()?,
        metal_available,
    })
}

/// Reads process-wide Metal allocator observations.
///
/// # Errors
///
/// Returns [`MlxError`] if MLX cannot report any allocator counter.
pub fn metal_memory_stats() -> Result<MetalMemoryStats, MlxError> {
    let mut active_nbytes = 0;
    check(
        unsafe { sys::mlx_get_active_memory(&raw mut active_nbytes) },
        "mlx_get_active_memory",
    )?;
    let mut cache_nbytes = 0;
    check(
        unsafe { sys::mlx_get_cache_memory(&raw mut cache_nbytes) },
        "mlx_get_cache_memory",
    )?;
    let mut peak_nbytes = 0;
    check(
        unsafe { sys::mlx_get_peak_memory(&raw mut peak_nbytes) },
        "mlx_get_peak_memory",
    )?;
    Ok(MetalMemoryStats {
        active_nbytes,
        cache_nbytes,
        peak_nbytes,
    })
}

/// MLX element type admitted by the native engine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    Bool,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
    BFloat16,
    Complex64,
}

impl DType {
    fn from_raw(raw: sys::mlx_dtype) -> Self {
        match raw {
            0 => Self::Bool,
            1 => Self::Uint8,
            2 => Self::Uint16,
            3 => Self::Uint32,
            4 => Self::Uint64,
            5 => Self::Int8,
            6 => Self::Int16,
            7 => Self::Int32,
            8 => Self::Int64,
            9 => Self::Float16,
            10 => Self::Float32,
            11 => Self::Float64,
            12 => Self::BFloat16,
            13 => Self::Complex64,
            _ => unreachable!("MLX returned an unknown dtype code: {raw}"),
        }
    }

    fn raw(self) -> sys::mlx_dtype {
        match self {
            Self::Bool => 0,
            Self::Uint8 => 1,
            Self::Uint16 => 2,
            Self::Uint32 => 3,
            Self::Uint64 => 4,
            Self::Int8 => 5,
            Self::Int16 => 6,
            Self::Int32 => 7,
            Self::Int64 => 8,
            Self::Float16 => 9,
            Self::Float32 => 10,
            Self::Float64 => 11,
            Self::BFloat16 => 12,
            Self::Complex64 => 13,
        }
    }

    /// Stable human-readable dtype name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::Uint8 => "uint8",
            Self::Uint16 => "uint16",
            Self::Uint32 => "uint32",
            Self::Uint64 => "uint64",
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::Float16 => "float16",
            Self::Float32 => "float32",
            Self::Float64 => "float64",
            Self::BFloat16 => "bfloat16",
            Self::Complex64 => "complex64",
        }
    }
}

/// Parameters for one quantized matrix multiplication.
#[derive(Clone, Copy, Debug)]
pub struct QuantizedMatmulConfig<'a> {
    pub biases: Option<&'a Array>,
    pub transpose: bool,
    pub group_size: i32,
    pub bits: i32,
    pub mode: &'a str,
}

/// Parameters for one expert-indexed quantized matrix multiplication.
#[derive(Clone, Copy, Debug)]
pub struct GatherQuantizedMatmulConfig<'a> {
    pub biases: Option<&'a Array>,
    pub rhs_indices: &'a Array,
    pub transpose: bool,
    pub group_size: i32,
    pub bits: i32,
    pub mode: &'a str,
    pub sorted_indices: bool,
}

/// Parameters for dequantizing one packed affine tensor.
#[derive(Clone, Copy, Debug)]
pub struct DequantizeConfig<'a> {
    pub biases: Option<&'a Array>,
    pub group_size: i32,
    pub bits: i32,
    pub mode: &'a str,
    pub dtype: Option<DType>,
}

/// Parameters for MLX's fused rotary position encoding.
#[derive(Clone, Copy, Debug)]
pub struct RopeConfig<'a> {
    pub dimensions: i32,
    pub traditional: bool,
    pub base: Option<f32>,
    pub scale: f32,
    pub offset: i32,
    pub frequencies: Option<&'a Array>,
}

struct GpuStream(sys::mlx_stream);

impl GpuStream {
    fn new() -> Self {
        Self(unsafe { sys::mlx_default_gpu_stream_new() })
    }
}

struct CpuStream(sys::mlx_stream);

impl CpuStream {
    fn new() -> Self {
        Self(unsafe { sys::mlx_default_cpu_stream_new() })
    }
}

impl Drop for CpuStream {
    fn drop(&mut self) {
        let _ = unsafe { sys::mlx_stream_free(self.0) };
    }
}

impl Drop for GpuStream {
    fn drop(&mut self) {
        let _ = unsafe { sys::mlx_stream_free(self.0) };
    }
}

/// One reusable handle to MLX's default GPU stream.
pub struct Gpu {
    stream: GpuStream,
    silu_function: OnceCell<CompiledClosure>,
    swiglu_function: OnceCell<CompiledClosure>,
    precise_swiglu_function: OnceCell<CompiledClosure>,
    compute_g_function: OnceCell<CompiledClosure>,
}

impl Gpu {
    /// Uses MLX's default GPU stream for all operations issued through this
    /// handle.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stream: GpuStream::new(),
            silu_function: OnceCell::new(),
            swiglu_function: OnceCell::new(),
            precise_swiglu_function: OnceCell::new(),
            compute_g_function: OnceCell::new(),
        }
    }

    /// Reports trace counts for the compiled graphs initialized on this GPU.
    #[must_use]
    pub fn compiled_trace_counts(&self) -> CompiledTraceCounts {
        CompiledTraceCounts {
            silu: self
                .silu_function
                .get()
                .map_or(0, CompiledClosure::trace_count),
            swiglu: self
                .swiglu_function
                .get()
                .map_or(0, CompiledClosure::trace_count),
            precise_swiglu: self
                .precise_swiglu_function
                .get()
                .map_or(0, CompiledClosure::trace_count),
            compute_g: self
                .compute_g_function
                .get()
                .map_or(0, CompiledClosure::trace_count),
        }
    }

    /// Materializes a set of lazy arrays as one MLX evaluation boundary.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] when no arrays are supplied or MLX cannot evaluate
    /// the graph.
    pub fn eval(&self, arrays: &[&Array]) -> Result<(), MlxError> {
        if arrays.is_empty() {
            return Err(MlxError::detail(
                "mlx_eval",
                "at least one array is required",
            ));
        }
        let arrays = ArrayVector::from_arrays(arrays);
        check(unsafe { sys::mlx_eval(arrays.0) }, "mlx_eval")
    }

    /// Schedules a set of lazy arrays for asynchronous evaluation.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] when no arrays are supplied or MLX cannot
    /// schedule the graph.
    pub fn async_eval(&self, arrays: &[&Array]) -> Result<(), MlxError> {
        if arrays.is_empty() {
            return Err(MlxError::detail(
                "mlx_async_eval",
                "at least one array is required",
            ));
        }
        let arrays = ArrayVector::from_arrays(arrays);
        check(unsafe { sys::mlx_async_eval(arrays.0) }, "mlx_async_eval")
    }

    /// Waits until all work issued to this GPU stream has completed.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] when MLX cannot synchronize the stream.
    pub fn synchronize(&self) -> Result<(), MlxError> {
        check(
            unsafe { sys::mlx_synchronize(self.stream.0) },
            "mlx_synchronize",
        )
    }

    /// Creates a zero-filled array.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] for dimensions outside MLX's C ABI or an MLX
    /// operation failure.
    pub fn zeros(&self, shape: &[usize], dtype: DType) -> Result<Array, MlxError> {
        let shape = checked_shape(shape, "mlx_zeros")?;
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_zeros(
                    &raw mut result.0,
                    shape.as_ptr(),
                    shape.len(),
                    dtype.raw(),
                    self.stream.0,
                )
            },
            "mlx_zeros",
        )?;
        Ok(result)
    }

    /// Casts an array to a different element type.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the cast.
    pub fn astype(&self, array: &Array, dtype: DType) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe { sys::mlx_astype(&raw mut result.0, array.0, dtype.raw(), self.stream.0) },
            "mlx_astype",
        )?;
        Ok(result)
    }

    /// Reshapes an array without changing its logical values.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] for an invalid shape or MLX operation failure.
    pub fn reshape(&self, array: &Array, shape: &[usize]) -> Result<Array, MlxError> {
        let shape = checked_shape(shape, "mlx_reshape")?;
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_reshape(
                    &raw mut result.0,
                    array.0,
                    shape.as_ptr(),
                    shape.len(),
                    self.stream.0,
                )
            },
            "mlx_reshape",
        )?;
        Ok(result)
    }

    /// Permutes all axes using one explicit order.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] for an invalid permutation or MLX operation
    /// failure.
    pub fn transpose(&self, array: &Array, axes: &[i32]) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_transpose_axes(
                    &raw mut result.0,
                    array.0,
                    axes.as_ptr(),
                    axes.len(),
                    self.stream.0,
                )
            },
            "mlx_transpose_axes",
        )?;
        Ok(result)
    }

    /// Concatenates arrays along one axis.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] when there are no arrays or MLX rejects their
    /// shapes, dtypes, or axis.
    pub fn concatenate(&self, arrays: &[&Array], axis: i32) -> Result<Array, MlxError> {
        if arrays.is_empty() {
            return Err(MlxError::detail(
                "mlx_concatenate_axis",
                "at least one array is required",
            ));
        }
        let arrays = ArrayVector::from_arrays(arrays);
        let mut result = Array::empty();
        check(
            unsafe { sys::mlx_concatenate_axis(&raw mut result.0, arrays.0, axis, self.stream.0) },
            "mlx_concatenate_axis",
        )?;
        Ok(result)
    }

    /// Takes a static multi-axis slice.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] for mismatched slice specifications or an MLX
    /// operation failure.
    pub fn slice(
        &self,
        array: &Array,
        start: &[i32],
        stop: &[i32],
        strides: &[i32],
    ) -> Result<Array, MlxError> {
        if start.len() != stop.len() || start.len() != strides.len() {
            return Err(MlxError::detail(
                "mlx_slice",
                "start, stop, and strides must have equal lengths",
            ));
        }
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_slice(
                    &raw mut result.0,
                    array.0,
                    start.as_ptr(),
                    start.len(),
                    stop.as_ptr(),
                    stop.len(),
                    strides.as_ptr(),
                    strides.len(),
                    self.stream.0,
                )
            },
            "mlx_slice",
        )?;
        Ok(result)
    }

    /// Replaces one static multi-axis slice while preserving the source shape.
    ///
    /// MLX may donate the source buffer when it is no longer referenced. This
    /// is the primitive used by capacity-backed KV caches to avoid rebuilding
    /// the complete cache on every decoded token.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] for mismatched slice specifications or an MLX
    /// operation failure.
    pub fn slice_update(
        &self,
        source: &Array,
        update: &Array,
        start: &[i32],
        stop: &[i32],
        strides: &[i32],
    ) -> Result<Array, MlxError> {
        if start.len() != stop.len() || start.len() != strides.len() {
            return Err(MlxError::detail(
                "mlx_slice_update",
                "start, stop, and strides must have equal lengths",
            ));
        }
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_slice_update(
                    &raw mut result.0,
                    source.0,
                    update.0,
                    start.as_ptr(),
                    start.len(),
                    stop.as_ptr(),
                    stop.len(),
                    strides.as_ptr(),
                    strides.len(),
                    self.stream.0,
                )
            },
            "mlx_slice_update",
        )?;
        Ok(result)
    }

    /// Requests a row-contiguous representation.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX cannot construct the representation.
    pub fn contiguous(&self, array: &Array) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe { sys::mlx_contiguous(&raw mut result.0, array.0, false, self.stream.0) },
            "mlx_contiguous",
        )?;
        Ok(result)
    }

    /// Executes MLX's grouped one-dimensional convolution.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the convolution layout.
    pub fn conv1d(
        &self,
        input: &Array,
        weight: &Array,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_conv1d(
                    &raw mut result.0,
                    input.0,
                    weight.0,
                    stride,
                    padding,
                    dilation,
                    groups,
                    self.stream.0,
                )
            },
            "mlx_conv1d",
        )?;
        Ok(result)
    }

    /// Executes an affine quantized matrix multiplication.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] for invalid quantization parameters, strings, or
    /// an MLX operation failure.
    pub fn quantized_matmul(
        &self,
        input: &Array,
        weight: &Array,
        scales: &Array,
        config: QuantizedMatmulConfig<'_>,
    ) -> Result<Array, MlxError> {
        let mode = CString::new(config.mode)
            .map_err(|error| MlxError::detail("quantization mode", error.to_string()))?;
        let group_size = sys::mlx_optional_int_ {
            value: config.group_size,
            has_value: true,
        };
        let bits = sys::mlx_optional_int_ {
            value: config.bits,
            has_value: true,
        };
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_quantized_matmul(
                    &raw mut result.0,
                    input.0,
                    weight.0,
                    scales.0,
                    optional_array(config.biases),
                    config.transpose,
                    group_size,
                    bits,
                    mode.as_ptr(),
                    self.stream.0,
                )
            },
            "mlx_quantized_matmul",
        )?;
        Ok(result)
    }

    /// Dequantizes one packed affine tensor.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] for invalid quantization parameters, strings, or
    /// an MLX operation failure.
    pub fn dequantize(
        &self,
        weight: &Array,
        scales: &Array,
        config: DequantizeConfig<'_>,
    ) -> Result<Array, MlxError> {
        let mode = CString::new(config.mode)
            .map_err(|error| MlxError::detail("quantization mode", error.to_string()))?;
        let group_size = sys::mlx_optional_int_ {
            value: config.group_size,
            has_value: true,
        };
        let bits = sys::mlx_optional_int_ {
            value: config.bits,
            has_value: true,
        };
        let dtype = sys::mlx_optional_dtype_ {
            value: config.dtype.unwrap_or(DType::Float32).raw(),
            has_value: config.dtype.is_some(),
        };
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_dequantize(
                    &raw mut result.0,
                    weight.0,
                    scales.0,
                    optional_array(config.biases),
                    group_size,
                    bits,
                    mode.as_ptr(),
                    optional_array(None),
                    dtype,
                    self.stream.0,
                )
            },
            "mlx_dequantize",
        )?;
        Ok(result)
    }

    /// Executes an expert-indexed affine quantized matrix multiplication.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] for invalid quantization parameters, strings, or
    /// an MLX operation failure.
    pub fn gather_quantized_matmul(
        &self,
        input: &Array,
        weight: &Array,
        scales: &Array,
        config: GatherQuantizedMatmulConfig<'_>,
    ) -> Result<Array, MlxError> {
        let mode = CString::new(config.mode)
            .map_err(|error| MlxError::detail("quantization mode", error.to_string()))?;
        let group_size = sys::mlx_optional_int_ {
            value: config.group_size,
            has_value: true,
        };
        let bits = sys::mlx_optional_int_ {
            value: config.bits,
            has_value: true,
        };
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_gather_qmm(
                    &raw mut result.0,
                    input.0,
                    weight.0,
                    scales.0,
                    optional_array(config.biases),
                    optional_array(None),
                    config.rhs_indices.0,
                    config.transpose,
                    group_size,
                    bits,
                    mode.as_ptr(),
                    config.sorted_indices,
                    self.stream.0,
                )
            },
            "mlx_gather_qmm",
        )?;
        Ok(result)
    }

    /// Applies fused rotary position encoding.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the shape, rotary dimensions, base,
    /// offset, or optional frequency tensor.
    pub fn rope(&self, input: &Array, config: RopeConfig<'_>) -> Result<Array, MlxError> {
        let base = sys::mlx_optional_float_ {
            value: config.base.unwrap_or_default(),
            has_value: config.base.is_some(),
        };
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_fast_rope(
                    &raw mut result.0,
                    input.0,
                    config.dimensions,
                    config.traditional,
                    base,
                    config.scale,
                    config.offset,
                    optional_array(config.frequencies),
                    self.stream.0,
                )
            },
            "mlx_fast_rope",
        )?;
        Ok(result)
    }

    /// Applies fused grouped-query scaled dot-product attention.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the query/key/value shapes or mask
    /// mode.
    pub fn scaled_dot_product_attention(
        &self,
        queries: &Array,
        keys: &Array,
        values: &Array,
        scale: f32,
        causal: bool,
    ) -> Result<Array, MlxError> {
        let mask_mode = CString::new(if causal { "causal" } else { "" })
            .map_err(|error| MlxError::detail("attention mask mode", error.to_string()))?;
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_fast_scaled_dot_product_attention(
                    &raw mut result.0,
                    queries.0,
                    keys.0,
                    values.0,
                    scale,
                    mask_mode.as_ptr(),
                    optional_array(None),
                    optional_array(None),
                    self.stream.0,
                )
            },
            "mlx_fast_scaled_dot_product_attention",
        )?;
        Ok(result)
    }

    /// Computes a precise softmax along one axis.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the axis or input.
    pub fn softmax_axis(&self, input: &Array, axis: i32, precise: bool) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_softmax_axis(&raw mut result.0, input.0, axis, precise, self.stream.0)
            },
            "mlx_softmax_axis",
        )?;
        Ok(result)
    }

    /// Returns indices partitioned around `kth` along one axis.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the rank, axis, or partition index.
    pub fn argpartition_axis(&self, input: &Array, kth: i32, axis: i32) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_argpartition_axis(&raw mut result.0, input.0, kth, axis, self.stream.0)
            },
            "mlx_argpartition_axis",
        )?;
        Ok(result)
    }

    /// Returns an integer range on this GPU stream.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if the bounds do not fit MLX or MLX rejects the
    /// requested dtype.
    pub fn arange(&self, stop: usize, dtype: DType) -> Result<Array, MlxError> {
        let stop = u32::try_from(stop)
            .map_err(|error| MlxError::detail("mlx_arange", error.to_string()))?;
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_arange(
                    &raw mut result.0,
                    0.0,
                    f64::from(stop),
                    1.0,
                    dtype.raw(),
                    self.stream.0,
                )
            },
            "mlx_arange",
        )?;
        Ok(result)
    }

    /// Returns maximum-value indices along one axis.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the rank or axis.
    pub fn argmax_axis(&self, input: &Array, axis: i32, keepdims: bool) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_argmax_axis(&raw mut result.0, input.0, axis, keepdims, self.stream.0)
            },
            "mlx_argmax_axis",
        )?;
        Ok(result)
    }

    /// Returns ascending sort indices for a one-dimensional input.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the input.
    pub fn argsort(&self, input: &Array) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe { sys::mlx_argsort(&raw mut result.0, input.0, self.stream.0) },
            "mlx_argsort",
        )?;
        Ok(result)
    }

    /// Returns ascending sort indices along one axis.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the input or axis.
    pub fn argsort_axis(&self, input: &Array, axis: i32) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe { sys::mlx_argsort_axis(&raw mut result.0, input.0, axis, self.stream.0) },
            "mlx_argsort_axis",
        )?;
        Ok(result)
    }

    /// Takes flattened values at the supplied indices.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the input or index dtype.
    pub fn take(&self, input: &Array, indices: &Array) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe { sys::mlx_take(&raw mut result.0, input.0, indices.0, self.stream.0) },
            "mlx_take",
        )?;
        Ok(result)
    }

    /// Takes slices from one axis at the supplied indices.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the input, indices, or axis.
    pub fn take_axis(&self, input: &Array, indices: &Array, axis: i32) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_take_axis(&raw mut result.0, input.0, indices.0, axis, self.stream.0)
            },
            "mlx_take_axis",
        )?;
        Ok(result)
    }

    /// Takes values using same-rank indices along one axis.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the shapes, dtype, or axis.
    pub fn take_along_axis(
        &self,
        input: &Array,
        indices: &Array,
        axis: i32,
    ) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_take_along_axis(&raw mut result.0, input.0, indices.0, axis, self.stream.0)
            },
            "mlx_take_along_axis",
        )?;
        Ok(result)
    }

    /// Replaces values at same-rank indices along one axis.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the shapes, dtype, or axis.
    pub fn put_along_axis(
        &self,
        input: &Array,
        indices: &Array,
        values: &Array,
        axis: i32,
    ) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_put_along_axis(
                    &raw mut result.0,
                    input.0,
                    indices.0,
                    values.0,
                    axis,
                    self.stream.0,
                )
            },
            "mlx_put_along_axis",
        )?;
        Ok(result)
    }

    /// Computes an inclusive cumulative sum along one axis.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the axis or input.
    pub fn cumsum(&self, input: &Array, axis: i32) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_cumsum(&raw mut result.0, input.0, axis, false, true, self.stream.0)
            },
            "mlx_cumsum",
        )?;
        Ok(result)
    }

    /// Computes log-sum-exp along one axis.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the axis or input.
    pub fn logsumexp_axis(
        &self,
        input: &Array,
        axis: i32,
        keepdims: bool,
    ) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_logsumexp_axis(&raw mut result.0, input.0, axis, keepdims, self.stream.0)
            },
            "mlx_logsumexp_axis",
        )?;
        Ok(result)
    }

    /// Compares two arrays elementwise with `>`.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX cannot broadcast or compare the arrays.
    pub fn greater(&self, left: &Array, right: &Array) -> Result<Array, MlxError> {
        self.binary(
            left,
            right,
            "mlx_greater",
            |result, left, right, stream| unsafe { sys::mlx_greater(result, left, right, stream) },
        )
    }

    /// Selects values elementwise according to a boolean condition.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX cannot broadcast the supplied arrays.
    pub fn where_condition(
        &self,
        condition: &Array,
        when_true: &Array,
        when_false: &Array,
    ) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_where(
                    &raw mut result.0,
                    condition.0,
                    when_true.0,
                    when_false.0,
                    self.stream.0,
                )
            },
            "mlx_where",
        )?;
        Ok(result)
    }

    /// Creates a zero array matching another array's shape and dtype.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the input.
    pub fn zeros_like(&self, input: &Array) -> Result<Array, MlxError> {
        self.unary(input, "mlx_zeros_like", |result, input, stream| unsafe {
            sys::mlx_zeros_like(result, input, stream)
        })
    }

    /// Draws one categorical sample using an explicit functional random key.
    ///
    /// The key is request-owned so sampling one E.C.H.O. instance cannot
    /// perturb another instance's random stream.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the logits, axis, or key.
    pub fn categorical_with_seed(
        &self,
        logits: &Array,
        axis: i32,
        seed: u64,
    ) -> Result<Array, MlxError> {
        let mut key = Array::empty();
        check(
            unsafe { sys::mlx_random_key(&raw mut key.0, seed) },
            "mlx_random_key",
        )?;
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_random_categorical(&raw mut result.0, logits.0, axis, key.0, self.stream.0)
            },
            "mlx_random_categorical",
        )?;
        Ok(result)
    }

    /// Sums one axis while optionally retaining its dimension.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the axis or input.
    pub fn sum_axis(&self, input: &Array, axis: i32, keepdims: bool) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe { sys::mlx_sum_axis(&raw mut result.0, input.0, axis, keepdims, self.stream.0) },
            "mlx_sum_axis",
        )?;
        Ok(result)
    }

    /// Executes MLX's fused RMS normalization.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the input or weight.
    pub fn rms_norm(
        &self,
        input: &Array,
        weight: Option<&Array>,
        epsilon: f32,
    ) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe {
                sys::mlx_fast_rms_norm(
                    &raw mut result.0,
                    input.0,
                    optional_array(weight),
                    epsilon,
                    self.stream.0,
                )
            },
            "mlx_fast_rms_norm",
        )?;
        Ok(result)
    }

    /// Element-wise absolute value.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if the MLX operation fails.
    pub fn abs(&self, input: &Array) -> Result<Array, MlxError> {
        self.unary(input, "mlx_abs", |result, array, stream| unsafe {
            sys::mlx_abs(result, array, stream)
        })
    }

    /// Element-wise exponential.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if the MLX operation fails.
    pub fn exp(&self, input: &Array) -> Result<Array, MlxError> {
        self.unary(input, "mlx_exp", |result, array, stream| unsafe {
            sys::mlx_exp(result, array, stream)
        })
    }

    /// Element-wise negation.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if the MLX operation fails.
    pub fn negative(&self, input: &Array) -> Result<Array, MlxError> {
        self.unary(input, "mlx_negative", |result, array, stream| unsafe {
            sys::mlx_negative(result, array, stream)
        })
    }

    /// Element-wise sigmoid.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if the MLX operation fails.
    pub fn sigmoid(&self, input: &Array) -> Result<Array, MlxError> {
        self.unary(input, "mlx_sigmoid", |result, array, stream| unsafe {
            sys::mlx_sigmoid(result, array, stream)
        })
    }

    /// Element-wise addition with MLX broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if the MLX operation fails.
    pub fn add(&self, left: &Array, right: &Array) -> Result<Array, MlxError> {
        self.binary(left, right, "mlx_add", |result, a, b, stream| unsafe {
            sys::mlx_add(result, a, b, stream)
        })
    }

    /// Element-wise subtraction with MLX broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if the MLX operation fails.
    pub fn subtract(&self, left: &Array, right: &Array) -> Result<Array, MlxError> {
        self.binary(left, right, "mlx_subtract", |result, a, b, stream| unsafe {
            sys::mlx_subtract(result, a, b, stream)
        })
    }

    /// Element-wise multiplication with MLX broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if the MLX operation fails.
    pub fn multiply(&self, left: &Array, right: &Array) -> Result<Array, MlxError> {
        self.binary(left, right, "mlx_multiply", |result, a, b, stream| unsafe {
            sys::mlx_multiply(result, a, b, stream)
        })
    }

    /// Element-wise division with MLX broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if the MLX operation fails.
    pub fn divide(&self, left: &Array, right: &Array) -> Result<Array, MlxError> {
        self.binary(left, right, "mlx_divide", |result, a, b, stream| unsafe {
            sys::mlx_divide(result, a, b, stream)
        })
    }

    /// Element-wise floor division with MLX broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if the MLX operation fails.
    pub fn floor_divide(&self, left: &Array, right: &Array) -> Result<Array, MlxError> {
        self.binary(
            left,
            right,
            "mlx_floor_divide",
            |result, a, b, stream| unsafe { sys::mlx_floor_divide(result, a, b, stream) },
        )
    }

    /// Stable element-wise `log(exp(a) + exp(b))`.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if the MLX operation fails.
    pub fn logaddexp(&self, left: &Array, right: &Array) -> Result<Array, MlxError> {
        self.binary(
            left,
            right,
            "mlx_logaddexp",
            |result, a, b, stream| unsafe { sys::mlx_logaddexp(result, a, b, stream) },
        )
    }

    /// Reduces all dimensions to their maximum.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if the MLX operation fails.
    pub fn max(&self, input: &Array) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe { sys::mlx_max(&raw mut result.0, input.0, false, self.stream.0) },
            "mlx_max",
        )?;
        Ok(result)
    }

    /// Fused `SiLU` used by MLX's compiled `nn.silu`.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX cannot compile or execute the fused graph.
    pub fn silu(&self, input: &Array) -> Result<Array, MlxError> {
        let function = if let Some(function) = self.silu_function.get() {
            function
        } else {
            let function = CompiledClosure::silu(self.stream.0)?;
            self.silu_function.set(function).map_err(|_| {
                MlxError::detail("compiled_silu", "kernel cache was initialized concurrently")
            })?;
            self.silu_function
                .get()
                .ok_or_else(|| MlxError::detail("compiled_silu", "kernel cache stayed empty"))?
        };
        let outputs = function.apply(&[input])?;
        outputs
            .into_iter()
            .next()
            .ok_or_else(|| MlxError::detail("compiled_silu", "compiled graph omitted its output"))
    }

    /// Compiled `SiLU(gate) * input` used by Qwen's routed and shared experts.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX cannot compile or execute the fused graph.
    pub fn swiglu(&self, gate: &Array, input: &Array) -> Result<Array, MlxError> {
        let function = if let Some(function) = self.swiglu_function.get() {
            function
        } else {
            let function = CompiledClosure::new(self.stream.0, compiled_swiglu, "compiled_swiglu")?;
            self.swiglu_function.set(function).map_err(|_| {
                MlxError::detail(
                    "compiled_swiglu",
                    "function cache was initialized concurrently",
                )
            })?;
            self.swiglu_function
                .get()
                .ok_or_else(|| MlxError::detail("compiled_swiglu", "function cache stayed empty"))?
        };
        one_compiled_output("compiled_swiglu", function.apply(&[gate, input])?)
    }

    /// Float32-accumulating compiled `SwiGLU` used by Qwen's gated GDN norm.
    ///
    /// `reference` supplies the output dtype exactly as in the official
    /// `_precise_swiglu(hidden, gate, normalized)` helper.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX cannot compile or execute the fused graph.
    pub fn precise_swiglu(
        &self,
        reference: &Array,
        gate: &Array,
        input: &Array,
    ) -> Result<Array, MlxError> {
        let function = if let Some(function) = self.precise_swiglu_function.get() {
            function
        } else {
            let function = CompiledClosure::new(
                self.stream.0,
                compiled_precise_swiglu,
                "compiled_precise_swiglu",
            )?;
            self.precise_swiglu_function.set(function).map_err(|_| {
                MlxError::detail(
                    "compiled_precise_swiglu",
                    "function cache was initialized concurrently",
                )
            })?;
            self.precise_swiglu_function.get().ok_or_else(|| {
                MlxError::detail("compiled_precise_swiglu", "function cache stayed empty")
            })?
        };
        one_compiled_output(
            "compiled_precise_swiglu",
            function.apply(&[reference, gate, input])?,
        )
    }

    /// Compiled Qwen GDN decay `exp(-exp(A_log.float()) * softplus(a + bias))`.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX cannot compile or execute the fused graph.
    pub fn compute_g(&self, a_log: &Array, a: &Array, dt_bias: &Array) -> Result<Array, MlxError> {
        let function = if let Some(function) = self.compute_g_function.get() {
            function
        } else {
            let function =
                CompiledClosure::new(self.stream.0, compiled_compute_g, "compiled_compute_g")?;
            self.compute_g_function.set(function).map_err(|_| {
                MlxError::detail(
                    "compiled_compute_g",
                    "function cache was initialized concurrently",
                )
            })?;
            self.compute_g_function.get().ok_or_else(|| {
                MlxError::detail("compiled_compute_g", "function cache stayed empty")
            })?
        };
        one_compiled_output("compiled_compute_g", function.apply(&[a_log, a, dt_bias])?)
    }

    /// Exact softplus construction used by MLX `nn.softplus`.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX cannot create or combine the arrays.
    pub fn softplus(&self, input: &Array) -> Result<Array, MlxError> {
        let zero = self.scalar_like(0.0, input.dtype())?;
        self.logaddexp(input, &zero)
    }

    /// Creates a scalar cast to the requested dtype.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the cast.
    pub fn scalar_like(&self, value: f32, dtype: DType) -> Result<Array, MlxError> {
        let scalar = Array(unsafe { sys::mlx_array_new_float32(value) });
        self.astype(&scalar, dtype)
    }

    /// Creates one int32 scalar.
    #[must_use]
    pub fn scalar_i32(&self, value: i32) -> Array {
        Array(unsafe { sys::mlx_array_new_int(value) })
    }

    /// Computes an eagerly materialized maximum absolute difference in
    /// float32.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] for shape mismatch or any failed MLX operation.
    pub fn max_abs_difference(&self, left: &Array, right: &Array) -> Result<f32, MlxError> {
        if left.shape() != right.shape() {
            return Err(MlxError::detail(
                "max_abs_difference",
                format!(
                    "shape mismatch: left {:?}, right {:?}",
                    left.shape(),
                    right.shape()
                ),
            ));
        }
        let left = self.astype(left, DType::Float32)?;
        let right = self.astype(right, DType::Float32)?;
        let difference = self.subtract(&left, &right)?;
        let difference = self.abs(&difference)?;
        let maximum = self.max(&difference)?;
        maximum.item_f32()
    }

    fn unary(
        &self,
        input: &Array,
        operation: &'static str,
        apply: impl FnOnce(*mut sys::mlx_array, sys::mlx_array, sys::mlx_stream) -> i32,
    ) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(apply(&raw mut result.0, input.0, self.stream.0), operation)?;
        Ok(result)
    }

    fn binary(
        &self,
        left: &Array,
        right: &Array,
        operation: &'static str,
        apply: impl FnOnce(*mut sys::mlx_array, sys::mlx_array, sys::mlx_array, sys::mlx_stream) -> i32,
    ) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            apply(&raw mut result.0, left.0, right.0, self.stream.0),
            operation,
        )?;
        Ok(result)
    }
}

impl Default for Gpu {
    fn default() -> Self {
        Self::new()
    }
}

/// Owned MLX array handle.
#[derive(Debug)]
pub struct Array(sys::mlx_array);

impl Array {
    fn empty() -> Self {
        Self(unsafe { sys::mlx_array_new() })
    }

    /// Creates an eager `int32` array by copying one host slice.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] when the shape does not describe exactly the
    /// supplied number of elements, a dimension exceeds MLX's C ABI, or MLX
    /// cannot create the array.
    pub fn from_i32_slice(values: &[i32], shape: &[usize]) -> Result<Self, MlxError> {
        let shape_i32 = checked_shape(shape, "mlx_array_new_data")?;
        let element_count = shape.iter().try_fold(1usize, |count, dimension| {
            count.checked_mul(*dimension).ok_or_else(|| {
                MlxError::detail("mlx_array_new_data", "shape element count overflow")
            })
        })?;
        if element_count != values.len() {
            return Err(MlxError::detail(
                "mlx_array_new_data",
                format!(
                    "shape {shape:?} describes {element_count} elements, supplied {}",
                    values.len()
                ),
            ));
        }
        let raw = unsafe {
            sys::mlx_array_new_data(
                values.as_ptr().cast(),
                shape_i32.as_ptr(),
                shape_i32
                    .len()
                    .try_into()
                    .map_err(|error| MlxError::detail("mlx_array_new_data", format!("{error}")))?,
                DType::Int32.raw(),
            )
        };
        if raw.ctx.is_null() {
            return Err(MlxError::detail(
                "mlx_array_new_data",
                "MLX returned an empty array handle",
            ));
        }
        Ok(Self(raw))
    }

    /// Creates an eager `float32` array by copying one host slice.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] when the shape does not describe exactly the
    /// supplied number of elements, a dimension exceeds MLX's C ABI, or MLX
    /// cannot create the array.
    pub fn from_f32_slice(values: &[f32], shape: &[usize]) -> Result<Self, MlxError> {
        let shape_i32 = checked_shape(shape, "mlx_array_new_data")?;
        let element_count = shape.iter().try_fold(1usize, |count, dimension| {
            count.checked_mul(*dimension).ok_or_else(|| {
                MlxError::detail("mlx_array_new_data", "shape element count overflow")
            })
        })?;
        if element_count != values.len() {
            return Err(MlxError::detail(
                "mlx_array_new_data",
                format!(
                    "shape {shape:?} describes {element_count} elements, supplied {}",
                    values.len()
                ),
            ));
        }
        let raw = unsafe {
            sys::mlx_array_new_data(
                values.as_ptr().cast(),
                shape_i32.as_ptr(),
                shape_i32
                    .len()
                    .try_into()
                    .map_err(|error| MlxError::detail("mlx_array_new_data", format!("{error}")))?,
                DType::Float32.raw(),
            )
        };
        if raw.ctx.is_null() {
            return Err(MlxError::detail(
                "mlx_array_new_data",
                "MLX returned an empty array handle",
            ));
        }
        Ok(Self(raw))
    }

    /// Returns the logical tensor shape.
    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        let dimensions = unsafe { sys::mlx_array_ndim(self.0) };
        let pointer = unsafe { sys::mlx_array_shape(self.0) };
        if pointer.is_null() || dimensions == 0 {
            return Vec::new();
        }
        unsafe { std::slice::from_raw_parts(pointer, dimensions) }
            .iter()
            .map(|dimension| usize::try_from(*dimension).unwrap_or_default())
            .collect()
    }

    /// Returns logical bytes represented by the array.
    #[must_use]
    pub fn nbytes(&self) -> usize {
        unsafe { sys::mlx_array_nbytes(self.0) }
    }

    /// Returns the array element type.
    #[must_use]
    pub fn dtype(&self) -> DType {
        DType::from_raw(unsafe { sys::mlx_array_dtype(self.0) })
    }

    /// Returns a stable human-readable dtype name.
    #[must_use]
    pub fn dtype_name(&self) -> &'static str {
        self.dtype().name()
    }

    /// Creates another owned handle to the same immutable MLX graph node.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX cannot retain the handle.
    pub fn try_clone(&self) -> Result<Self, MlxError> {
        let mut result = Self::empty();
        check(
            unsafe { sys::mlx_array_set(&raw mut result.0, self.0) },
            "mlx_array_set",
        )?;
        Ok(result)
    }

    /// Materializes and returns one `uint32` scalar.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] when the array is not a `uint32` scalar or MLX
    /// cannot materialize it.
    pub fn item_u32(&self) -> Result<u32, MlxError> {
        if self.dtype() != DType::Uint32 || self.shape() != Vec::<usize>::new() {
            return Err(MlxError::detail(
                "mlx_array_item_uint32",
                format!(
                    "expected uint32 scalar, observed {} {:?}",
                    self.dtype_name(),
                    self.shape()
                ),
            ));
        }
        let mut result = 0;
        check(
            unsafe { sys::mlx_array_item_uint32(&raw mut result, self.0) },
            "mlx_array_item_uint32",
        )?;
        Ok(result)
    }

    fn item_f32(&self) -> Result<f32, MlxError> {
        if self.dtype() != DType::Float32 || self.shape() != Vec::<usize>::new() {
            return Err(MlxError::detail(
                "mlx_array_item_float32",
                format!(
                    "expected float32 scalar, observed {} {:?}",
                    self.dtype_name(),
                    self.shape()
                ),
            ));
        }
        let mut result = 0.0;
        check(
            unsafe { sys::mlx_array_item_float32(&raw mut result, self.0) },
            "mlx_array_item_float32",
        )?;
        Ok(result)
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        let _ = unsafe { sys::mlx_array_free(self.0) };
    }
}

fn optional_array(array: Option<&Array>) -> sys::mlx_array {
    array.map_or(
        sys::mlx_array_ {
            ctx: ptr::null_mut(),
        },
        |value| value.0,
    )
}

fn checked_shape(shape: &[usize], operation: &'static str) -> Result<Vec<i32>, MlxError> {
    shape
        .iter()
        .map(|dimension| {
            i32::try_from(*dimension)
                .map_err(|error| MlxError::detail(operation, error.to_string()))
        })
        .collect()
}

fn checked_i32(value: usize, operation: &'static str) -> Result<i32, MlxError> {
    i32::try_from(value).map_err(|error| MlxError::detail(operation, error.to_string()))
}

struct ArrayVector(sys::mlx_vector_array);

impl ArrayVector {
    fn new() -> Self {
        Self(unsafe { sys::mlx_vector_array_new() })
    }

    fn from_arrays(arrays: &[&Array]) -> Self {
        let handles = arrays.iter().map(|array| array.0).collect::<Vec<_>>();
        Self(unsafe { sys::mlx_vector_array_new_data(handles.as_ptr(), handles.len()) })
    }

    fn get(&self, index: usize) -> Result<Array, MlxError> {
        let mut result = Array::empty();
        check(
            unsafe { sys::mlx_vector_array_get(&raw mut result.0, self.0, index) },
            "mlx_vector_array_get",
        )?;
        Ok(result)
    }

    fn len(&self) -> usize {
        unsafe { sys::mlx_vector_array_size(self.0) }
    }
}

impl Drop for ArrayVector {
    fn drop(&mut self) {
        let _ = unsafe { sys::mlx_vector_array_free(self.0) };
    }
}

struct CompiledStream {
    stream: sys::mlx_stream,
    trace_count: Arc<AtomicUsize>,
}

unsafe extern "C" fn drop_compiled_stream(payload: *mut c_void) {
    if !payload.is_null() {
        drop(unsafe { Box::from_raw(payload.cast::<CompiledStream>()) });
    }
}

unsafe extern "C" fn compiled_silu(
    result: *mut sys::mlx_vector_array,
    inputs: sys::mlx_vector_array,
    payload: *mut c_void,
) -> i32 {
    if result.is_null() || payload.is_null() || unsafe { sys::mlx_vector_array_size(inputs) } != 1 {
        return 1;
    }
    let compiled_stream = unsafe { &*payload.cast::<CompiledStream>() };
    compiled_stream.trace_count.fetch_add(1, Ordering::Relaxed);
    let stream = compiled_stream.stream;
    let mut input = Array::empty();
    if unsafe { sys::mlx_vector_array_get(&raw mut input.0, inputs, 0) } != 0 {
        return 1;
    }
    let mut sigmoid = Array::empty();
    if unsafe { sys::mlx_sigmoid(&raw mut sigmoid.0, input.0, stream) } != 0 {
        return 1;
    }
    let mut output = Array::empty();
    if unsafe { sys::mlx_multiply(&raw mut output.0, input.0, sigmoid.0, stream) } != 0 {
        return 1;
    }
    unsafe { sys::mlx_vector_array_set_value(result, output.0) }
}

unsafe extern "C" fn compiled_swiglu(
    result: *mut sys::mlx_vector_array,
    inputs: sys::mlx_vector_array,
    payload: *mut c_void,
) -> i32 {
    if result.is_null() || payload.is_null() || unsafe { sys::mlx_vector_array_size(inputs) } != 2 {
        return 1;
    }
    let compiled_stream = unsafe { &*payload.cast::<CompiledStream>() };
    compiled_stream.trace_count.fetch_add(1, Ordering::Relaxed);
    let stream = compiled_stream.stream;
    let mut gate = Array::empty();
    let mut input = Array::empty();
    if unsafe { sys::mlx_vector_array_get(&raw mut gate.0, inputs, 0) } != 0
        || unsafe { sys::mlx_vector_array_get(&raw mut input.0, inputs, 1) } != 0
    {
        return 1;
    }
    let mut sigmoid = Array::empty();
    if unsafe { sys::mlx_sigmoid(&raw mut sigmoid.0, gate.0, stream) } != 0 {
        return 1;
    }
    let mut activated = Array::empty();
    if unsafe { sys::mlx_multiply(&raw mut activated.0, gate.0, sigmoid.0, stream) } != 0 {
        return 1;
    }
    let mut output = Array::empty();
    if unsafe { sys::mlx_multiply(&raw mut output.0, activated.0, input.0, stream) } != 0 {
        return 1;
    }
    unsafe { sys::mlx_vector_array_set_value(result, output.0) }
}

unsafe extern "C" fn compiled_precise_swiglu(
    result: *mut sys::mlx_vector_array,
    inputs: sys::mlx_vector_array,
    payload: *mut c_void,
) -> i32 {
    if result.is_null() || payload.is_null() || unsafe { sys::mlx_vector_array_size(inputs) } != 3 {
        return 1;
    }
    let compiled_stream = unsafe { &*payload.cast::<CompiledStream>() };
    compiled_stream.trace_count.fetch_add(1, Ordering::Relaxed);
    let stream = compiled_stream.stream;
    let mut reference = Array::empty();
    let mut gate = Array::empty();
    let mut input = Array::empty();
    if unsafe { sys::mlx_vector_array_get(&raw mut reference.0, inputs, 0) } != 0
        || unsafe { sys::mlx_vector_array_get(&raw mut gate.0, inputs, 1) } != 0
        || unsafe { sys::mlx_vector_array_get(&raw mut input.0, inputs, 2) } != 0
    {
        return 1;
    }
    let output_dtype = unsafe { sys::mlx_array_dtype(reference.0) };
    let mut gate_f32 = Array::empty();
    if unsafe { sys::mlx_astype(&raw mut gate_f32.0, gate.0, DType::Float32.raw(), stream) } != 0 {
        return 1;
    }
    let mut sigmoid = Array::empty();
    if unsafe { sys::mlx_sigmoid(&raw mut sigmoid.0, gate_f32.0, stream) } != 0 {
        return 1;
    }
    let mut activated = Array::empty();
    if unsafe { sys::mlx_multiply(&raw mut activated.0, gate_f32.0, sigmoid.0, stream) } != 0 {
        return 1;
    }
    let mut input_f32 = Array::empty();
    if unsafe { sys::mlx_astype(&raw mut input_f32.0, input.0, DType::Float32.raw(), stream) } != 0
    {
        return 1;
    }
    let mut product = Array::empty();
    if unsafe { sys::mlx_multiply(&raw mut product.0, activated.0, input_f32.0, stream) } != 0 {
        return 1;
    }
    let mut output = Array::empty();
    if unsafe { sys::mlx_astype(&raw mut output.0, product.0, output_dtype, stream) } != 0 {
        return 1;
    }
    unsafe { sys::mlx_vector_array_set_value(result, output.0) }
}

unsafe extern "C" fn compiled_compute_g(
    result: *mut sys::mlx_vector_array,
    inputs: sys::mlx_vector_array,
    payload: *mut c_void,
) -> i32 {
    if result.is_null() || payload.is_null() || unsafe { sys::mlx_vector_array_size(inputs) } != 3 {
        return 1;
    }
    let compiled_stream = unsafe { &*payload.cast::<CompiledStream>() };
    compiled_stream.trace_count.fetch_add(1, Ordering::Relaxed);
    let stream = compiled_stream.stream;
    let mut a_log = Array::empty();
    let mut a = Array::empty();
    let mut dt_bias = Array::empty();
    if unsafe { sys::mlx_vector_array_get(&raw mut a_log.0, inputs, 0) } != 0
        || unsafe { sys::mlx_vector_array_get(&raw mut a.0, inputs, 1) } != 0
        || unsafe { sys::mlx_vector_array_get(&raw mut dt_bias.0, inputs, 2) } != 0
    {
        return 1;
    }
    let mut a_log_f32 = Array::empty();
    if unsafe { sys::mlx_astype(&raw mut a_log_f32.0, a_log.0, DType::Float32.raw(), stream) } != 0
    {
        return 1;
    }
    let mut exponentiated_a_log = Array::empty();
    if unsafe { sys::mlx_exp(&raw mut exponentiated_a_log.0, a_log_f32.0, stream) } != 0 {
        return 1;
    }
    let mut a_with_bias = Array::empty();
    if unsafe { sys::mlx_add(&raw mut a_with_bias.0, a.0, dt_bias.0, stream) } != 0 {
        return 1;
    }
    let zero_f32 = Array(unsafe { sys::mlx_array_new_float32(0.0) });
    let mut zero = Array::empty();
    let a_dtype = unsafe { sys::mlx_array_dtype(a_with_bias.0) };
    if unsafe { sys::mlx_astype(&raw mut zero.0, zero_f32.0, a_dtype, stream) } != 0 {
        return 1;
    }
    let mut softplus = Array::empty();
    if unsafe { sys::mlx_logaddexp(&raw mut softplus.0, a_with_bias.0, zero.0, stream) } != 0 {
        return 1;
    }
    let mut product = Array::empty();
    if unsafe {
        sys::mlx_multiply(
            &raw mut product.0,
            exponentiated_a_log.0,
            softplus.0,
            stream,
        )
    } != 0
    {
        return 1;
    }
    let mut negative = Array::empty();
    if unsafe { sys::mlx_negative(&raw mut negative.0, product.0, stream) } != 0 {
        return 1;
    }
    let mut output = Array::empty();
    if unsafe { sys::mlx_exp(&raw mut output.0, negative.0, stream) } != 0 {
        return 1;
    }
    unsafe { sys::mlx_vector_array_set_value(result, output.0) }
}

struct CompiledClosure {
    closure: sys::mlx_closure,
    trace_count: Arc<AtomicUsize>,
}

impl CompiledClosure {
    fn silu(stream: sys::mlx_stream) -> Result<Self, MlxError> {
        Self::new(stream, compiled_silu, "compiled_silu")
    }

    fn new(
        stream: sys::mlx_stream,
        function: unsafe extern "C" fn(
            *mut sys::mlx_vector_array,
            sys::mlx_vector_array,
            *mut c_void,
        ) -> i32,
        operation: &'static str,
    ) -> Result<Self, MlxError> {
        let trace_count = Arc::new(AtomicUsize::new(0));
        let payload = Box::new(CompiledStream {
            stream: sys::mlx_stream { ctx: stream.ctx },
            trace_count: Arc::clone(&trace_count),
        });
        let closure = unsafe {
            sys::mlx_closure_new_func_payload(
                Some(function),
                Box::into_raw(payload).cast(),
                Some(drop_compiled_stream),
            )
        };
        if closure.ctx.is_null() {
            return Err(MlxError::detail(
                "mlx_closure_new_func_payload",
                "returned an empty closure",
            ));
        }
        let mut compiled = unsafe { sys::mlx_closure_new() };
        let compile_status = unsafe { sys::mlx_compile(&raw mut compiled, closure, true) };
        let free_status = unsafe { sys::mlx_closure_free(closure) };
        if compile_status != 0 {
            let _ = unsafe { sys::mlx_closure_free(compiled) };
            return Err(MlxError::status(operation, compile_status));
        }
        check(free_status, "mlx_closure_free")?;
        if compiled.ctx.is_null() {
            return Err(MlxError::detail(
                operation,
                "returned an empty compiled closure",
            ));
        }
        Ok(Self {
            closure: compiled,
            trace_count,
        })
    }

    fn apply(&self, inputs: &[&Array]) -> Result<Vec<Array>, MlxError> {
        if inputs.is_empty() {
            return Err(MlxError::detail(
                "mlx_closure_apply",
                "at least one input is required",
            ));
        }
        let inputs = ArrayVector::from_arrays(inputs);
        let mut outputs = ArrayVector::new();
        check(
            unsafe { sys::mlx_closure_apply(&raw mut outputs.0, self.closure, inputs.0) },
            "mlx_closure_apply",
        )?;
        (0..outputs.len()).map(|index| outputs.get(index)).collect()
    }

    fn trace_count(&self) -> usize {
        self.trace_count.load(Ordering::Relaxed)
    }
}

impl Drop for CompiledClosure {
    fn drop(&mut self) {
        let _ = unsafe { sys::mlx_closure_free(self.closure) };
    }
}

fn one_compiled_output(operation: &'static str, outputs: Vec<Array>) -> Result<Array, MlxError> {
    let mut outputs = outputs.into_iter();
    let output = outputs
        .next()
        .ok_or_else(|| MlxError::detail(operation, "compiled graph omitted its output"))?;
    if outputs.next().is_some() {
        return Err(MlxError::detail(
            operation,
            "compiled graph returned excess outputs",
        ));
    }
    Ok(output)
}

struct StringVector(sys::mlx_vector_string);

impl StringVector {
    fn new(values: &[&str]) -> Result<Self, MlxError> {
        let values = values
            .iter()
            .map(|value| {
                CString::new(*value)
                    .map_err(|error| MlxError::detail("metal kernel name", error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut pointers = values
            .iter()
            .map(|value| value.as_ptr())
            .collect::<Vec<_>>();
        Ok(Self(unsafe {
            sys::mlx_vector_string_new_data(pointers.as_mut_ptr(), pointers.len())
        }))
    }
}

impl Drop for StringVector {
    fn drop(&mut self) {
        let _ = unsafe { sys::mlx_vector_string_free(self.0) };
    }
}

/// One custom Metal kernel output declaration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetalOutput {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

/// One compile-time Metal kernel template argument.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalTemplate<'a> {
    DType(&'a str, DType),
    Int(&'a str, i32),
    Bool(&'a str, bool),
}

struct MetalKernelConfig(sys::mlx_fast_metal_kernel_config);

impl MetalKernelConfig {
    fn new() -> Result<Self, MlxError> {
        let config = Self(unsafe { sys::mlx_fast_metal_kernel_config_new() });
        if config.0.ctx.is_null() {
            return Err(MlxError::detail(
                "mlx_fast_metal_kernel_config_new",
                "returned an empty config",
            ));
        }
        Ok(config)
    }

    fn add_output(&self, output: &MetalOutput) -> Result<(), MlxError> {
        let shape = checked_shape(&output.shape, "metal output shape")?;
        check(
            unsafe {
                sys::mlx_fast_metal_kernel_config_add_output_arg(
                    self.0,
                    shape.as_ptr(),
                    shape.len(),
                    output.dtype.raw(),
                )
            },
            "mlx_fast_metal_kernel_config_add_output_arg",
        )
    }

    fn set_dispatch(&self, grid: [usize; 3], threadgroup: [usize; 3]) -> Result<(), MlxError> {
        check(
            unsafe {
                sys::mlx_fast_metal_kernel_config_set_grid(
                    self.0,
                    checked_i32(grid[0], "metal grid")?,
                    checked_i32(grid[1], "metal grid")?,
                    checked_i32(grid[2], "metal grid")?,
                )
            },
            "mlx_fast_metal_kernel_config_set_grid",
        )?;
        check(
            unsafe {
                sys::mlx_fast_metal_kernel_config_set_thread_group(
                    self.0,
                    checked_i32(threadgroup[0], "metal threadgroup")?,
                    checked_i32(threadgroup[1], "metal threadgroup")?,
                    checked_i32(threadgroup[2], "metal threadgroup")?,
                )
            },
            "mlx_fast_metal_kernel_config_set_thread_group",
        )
    }

    fn add_template(&self, template: MetalTemplate<'_>) -> Result<(), MlxError> {
        match template {
            MetalTemplate::DType(name, dtype) => {
                let name = CString::new(name)
                    .map_err(|error| MlxError::detail("metal dtype template", error.to_string()))?;
                check(
                    unsafe {
                        sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                            self.0,
                            name.as_ptr(),
                            dtype.raw(),
                        )
                    },
                    "mlx_fast_metal_kernel_config_add_template_arg_dtype",
                )
            }
            MetalTemplate::Int(name, value) => {
                let name = CString::new(name)
                    .map_err(|error| MlxError::detail("metal int template", error.to_string()))?;
                check(
                    unsafe {
                        sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                            self.0,
                            name.as_ptr(),
                            value,
                        )
                    },
                    "mlx_fast_metal_kernel_config_add_template_arg_int",
                )
            }
            MetalTemplate::Bool(name, value) => {
                let name = CString::new(name)
                    .map_err(|error| MlxError::detail("metal bool template", error.to_string()))?;
                check(
                    unsafe {
                        sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
                            self.0,
                            name.as_ptr(),
                            value,
                        )
                    },
                    "mlx_fast_metal_kernel_config_add_template_arg_bool",
                )
            }
        }
    }
}

impl Drop for MetalKernelConfig {
    fn drop(&mut self) {
        unsafe { sys::mlx_fast_metal_kernel_config_free(self.0) };
    }
}

/// Reusable output and dispatch metadata for one fixed custom-kernel shape.
///
/// Keeping this configuration resident avoids rebuilding identical C vectors
/// for every invocation in shape-stable decode loops.
pub struct MetalKernelDispatch {
    config: MetalKernelConfig,
    output_count: usize,
}

/// Ownership-safe wrapper around `mx.fast.metal_kernel`.
pub struct MetalKernel(sys::mlx_fast_metal_kernel);

impl MetalKernel {
    /// Compiles a custom Metal kernel description lazily.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] for strings containing an interior NUL or when
    /// MLX rejects the kernel description.
    pub fn new(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
    ) -> Result<Self, MlxError> {
        let name = CString::new(name)
            .map_err(|error| MlxError::detail("metal kernel name", error.to_string()))?;
        let source = CString::new(source)
            .map_err(|error| MlxError::detail("metal kernel source", error.to_string()))?;
        let header = CString::new("")
            .map_err(|error| MlxError::detail("metal kernel header", error.to_string()))?;
        let input_names = StringVector::new(input_names)?;
        let output_names = StringVector::new(output_names)?;
        let kernel = Self(unsafe {
            sys::mlx_fast_metal_kernel_new(
                name.as_ptr(),
                input_names.0,
                output_names.0,
                source.as_ptr(),
                header.as_ptr(),
                true,
                false,
            )
        });
        if kernel.0.ctx.is_null() {
            return Err(MlxError::detail(
                "mlx_fast_metal_kernel_new",
                "returned an empty kernel",
            ));
        }
        Ok(kernel)
    }

    /// Applies the custom kernel on an MLX GPU stream.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] for invalid dimensions, template names, output
    /// arity, or an MLX compilation/execution failure.
    pub fn apply(
        &self,
        gpu: &Gpu,
        inputs: &[&Array],
        outputs: &[MetalOutput],
        templates: &[MetalTemplate<'_>],
        grid: [usize; 3],
        threadgroup: [usize; 3],
    ) -> Result<Vec<Array>, MlxError> {
        let dispatch = Self::prepare_dispatch(outputs, templates, grid, threadgroup)?;
        self.apply_prepared(gpu, inputs, &dispatch)
    }

    /// Prepares reusable output, template, and dispatch metadata.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] for invalid dimensions, template names, or empty
    /// output declarations.
    pub fn prepare_dispatch(
        outputs: &[MetalOutput],
        templates: &[MetalTemplate<'_>],
        grid: [usize; 3],
        threadgroup: [usize; 3],
    ) -> Result<MetalKernelDispatch, MlxError> {
        if outputs.is_empty() {
            return Err(MlxError::detail(
                "mlx_fast_metal_kernel_prepare",
                "at least one output is required",
            ));
        }
        let config = MetalKernelConfig::new()?;
        for output in outputs {
            config.add_output(output)?;
        }
        config.set_dispatch(grid, threadgroup)?;
        for template in templates {
            config.add_template(*template)?;
        }
        Ok(MetalKernelDispatch {
            config,
            output_count: outputs.len(),
        })
    }

    /// Applies the custom kernel using previously prepared fixed-shape
    /// metadata.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] if MLX rejects the inputs or execution, or returns
    /// an output arity different from the prepared declaration.
    pub fn apply_prepared(
        &self,
        gpu: &Gpu,
        inputs: &[&Array],
        dispatch: &MetalKernelDispatch,
    ) -> Result<Vec<Array>, MlxError> {
        let inputs = ArrayVector::from_arrays(inputs);
        let mut result = ArrayVector::new();
        check(
            unsafe {
                sys::mlx_fast_metal_kernel_apply(
                    &raw mut result.0,
                    self.0,
                    inputs.0,
                    dispatch.config.0,
                    gpu.stream.0,
                )
            },
            "mlx_fast_metal_kernel_apply",
        )?;
        if result.len() != dispatch.output_count {
            return Err(MlxError::detail(
                "mlx_fast_metal_kernel_apply",
                format!(
                    "output arity mismatch: expected {}, observed {}",
                    dispatch.output_count,
                    result.len()
                ),
            ));
        }
        (0..result.len()).map(|index| result.get(index)).collect()
    }
}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe { sys::mlx_fast_metal_kernel_free(self.0) };
    }
}

struct ArrayMap(sys::mlx_map_string_to_array);

impl ArrayMap {
    fn new() -> Self {
        Self(unsafe { sys::mlx_map_string_to_array_new() })
    }
}

impl Drop for ArrayMap {
    fn drop(&mut self) {
        let _ = unsafe { sys::mlx_map_string_to_array_free(self.0) };
    }
}

struct StringMap(sys::mlx_map_string_to_string);

impl StringMap {
    fn new() -> Self {
        Self(unsafe { sys::mlx_map_string_to_string_new() })
    }
}

impl Drop for StringMap {
    fn drop(&mut self) {
        let _ = unsafe { sys::mlx_map_string_to_string_free(self.0) };
    }
}

struct ArrayMapIterator(sys::mlx_map_string_to_array_iterator);

impl ArrayMapIterator {
    fn new(map: sys::mlx_map_string_to_array) -> Self {
        Self(unsafe { sys::mlx_map_string_to_array_iterator_new(map) })
    }
}

impl Drop for ArrayMapIterator {
    fn drop(&mut self) {
        let _ = unsafe { sys::mlx_map_string_to_array_iterator_free(self.0) };
    }
}

struct StringMapIterator(sys::mlx_map_string_to_string_iterator);

impl StringMapIterator {
    fn new(map: sys::mlx_map_string_to_string) -> Self {
        Self(unsafe { sys::mlx_map_string_to_string_iterator_new(map) })
    }
}

impl Drop for StringMapIterator {
    fn drop(&mut self) {
        let _ = unsafe { sys::mlx_map_string_to_string_iterator_free(self.0) };
    }
}

/// Owned tensors and metadata loaded from one safetensors file.
#[derive(Debug)]
pub struct SafeTensors {
    tensors: BTreeMap<String, Array>,
    metadata: BTreeMap<String, String>,
}

impl SafeTensors {
    /// Saves named arrays and metadata as one safetensors payload through MLX C.
    ///
    /// The caller must materialize and synchronize GPU-produced arrays before
    /// using the resulting file as a durable checkpoint.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] for an empty or duplicate tensor map, strings that
    /// cannot cross the C ABI, or an MLX serialization failure.
    pub fn save(
        path: &Path,
        tensors: &[(&str, &Array)],
        metadata: &[(&str, &str)],
    ) -> Result<(), MlxError> {
        if tensors.is_empty() {
            return Err(MlxError::detail(
                "mlx_save_safetensors",
                "at least one tensor is required",
            ));
        }
        let path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|error| MlxError::detail("safetensors path", error.to_string()))?;
        let tensor_map = ArrayMap::new();
        let mut tensor_names = BTreeSet::new();
        for (name, array) in tensors {
            if !tensor_names.insert(*name) {
                return Err(MlxError::detail(
                    "mlx_save_safetensors",
                    format!("duplicate tensor name: {name}"),
                ));
            }
            let name = CString::new(*name)
                .map_err(|error| MlxError::detail("safetensors tensor name", error.to_string()))?;
            check(
                unsafe {
                    sys::mlx_map_string_to_array_insert(tensor_map.0, name.as_ptr(), array.0)
                },
                "mlx_map_string_to_array_insert",
            )?;
        }

        let metadata_map = StringMap::new();
        let mut metadata_names = BTreeSet::new();
        for (name, value) in metadata {
            if !metadata_names.insert(*name) {
                return Err(MlxError::detail(
                    "mlx_save_safetensors",
                    format!("duplicate metadata name: {name}"),
                ));
            }
            let name = CString::new(*name).map_err(|error| {
                MlxError::detail("safetensors metadata name", error.to_string())
            })?;
            let value = CString::new(*value).map_err(|error| {
                MlxError::detail("safetensors metadata value", error.to_string())
            })?;
            check(
                unsafe {
                    sys::mlx_map_string_to_string_insert(
                        metadata_map.0,
                        name.as_ptr(),
                        value.as_ptr(),
                    )
                },
                "mlx_map_string_to_string_insert",
            )?;
        }

        check(
            unsafe { sys::mlx_save_safetensors(path.as_ptr(), tensor_map.0, metadata_map.0) },
            "mlx_save_safetensors",
        )
    }

    /// Loads one safetensors file through MLX C.
    ///
    /// # Errors
    ///
    /// Returns [`MlxError`] for invalid paths, load failures, or malformed map
    /// iteration results.
    pub fn load(path: &Path) -> Result<Self, MlxError> {
        let path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|error| MlxError::detail("safetensors path", error.to_string()))?;
        let stream = CpuStream::new();
        let mut tensors = ArrayMap::new();
        let mut metadata = StringMap::new();
        check(
            unsafe {
                sys::mlx_load_safetensors(
                    &raw mut tensors.0,
                    &raw mut metadata.0,
                    path.as_ptr(),
                    stream.0,
                )
            },
            "mlx_load_safetensors",
        )?;

        Ok(Self {
            tensors: collect_arrays(tensors.0)?,
            metadata: collect_strings(metadata.0)?,
        })
    }

    /// Number of tensors in the file.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the file contains no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Sum of logical tensor bytes.
    #[must_use]
    pub fn total_nbytes(&self) -> usize {
        self.tensors.values().map(Array::nbytes).sum()
    }

    /// Returns a tensor by exact name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&Array> {
        self.tensors.get(name)
    }

    /// Iterates tensor names and handles in deterministic order.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &Array)> {
        self.tensors
            .iter()
            .map(|(name, array)| (name.as_str(), array))
    }

    /// Returns safetensors metadata by exact key.
    #[must_use]
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    /// Iterates metadata in deterministic order.
    pub fn metadata_entries(&self) -> impl Iterator<Item = (&str, &str)> {
        self.metadata
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
    }
}

fn collect_arrays(map: sys::mlx_map_string_to_array) -> Result<BTreeMap<String, Array>, MlxError> {
    let iterator = ArrayMapIterator::new(map);
    let mut result = BTreeMap::new();
    loop {
        let mut key: *const c_char = ptr::null();
        let mut array = Array::empty();
        let status = unsafe {
            sys::mlx_map_string_to_array_iterator_next(&raw mut key, &raw mut array.0, iterator.0)
        };
        if status == ITERATOR_END {
            break;
        }
        check(status, "mlx_map_string_to_array_iterator_next")?;
        if key.is_null() {
            return Err(MlxError::detail(
                "mlx_map_string_to_array_iterator_next",
                "returned a null key",
            ));
        }
        let name = unsafe { CStr::from_ptr(key) }
            .to_str()
            .map_err(|error| {
                MlxError::detail("mlx_map_string_to_array_iterator_next", error.to_string())
            })?
            .to_owned();
        result.insert(name, array);
    }
    Ok(result)
}

fn collect_strings(
    map: sys::mlx_map_string_to_string,
) -> Result<BTreeMap<String, String>, MlxError> {
    let iterator = StringMapIterator::new(map);
    let mut result = BTreeMap::new();
    loop {
        let mut key: *const c_char = ptr::null();
        let mut value: *const c_char = ptr::null();
        let status = unsafe {
            sys::mlx_map_string_to_string_iterator_next(&raw mut key, &raw mut value, iterator.0)
        };
        if status == ITERATOR_END {
            break;
        }
        check(status, "mlx_map_string_to_string_iterator_next")?;
        if key.is_null() || value.is_null() {
            return Err(MlxError::detail(
                "mlx_map_string_to_string_iterator_next",
                "returned a null key or value",
            ));
        }
        let key = unsafe { CStr::from_ptr(key) }
            .to_str()
            .map_err(|error| {
                MlxError::detail("mlx_map_string_to_string_iterator_next", error.to_string())
            })?
            .to_owned();
        let value = unsafe { CStr::from_ptr(value) }
            .to_str()
            .map_err(|error| {
                MlxError::detail("mlx_map_string_to_string_iterator_next", error.to_string())
            })?
            .to_owned();
        result.insert(key, value);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn compiled_kv_update_reshape_probe(
        result: *mut sys::mlx_vector_array,
        inputs: sys::mlx_vector_array,
        payload: *mut c_void,
    ) -> i32 {
        if result.is_null()
            || payload.is_null()
            || unsafe { sys::mlx_vector_array_size(inputs) } != 3
        {
            return 1;
        }
        let compiled_stream = unsafe { &*payload.cast::<CompiledStream>() };
        compiled_stream.trace_count.fetch_add(1, Ordering::Relaxed);
        let stream = compiled_stream.stream;
        let mut source = Array::empty();
        let mut update = Array::empty();
        let mut start = Array::empty();
        if unsafe { sys::mlx_vector_array_get(&raw mut source.0, inputs, 0) } != 0
            || unsafe { sys::mlx_vector_array_get(&raw mut update.0, inputs, 1) } != 0
            || unsafe { sys::mlx_vector_array_get(&raw mut start.0, inputs, 2) } != 0
        {
            return 1;
        }
        let axes = [2];
        let mut updated = Array::empty();
        if unsafe {
            sys::mlx_slice_update_dynamic(
                &raw mut updated.0,
                source.0,
                update.0,
                start.0,
                axes.as_ptr(),
                axes.len(),
                stream,
            )
        } != 0
        {
            return 1;
        }
        let flattened_shape = [-1, 4];
        let mut flattened = Array::empty();
        if unsafe {
            sys::mlx_reshape(
                &raw mut flattened.0,
                updated.0,
                flattened_shape.as_ptr(),
                flattened_shape.len(),
                stream,
            )
        } != 0
        {
            return 1;
        }
        unsafe { sys::mlx_vector_array_set_value(result, flattened.0) }
    }

    fn expected_kv_update(capacity: usize, offset: usize, update: &[f32]) -> Array {
        let head_count = 2;
        let head_dimension = 4;
        let mut values = vec![0.0; head_count * capacity * head_dimension];
        for head in 0..head_count {
            for feature in 0..head_dimension {
                let output_index = (head * capacity + offset) * head_dimension + feature;
                values[output_index] = update[head * head_dimension + feature];
            }
        }
        Array::from_f32_slice(&values, &[head_count * capacity, head_dimension])
            .expect("expected KV update")
    }

    #[test]
    fn reports_metal_allocator_counters() {
        let stats = metal_memory_stats().expect("Metal allocator counters");
        assert!(stats.peak_nbytes >= stats.active_nbytes);
    }

    #[test]
    fn asynchronously_materializes_arrays() {
        let gpu = Gpu::new();
        let input = Array::from_f32_slice(&[1.0, 2.0, 3.0], &[3]).expect("input");
        let output = gpu.cumsum(&input, -1).expect("cumulative sum");
        gpu.async_eval(&[&output])
            .expect("schedule asynchronous evaluation");
        let expected = Array::from_f32_slice(&[1.0, 3.0, 6.0], &[3]).expect("expected");
        let difference = gpu
            .max_abs_difference(&output, &expected)
            .expect("asynchronous evaluation difference");

        assert!(
            difference < f32::EPSILON,
            "asynchronous evaluation drifted by {difference}"
        );
    }

    #[test]
    fn cumulative_sum_includes_the_current_element() {
        let gpu = Gpu::new();
        let input = Array::from_f32_slice(&[1.0, 2.0, 3.0], &[3]).expect("input");
        let expected = Array::from_f32_slice(&[1.0, 3.0, 6.0], &[3]).expect("expected");
        let actual = gpu.cumsum(&input, -1).expect("cumulative sum");
        let difference = gpu
            .max_abs_difference(&actual, &expected)
            .expect("cumulative sum difference");

        assert!(
            difference < f32::EPSILON,
            "inclusive cumulative sum drifted by {difference}"
        );
    }

    #[test]
    fn static_slice_update_preserves_capacity_and_replaces_only_the_target() {
        let gpu = Gpu::new();
        let source = Array::from_f32_slice(&[0.0, 0.0, 0.0, 0.0], &[1, 4]).expect("source");
        let update = Array::from_f32_slice(&[2.0, 3.0], &[1, 2]).expect("update");
        let actual = gpu
            .slice_update(&source, &update, &[0, 1], &[1, 3], &[1, 1])
            .expect("slice update");
        let expected = Array::from_f32_slice(&[0.0, 2.0, 3.0, 0.0], &[1, 4]).expect("expected");
        let difference = gpu
            .max_abs_difference(&actual, &expected)
            .expect("slice-update difference");

        assert_eq!(actual.shape(), vec![1, 4]);
        assert!(
            difference < f32::EPSILON,
            "static slice update drifted by {difference}"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn prepared_metal_dispatch_is_reused_without_numerical_drift() {
        let gpu = Gpu::new();
        let kernel = MetalKernel::new(
            "prepared_dispatch_add_one",
            &["input"],
            &["output"],
            "
                auto index = thread_position_in_grid.x;
                output[index] = input[index] + 1.0f;
            ",
        )
        .expect("prepared-dispatch kernel");
        let dispatch = MetalKernel::prepare_dispatch(
            &[MetalOutput {
                shape: vec![4],
                dtype: DType::Float32,
            }],
            &[],
            [4, 1, 1],
            [4, 1, 1],
        )
        .expect("prepared dispatch");
        let input = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).expect("input");
        let expected = Array::from_f32_slice(&[2.0, 3.0, 4.0, 5.0], &[4]).expect("expected");

        for invocation in 1..=2 {
            let mut outputs = kernel
                .apply_prepared(&gpu, &[&input], &dispatch)
                .expect("prepared dispatch invocation");
            let output = outputs.pop().expect("prepared dispatch output");
            assert!(outputs.is_empty());
            let difference = gpu
                .max_abs_difference(&output, &expected)
                .expect("prepared-dispatch difference");
            assert_eq!(
                difference, 0.0,
                "prepared dispatch invocation {invocation} drifted"
            );
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn compiled_silu_reuses_same_rank_and_retraces_new_rank_exactly() {
        let gpu = Gpu::new();
        let input = Array::from_f32_slice(&[-3.0, -0.5, 0.0, 1.5], &[1, 4]).expect("SiLU input");
        let first = gpu.silu(&input).expect("first compiled SiLU");
        let compiled = gpu
            .silu_function
            .get()
            .map(std::ptr::from_ref)
            .expect("compiled SiLU cache");
        let sigmoid = gpu.sigmoid(&input).expect("reference sigmoid");
        let reference = gpu
            .multiply(&input, &sigmoid)
            .expect("reference SiLU product");
        let first_difference = gpu
            .max_abs_difference(&first, &reference)
            .expect("first SiLU difference");
        assert_eq!(first_difference, 0.0);
        assert_eq!(gpu.compiled_trace_counts().silu, 1);

        let second = gpu.silu(&input).expect("second compiled SiLU");
        let second_difference = gpu
            .max_abs_difference(&second, &reference)
            .expect("second SiLU difference");
        assert_eq!(second_difference, 0.0);
        assert_eq!(gpu.compiled_trace_counts().silu, 1);
        assert_eq!(
            compiled,
            gpu.silu_function
                .get()
                .map(std::ptr::from_ref)
                .expect("reused compiled SiLU cache")
        );

        let reshaped_input =
            Array::from_f32_slice(&[-3.0, -0.5, 0.0, 1.5], &[2, 2]).expect("reshaped input");
        let reshaped = gpu.silu(&reshaped_input).expect("reshaped compiled SiLU");
        let reshaped_sigmoid = gpu
            .sigmoid(&reshaped_input)
            .expect("reshaped reference sigmoid");
        let reshaped_reference = gpu
            .multiply(&reshaped_input, &reshaped_sigmoid)
            .expect("reshaped reference SiLU product");
        let reshaped_difference = gpu
            .max_abs_difference(&reshaped, &reshaped_reference)
            .expect("reshaped SiLU difference");
        assert_eq!(reshaped_difference, 0.0);
        assert_eq!(gpu.compiled_trace_counts().silu, 1);

        let rank_changed_input =
            Array::from_f32_slice(&[-3.0, -0.5, 0.0, 1.5], &[4]).expect("rank-changed input");
        let rank_changed = gpu
            .silu(&rank_changed_input)
            .expect("rank-changed compiled SiLU");
        let rank_changed_sigmoid = gpu
            .sigmoid(&rank_changed_input)
            .expect("rank-changed reference sigmoid");
        let rank_changed_reference = gpu
            .multiply(&rank_changed_input, &rank_changed_sigmoid)
            .expect("rank-changed reference SiLU product");
        let rank_changed_difference = gpu
            .max_abs_difference(&rank_changed, &rank_changed_reference)
            .expect("rank-changed SiLU difference");
        assert_eq!(rank_changed_difference, 0.0);
        assert_eq!(gpu.compiled_trace_counts().silu, 2);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn compiled_dynamic_kv_update_and_reshape_are_shape_polymorphic() {
        let gpu = Gpu::new();
        let function = CompiledClosure::new(
            gpu.stream.0,
            compiled_kv_update_reshape_probe,
            "compiled_kv_update_reshape_probe",
        )
        .expect("compiled KV update probe");
        let update_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let update = Array::from_f32_slice(&update_values, &[1, 2, 1, 4]).expect("KV update");

        for (capacity, offset) in [(8, 2), (8, 5), (16, 9)] {
            let source = Array::from_f32_slice(&vec![0.0; 2 * capacity * 4], &[1, 2, capacity, 4])
                .expect("KV source");
            let start = Array::from_i32_slice(&[i32::try_from(offset).expect("KV offset")], &[1])
                .expect("dynamic KV start");
            let actual = one_compiled_output(
                "compiled_kv_update_reshape_probe",
                function
                    .apply(&[&source, &update, &start])
                    .expect("compiled KV update"),
            )
            .expect("compiled KV output");
            let expected = expected_kv_update(capacity, offset, &update_values);
            let difference = gpu
                .max_abs_difference(&actual, &expected)
                .expect("compiled KV difference");

            assert_eq!(actual.shape(), vec![2 * capacity, 4]);
            assert_eq!(difference, 0.0);
            assert_eq!(
                function.trace_count(),
                1,
                "compatible KV shape retraced at capacity {capacity}"
            );
        }
    }

    #[test]
    fn saves_and_loads_named_arrays_and_metadata() {
        let path = std::env::temp_dir().join(format!(
            "echo-mlx-save-test-{}-{}.safetensors",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let array = Array::from_i32_slice(&[17, 19], &[2]).expect("test array");
        SafeTensors::save(
            &path,
            &[("state.tokens", &array)],
            &[("echo_schema_version", "1")],
        )
        .expect("save");

        let loaded = SafeTensors::load(&path).expect("load");
        let restored = loaded.tensor("state.tokens").expect("restored tensor");
        assert_eq!(restored.shape(), vec![2]);
        assert_eq!(restored.dtype(), DType::Int32);
        assert_eq!(restored.nbytes(), 8);
        assert_eq!(loaded.metadata("echo_schema_version"), Some("1"));

        std::fs::remove_file(path).expect("remove test payload");
    }

    #[test]
    fn rejects_duplicate_tensor_names_before_writing() {
        let array = Array::from_i32_slice(&[1], &[1]).expect("test array");
        let error = SafeTensors::save(
            Path::new("unused.safetensors"),
            &[("duplicate", &array), ("duplicate", &array)],
            &[],
        )
        .expect_err("duplicate name must fail");
        assert!(error.to_string().contains("duplicate tensor name"));
    }
}
