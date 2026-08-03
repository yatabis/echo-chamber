use std::env;
use std::path::{Path, PathBuf};

fn required_directory(name: &str) -> PathBuf {
    let value = env::var_os(name).unwrap_or_else(|| panic!("{name} must be set"));
    let path = PathBuf::from(value);
    assert!(
        path.is_dir(),
        "{name} is not a directory: {}",
        path.display()
    );
    path
}

fn emit_link_search(path: &Path) {
    println!("cargo:rustc-link-search=native={}", path.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
}

fn main() {
    let include_dir = required_directory("MLX_C_INCLUDE_DIR");
    let mlx_c_lib_dir = required_directory("MLX_C_LIB_DIR");
    let mlx_lib_dir = required_directory("MLX_LIB_DIR");
    let header = include_dir.join("mlx/c/mlx.h");
    assert!(
        header.is_file(),
        "MLX_C_INCLUDE_DIR does not contain mlx/c/mlx.h: {}",
        header.display()
    );

    println!("cargo:rerun-if-env-changed=MLX_C_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=MLX_C_LIB_DIR");
    println!("cargo:rerun-if-env-changed=MLX_LIB_DIR");
    println!("cargo:rerun-if-changed={}", header.display());
    emit_link_search(&mlx_c_lib_dir);
    emit_link_search(&mlx_lib_dir);
    println!("cargo:rustc-link-lib=dylib=mlxc");
    println!("cargo:rustc-link-lib=dylib=mlx");

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .clang_arg(format!("-I{}", include_dir.display()))
        .allowlist_function("mlx_array_.*")
        .allowlist_function("mlx_default_cpu_stream_new")
        .allowlist_function("mlx_default_gpu_stream_new")
        .allowlist_function("mlx_compile")
        .allowlist_function("mlx_closure_.*")
        .allowlist_function("mlx_async_eval")
        .allowlist_function("mlx_eval")
        .allowlist_function(
            "mlx_(abs|add|arange|argmax_axis|argpartition_axis|argsort|argsort_axis|astype|concatenate_axis|contiguous|conv1d|cumsum|dequantize|divide|exp|fast_.*|floor_divide|gather_qmm|greater|logaddexp|logsumexp_axis|max|multiply|negative|put_along_axis|quantized_matmul|reshape|sigmoid|slice|slice_update|softmax_axis|subtract|sum_axis|take|take_along_axis|take_axis|transpose_axes|vector_array_.*|vector_string_.*|where|zeros|zeros_like)",
        )
        .allowlist_function("mlx_slice_update_dynamic")
        .allowlist_function("mlx_random_(categorical|key)")
        .allowlist_function("mlx_(load|save)_safetensors")
        .allowlist_function("mlx_map_string_to_array_.*")
        .allowlist_function("mlx_map_string_to_string_.*")
        .allowlist_function("mlx_get_(active|cache|peak)_memory")
        .allowlist_function("mlx_metal_is_available")
        .allowlist_function("mlx_stream_free")
        .allowlist_function("mlx_synchronize")
        .allowlist_function("mlx_string_.*")
        .allowlist_function("mlx_version")
        .allowlist_type("mlx_array")
        .allowlist_type("mlx_closure")
        .allowlist_type("mlx_dtype.*")
        .allowlist_type("mlx_fast_metal_kernel.*")
        .allowlist_type("mlx_map_string_to_array.*")
        .allowlist_type("mlx_map_string_to_string.*")
        .allowlist_type("mlx_optional_int")
        .allowlist_type("mlx_optional_float")
        .allowlist_type("mlx_optional_dtype")
        .allowlist_type("mlx_stream")
        .allowlist_type("mlx_string")
        .allowlist_type("mlx_vector_array")
        .allowlist_type("mlx_vector_string")
        .allowlist_var("MLX_.*")
        .derive_debug(true)
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("official MLX C headers must generate Rust bindings");

    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set"));
    bindings
        .write_to_file(output.join("bindings.rs"))
        .expect("generated MLX bindings must be writable");
}
