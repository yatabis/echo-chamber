use std::env;
use std::error::Error;
use std::path::Path;
use std::process::ExitCode;

#[cfg(feature = "moe-performance-diagnostics")]
use echo_inference::run_moe_performance_diagnostic;
use echo_inference::{
    LocalServerConfig, inspect_model, produce_durable_state_parity, restore_durable_state_parity,
    run_attention_layer_parity, run_chat_template_parity, run_decoder_layer_parity,
    run_durable_state_parity, run_full_model_parity, run_gdn_layer_parity, run_hybrid_block_parity,
    run_live_state_parity, run_new_session_parity, run_resident_runtime_parity,
    run_sampling_parity, serve_local_stdio,
};
use echo_mlx::SafeTensors;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let command = arguments.next().ok_or("missing command")?;
    match command.as_str() {
        "probe-mlx" => {
            reject_extra_arguments(arguments)?;
            let info = echo_mlx::runtime_info()?;
            println!("mlx_version={}", info.version);
            println!("metal_available={}", info.metal_available);
        }
        "inspect-model" => {
            let model_directory = arguments.next().ok_or("missing model directory")?;
            reject_extra_arguments(arguments)?;
            let inspection = inspect_model(Path::new(&model_directory))?;
            println!("architecture={}", inspection.plan.architecture);
            println!("layers={}", inspection.plan.layer_count);
            println!("recurrent_layers={}", inspection.plan.recurrent_layer_count);
            println!(
                "full_attention_layers={}",
                inspection.plan.full_attention_layer_count
            );
            println!("experts={}", inspection.plan.expert_count);
            println!("experts_per_token={}", inspection.plan.experts_per_token);
            println!("shards={}", inspection.shard_count);
            println!("tensors={}", inspection.tensor_count);
            println!("logical_nbytes={}", inspection.logical_nbytes);
            println!(
                "has_first_gdn_projection={}",
                inspection.has_first_gdn_projection
            );
        }
        "inspect-checkpoint" => {
            inspect_checkpoint_command(&mut arguments)?;
        }
        "run-gdn-layer-parity" => {
            let model_directory = arguments.next().ok_or("missing model directory")?;
            let fixture_path = arguments.next().ok_or("missing fixture path")?;
            let manifest_path = arguments.next().ok_or("missing fixture manifest path")?;
            reject_extra_arguments(arguments)?;
            let result = run_gdn_layer_parity(
                Path::new(&model_directory),
                Path::new(&fixture_path),
                Path::new(&manifest_path),
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            if !result.exact {
                return Err("Rust GDN layer output is not exactly equal to the oracle".into());
            }
        }
        "run-decoder-layer-parity" => {
            let model_directory = arguments.next().ok_or("missing model directory")?;
            let fixture_path = arguments.next().ok_or("missing fixture path")?;
            let manifest_path = arguments.next().ok_or("missing fixture manifest path")?;
            reject_extra_arguments(arguments)?;
            let result = run_decoder_layer_parity(
                Path::new(&model_directory),
                Path::new(&fixture_path),
                Path::new(&manifest_path),
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            if !result.exact {
                return Err("Rust decoder layer output is not exactly equal to the oracle".into());
            }
        }
        "run-attention-layer-parity" => {
            run_attention_layer_command(&mut arguments)?;
        }
        "run-hybrid-block-parity" => {
            run_hybrid_block_command(&mut arguments)?;
        }
        "run-full-model-parity" => {
            run_full_model_command(&mut arguments)?;
        }
        "run-live-state-parity" => {
            run_live_state_command(&mut arguments)?;
        }
        "run-chat-template-parity" => {
            run_chat_template_command(&mut arguments)?;
        }
        "run-sampling-parity" => {
            let fixture_path = arguments.next().ok_or("missing sampling fixture path")?;
            reject_extra_arguments(arguments)?;
            let result = run_sampling_parity(Path::new(&fixture_path))?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            if !result.exact {
                return Err("Rust production sampling differs from the Python/MLX oracle".into());
            }
        }
        #[cfg(feature = "moe-performance-diagnostics")]
        "run-moe-performance-diagnostic" => run_moe_performance_command(&mut arguments)?,
        "serve-stdio" => {
            serve_stdio_command(&mut arguments)?;
        }
        "run-resident-runtime-parity" => run_resident_runtime_command(&mut arguments)?,
        "run-new-session-parity" => run_new_session_command(&mut arguments)?,
        "run-durable-state-parity"
        | "produce-durable-state-parity"
        | "restore-durable-state-parity" => run_durable_command(&command, &mut arguments)?,
        _ => return Err(format!("unknown command: {command}").into()),
    }
    Ok(())
}

#[cfg(feature = "moe-performance-diagnostics")]
fn run_moe_performance_command(
    arguments: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    let model_directory = arguments.next().ok_or("missing model directory")?;
    let warmup_runs = arguments
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    let measured_runs = arguments
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(3);
    let max_new_tokens = arguments
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(128);
    let output_path = arguments.next();
    reject_extra_arguments(arguments)?;
    let result = run_moe_performance_diagnostic(
        Path::new(&model_directory),
        warmup_runs,
        measured_runs,
        max_new_tokens,
    )?;
    let serialized = serde_json::to_string_pretty(&result)?;
    if let Some(output_path) = output_path {
        std::fs::write(&output_path, serialized.as_bytes())?;
        println!("wrote {output_path}");
    } else {
        println!("{serialized}");
    }
    Ok(())
}

fn serve_stdio_command(arguments: &mut impl Iterator<Item = String>) -> Result<(), Box<dyn Error>> {
    let model_directory = arguments.next().ok_or("missing model directory")?;
    let max_outstanding_requests = arguments
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(LocalServerConfig::default().max_outstanding_requests);
    reject_extra_arguments(arguments)?;
    serve_local_stdio(
        Path::new(&model_directory),
        LocalServerConfig {
            max_outstanding_requests,
            ..LocalServerConfig::default()
        },
    )?;
    Ok(())
}

fn run_chat_template_command(
    arguments: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    let model_directory = arguments.next().ok_or("missing model directory")?;
    let fixture_path = arguments
        .next()
        .ok_or("missing chat-template fixture path")?;
    reject_extra_arguments(arguments)?;
    let result = run_chat_template_parity(Path::new(&model_directory), Path::new(&fixture_path))?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    if !result.exact {
        return Err("Rust chat-template or tokenizer output differs from the oracle".into());
    }
    Ok(())
}

fn inspect_checkpoint_command(
    arguments: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    let safetensors_path = arguments.next().ok_or("missing safetensors path")?;
    reject_extra_arguments(arguments)?;
    let checkpoint = SafeTensors::load(Path::new(&safetensors_path))?;
    for (name, value) in checkpoint.metadata_entries() {
        println!("{name}={value}");
    }
    println!("tensors={}", checkpoint.len());
    println!("logical_nbytes={}", checkpoint.total_nbytes());
    Ok(())
}

fn run_durable_command(
    command: &str,
    arguments: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    match command {
        "run-durable-state-parity" => run_durable_state_command(arguments),
        "produce-durable-state-parity" => produce_durable_state_command(arguments),
        "restore-durable-state-parity" => restore_durable_state_command(arguments),
        _ => Err(format!("unknown durable-state command: {command}").into()),
    }
}

fn run_attention_layer_command(
    arguments: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    let model_directory = arguments.next().ok_or("missing model directory")?;
    let fixture_path = arguments.next().ok_or("missing fixture path")?;
    let manifest_path = arguments.next().ok_or("missing fixture manifest path")?;
    reject_extra_arguments(arguments)?;
    let result = run_attention_layer_parity(
        Path::new(&model_directory),
        Path::new(&fixture_path),
        Path::new(&manifest_path),
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    if !result.exact {
        return Err(
            "Rust full-attention decoder layer output is not exactly equal to the oracle".into(),
        );
    }
    Ok(())
}

fn run_hybrid_block_command(
    arguments: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    let model_directory = arguments.next().ok_or("missing model directory")?;
    let fixture_path = arguments.next().ok_or("missing fixture path")?;
    let manifest_path = arguments.next().ok_or("missing fixture manifest path")?;
    reject_extra_arguments(arguments)?;
    let result = run_hybrid_block_parity(
        Path::new(&model_directory),
        Path::new(&fixture_path),
        Path::new(&manifest_path),
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    if !result.exact {
        return Err("Rust hybrid decoder block is not exactly equal to the oracle".into());
    }
    Ok(())
}

fn run_full_model_command(
    arguments: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    let model_directory = arguments.next().ok_or("missing model directory")?;
    let fixture_path = arguments.next().ok_or("missing fixture path")?;
    let manifest_path = arguments.next().ok_or("missing fixture manifest path")?;
    reject_extra_arguments(arguments)?;
    let result = run_full_model_parity(
        Path::new(&model_directory),
        Path::new(&fixture_path),
        Path::new(&manifest_path),
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    if !result.exact {
        return Err(
            "Rust full model or greedy generation is not exactly equal to the oracle".into(),
        );
    }
    Ok(())
}

fn run_live_state_command(
    arguments: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    let model_directory = arguments.next().ok_or("missing model directory")?;
    let fixture_path = arguments.next().ok_or("missing fixture path")?;
    let manifest_path = arguments.next().ok_or("missing fixture manifest path")?;
    reject_extra_arguments(arguments)?;
    let result = run_live_state_parity(
        Path::new(&model_directory),
        Path::new(&fixture_path),
        Path::new(&manifest_path),
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    if !result.exact {
        return Err(
            "committed/restored MLX state is not exactly equal to the direct path and oracle"
                .into(),
        );
    }
    Ok(())
}

fn run_resident_runtime_command(
    arguments: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    let model_directory = arguments.next().ok_or("missing model directory")?;
    let fixture_path = arguments.next().ok_or("missing fixture path")?;
    let manifest_path = arguments.next().ok_or("missing fixture manifest path")?;
    reject_extra_arguments(arguments)?;
    let result = run_resident_runtime_parity(
        Path::new(&model_directory),
        Path::new(&fixture_path),
        Path::new(&manifest_path),
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    if !result.exact {
        return Err(
            "resident runtime did not preserve FIFO, state, rollback, and oracle invariants".into(),
        );
    }
    Ok(())
}

fn run_new_session_command(
    arguments: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    let model_directory = arguments.next().ok_or("missing model directory")?;
    let fixture_path = arguments.next().ok_or("missing fixture path")?;
    let manifest_path = arguments.next().ok_or("missing fixture manifest path")?;
    reject_extra_arguments(arguments)?;
    let result = run_new_session_parity(
        Path::new(&model_directory),
        Path::new(&fixture_path),
        Path::new(&manifest_path),
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    if !result.exact {
        return Err(
            "new-session transition did not preserve GDN, clear KV, and replace lineage exactly"
                .into(),
        );
    }
    Ok(())
}

fn run_durable_state_command(
    arguments: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    let model_directory = arguments.next().ok_or("missing model directory")?;
    let fixture_path = arguments.next().ok_or("missing fixture path")?;
    let manifest_path = arguments.next().ok_or("missing fixture manifest path")?;
    let snapshot_root = arguments.next().ok_or("missing snapshot root")?;
    reject_extra_arguments(arguments)?;
    let executable = env::current_exe()?;
    let result = run_durable_state_parity(
        &executable,
        Path::new(&model_directory),
        Path::new(&fixture_path),
        Path::new(&manifest_path),
        Path::new(&snapshot_root),
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    if !result.exact {
        return Err(
            "durable MLX state did not survive atomic publication and process restart exactly"
                .into(),
        );
    }
    Ok(())
}

fn produce_durable_state_command(
    arguments: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    let model_directory = arguments.next().ok_or("missing model directory")?;
    let fixture_path = arguments.next().ok_or("missing fixture path")?;
    let manifest_path = arguments.next().ok_or("missing fixture manifest path")?;
    let snapshot_root = arguments.next().ok_or("missing snapshot root")?;
    reject_extra_arguments(arguments)?;
    let result = produce_durable_state_parity(
        Path::new(&model_directory),
        Path::new(&fixture_path),
        Path::new(&manifest_path),
        Path::new(&snapshot_root),
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    if !result.exact {
        return Err("durable state producer differed from the admitted oracle".into());
    }
    Ok(())
}

fn restore_durable_state_command(
    arguments: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    let model_directory = arguments.next().ok_or("missing model directory")?;
    let fixture_path = arguments.next().ok_or("missing fixture path")?;
    let manifest_path = arguments.next().ok_or("missing fixture manifest path")?;
    let snapshot_root = arguments.next().ok_or("missing snapshot root")?;
    reject_extra_arguments(arguments)?;
    let result = restore_durable_state_parity(
        Path::new(&model_directory),
        Path::new(&fixture_path),
        Path::new(&manifest_path),
        Path::new(&snapshot_root),
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    if !result.exact {
        return Err("durable state restorer differed from the admitted oracle".into());
    }
    Ok(())
}

fn reject_extra_arguments(
    mut arguments: impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    if let Some(argument) = arguments.next() {
        return Err(format!("unexpected argument: {argument}").into());
    }
    Ok(())
}
