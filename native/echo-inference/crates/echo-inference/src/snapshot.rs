//! Crash-consistent storage for one current state per E.C.H.O. instance.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use echo_inference_state::{CommittedState, InstanceId, ModelIdentity};
use echo_mlx::{Array, Gpu, SafeTensors};

use super::model_state::{LayerState, MlxInferenceState};
use super::{EngineError, ModelPlan};

/// Durable format understood by this implementation.
const DURABLE_SCHEMA_VERSION: u32 = 1;
/// The sole authoritative state payload inside one instance directory.
pub const CURRENT_STATE_FILE: &str = "current.safetensors";
const OWNER_LOCK_FILE: &str = ".owner.lock";
const STAGING_PREFIX: &str = ".current.safetensors.tmp-";
const STAGING_SUFFIX: &str = ".safetensors";
const LEGACY_POINTER_FILE: &str = "current.json";
const SCHEMA_METADATA: &str = "echo_schema_version";
const INSTANCE_METADATA: &str = "echo_instance_id";
const MODEL_METADATA: &str = "echo_model_identity";
static STAGING_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Result of atomically replacing an instance's durable current state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublishedMlxCheckpoint {
    /// Fixed path that now names the committed state.
    pub path: PathBuf,
    /// On-disk byte length of the safetensors payload.
    pub physical_nbytes: u64,
    /// E.C.H.O. existence bound into the payload metadata.
    pub instance_id: InstanceId,
}

/// Authenticated durable metadata plus newly owned MLX state handles.
#[derive(Debug)]
pub struct RestoredMlxCheckpoint {
    pub instance_id: InstanceId,
    pub model: ModelIdentity,
    pub state: MlxInferenceState,
}

/// Exclusive process-lifetime ownership of one instance's durable state root.
///
/// The advisory lock is deliberately retained for as long as this value is
/// retained by the resident engine. Publishing and loading through a borrowed
/// owner therefore cannot race another conforming engine process.
#[derive(Debug)]
pub struct CurrentStateOwner {
    root: PathBuf,
    _lock: File,
}

impl CurrentStateOwner {
    /// Acquires exclusive ownership of an instance state directory.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] when the directory cannot be created, a second
    /// owner already holds it, or legacy `current.json` authority is present.
    pub fn acquire(root: &Path) -> Result<Self, EngineError> {
        fs::create_dir_all(root).map_err(|source| EngineError::Io {
            path: root.to_path_buf(),
            source,
        })?;
        sync_directory(root)?;
        if let Some(parent) = root.parent() {
            sync_directory(parent)?;
        }

        let lock_path = root.join(OWNER_LOCK_FILE);
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)
            .map_err(|source| EngineError::Io {
                path: lock_path.clone(),
                source,
            })?;
        lock.try_lock().map_err(|source| EngineError::Io {
            path: lock_path,
            source: source.into(),
        })?;

        let legacy_pointer = root.join(LEGACY_POINTER_FILE);
        if legacy_pointer
            .try_exists()
            .map_err(|source| EngineError::Io {
                path: legacy_pointer.clone(),
                source,
            })?
        {
            return Err(EngineError::Unsupported(format!(
                "legacy durable authority {} must be archived or migrated before opening this instance",
                legacy_pointer.display()
            )));
        }

        prune_stale_staging_files(root)?;
        Ok(Self {
            root: root.to_path_buf(),
            _lock: lock,
        })
    }

    /// Returns the directory exclusively owned by this process.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Returns the fixed authoritative payload path.
    #[must_use]
    pub fn current_path(&self) -> PathBuf {
        self.root.join(CURRENT_STATE_FILE)
    }

    /// Loads the current state when one has previously been published.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] for malformed safetensors, metadata identity
    /// drift, missing or extra tensors, or an invalid model-specific layout.
    pub fn load_current(
        &self,
        instance_id: &InstanceId,
        plan: &ModelPlan,
        expected_model: &ModelIdentity,
        batch_size: usize,
    ) -> Result<Option<RestoredMlxCheckpoint>, EngineError> {
        let path = self.current_path();
        match path.try_exists() {
            Ok(false) => return Ok(None),
            Ok(true) => {}
            Err(source) => return Err(EngineError::Io { path, source }),
        }

        let tensors = SafeTensors::load(&path).map_err(EngineError::Mlx)?;
        let expected_metadata = embedded_metadata(instance_id, expected_model)?;
        let observed_metadata = tensors
            .metadata_entries()
            .map(|(name, value)| (name.to_owned(), value.to_owned()))
            .collect::<BTreeMap<_, _>>();
        if observed_metadata != expected_metadata {
            return Err(EngineError::Unsupported(format!(
                "durable state metadata at {} does not match this instance and model",
                path.display()
            )));
        }
        if tensors.len() != plan.layer_count * 2 {
            return Err(EngineError::Unsupported(format!(
                "durable state tensor count mismatch: expected {}, observed {}",
                plan.layer_count * 2,
                tensors.len()
            )));
        }

        let state = restore_named_state(&tensors, plan)?;
        state.validate(plan, batch_size)?;
        Ok(Some(RestoredMlxCheckpoint {
            instance_id: instance_id.clone(),
            model: expected_model.clone(),
            state,
        }))
    }

    /// Materializes and atomically replaces `current.safetensors`.
    ///
    /// The temporary payload is synchronized before rename, then the state
    /// directory is synchronized after rename. A crash therefore exposes
    /// either the preceding complete file or the replacement complete file.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError`] for model/layout mismatch, MLX evaluation or
    /// serialization failure, or a filesystem durability failure.
    pub fn publish(
        &self,
        committed: &CommittedState<MlxInferenceState>,
        plan: &ModelPlan,
        batch_size: usize,
        gpu: &Gpu,
    ) -> Result<PublishedMlxCheckpoint, EngineError> {
        if committed.model.architecture != plan.architecture
            || !model_identity_is_complete(&committed.model)
        {
            return Err(EngineError::Unsupported(
                "durable state model identity does not match the admitted plan".into(),
            ));
        }
        committed.payload.validate(plan, batch_size)?;

        let named_tensors = named_state_tensors(&committed.payload);
        if named_tensors.len() != plan.layer_count * 2 {
            return Err(EngineError::Unsupported(format!(
                "durable state tensor count mismatch: expected {}, observed {}",
                plan.layer_count * 2,
                named_tensors.len()
            )));
        }
        let arrays = named_tensors
            .iter()
            .map(|(_, array)| *array)
            .collect::<Vec<_>>();
        gpu.eval(&arrays).map_err(EngineError::Mlx)?;
        gpu.synchronize().map_err(EngineError::Mlx)?;

        let metadata = embedded_metadata(&committed.instance_id, &committed.model)?;
        let tensor_references = named_tensors
            .iter()
            .map(|(name, array)| (name.as_str(), *array))
            .collect::<Vec<_>>();
        let metadata_references = metadata
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect::<Vec<_>>();
        let mut staging = StagingFile::allocate(&self.root)?;
        SafeTensors::save(&staging.path, &tensor_references, &metadata_references)
            .map_err(EngineError::Mlx)?;
        sync_file(&staging.path)?;

        let destination = self.current_path();
        fs::rename(&staging.path, &destination).map_err(|source| EngineError::Io {
            path: destination.clone(),
            source,
        })?;
        staging.published = true;
        sync_directory(&self.root)?;
        let physical_nbytes = fs::metadata(&destination)
            .map_err(|source| EngineError::Io {
                path: destination.clone(),
                source,
            })?
            .len();
        Ok(PublishedMlxCheckpoint {
            path: destination,
            physical_nbytes,
            instance_id: committed.instance_id.clone(),
        })
    }
}

fn named_state_tensors(state: &MlxInferenceState) -> Vec<(String, &Array)> {
    let mut tensors = Vec::with_capacity(state.tensor_count());
    for (layer_index, layer) in state.layers().iter().enumerate() {
        match layer {
            LayerState::Gdn {
                convolution,
                recurrent,
            } => {
                tensors.push((tensor_name(layer_index, "gdn", "convolution"), convolution));
                tensors.push((tensor_name(layer_index, "gdn", "recurrent"), recurrent));
            }
            LayerState::Attention { keys, values } => {
                tensors.push((tensor_name(layer_index, "attention", "keys"), keys));
                tensors.push((tensor_name(layer_index, "attention", "values"), values));
            }
        }
    }
    tensors
}

pub(crate) fn restore_named_state(
    tensors: &SafeTensors,
    plan: &ModelPlan,
) -> Result<MlxInferenceState, EngineError> {
    let expected_names = (0..plan.layer_count)
        .flat_map(|layer_index| {
            if (layer_index + 1).is_multiple_of(plan.full_attention_interval) {
                [
                    tensor_name(layer_index, "attention", "keys"),
                    tensor_name(layer_index, "attention", "values"),
                ]
            } else {
                [
                    tensor_name(layer_index, "gdn", "convolution"),
                    tensor_name(layer_index, "gdn", "recurrent"),
                ]
            }
        })
        .collect::<Vec<_>>();
    let observed_names = tensors
        .tensors()
        .map(|(name, _)| name.to_owned())
        .collect::<Vec<_>>();
    if observed_names != expected_names {
        return Err(EngineError::Unsupported(
            "durable state tensor names differ from the admitted hybrid layout".into(),
        ));
    }

    let mut layers = Vec::with_capacity(plan.layer_count);
    for layer_index in 0..plan.layer_count {
        if (layer_index + 1).is_multiple_of(plan.full_attention_interval) {
            layers.push(LayerState::Attention {
                keys: require_tensor(tensors, &tensor_name(layer_index, "attention", "keys"))?,
                values: require_tensor(tensors, &tensor_name(layer_index, "attention", "values"))?,
            });
        } else {
            layers.push(LayerState::Gdn {
                convolution: require_tensor(
                    tensors,
                    &tensor_name(layer_index, "gdn", "convolution"),
                )?,
                recurrent: require_tensor(tensors, &tensor_name(layer_index, "gdn", "recurrent"))?,
            });
        }
    }
    Ok(MlxInferenceState::new(layers))
}

fn require_tensor(tensors: &SafeTensors, name: &str) -> Result<Array, EngineError> {
    tensors
        .tensor(name)
        .ok_or_else(|| EngineError::Unsupported(format!("durable state is missing {name}")))?
        .try_clone()
        .map_err(EngineError::Mlx)
}

fn tensor_name(layer_index: usize, kind: &str, component: &str) -> String {
    format!("layer.{layer_index:02}.{kind}.{component}")
}

fn embedded_metadata(
    instance_id: &InstanceId,
    model: &ModelIdentity,
) -> Result<BTreeMap<String, String>, EngineError> {
    let model = serde_json::to_string(model).map_err(|source| EngineError::Json {
        path: PathBuf::from("<embedded model identity>"),
        source,
    })?;
    Ok([
        (
            SCHEMA_METADATA.to_owned(),
            DURABLE_SCHEMA_VERSION.to_string(),
        ),
        (
            INSTANCE_METADATA.to_owned(),
            instance_id.as_str().to_owned(),
        ),
        (MODEL_METADATA.to_owned(), model),
    ]
    .into_iter()
    .collect())
}

fn model_identity_is_complete(identity: &ModelIdentity) -> bool {
    !identity.architecture.trim().is_empty()
        && is_sha256(&identity.config_digest)
        && is_sha256(&identity.weights_digest)
        && is_sha256(&identity.tokenizer_digest)
        && is_sha256(&identity.template_digest)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn prune_stale_staging_files(root: &Path) -> Result<(), EngineError> {
    for entry in fs::read_dir(root).map_err(|source| EngineError::Io {
        path: root.to_path_buf(),
        source,
    })? {
        let entry = entry.map_err(|source| EngineError::Io {
            path: root.to_path_buf(),
            source,
        })?;
        let name = entry.file_name();
        if name.to_string_lossy().starts_with(STAGING_PREFIX) {
            let path = entry.path();
            fs::remove_file(&path).map_err(|source| EngineError::Io { path, source })?;
        }
    }
    Ok(())
}

fn sync_file(path: &Path) -> Result<(), EngineError> {
    File::open(path)
        .and_then(|file| file.sync_all())
        .map_err(|source| EngineError::Io {
            path: path.to_path_buf(),
            source,
        })
}

fn sync_directory(path: &Path) -> Result<(), EngineError> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| EngineError::Io {
            path: path.to_path_buf(),
            source,
        })
}

struct StagingFile {
    path: PathBuf,
    published: bool,
}

impl StagingFile {
    fn allocate(root: &Path) -> Result<Self, EngineError> {
        for _ in 0..128 {
            let counter = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
            // MLX appends `.safetensors` when the supplied path does not
            // already have that extension. Keep it explicit so the path we
            // synchronize and atomically rename is the file MLX actually
            // writes.
            let path = root.join(format!(
                "{STAGING_PREFIX}{}-{counter}{STAGING_SUFFIX}",
                std::process::id()
            ));
            match path.try_exists() {
                Ok(false) => {
                    return Ok(Self {
                        path,
                        published: false,
                    });
                }
                Ok(true) => {}
                Err(source) if source.kind() == ErrorKind::NotFound => {}
                Err(source) => return Err(EngineError::Io { path, source }),
            }
        }
        Err(EngineError::Unsupported(
            "could not allocate a unique current-state staging file".into(),
        ))
    }
}

impl Drop for StagingFile {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_root(label: &str) -> PathBuf {
        let counter = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "echo-current-state-{label}-{}-{counter}",
            std::process::id()
        ))
    }

    #[test]
    fn owner_lock_is_retained_until_owner_is_dropped() {
        let root = test_root("lock");
        let owner = CurrentStateOwner::acquire(&root).expect("first owner");
        let error = CurrentStateOwner::acquire(&root).expect_err("second owner must fail");
        assert!(matches!(error, EngineError::Io { .. }));
        drop(owner);
        CurrentStateOwner::acquire(&root).expect("lock released on drop");
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn owner_rejects_legacy_authority_without_deleting_it() {
        let root = test_root("legacy");
        fs::create_dir_all(&root).expect("root");
        fs::write(root.join(LEGACY_POINTER_FILE), b"{}").expect("legacy pointer");
        let error = CurrentStateOwner::acquire(&root).expect_err("legacy authority must fail");
        assert!(matches!(error, EngineError::Unsupported(_)));
        assert!(root.join(LEGACY_POINTER_FILE).exists());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn owner_removes_only_managed_staging_files() {
        let root = test_root("staging");
        fs::create_dir_all(&root).expect("root");
        let stale = root.join(format!("{STAGING_PREFIX}stale{STAGING_SUFFIX}"));
        let unknown = root.join("notes.txt");
        fs::write(&stale, b"partial").expect("stale staging");
        fs::write(&unknown, b"keep").expect("unknown file");

        let owner = CurrentStateOwner::acquire(&root).expect("owner");
        assert!(!stale.exists());
        assert!(unknown.exists());
        drop(owner);
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn staging_path_is_the_exact_file_written_by_mlx() {
        let root = test_root("mlx-staging-path");
        fs::create_dir_all(&root).expect("root");
        let staging = StagingFile::allocate(&root).expect("staging path");
        assert_eq!(
            staging.path.extension().and_then(|value| value.to_str()),
            Some("safetensors")
        );

        let array = Array::from_i32_slice(&[17], &[1]).expect("test tensor");
        SafeTensors::save(&staging.path, &[("state", &array)], &[]).expect("save staging");
        assert!(staging.path.is_file());

        let staging_path = staging.path.clone();
        drop(staging);
        assert!(!staging_path.exists());
        fs::remove_dir_all(root).expect("cleanup");
    }

    #[test]
    fn durable_metadata_is_minimal_and_composite() {
        let instance = InstanceId::new("echo:rin").expect("instance");
        let digest = "a".repeat(64);
        let model = ModelIdentity {
            architecture: "qwen3_5_moe".into(),
            config_digest: digest.clone(),
            weights_digest: digest.clone(),
            tokenizer_digest: digest.clone(),
            template_digest: digest,
        };
        let metadata = embedded_metadata(&instance, &model).expect("metadata");
        assert_eq!(metadata.len(), 3);
        assert_eq!(metadata[SCHEMA_METADATA], "1");
        assert_eq!(metadata[INSTANCE_METADATA], "echo:rin");
        assert_eq!(
            serde_json::from_str::<ModelIdentity>(&metadata[MODEL_METADATA]).expect("identity"),
            model
        );
    }
}
