//! Single-current-state, fail-closed ownership for E.C.H.O. instances.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use serde::de::Error as DeserializeError;
use serde::{Deserialize, Deserializer, Serialize};

/// Stable identity of one E.C.H.O. existence.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(transparent)]
pub struct InstanceId(String);

impl InstanceId {
    /// Creates a non-empty instance identifier.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidInstanceId`] when the trimmed value is empty.
    pub fn new(value: impl Into<String>) -> Result<Self, InvalidInstanceId> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(InvalidInstanceId);
        }
        Ok(Self(value))
    }

    /// Returns the wire representation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// An empty instance identifier is never a valid state owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvalidInstanceId;

impl fmt::Display for InvalidInstanceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("instance id must not be empty")
    }
}

impl Error for InvalidInstanceId {}

impl<'de> Deserialize<'de> for InstanceId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(D::Error::custom)
    }
}

/// Compatibility identity that must match before a state can advance.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelIdentity {
    pub architecture: String,
    pub config_digest: String,
    pub weights_digest: String,
    pub tokenizer_digest: String,
    pub template_digest: String,
}

/// State produced by an inference operation but not yet committed.
#[derive(Debug)]
pub struct PreparedState<P> {
    pub model: ModelIdentity,
    pub payload: P,
}

/// The atomically visible inference state for one E.C.H.O. instance.
#[derive(Debug)]
pub struct CommittedState<P> {
    pub instance_id: InstanceId,
    pub model: ModelIdentity,
    pub payload: P,
}

/// Process-local presence precondition for beginning a state transaction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpectedState {
    /// The instance must not have a committed state yet.
    Absent,
    /// The instance must already have its one current state.
    Present,
}

#[derive(Debug)]
struct StoreInner<P> {
    committed: HashMap<InstanceId, Arc<CommittedState<P>>>,
    active_leases: HashMap<InstanceId, u64>,
    next_lease_id: u64,
}

impl<P> Default for StoreInner<P> {
    fn default() -> Self {
        Self {
            committed: HashMap::new(),
            active_leases: HashMap::new(),
            next_lease_id: 1,
        }
    }
}

/// Process-local owner of one current inference state per E.C.H.O. instance.
///
/// The payload is backend-specific, but the lifecycle is fixed: one writer,
/// explicit absent/present preconditions, atomic replacement, model identity
/// continuity, and rollback-on-drop. No version or rollback generation is
/// exposed because same-instance execution is serialized by the owner.
#[derive(Debug)]
pub struct StateStore<P> {
    inner: Arc<Mutex<StoreInner<P>>>,
}

impl<P> Clone for StateStore<P> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<P> Default for StateStore<P> {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(StoreInner::default())),
        }
    }
}

impl<P> StateStore<P> {
    /// Returns the current committed state, if one exists.
    #[must_use]
    pub fn current(&self, instance_id: &InstanceId) -> Option<Arc<CommittedState<P>>> {
        self.lock().committed.get(instance_id).cloned()
    }

    /// Restores one durable current state into an empty process-local slot.
    ///
    /// # Errors
    ///
    /// Fails closed when the instance has an active writer or already has a
    /// process-local state.
    pub fn restore(
        &self,
        committed: CommittedState<P>,
    ) -> Result<Arc<CommittedState<P>>, RestoreError> {
        let mut inner = self.lock();
        if inner.active_leases.contains_key(&committed.instance_id) {
            return Err(RestoreError::Busy {
                instance_id: committed.instance_id,
            });
        }
        if inner.committed.contains_key(&committed.instance_id) {
            return Err(RestoreError::UnexpectedExisting {
                instance_id: committed.instance_id,
            });
        }

        let instance_id = committed.instance_id.clone();
        let committed = Arc::new(committed);
        inner.committed.insert(instance_id, Arc::clone(&committed));
        Ok(committed)
    }

    /// Begins an exclusive transaction for one instance.
    ///
    /// # Errors
    ///
    /// Fails when another writer is active or the caller's presence
    /// precondition does not match the one current state.
    pub fn begin(
        &self,
        instance_id: InstanceId,
        expected: ExpectedState,
    ) -> Result<StateLease<P>, BeginError> {
        let mut inner = self.lock();
        if inner.active_leases.contains_key(&instance_id) {
            return Err(BeginError::Busy {
                instance_id: instance_id.clone(),
            });
        }

        let base = inner.committed.get(&instance_id).cloned();
        match (expected, base.as_ref()) {
            (ExpectedState::Absent, Some(_)) => {
                return Err(BeginError::UnexpectedExisting { instance_id });
            }
            (ExpectedState::Present, None) => {
                return Err(BeginError::Missing { instance_id });
            }
            (ExpectedState::Absent, None) | (ExpectedState::Present, Some(_)) => {}
        }

        let lease_id = inner.next_lease_id;
        inner.next_lease_id = inner.next_lease_id.checked_add(1).unwrap_or(1);
        inner.active_leases.insert(instance_id.clone(), lease_id);
        drop(inner);

        Ok(StateLease {
            store: self.clone(),
            instance_id,
            lease_id,
            base,
            finished: false,
        })
    }

    fn lock(&self) -> MutexGuard<'_, StoreInner<P>> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn release(&self, instance_id: &InstanceId, lease_id: u64) {
        let mut inner = self.lock();
        if inner.active_leases.get(instance_id) == Some(&lease_id) {
            inner.active_leases.remove(instance_id);
        }
    }
}

/// Exclusive pending inference operation for one instance.
#[derive(Debug)]
pub struct StateLease<P> {
    store: StateStore<P>,
    instance_id: InstanceId,
    lease_id: u64,
    base: Option<Arc<CommittedState<P>>>,
    finished: bool,
}

impl<P> StateLease<P> {
    /// Returns the current committed base captured when the lease began.
    #[must_use]
    pub fn base(&self) -> Option<&CommittedState<P>> {
        self.base.as_deref()
    }

    /// Atomically replaces the instance's one current state.
    ///
    /// # Errors
    ///
    /// Fails closed on a lost lease or model identity mismatch. On error,
    /// dropping the lease releases the writer without changing committed state.
    pub fn commit(
        mut self,
        prepared: PreparedState<P>,
    ) -> Result<Arc<CommittedState<P>>, CommitError> {
        let mut inner = self.store.lock();
        if inner.active_leases.get(&self.instance_id) != Some(&self.lease_id) {
            return Err(CommitError::LostLease {
                instance_id: self.instance_id.clone(),
            });
        }
        if self
            .base
            .as_deref()
            .is_some_and(|base| base.model != prepared.model)
        {
            return Err(CommitError::ModelMismatch {
                instance_id: self.instance_id.clone(),
            });
        }

        let committed = Arc::new(CommittedState {
            instance_id: self.instance_id.clone(),
            model: prepared.model,
            payload: prepared.payload,
        });
        inner
            .committed
            .insert(self.instance_id.clone(), Arc::clone(&committed));
        inner.active_leases.remove(&self.instance_id);
        drop(inner);
        self.finished = true;
        Ok(committed)
    }

    /// Explicitly abandons pending work without changing committed state.
    pub fn rollback(mut self) {
        self.store.release(&self.instance_id, self.lease_id);
        self.finished = true;
    }
}

impl<P> Drop for StateLease<P> {
    fn drop(&mut self) {
        if !self.finished {
            self.store.release(&self.instance_id, self.lease_id);
        }
    }
}

/// Failure to acquire a state transaction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BeginError {
    Busy { instance_id: InstanceId },
    Missing { instance_id: InstanceId },
    UnexpectedExisting { instance_id: InstanceId },
}

impl fmt::Display for BeginError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Busy { instance_id } => write!(
                formatter,
                "instance {} already has an active writer",
                instance_id.as_str()
            ),
            Self::Missing { instance_id } => write!(
                formatter,
                "instance {} has no current state",
                instance_id.as_str()
            ),
            Self::UnexpectedExisting { instance_id } => write!(
                formatter,
                "instance {} already has a current state",
                instance_id.as_str()
            ),
        }
    }
}

impl Error for BeginError {}

/// Failure to restore a durable current state into process-local ownership.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RestoreError {
    Busy { instance_id: InstanceId },
    UnexpectedExisting { instance_id: InstanceId },
}

impl fmt::Display for RestoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Busy { instance_id } => write!(
                formatter,
                "instance {} already has an active writer",
                instance_id.as_str()
            ),
            Self::UnexpectedExisting { instance_id } => write!(
                formatter,
                "instance {} already has a current state",
                instance_id.as_str()
            ),
        }
    }
}

impl Error for RestoreError {}

/// Failure to atomically publish a prepared state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CommitError {
    LostLease { instance_id: InstanceId },
    ModelMismatch { instance_id: InstanceId },
}

impl fmt::Display for CommitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LostLease { instance_id } => write!(
                formatter,
                "instance {} lease is no longer active",
                instance_id.as_str()
            ),
            Self::ModelMismatch { instance_id } => write!(
                formatter,
                "instance {} attempted to mix incompatible model state",
                instance_id.as_str()
            ),
        }
    }
}

impl Error for CommitError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Eq, PartialEq)]
    struct Payload(&'static str);

    fn instance(value: &str) -> InstanceId {
        InstanceId::new(value).expect("test instance id is valid")
    }

    fn model(label: &str) -> ModelIdentity {
        ModelIdentity {
            architecture: "qwen3_5_moe".into(),
            config_digest: format!("config-{label}"),
            weights_digest: format!("weights-{label}"),
            tokenizer_digest: "tokenizer".into(),
            template_digest: "template".into(),
        }
    }

    fn prepared(label: &'static str) -> PreparedState<Payload> {
        PreparedState {
            model: model("a"),
            payload: Payload(label),
        }
    }

    fn durable(id: InstanceId, label: &'static str) -> CommittedState<Payload> {
        CommittedState {
            instance_id: id,
            model: model("a"),
            payload: Payload(label),
        }
    }

    #[test]
    fn initializes_and_replaces_one_current_state() {
        let store = StateStore::default();
        let id = instance("echo:rin");

        let first = store
            .begin(id.clone(), ExpectedState::Absent)
            .expect("cold lease")
            .commit(prepared("first"))
            .expect("first commit");
        assert_eq!(first.payload, Payload("first"));

        let second_lease = store
            .begin(id.clone(), ExpectedState::Present)
            .expect("continuation lease");
        assert_eq!(
            second_lease.base().expect("base state exists").payload,
            Payload("first")
        );
        let second = second_lease
            .commit(prepared("second"))
            .expect("second commit");

        assert_eq!(second.payload, Payload("second"));
        assert_eq!(
            store.current(&id).expect("current state").payload,
            Payload("second")
        );
    }

    #[test]
    fn restores_current_state_then_replaces_it_normally() {
        let store = StateStore::default();
        let id = instance("echo:rin");
        let restored = store
            .restore(durable(id.clone(), "restored"))
            .expect("durable restore");
        assert_eq!(restored.payload, Payload("restored"));

        let advanced = store
            .begin(id, ExpectedState::Present)
            .expect("lease from durable base")
            .commit(prepared("advanced"))
            .expect("commit after restore");
        assert_eq!(advanced.payload, Payload("advanced"));
    }

    #[test]
    fn durable_restore_never_replaces_live_state_or_active_writer() {
        let store = StateStore::default();
        let id = instance("echo:rin");
        let lease = store
            .begin(id.clone(), ExpectedState::Absent)
            .expect("active writer");
        assert!(matches!(
            store.restore(durable(id.clone(), "busy")),
            Err(RestoreError::Busy { instance_id }) if instance_id == id
        ));
        lease.rollback();

        store
            .restore(durable(id.clone(), "first"))
            .expect("first restore");
        assert!(matches!(
            store.restore(durable(id.clone(), "replacement")),
            Err(RestoreError::UnexpectedExisting { instance_id }) if instance_id == id
        ));
        assert_eq!(
            store
                .current(&id)
                .expect("original restore remains")
                .payload,
            Payload("first")
        );
    }

    #[test]
    fn rollback_after_failure_preserves_commit_and_releases_writer() {
        let store = StateStore::default();
        let id = instance("echo:rin");
        store
            .begin(id.clone(), ExpectedState::Absent)
            .expect("cold lease")
            .commit(prepared("committed"))
            .expect("commit");

        let lease = store
            .begin(id.clone(), ExpectedState::Present)
            .expect("pending lease");
        drop(lease);

        assert_eq!(
            store.current(&id).expect("committed state remains").payload,
            Payload("committed")
        );
        store
            .begin(id, ExpectedState::Present)
            .expect("writer was released")
            .rollback();
    }

    #[test]
    fn rejects_concurrent_writer_for_same_instance_but_not_another_instance() {
        let store = StateStore::<Payload>::default();
        let rin = instance("echo:rin");
        let marie = instance("echo:marie");

        let rin_lease = store
            .begin(rin.clone(), ExpectedState::Absent)
            .expect("rin lease");
        assert!(matches!(
            store.begin(rin.clone(), ExpectedState::Absent),
            Err(BeginError::Busy { instance_id }) if instance_id == rin
        ));
        store
            .begin(marie, ExpectedState::Absent)
            .expect("another instance is isolated")
            .rollback();
        rin_lease.rollback();
    }

    #[test]
    fn presence_preconditions_fail_closed() {
        let store = StateStore::default();
        let id = instance("echo:rin");
        assert!(matches!(
            store.begin(id.clone(), ExpectedState::Present),
            Err(BeginError::Missing { .. })
        ));
        store
            .begin(id.clone(), ExpectedState::Absent)
            .expect("initial lease")
            .commit(prepared("first"))
            .expect("initial commit");
        assert!(matches!(
            store.begin(id, ExpectedState::Absent),
            Err(BeginError::UnexpectedExisting { .. })
        ));
    }

    #[test]
    fn model_mismatch_does_not_replace_current_state() {
        let store = StateStore::default();
        let id = instance("echo:rin");
        store
            .begin(id.clone(), ExpectedState::Absent)
            .expect("cold lease")
            .commit(prepared("first"))
            .expect("commit");

        let mut incompatible = prepared("wrong-model");
        incompatible.model = model("b");
        let error = store
            .begin(id.clone(), ExpectedState::Present)
            .expect("lease")
            .commit(incompatible)
            .expect_err("model mismatch");
        assert!(matches!(error, CommitError::ModelMismatch { .. }));
        assert_eq!(
            store.current(&id).expect("original commit remains").payload,
            Payload("first")
        );
    }

    #[test]
    fn instance_state_never_crosses_owner_boundary() {
        let store = StateStore::default();
        let rin = instance("echo:rin");
        let marie = instance("echo:marie");

        store
            .begin(rin.clone(), ExpectedState::Absent)
            .expect("rin lease")
            .commit(prepared("rin"))
            .expect("rin commit");
        store
            .begin(marie.clone(), ExpectedState::Absent)
            .expect("marie lease")
            .commit(prepared("marie"))
            .expect("marie commit");

        assert_eq!(store.current(&rin).expect("rin").payload, Payload("rin"));
        assert_eq!(
            store.current(&marie).expect("marie").payload,
            Payload("marie")
        );
    }

    #[test]
    fn wire_identity_rejects_an_empty_instance() {
        let error = serde_json::from_str::<InstanceId>(r#""""#).expect_err("empty owner must fail");
        assert!(error.to_string().contains("must not be empty"));
    }
}
