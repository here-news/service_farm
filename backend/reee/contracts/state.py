"""
State Contracts - Kernel computation state.

These represent the current state of surfaces/incidents
that the kernel operates on.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, Dict, FrozenSet, Tuple
import hashlib


@dataclass(frozen=True)
class SurfaceKey:
    """Stable identity for a surface.

    Surface identity = (scope_id, question_key)
    - scope_id: derived from anchor entities (which real-world referent)
    - question_key: semantic type of question (what is being asked)
    """

    scope_id: str
    question_key: str

    @property
    def signature(self) -> str:
        """Deterministic merge key for persistence.

        Used for MERGE operations to ensure convergence.
        """
        content = f"surface|{self.scope_id}|{self.question_key}"
        return f"sf_{hashlib.sha256(content.encode()).hexdigest()[:12]}"

    def __str__(self) -> str:
        return f"{self.scope_id}::{self.question_key}"


@dataclass
class SurfaceState:
    """Surface state for kernel computation.

    Mutable during kernel.step(), but immutable in persistence.
    """

    # Identity (stable)
    key: SurfaceKey

    # Claim membership
    claim_ids: FrozenSet[str] = frozenset()

    # Entity sets
    entities: FrozenSet[str] = frozenset()
    anchor_entities: FrozenSet[str] = frozenset()
    sources: FrozenSet[str] = frozenset()

    # Time bounds
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None

    # Belief state (Jaynes)
    posterior_entropy: float = 0.0
    posterior_map: Optional[Any] = None
    observation_count: int = 0

    # Versioning
    kernel_version: str = ""
    params_hash: str = ""

    @property
    def signature(self) -> str:
        """Delegate to key's signature."""
        return self.key.signature

    @property
    def scope_id(self) -> str:
        """Delegate to key's scope_id."""
        return self.key.scope_id

    @property
    def question_key(self) -> str:
        """Delegate to key's question_key."""
        return self.key.question_key

    @property
    def companion_entities(self) -> FrozenSet[str]:
        """Entities that are not anchors."""
        return self.entities - self.anchor_entities

    def with_claim(
        self,
        claim_id: str,
        entities: FrozenSet[str],
        anchors: FrozenSet[str],
        source_id: Optional[str],
        claim_time: Optional[datetime],
    ) -> "SurfaceState":
        """Create new SurfaceState with claim added."""
        new_sources = self.sources | ({source_id} if source_id else set())

        # Update time bounds
        new_time_start = self.time_start
        new_time_end = self.time_end
        if claim_time:
            if new_time_start is None or claim_time < new_time_start:
                new_time_start = claim_time
            if new_time_end is None or claim_time > new_time_end:
                new_time_end = claim_time

        return SurfaceState(
            key=self.key,
            claim_ids=self.claim_ids | {claim_id},
            entities=self.entities | entities,
            anchor_entities=self.anchor_entities | anchors,
            sources=frozenset(new_sources),
            time_start=new_time_start,
            time_end=new_time_end,
            posterior_entropy=self.posterior_entropy,
            posterior_map=self.posterior_map,
            observation_count=self.observation_count,
            kernel_version=self.kernel_version,
            params_hash=self.params_hash,
        )


def compute_incident_signature(
    anchors: FrozenSet[str], time_start: Optional[datetime]
) -> str:
    """Compute deterministic incident signature.

    NOTE: time_start (not bin) used for signature.
    Sliding window is for membership, not identity.
    """
    sorted_anchors = ",".join(sorted(anchors)[:10])
    # Use ISO week for time component (coarse enough to be stable)
    time_component = "unknown"
    if time_start:
        time_component = f"{time_start.year}-W{time_start.isocalendar()[1]:02d}"
    content = f"incident|{sorted_anchors}|{time_component}"
    return f"inc_{hashlib.sha256(content.encode()).hexdigest()[:12]}"


@dataclass
class IncidentState:
    """Incident state for kernel computation."""

    # Identity
    id: str  # Generated ID for internal references
    signature: str  # Deterministic merge key
    scope_id: str = ""  # Scope for incident lookup

    # Membership
    surface_ids: FrozenSet[str] = frozenset()
    anchor_entities: FrozenSet[str] = frozenset()
    companion_entities: FrozenSet[str] = frozenset()

    # Motifs (L3→L4 contract)
    core_motifs: Tuple[Dict, ...] = ()

    # Time bounds
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None

    # Versioning
    kernel_version: str = ""
    params_hash: str = ""

    def with_surface(
        self,
        surface_id: str,
        anchors: FrozenSet[str],
        companions: FrozenSet[str],
        surface_time: Optional[datetime],
    ) -> "IncidentState":
        """Create new IncidentState with surface added."""
        new_time_start = self.time_start
        new_time_end = self.time_end
        if surface_time:
            if new_time_start is None or surface_time < new_time_start:
                new_time_start = surface_time
            if new_time_end is None or surface_time > new_time_end:
                new_time_end = surface_time

        return IncidentState(
            id=self.id,
            signature=self.signature,
            scope_id=self.scope_id,
            surface_ids=self.surface_ids | {surface_id},
            anchor_entities=self.anchor_entities | anchors,
            companion_entities=self.companion_entities | companions,
            core_motifs=self.core_motifs,
            time_start=new_time_start,
            time_end=new_time_end,
            kernel_version=self.kernel_version,
            params_hash=self.params_hash,
        )


@dataclass(frozen=True)
class PartitionSnapshot:
    """Snapshot of a scope partition for kernel computation.

    Passed to kernel.step() as the current state.
    Kernel returns TopologyDelta describing changes.
    """

    scope_id: str
    surfaces: Tuple[SurfaceState, ...] = ()
    incidents: Tuple[IncidentState, ...] = ()

    def get_surface_by_key(self, key: SurfaceKey) -> Optional[SurfaceState]:
        """Find surface by key."""
        for s in self.surfaces:
            if s.key == key:
                return s
        return None

    def get_incident_by_id(self, incident_id: str) -> Optional[IncidentState]:
        """Find incident by ID."""
        for i in self.incidents:
            if i.id == incident_id:
                return i
        return None

    @property
    def surface_by_signature(self) -> Dict[str, SurfaceState]:
        """Index surfaces by signature."""
        return {s.signature: s for s in self.surfaces}

    @property
    def incident_by_signature(self) -> Dict[str, IncidentState]:
        """Index incidents by signature."""
        return {i.signature: i for i in self.incidents}
