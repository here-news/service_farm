"""
Core Types for Epistemic Unit Architecture
==========================================

This module contains pure data structures with no algorithms.
All computation is in separate modules.

Layers:
  L0 Claim: Atomic, immutable observations with provenance
  L1 Proposition: Not yet implemented (future: version chains)
  L2 Surface: Bundle of claims connected by identity edges
  L3 Event: Cluster of surfaces connected by aboutness edges
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Set, Optional, Tuple, Any, Literal


# =============================================================================
# ENUMS
# =============================================================================

class Relation(Enum):
    """Level 0 relations (claim-to-claim only)."""
    CONFIRMS = "confirms"       # Same fact, different source
    REFINES = "refines"         # Adds detail to same fact
    SUPERSEDES = "supersedes"   # Updates/corrects prior claim
    CONFLICTS = "conflicts"     # Contradicts existing claim
    UNRELATED = "unrelated"     # Different facts


class Association(Enum):
    """Higher-level associations (surface-to-surface, event-to-event)."""
    SAME = "same"               # Identity: should merge
    RELATED = "related"         # Association: edge only
    DISTINCT = "distinct"       # No connection


class MembershipLevel(Enum):
    """Membership tier for surface->event attachment."""
    CORE = "core"               # High confidence, multiple strong signals
    PERIPHERY = "periphery"     # Moderate confidence, some evidence
    QUARANTINE = "quarantine"   # Weak evidence, pending more data


# =============================================================================
# PARAMETER VERSIONING (Invariant 2)
# =============================================================================

@dataclass
class ParameterChange:
    """
    System action that affects L1-L5 computation.

    Parameters are versioned and attributed because they change
    derived layer outcomes without new evidence.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # What changed
    parameter: str = ""
    old_value: Any = None
    new_value: Any = None

    # Provenance (who/what/why)
    actor: str = "system"
    trigger: Optional[str] = None
    rationale: str = ""

    # Reproducibility
    topology_version: Optional[str] = None
    affects_layers: List[str] = field(default_factory=list)


@dataclass
class Parameters:
    """
    Versioned parameter set for epistemic computation.

    All derived state (L1-L5) is deterministic given (L0, params@version).
    """
    version: int = 1

    # L2 Surface formation (identity edges)
    identity_confidence_threshold: float = 0.5

    # L3 Event formation (aboutness edges)
    hub_max_df: int = 5
    aboutness_min_signals: int = 2
    aboutness_threshold: float = 0.15

    # Tension detection
    high_entropy_threshold: float = 0.6

    # History
    changes: List[ParameterChange] = field(default_factory=list)

    def update(
        self,
        parameter: str,
        new_value: Any,
        actor: str = "system",
        trigger: Optional[str] = None,
        rationale: str = ""
    ) -> ParameterChange:
        """Update a parameter with full provenance tracking."""
        old_value = getattr(self, parameter, None)

        change = ParameterChange(
            parameter=parameter,
            old_value=old_value,
            new_value=new_value,
            actor=actor,
            trigger=trigger,
            rationale=rationale,
            topology_version=f"v{self.version}",
            affects_layers=self._affected_layers(parameter)
        )

        setattr(self, parameter, new_value)
        self.version += 1
        self.changes.append(change)

        return change

    def _affected_layers(self, parameter: str) -> List[str]:
        """Determine which layers are affected by a parameter change."""
        if parameter.startswith("identity"):
            return ["L2", "L3"]
        elif parameter.startswith("aboutness") or parameter == "hub_max_df":
            return ["L3"]
        elif parameter.startswith("high_entropy"):
            return []
        return ["L2", "L3"]


# =============================================================================
# META-CLAIMS (Invariant 6)
# =============================================================================

MetaClaimType = Literal[
    "high_stakes_low_evidence",
    "unresolved_conflict",
    "single_source_only",
    "high_entropy_surface",
    "bridge_node_detected",
    "stale_event",
]


@dataclass
class MetaClaim:
    """
    Observation about the epistemic state itself.

    These are NOT truth claims about the world. They are observations
    about the topology that may trigger operational actions.
    """
    id: str = field(default_factory=lambda: f"mc_{uuid.uuid4().hex[:8]}")
    type: MetaClaimType = "high_entropy_surface"
    target_id: str = ""
    target_type: str = "surface"

    evidence: Dict = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    params_version: int = 1

    resolved: bool = False
    resolution: Optional[str] = None


# =============================================================================
# L0: CLAIM
# =============================================================================

@dataclass
class Claim:
    """
    L0: Atomic epistemic unit. Append-only, immutable.

    The claim contains data only. All relationship computation
    is in identity/linker.py.
    """
    id: str
    text: str
    source: str
    embedding: Optional[List[float]] = None
    entities: Set[str] = field(default_factory=set)
    anchor_entities: Set[str] = field(default_factory=set)
    timestamp: Optional[datetime] = None

    # Metadata
    page_id: Optional[str] = None
    event_time: Optional[datetime] = None

    # Question Key (q1/q2 pattern)
    question_key: Optional[str] = None
    extracted_value: Optional[Any] = None
    value_unit: Optional[str] = None
    has_update_language: bool = False
    is_monotonic: Optional[bool] = None

    def __hash__(self):
        return hash(self.id)


# =============================================================================
# L2: SURFACE
# =============================================================================

@dataclass
class AboutnessLink:
    """
    Soft aboutness edge between surfaces (Tier-2).

    These edges represent "same event, different aspect" associations.
    They are NOT identity edges.
    """
    target_id: str
    score: float
    evidence: Dict = field(default_factory=dict)


@dataclass
class Surface:
    """
    L2: Bundle of claims connected by IDENTITY edges.

    Internal edges are identity relations (CONFIRMS/REFINES/SUPERSEDES/CONFLICTS).
    Aboutness (L3 event-level) is stored in about_links.
    """
    id: str
    claim_ids: Set[str] = field(default_factory=set)

    # Computed properties
    centroid: Optional[List[float]] = None
    entropy: float = 0.0
    mass: float = 0.0
    sources: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    anchor_entities: Set[str] = field(default_factory=set)
    time_window: Tuple[Optional[datetime], Optional[datetime]] = (None, None)

    # Semantic properties (from LLM interpretation)
    canonical_title: Optional[str] = None
    description: Optional[str] = None
    key_facts: List[str] = field(default_factory=list)

    # Internal structure
    internal_edges: List[Tuple[str, str, Relation]] = field(default_factory=list)

    # External structure (aboutness edges to other surfaces)
    about_links: List[AboutnessLink] = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)


# =============================================================================
# L3: EVENT
# =============================================================================

@dataclass
class EventSignature:
    """
    Event signature - the profile that surfaces are matched against.
    """
    anchor_weights: Dict[str, float] = field(default_factory=dict)
    entity_weights: Dict[str, float] = field(default_factory=dict)
    centroid: Optional[List[float]] = None
    centroid_dispersion: float = 0.0
    time_model: Literal["incident", "case"] = "incident"
    time_window: Tuple[Optional[datetime], Optional[datetime]] = (None, None)
    source_count: int = 0
    source_diversity: float = 0.0


@dataclass
class SurfaceMembership:
    """Record of a surface's membership in an event."""
    surface_id: str
    level: MembershipLevel
    score: float
    evidence: Dict = field(default_factory=dict)
    attached_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Event:
    """
    L3: Higher-level virtual unit. Emergent from surface relationships.
    """
    id: str
    surface_ids: Set[str] = field(default_factory=set)

    # Computed from surfaces
    centroid: Optional[List[float]] = None
    total_claims: int = 0
    total_sources: int = 0
    entities: Set[str] = field(default_factory=set)
    anchor_entities: Set[str] = field(default_factory=set)
    time_window: Tuple[Optional[datetime], Optional[datetime]] = (None, None)

    # Metabolic state
    signature: Optional[EventSignature] = None
    memberships: Dict[str, SurfaceMembership] = field(default_factory=dict)

    # Semantic interpretation
    canonical_title: Optional[str] = None
    narrative: Optional[str] = None
    timeline: List[Dict] = field(default_factory=list)

    def core_surfaces(self) -> Set[str]:
        """Return surface IDs with CORE membership."""
        return {
            sid for sid, m in self.memberships.items()
            if m.level == MembershipLevel.CORE
        }

    def periphery_surfaces(self) -> Set[str]:
        """Return surface IDs with PERIPHERY membership."""
        return {
            sid for sid, m in self.memberships.items()
            if m.level == MembershipLevel.PERIPHERY
        }

    def quarantine_surfaces(self) -> Set[str]:
        """Return surface IDs with QUARANTINE membership."""
        return {
            sid for sid, m in self.memberships.items()
            if m.level == MembershipLevel.QUARANTINE
        }
