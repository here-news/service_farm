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


class ConstraintType(Enum):
    """Types of constraints in the ledger."""
    STRUCTURAL = "structural"   # Motif sharing, co-occurrence
    SEMANTIC = "semantic"       # Embedding similarity, LLM-derived
    TYPED = "typed"             # Typed variable observations
    TEMPORAL = "temporal"       # Time-based constraints
    META = "meta"               # Observations about inference state
    LOCATION_EVENT = "location_event"  # Specific location + time = same physical event


# =============================================================================
# CONSTRAINT LEDGER (Principled Emergence)
# =============================================================================

@dataclass
class Constraint:
    """
    A single constraint with provenance.

    This is the atomic unit of evidence - everything flows through constraints.
    All decisions are derived from constraints, not computed directly.
    """
    id: str = field(default_factory=lambda: f"cst_{uuid.uuid4().hex[:8]}")
    constraint_type: ConstraintType = ConstraintType.STRUCTURAL

    # What does this constraint assert?
    assertion: str = ""

    # Evidence and provenance
    evidence: Dict[str, Any] = field(default_factory=dict)
    provenance: str = ""  # Where did this come from?

    # Confidence (for semantic constraints)
    confidence: float = 1.0

    # Scope (what pair/entity/surface does this apply to?)
    scope: Optional[str] = None

    # Timestamp
    created_at: datetime = field(default_factory=datetime.utcnow)

    def is_structural(self) -> bool:
        return self.constraint_type == ConstraintType.STRUCTURAL

    def is_semantic(self) -> bool:
        return self.constraint_type == ConstraintType.SEMANTIC


@dataclass
class ConstraintLedger:
    """
    The constraint ledger - replaces score soup.

    All decisions are derived from constraints, not computed directly.
    This enables full auditability and reproducibility.
    """
    constraints: List[Constraint] = field(default_factory=list)

    # Index by type
    _by_type: Dict[str, List[Constraint]] = field(default_factory=dict)

    # Index by scope (surface/event ID or pair key)
    _by_scope: Dict[str, List[Constraint]] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize indices
        if not self._by_type:
            self._by_type = {t.value: [] for t in ConstraintType}
        if not self._by_scope:
            self._by_scope = {}

    def add(self, constraint: Constraint, scope: str = None):
        """Add constraint to ledger."""
        self.constraints.append(constraint)

        # Index by type
        type_key = constraint.constraint_type.value
        if type_key not in self._by_type:
            self._by_type[type_key] = []
        self._by_type[type_key].append(constraint)

        # Index by scope
        effective_scope = scope or constraint.scope
        if effective_scope:
            constraint.scope = effective_scope
            if effective_scope not in self._by_scope:
                self._by_scope[effective_scope] = []
            self._by_scope[effective_scope].append(constraint)

    def get_structural(self) -> List[Constraint]:
        return self._by_type.get(ConstraintType.STRUCTURAL.value, [])

    def get_semantic(self) -> List[Constraint]:
        return self._by_type.get(ConstraintType.SEMANTIC.value, [])

    def for_scope(self, scope: str) -> List[Constraint]:
        return self._by_scope.get(scope, [])

    def can_form_core(self, scope1: str, scope2: str) -> Tuple[bool, str]:
        """
        ANTI-TRAP RULE: Core edges require ≥2 constraints, ≥1 non-semantic.

        This prevents LLM-only similarity from forming event cores.
        """
        pair_key = f"{min(scope1, scope2)}:{max(scope1, scope2)}"
        constraints = self.for_scope(pair_key)

        if len(constraints) < 2:
            return False, f"Only {len(constraints)} constraints (need ≥2)"

        non_semantic = [c for c in constraints if not c.is_semantic()]
        if not non_semantic:
            return False, "All constraints are semantic (need ≥1 structural/temporal)"

        return True, f"Valid: {len(non_semantic)} structural + {len(constraints) - len(non_semantic)} semantic"

    def to_dict(self) -> Dict:
        """Serialize ledger for storage."""
        return {
            "constraints": [
                {
                    "id": c.id,
                    "type": c.constraint_type.value,
                    "assertion": c.assertion,
                    "evidence": c.evidence,
                    "provenance": c.provenance,
                    "confidence": c.confidence,
                    "scope": c.scope,
                }
                for c in self.constraints
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ConstraintLedger":
        """Deserialize ledger from storage."""
        ledger = cls()
        for c_data in data.get("constraints", []):
            constraint = Constraint(
                id=c_data.get("id", f"cst_{uuid.uuid4().hex[:8]}"),
                constraint_type=ConstraintType(c_data.get("type", "structural")),
                assertion=c_data.get("assertion", ""),
                evidence=c_data.get("evidence", {}),
                provenance=c_data.get("provenance", ""),
                confidence=c_data.get("confidence", 1.0),
                scope=c_data.get("scope"),
            )
            ledger.add(constraint)
        return ledger


@dataclass
class Motif:
    """
    A k-set of entities that appears in multiple claims.

    Motifs are the structural building blocks for surface formation.
    They replace embedding similarity as the primary clustering signal.
    """
    id: str = field(default_factory=lambda: f"mtf_{uuid.uuid4().hex[:8]}")
    entities: Set[str] = field(default_factory=set)
    support: int = 0  # Number of claims containing this motif
    weight: float = 0.0  # Graded evidence weight (log(support+1) * k_bonus)
    claim_ids: Set[str] = field(default_factory=set)

    @property
    def k(self) -> int:
        """Motif size (number of entities)."""
        return len(self.entities)

    def __hash__(self):
        return hash(frozenset(self.entities))


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

    # Temporal gate for incident-level events
    temporal_window_days: int = 7  # Surfaces must be within Δ days to link
    temporal_unknown_penalty: float = 0.5  # Cap edge weight when time unknown

    # Discriminative anchor requirement
    require_discriminative_anchor: bool = True  # Require high-IDF shared anchor
    discriminative_idf_threshold: float = 1.5  # Min IDF to be "discriminative"

    # Dispersion-based hub detection (set by IncidentEventView)
    # These anchors are high-frequency but high-dispersion (bridge unrelated surfaces)
    hub_anchors: Set[str] = field(default_factory=set)

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
    # Incompatible constraints (typed)
    "typed_value_conflict",      # Multiple distinct values for same (S, X)
    "unresolved_conflict",       # Explicit CONFLICTS edge between claims

    # Weak constraints (typed) - REQUIRE typed coverage > 0
    "high_entropy_value",        # H(X|E,S) > threshold (Jaynes posterior entropy over typed variable)
                                 # INVARIANT: Never emit if typed_coverage_zero
    "single_source_only",        # coverage=1, needs corroboration
    "high_stakes_low_evidence",  # High priority but weak evidence

    # Weak constraints (geometric) - about surface structure, not typed values
    "high_dispersion_surface",   # D(surface) > threshold (semantic dispersion from embeddings)
                                 # This is geometric, NOT Jaynes entropy. Indicates scope/mixing risk.

    # Missing constraints (gap detection) - REQUIRE typed coverage > 0
    "coverage_gap",              # coverage=0, expectedness high
    "extraction_gap",            # Cues present in text, typed value absent

    # Structural
    "bridge_node_detected",      # Hub entity risks percolation
    "stale_event",               # No new evidence in time window

    # Constraint availability (self-awareness about inference blockers)
    "typed_coverage_zero",       # No claims with question_key in scope → blocks H(X|E,S) inference
    "missing_time_high_rate",    # High % surfaces with no time → blocks temporal binding
    "missing_semantic_signal",   # No centroids/embeddings → blocks semantic similarity signal

    # Binding diagnostics (event-level structural issues)
    "insufficient_binding_evidence",  # Edges blocked by gates (signals_met<min, no_discriminative)
    "quarantine_dominates_event",     # Event held together mostly by weak anchor-only attachment
]


@dataclass
class MetaClaim:
    """
    Observation about the epistemic state itself.

    These are NOT truth claims about the world. They are observations
    about the topology that may trigger operational actions.
    """
    id: str = field(default_factory=lambda: f"mc_{uuid.uuid4().hex[:8]}")
    type: MetaClaimType = "high_entropy_value"
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

    SURFACE IDENTITY (L2 Definition):
    ================================
    Surfaces are keyed by QUESTION_KEY (proposition identity).
    Claims with the same question_key belong to the same surface.

    question_key examples:
    - "death_toll_hong_kong_fire_2025" → all claims about this count
    - "policy_announcement_john_lee" → all claims about this announcement
    - "fire_cause_tai_po" → all claims about the cause

    This means:
    - Same question_key → same surface (even if entities differ slightly)
    - Different question_key → different surfaces (even if entities overlap)
    - Conflicts stay INSIDE one surface (same question, different answers)

    MOTIF CLUSTERING (L3 Evidence):
    ================================
    Motifs (k-sets of co-occurring entities) are used for L3 incident formation,
    NOT for L2 surface identity. A motif like {Do Kwon, Terraform Labs} helps
    link surfaces into incidents, but doesn't define surface boundaries.

    The formation_method field tracks which algorithm created this surface:
    - "question_key": Principled L2 (authoritative)
    - "motif": Proto-surface from motif clustering (L3 evidence)
    - "legacy": Old embedding similarity (deprecated)

    The constraint_ledger tracks all evidence for membership decisions.
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

    # === PRINCIPLED EMERGENCE ADDITIONS ===

    # Constraint ledger: tracks all evidence for membership decisions
    constraint_ledger: Optional[ConstraintLedger] = None

    # Motif memberships: which motifs does this surface contain?
    motif_ids: Set[str] = field(default_factory=set)

    # Formation method (see docstring for semantics):
    # - "question_key": Principled L2 (authoritative)
    # - "motif": Proto-surface from motif clustering (L3 evidence)
    # - "legacy": Old embedding similarity (deprecated)
    formation_method: str = "legacy"

    # Question key: the proposition this surface answers
    # All claims with the same question_key belong together
    question_key: Optional[str] = None

    def __hash__(self):
        return hash(self.id)

    def get_ledger(self) -> ConstraintLedger:
        """Get or create constraint ledger."""
        if self.constraint_ledger is None:
            self.constraint_ledger = ConstraintLedger()
        return self.constraint_ledger


# =============================================================================
# L3: EVENT
# =============================================================================

@dataclass
class EventJustification:
    """
    Explanation bundle for an event - provides full explainability.

    This addresses the user's concern: "Using top entities as the event
    itself is cognitively wrong. Entities should be evidence, not the headline."

    Contains:
    1. Membrane proof (why it's one object)
    2. What happened (representative surfaces)
    3. Semantic summary (LLM-generated with citations)
    4. Why rejected (blocked bridges, underpowered edges)
    """

    # === MEMBRANE PROOF ===
    # Why these surfaces belong together

    # Core motifs that recur (k≥2) with support and time bins
    core_motifs: List[Dict[str, Any]] = field(default_factory=list)
    # Format: [{"entities": ["Do Kwon", "Terraform Labs"], "support": 5, "time_bin": "2025-12-10..12"}]

    # Context-compatibility passes (or "underpowered + motif-supported")
    context_passes: List[Dict[str, Any]] = field(default_factory=list)
    # Format: [{"surface1": "S001", "surface2": "S002", "overlap": 0.32, "status": "compatible"}]

    # Bridge blocks (what was rejected and why)
    blocked_bridges: List[Dict[str, Any]] = field(default_factory=list)
    # Format: [{"entity": "John Lee", "reason": "context Jaccard=0.00; powered", "surfaces": ["S001", "S042"]}]

    # Underpowered edges (treated as periphery only)
    underpowered_edges: List[Dict[str, Any]] = field(default_factory=list)
    # Format: [{"entity": "US", "reason": "companions too sparse", "treated_as": "periphery"}]

    # === WHAT HAPPENED ===
    # Representative surfaces that define this event

    # 1-3 representative surfaces with highest mass or typed variables
    representative_surfaces: List[str] = field(default_factory=list)

    # Their canonical titles and key facts
    representative_titles: List[str] = field(default_factory=list)
    representative_facts: List[str] = field(default_factory=list)

    # === SEMANTIC SUPPORT ===
    # Evidence from embeddings (periphery only, not core)

    # Average semantic similarity (for display, not merging)
    avg_semantic_similarity: float = 0.0

    # Model and embedding info
    semantic_model: str = ""
    semantic_evidence: str = ""  # e.g., "avg sim=0.71 (periphery), model=text-embedding-3-small"

    # === CANONICAL PROPOSITION HANDLE ===
    # The event label (NOT "top entities")

    # Format: (place + action + time) when typed is missing
    # e.g., "Do Kwon sentencing hearing (Judge Engelmayer)"
    canonical_handle: str = ""

    # Citations to representative surfaces
    handle_citations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize for API/storage."""
        return {
            "membrane_proof": {
                "core_motifs": self.core_motifs,
                "context_passes": self.context_passes,
                "blocked_bridges": self.blocked_bridges,
                "underpowered_edges": self.underpowered_edges,
            },
            "what_happened": {
                "representative_surfaces": self.representative_surfaces,
                "representative_titles": self.representative_titles,
                "representative_facts": self.representative_facts,
            },
            "semantic_support": {
                "avg_similarity": self.avg_semantic_similarity,
                "model": self.semantic_model,
                "evidence": self.semantic_evidence,
            },
            "canonical_handle": self.canonical_handle,
            "handle_citations": self.handle_citations,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EventJustification":
        """Deserialize from storage."""
        membrane = data.get("membrane_proof", {})
        what = data.get("what_happened", {})
        semantic = data.get("semantic_support", {})

        return cls(
            core_motifs=membrane.get("core_motifs", []),
            context_passes=membrane.get("context_passes", []),
            blocked_bridges=membrane.get("blocked_bridges", []),
            underpowered_edges=membrane.get("underpowered_edges", []),
            representative_surfaces=what.get("representative_surfaces", []),
            representative_titles=what.get("representative_titles", []),
            representative_facts=what.get("representative_facts", []),
            avg_semantic_similarity=semantic.get("avg_similarity", 0.0),
            semantic_model=semantic.get("model", ""),
            semantic_evidence=semantic.get("evidence", ""),
            canonical_handle=data.get("canonical_handle", ""),
            handle_citations=data.get("handle_citations", []),
        )


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

    # Justification bundle (explainability)
    justification: Optional[EventJustification] = None

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


# =============================================================================
# STORY: Unified L3/L4 Output Type
# =============================================================================

StoryScale = Literal["incident", "case"]


@dataclass
class Story:
    """
    User-facing L3/L4 object. One type, two scales.

    This is the ONLY type that should be exposed to API/frontend.
    Internal computation uses Event (L3) and views produce Story objects.

    Scale semantics:
    - "incident": Single coherent happening (L3 membrane)
      Examples: a fire, an arrest, a policy announcement
    - "case": Grouped related happenings (L4 membrane)
      Examples: Hong Kong fire + response + investigation

    STABILITY:
    - scope_signature provides deterministic identity across rebuilds
    - Same anchors + time bin + scale = same ID
    """
    id: str
    scale: StoryScale

    # === STABLE IDENTITY ===
    # Deterministic hash for rebuild stability
    scope_signature: str = ""
    params_version: int = 1

    # === CONTENT ===
    title: str = ""
    description: str = ""

    # === ENTITIES ===
    primary_entities: List[str] = field(default_factory=list)
    anchor_entities: Set[str] = field(default_factory=set)

    # === STRUCTURE ===
    surface_ids: Set[str] = field(default_factory=set)
    incident_ids: Optional[Set[str]] = None  # Only for scale="case"

    # === TEMPORAL ===
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None

    # === STATS ===
    surface_count: int = 0
    source_count: int = 0
    claim_count: int = 0
    incident_count: int = 0  # Only meaningful for scale="case"

    # === JUSTIFICATION ===
    justification: Optional[EventJustification] = None

    # === METADATA ===
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        """Ensure sets are proper sets."""
        if isinstance(self.anchor_entities, list):
            self.anchor_entities = set(self.anchor_entities)
        if isinstance(self.surface_ids, list):
            self.surface_ids = set(self.surface_ids)
        if self.incident_ids is not None and isinstance(self.incident_ids, list):
            self.incident_ids = set(self.incident_ids)

    def __hash__(self):
        return hash(self.id)

    def compute_scope_signature(self) -> str:
        """
        Compute deterministic scope signature for stable identity.

        The signature is based on:
        1. Scale (incident vs case)
        2. Sorted anchor entities (top 10)
        3. Time bin (week for incident, month for case)
        4. Params version

        This ensures the same story gets the same ID across rebuilds.
        """
        import hashlib

        # Sort anchors for determinism
        sorted_anchors = sorted(self.anchor_entities)[:10]

        # Time bin: week for incident, month for case
        time_bin = "unknown"
        if self.time_start:
            if self.scale == "incident":
                # Week bin: YYYY-Www
                time_bin = f"{self.time_start.year}-W{self.time_start.isocalendar()[1]:02d}"
            else:
                # Month bin: YYYY-MM
                time_bin = f"{self.time_start.year}-{self.time_start.month:02d}"

        # Build signature string
        sig_parts = [
            self.scale,
            ",".join(sorted_anchors),
            time_bin,
            f"v{self.params_version}"
        ]
        sig_string = "|".join(sig_parts)

        # Hash to fixed-length ID
        sig_hash = hashlib.sha256(sig_string.encode()).hexdigest()[:12]
        self.scope_signature = f"story_{sig_hash}"

        return self.scope_signature

    def generate_stable_id(self) -> str:
        """
        Generate stable ID from scope signature.

        Call compute_scope_signature() first if not already set.
        """
        if not self.scope_signature:
            self.compute_scope_signature()

        # Use scope signature as ID
        self.id = self.scope_signature
        return self.id

    @classmethod
    def from_event(
        cls,
        event: Event,
        scale: StoryScale = "incident",
        title: str = "",
        description: str = ""
    ) -> "Story":
        """
        Convert internal Event (L3) to Story.

        This is the bridge from internal computation to external API.
        """
        story = cls(
            id=event.id,
            scale=scale,
            title=title or event.canonical_title or "",
            description=description or event.narrative or "",
            primary_entities=list(event.anchor_entities)[:5],
            anchor_entities=event.anchor_entities.copy(),
            surface_ids=event.surface_ids.copy(),
            time_start=event.time_window[0],
            time_end=event.time_window[1],
            surface_count=len(event.surface_ids),
            source_count=event.total_sources,
            claim_count=event.total_claims,
            justification=event.justification,
        )

        # Compute stable ID
        story.compute_scope_signature()

        return story

    def to_dict(self) -> Dict:
        """Serialize for API/storage."""
        return {
            "id": self.id,
            "scale": self.scale,
            "scope_signature": self.scope_signature,
            "title": self.title,
            "description": self.description,
            "primary_entities": self.primary_entities,
            "anchor_entities": list(self.anchor_entities),
            "surface_ids": list(self.surface_ids),
            "incident_ids": list(self.incident_ids) if self.incident_ids else None,
            "time_start": self.time_start.isoformat() if self.time_start else None,
            "time_end": self.time_end.isoformat() if self.time_end else None,
            "surface_count": self.surface_count,
            "source_count": self.source_count,
            "claim_count": self.claim_count,
            "incident_count": self.incident_count,
            "justification": self.justification.to_dict() if self.justification else None,
            "params_version": self.params_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Story":
        """Deserialize from storage."""
        story = cls(
            id=data.get("id", ""),
            scale=data.get("scale", "incident"),
            scope_signature=data.get("scope_signature", ""),
            params_version=data.get("params_version", 1),
            title=data.get("title", ""),
            description=data.get("description", ""),
            primary_entities=data.get("primary_entities", []),
            anchor_entities=set(data.get("anchor_entities", [])),
            surface_ids=set(data.get("surface_ids", [])),
            incident_ids=set(data.get("incident_ids", [])) if data.get("incident_ids") else None,
            surface_count=data.get("surface_count", 0),
            source_count=data.get("source_count", 0),
            claim_count=data.get("claim_count", 0),
            incident_count=data.get("incident_count", 0),
        )

        # Parse times
        if data.get("time_start"):
            try:
                story.time_start = datetime.fromisoformat(data["time_start"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
        if data.get("time_end"):
            try:
                story.time_end = datetime.fromisoformat(data["time_end"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        # Parse justification
        if data.get("justification"):
            story.justification = EventJustification.from_dict(data["justification"])

        return story


# =============================================================================
# L5: ENTITY LENS
# =============================================================================

@dataclass(frozen=True)
class EntityLens:
    """
    Entity-centric navigation view (lens into the story graph).

    A Lens is NOT membership - it's a read-only projection of what we know
    about an entity across incidents. Unlike Stories (which partition incidents),
    Lenses can overlap (same incident visible through multiple entity lenses).

    Use cases:
    - Entity profile pages: "What do we know about Do Kwon?"
    - Navigation: Browse incidents involving an entity
    - Companion discovery: Who co-occurs with this entity?

    IMMUTABILITY: Lenses are frozen dataclasses. Consumers should not mutate
    lens fields. If you need to add derived data, build your own structure.

    CROSS-TIME: A lens spans all incidents for an entity, regardless of
    temporal mode. For time-scoped views, filter incident_ids by time.

    This replaces EntityCase from builders/case_builder.py with a minimal,
    stable type that doesn't depend on deprecated builder logic.
    """
    # Focal entity (canonical name)
    entity: str

    # Stable ID: hash of entity name (cross-time, deterministic)
    lens_id: str

    # All incident IDs involving this entity (core membership only)
    incident_ids: frozenset

    # Companion entities → co-occurrence count (immutable view)
    # Built by caller, not computed here
    companion_counts: tuple  # tuple of (entity, count) pairs for immutability

    @property
    def companion_entities(self) -> Dict[str, int]:
        """Dict view of companion counts for compatibility."""
        return dict(self.companion_counts)

    @classmethod
    def create(
        cls,
        entity: str,
        incident_ids: Set[str],
        companion_counts: Optional[Dict[str, int]] = None,
    ) -> "EntityLens":
        """
        Factory method to create an EntityLens.

        Args:
            entity: Focal entity canonical name
            incident_ids: Set of incident IDs (will be frozen)
            companion_counts: Optional dict of companion → count (will be frozen)
        """
        import hashlib
        lens_id = f"lens_{hashlib.sha256(entity.encode()).hexdigest()[:12]}"

        # Freeze companion_counts as sorted tuple for determinism
        companions = tuple(
            sorted((companion_counts or {}).items(), key=lambda x: (-x[1], x[0]))
        )

        return cls(
            entity=entity,
            lens_id=lens_id,
            incident_ids=frozenset(incident_ids),
            companion_counts=companions,
        )
