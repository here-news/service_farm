"""
Model Selection Weaver: Universal Grouping as Compression vs Contamination

This implements the key insight that grouping should only occur when it
compresses evidence while not increasing incoherence:

ΔScore = (information_gain + redundancy_removed) - (conflict_penalty + ambiguity_penalty + closure_risk)

Objects compete as explanations. The organism grows when "one organism explains
these parts better than separate ones."

Levels:
- Surface (L2): One latent variable. Can Merge/Couple/Split based on compression.
- Incident (L3): Container hosting surfaces about one happening instance.
- Case (L4): Organism = equivalence class over spine edges + metabolism graph.

Each object maintains:
- coherence metrics (entropy trend, conflict density, boundary pressure)
- support metrics (independent sources, consistent updates)
- closure risk (would merging create a giant component?)

Universal triggers:
- Promote: high support + stable coherence → create/strengthen higher-order object
- Quarantine: high boundary pressure → metabolic-only + inquiries
- Repair: accumulated DEFER → targeted adjudication
- Fission: persistent conflict → split
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, Set, List, Optional, Tuple, FrozenSet
import math
import hashlib


# =============================================================================
# Core Metrics: Universal measures of coherence and cost
# =============================================================================

@dataclass
class CoherenceMetrics:
    """Universal metrics for any object (surface, incident, case)."""

    # Entropy: lower is better (more certain beliefs)
    entropy: float = 1.0

    # Conflict density: lower is better (fewer contradictions per unit evidence)
    conflict_density: float = 0.0

    # Boundary pressure: lower is better (fewer ambiguous relations)
    boundary_pressure: float = 0.0

    # Support: higher is better (more independent sources)
    support_count: int = 0
    source_diversity: float = 0.0  # 0-1, how diverse are sources

    # Closure risk: probability that merging would chain to unrelated objects
    closure_risk: float = 0.0

    def compression_score(self) -> float:
        """How much does this object compress its evidence?"""
        # Higher support with lower entropy = good compression
        if self.support_count == 0:
            return 0.0
        return (1.0 - self.entropy) * math.log1p(self.support_count) * (1 + self.source_diversity)

    def contamination_cost(self) -> float:
        """Cost of maintaining this object's current state."""
        return (
            self.conflict_density * 2.0 +
            self.boundary_pressure * 1.5 +
            self.closure_risk * 3.0  # Closure risk is very expensive
        )

    def health_score(self) -> float:
        """Net health: compression benefit minus contamination cost."""
        return self.compression_score() - self.contamination_cost()


@dataclass
class GroupingDecision:
    """Result of evaluating whether to group two objects."""

    class Action(Enum):
        MERGE = auto()      # Same variable - combine into one
        COUPLE = auto()     # Related but distinct - add metabolic edge
        SPLIT = auto()      # Separate - evidence shows they're different
        DEFER = auto()      # Ambiguous - emit inquiry, don't decide yet

    action: Action
    delta_score: float  # Positive = grouping helps, negative = hurts
    information_gain: float
    redundancy_removed: float
    conflict_penalty: float
    ambiguity_penalty: float
    closure_risk: float
    rationale: str
    inquiry_seed: Optional[str] = None  # If DEFER, what to ask


def compute_grouping_score(
    a_metrics: CoherenceMetrics,
    b_metrics: CoherenceMetrics,
    overlap_strength: float,  # 0-1, how much evidence they share
    posterior_compatibility: float,  # 0-1, how consistent their beliefs
    closure_risk: float,  # 0-1, risk of chaining
) -> GroupingDecision:
    """
    Universal grouping objective: should A and B be grouped?

    ΔScore = (information_gain + redundancy_removed) -
             (conflict_penalty + ambiguity_penalty + closure_risk)
    """
    # Information gain: grouping reduces uncertainty if beliefs are compatible
    info_gain = overlap_strength * posterior_compatibility * (a_metrics.entropy + b_metrics.entropy) / 2

    # Redundancy removed: fewer duplicate variables
    redundancy = overlap_strength * min(a_metrics.support_count, b_metrics.support_count) * 0.1

    # Conflict penalty: combining incompatible beliefs creates conflict
    conflict = (1 - posterior_compatibility) * (a_metrics.support_count + b_metrics.support_count) * 0.2

    # Ambiguity penalty: weak overlap creates uncertain boundaries
    ambiguity = (1 - overlap_strength) * (a_metrics.boundary_pressure + b_metrics.boundary_pressure) / 2

    # Closure risk penalty: transitive closure dangers
    closure_cost = closure_risk * 3.0  # High weight - closure is very dangerous

    delta = (info_gain + redundancy) - (conflict + ambiguity + closure_cost)

    # Decision thresholds - tuned for incremental growth
    # Lower thresholds encourage more merging while still protecting against contamination
    if delta > 0.1 and posterior_compatibility > 0.4 and closure_risk < 0.5:
        action = GroupingDecision.Action.MERGE
        rationale = f"Compression gain ({delta:.2f}), compatible beliefs (compat={posterior_compatibility:.2f})"
    elif delta > -0.2 and closure_risk < 0.6:
        action = GroupingDecision.Action.COUPLE
        rationale = f"Marginal gain ({delta:.2f}), use metabolic edge for safety"
    elif posterior_compatibility < 0.2 or conflict > info_gain * 2:
        action = GroupingDecision.Action.SPLIT
        rationale = f"Incompatible beliefs (compat={posterior_compatibility:.2f}), conflict too high"
    else:
        action = GroupingDecision.Action.DEFER
        rationale = f"Ambiguous (delta={delta:.2f}), need more evidence"

    return GroupingDecision(
        action=action,
        delta_score=delta,
        information_gain=info_gain,
        redundancy_removed=redundancy,
        conflict_penalty=conflict,
        ambiguity_penalty=ambiguity,
        closure_risk=closure_risk,
        rationale=rationale,
        inquiry_seed="Need authoritative source to resolve" if action == GroupingDecision.Action.DEFER else None,
    )


# =============================================================================
# L2: Surface Dynamics
# =============================================================================

@dataclass
class Surface:
    """
    Surface = one latent variable about the world.

    Tracks its own coherence and can participate in Merge/Couple/Split dynamics.
    """
    surface_id: str
    proposition_key: str  # The question this surface answers

    # Evidence
    claim_ids: Set[str] = field(default_factory=set)
    sources: Set[str] = field(default_factory=set)

    # Semantic representation for metabolism
    description: str = ""  # Human-readable summary of what this surface represents
    embedding: Optional[List[float]] = None  # Semantic embedding for similarity
    claim_embeddings: Dict[str, List[float]] = field(default_factory=dict)  # claim_id -> embedding

    # Belief state (simplified - in reality would be full posterior)
    belief_entropy: float = 1.0  # 0 = certain, 1 = max uncertainty
    belief_mode: Optional[str] = None  # Most likely value

    # Temporal
    first_seen: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    # Relations to other surfaces (metabolic edges)
    coupled_surfaces: Dict[str, str] = field(default_factory=dict)  # id -> relation_type

    # Conflict tracking
    conflict_claims: Set[str] = field(default_factory=set)  # Claims that contradict

    def metrics(self) -> CoherenceMetrics:
        """Compute current coherence metrics."""
        n_claims = len(self.claim_ids)
        n_conflicts = len(self.conflict_claims)
        n_sources = len(self.sources)

        return CoherenceMetrics(
            entropy=self.belief_entropy,
            conflict_density=n_conflicts / max(n_claims, 1),
            boundary_pressure=len(self.coupled_surfaces) * 0.1,  # More relations = more ambiguity
            support_count=n_claims,
            source_diversity=min(n_sources / 5, 1.0),  # Normalize to 0-1
            closure_risk=0.0,  # Surfaces don't have closure risk (no transitive membership)
        )

    def posterior_compatibility(self, other: Surface) -> float:
        """How compatible are our beliefs with another surface?"""
        # Same proposition key = potentially same variable
        if self.proposition_key != other.proposition_key:
            return 0.0

        # Check for conflicts
        shared_claims = self.claim_ids & other.claim_ids
        if shared_claims:
            # Same claims = high compatibility
            return len(shared_claims) / max(len(self.claim_ids), len(other.claim_ids))

        # Different claims but same proposition - check entropy
        # If both are uncertain, they might be compatible
        # If both are certain and different, they're incompatible
        if self.belief_mode and other.belief_mode:
            if self.belief_mode == other.belief_mode:
                return 1.0 - max(self.belief_entropy, other.belief_entropy)
            else:
                return 0.0  # Different certain beliefs = incompatible

        # One or both uncertain - moderate compatibility
        return 0.5

    def semantic_similarity(self, other: Surface) -> float:
        """Compute semantic similarity using embeddings."""
        if self.embedding is None or other.embedding is None:
            return 0.5  # Unknown - moderate

        # Cosine similarity
        dot = sum(a * b for a, b in zip(self.embedding, other.embedding))
        norm_a = sum(x * x for x in self.embedding) ** 0.5
        norm_b = sum(x * x for x in other.embedding) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.5

        return (dot / (norm_a * norm_b) + 1) / 2  # Normalize to 0-1

    def compute_aggregate_embedding(self) -> Optional[List[float]]:
        """Compute aggregate embedding from claim embeddings."""
        if not self.claim_embeddings:
            return self.embedding

        # Average of claim embeddings
        n = len(self.claim_embeddings)
        dim = len(next(iter(self.claim_embeddings.values())))
        aggregate = [0.0] * dim

        for emb in self.claim_embeddings.values():
            for i, v in enumerate(emb):
                aggregate[i] += v / n

        return aggregate


class SurfaceInteraction(Enum):
    """Types of surface-to-surface interaction."""
    MERGE = auto()      # Same variable, combine
    CORROBORATE = auto()  # Different claims, same conclusion
    CONTRADICT = auto()   # Different claims, opposing conclusions
    CAUSE = auto()        # One variable influences another
    CONTEXT = auto()      # Background/ambient relation


@dataclass
class SurfaceAction:
    """Result of surface dynamics."""
    interaction: SurfaceInteraction
    target_surface_id: Optional[str]
    new_surface_id: Optional[str]  # For SPLIT
    inquiry: Optional[str]
    trace: str


# =============================================================================
# L3: Incident Dynamics
# =============================================================================

@dataclass
class Incident:
    """
    Incident = container hosting surfaces about one happening instance.

    The incident membrane decides what surfaces belong inside.
    """
    incident_id: str

    # Identity
    referents: FrozenSet[str] = frozenset()  # Who/what/where (identity witnesses)
    time_range: Optional[Tuple[datetime, datetime]] = None

    # Semantic representation for metabolism
    description: str = ""  # LLM-generated summary: "Fire at Wang Fuk Court kills 13"
    embedding: Optional[List[float]] = None  # Semantic embedding of incident
    headline: str = ""  # Short headline for this incident

    # Contents
    surface_ids: Set[str] = field(default_factory=set)

    # Metabolism (relations to other incidents)
    spine_edges: Dict[str, str] = field(default_factory=dict)  # incident_id -> edge_type (identity)
    metabolic_edges: Dict[str, str] = field(default_factory=dict)  # incident_id -> edge_type (related)

    # Coherence tracking
    internal_conflicts: int = 0
    boundary_disputes: int = 0  # How many "maybe related" edges

    def metrics(self, all_surfaces: Dict[str, Surface]) -> CoherenceMetrics:
        """Compute coherence from contained surfaces."""
        if not self.surface_ids:
            return CoherenceMetrics()

        surfaces = [all_surfaces[sid] for sid in self.surface_ids if sid in all_surfaces]
        if not surfaces:
            return CoherenceMetrics()

        # Aggregate surface metrics
        avg_entropy = sum(s.belief_entropy for s in surfaces) / len(surfaces)
        total_claims = sum(len(s.claim_ids) for s in surfaces)
        total_conflicts = sum(len(s.conflict_claims) for s in surfaces)
        all_sources = set()
        for s in surfaces:
            all_sources.update(s.sources)

        # Closure risk from spine edges
        n_spine = len(self.spine_edges)
        closure_risk = min(n_spine * 0.15, 1.0)  # Each spine edge adds risk

        return CoherenceMetrics(
            entropy=avg_entropy,
            conflict_density=total_conflicts / max(total_claims, 1),
            boundary_pressure=self.boundary_disputes * 0.2,
            support_count=total_claims,
            source_diversity=min(len(all_sources) / 10, 1.0),
            closure_risk=closure_risk,
        )

    def identity_overlap(self, other: Incident) -> float:
        """How much do our identities overlap?"""
        if not self.referents or not other.referents:
            return 0.0

        shared = self.referents & other.referents
        total = self.referents | other.referents
        return len(shared) / len(total) if total else 0.0

    def time_overlap(self, other: Incident) -> float:
        """How much do our time ranges overlap?"""
        if not self.time_range or not other.time_range:
            return 0.5  # Unknown = moderate overlap

        s1, e1 = self.time_range
        s2, e2 = other.time_range

        overlap_start = max(s1, s2)
        overlap_end = min(e1, e2)

        if overlap_start >= overlap_end:
            # No overlap - but could be sequential phases
            gap = (overlap_start - overlap_end).total_seconds()
            if gap < 86400 * 7:  # Within a week = possible phase
                return 0.3
            return 0.0

        overlap_duration = (overlap_end - overlap_start).total_seconds()
        total_duration = max((e1 - s1).total_seconds(), (e2 - s2).total_seconds())
        return overlap_duration / total_duration if total_duration > 0 else 0.5

    def compute_embedding(self, all_surfaces: Dict[str, "Surface"]) -> Optional[List[float]]:
        """Compute incident embedding as average of surface embeddings."""
        if self.embedding:
            return self.embedding

        embeddings = []
        for sid in self.surface_ids:
            surf = all_surfaces.get(sid)
            if surf and surf.embedding:
                embeddings.append(surf.embedding)

        if not embeddings:
            return None

        # Average of surface embeddings
        dim = len(embeddings[0])
        aggregate = [0.0] * dim
        for emb in embeddings:
            for i, v in enumerate(emb):
                aggregate[i] += v / len(embeddings)

        return aggregate

    def semantic_similarity(self, other: "Incident", all_surfaces: Dict[str, "Surface"]) -> float:
        """Compute semantic similarity with another incident."""
        emb_a = self.compute_embedding(all_surfaces)
        emb_b = other.compute_embedding(all_surfaces)

        if emb_a is None or emb_b is None:
            return 0.5  # Unknown

        dot = sum(a * b for a, b in zip(emb_a, emb_b))
        norm_a = sum(x * x for x in emb_a) ** 0.5
        norm_b = sum(x * x for x in emb_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.5

        return (dot / (norm_a * norm_b) + 1) / 2  # Normalize to 0-1

    def semantic_similarity_to_surface(self, surface: "Surface", all_surfaces: Dict[str, "Surface"]) -> float:
        """Compute semantic similarity between this incident and a surface."""
        inc_emb = self.compute_embedding(all_surfaces)
        surf_emb = surface.embedding

        if inc_emb is None or surf_emb is None:
            return 0.5  # Unknown - moderate similarity

        dot = sum(a * b for a, b in zip(inc_emb, surf_emb))
        norm_a = sum(x * x for x in inc_emb) ** 0.5
        norm_b = sum(x * x for x in surf_emb) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.5

        return (dot / (norm_a * norm_b) + 1) / 2  # Normalize to 0-1


class IncidentMembrane:
    """
    Decides whether new evidence should be assimilated into an incident.
    """

    def evaluate_surface(
        self,
        incident: Incident,
        surface: Surface,
        all_surfaces: Dict[str, Surface],
    ) -> Tuple[bool, str]:
        """Should this surface be assimilated into this incident?"""
        # Get incident's current surfaces
        existing = [all_surfaces[sid] for sid in incident.surface_ids if sid in all_surfaces]

        # Check proposition overlap - same questions = likely same incident
        existing_keys = {s.proposition_key for s in existing}
        if surface.proposition_key in existing_keys:
            return True, "Same proposition key - likely same variable"

        # Check source overlap - same sources often cover same incident
        existing_sources = set()
        for s in existing:
            existing_sources.update(s.sources)
        source_overlap = len(surface.sources & existing_sources) / max(len(surface.sources), 1)

        if source_overlap > 0.5:
            return True, f"High source overlap ({source_overlap:.1%})"

        return False, "No strong connection to incident"


class IncidentAction(Enum):
    """Membrane decisions for incident dynamics."""
    ASSIMILATE = auto()  # Surface belongs inside this incident
    RELATE = auto()      # Connected but not the same happening
    DEFER = auto()       # Ambiguous - emit inquiry
    REJECT = auto()      # Not related


# =============================================================================
# L4: Case Dynamics (Organism)
# =============================================================================

@dataclass
class Case:
    """
    Case = organism = equivalence class over spine edges + metabolism graph.

    Grows through model selection: adds incidents when it compresses evidence.
    """
    case_id: str

    # Membership (via spine edges = identity)
    incident_ids: Set[str] = field(default_factory=set)

    # The membrane: entities that define this case's identity
    membrane: Set[str] = field(default_factory=set)

    # Semantic representation for the whole story
    title: str = ""  # LLM-generated title: "Hong Kong High-Rise Fire Tragedy"
    summary: str = ""  # LLM-generated summary of the case
    embedding: Optional[List[float]] = None  # Semantic embedding for case-level similarity
    keywords: Set[str] = field(default_factory=set)  # Extracted keywords/themes

    # Spine edges (identity relations within case)
    spine_edges: Set[Tuple[str, str, str]] = field(default_factory=set)  # (from, to, type)

    # Metabolic edges (related but not identity)
    metabolic_edges: Set[Tuple[str, str, str]] = field(default_factory=set)

    # Health tracking
    last_fission_check: Optional[datetime] = None
    accumulated_conflicts: int = 0

    def compute_embedding(self, all_incidents: Dict[str, Incident], all_surfaces: Dict[str, Surface]) -> Optional[List[float]]:
        """Compute case embedding as average of incident embeddings."""
        if self.embedding:
            return self.embedding

        embeddings = []
        for iid in self.incident_ids:
            inc = all_incidents.get(iid)
            if inc:
                emb = inc.compute_embedding(all_surfaces)
                if emb:
                    embeddings.append(emb)

        if not embeddings:
            return None

        # Average of incident embeddings
        dim = len(embeddings[0])
        aggregate = [0.0] * dim
        for emb in embeddings:
            for i, v in enumerate(emb):
                aggregate[i] += v / len(embeddings)

        return aggregate

    def semantic_similarity(
        self,
        other: "Case",
        all_incidents: Dict[str, Incident],
        all_surfaces: Dict[str, Surface],
    ) -> float:
        """Compute semantic similarity with another case."""
        emb_a = self.compute_embedding(all_incidents, all_surfaces)
        emb_b = other.compute_embedding(all_incidents, all_surfaces)

        if emb_a is None or emb_b is None:
            return 0.5  # Unknown

        dot = sum(a * b for a, b in zip(emb_a, emb_b))
        norm_a = sum(x * x for x in emb_a) ** 0.5
        norm_b = sum(x * x for x in emb_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.5

        return (dot / (norm_a * norm_b) + 1) / 2  # Normalize to 0-1

    def semantic_similarity_to_incident(
        self,
        incident: Incident,
        all_incidents: Dict[str, Incident],
        all_surfaces: Dict[str, Surface],
    ) -> float:
        """Compute semantic similarity between this case and an incident."""
        case_emb = self.compute_embedding(all_incidents, all_surfaces)
        inc_emb = incident.compute_embedding(all_surfaces)

        if case_emb is None or inc_emb is None:
            return 0.5  # Unknown - moderate similarity

        dot = sum(a * b for a, b in zip(case_emb, inc_emb))
        norm_a = sum(x * x for x in case_emb) ** 0.5
        norm_b = sum(x * x for x in inc_emb) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.5

        return (dot / (norm_a * norm_b) + 1) / 2  # Normalize to 0-1

    def metrics(self, all_incidents: Dict[str, Incident], all_surfaces: Dict[str, Surface]) -> CoherenceMetrics:
        """Compute case-level coherence from incidents."""
        if not self.incident_ids:
            return CoherenceMetrics()

        incidents = [all_incidents[iid] for iid in self.incident_ids if iid in all_incidents]
        if not incidents:
            return CoherenceMetrics()

        # Aggregate incident metrics
        incident_metrics = [inc.metrics(all_surfaces) for inc in incidents]

        avg_entropy = sum(m.entropy for m in incident_metrics) / len(incident_metrics)
        total_support = sum(m.support_count for m in incident_metrics)
        max_diversity = max(m.source_diversity for m in incident_metrics)
        total_conflicts = sum(m.conflict_density * m.support_count for m in incident_metrics)

        # Case-level closure risk based on spine edge count and incident diversity
        n_incidents = len(incidents)
        n_spine = len(self.spine_edges)

        # If we have few incidents but many spine edges, closure risk is low
        # If we have many incidents, closure risk increases
        closure_risk = min(n_incidents * 0.05 + n_spine * 0.02, 1.0)

        # Boundary pressure from metabolic edges and membrane size
        boundary_pressure = len(self.metabolic_edges) * 0.05 + len(self.membrane) * 0.01

        return CoherenceMetrics(
            entropy=avg_entropy,
            conflict_density=total_conflicts / max(total_support, 1),
            boundary_pressure=boundary_pressure,
            support_count=total_support,
            source_diversity=max_diversity,
            closure_risk=closure_risk,
        )

    def should_admit_incident(
        self,
        incident: Incident,
        all_incidents: Dict[str, Incident],
        all_surfaces: Dict[str, Surface],
    ) -> GroupingDecision:
        """Should this incident join this case?"""
        if not self.incident_ids:
            # Empty case always admits
            return GroupingDecision(
                action=GroupingDecision.Action.MERGE,
                delta_score=1.0,
                information_gain=1.0,
                redundancy_removed=0.0,
                conflict_penalty=0.0,
                ambiguity_penalty=0.0,
                closure_risk=0.0,
                rationale="Empty case - first incident",
            )

        # Compute current case metrics
        current_metrics = self.metrics(all_incidents, all_surfaces)
        incident_metrics = incident.metrics(all_surfaces)

        # Identity overlap: does the incident share referents with our membrane?
        # Use Jaccard similarity for better normalization
        membrane_intersection = len(incident.referents & self.membrane)
        membrane_union = len(incident.referents | self.membrane)
        membrane_overlap = membrane_intersection / max(membrane_union, 1)

        # Boost overlap if we share the key spine entity
        if membrane_intersection >= 1:
            # At least one shared entity - significant overlap
            membrane_overlap = max(membrane_overlap, 0.5)

        # Check time consistency with existing incidents
        existing_incidents = [all_incidents[iid] for iid in self.incident_ids if iid in all_incidents]
        time_consistencies = [incident.time_overlap(ei) for ei in existing_incidents]
        avg_time_consistency = sum(time_consistencies) / len(time_consistencies) if time_consistencies else 0.5

        # Compute semantic similarity with existing incidents
        semantic_similarities = [
            incident.semantic_similarity(ei, all_surfaces)
            for ei in existing_incidents
        ]
        avg_semantic_sim = sum(semantic_similarities) / len(semantic_similarities) if semantic_similarities else 0.5

        # Posterior compatibility based on membrane + time + semantics
        # Weight: entity (0.4), semantic (0.35), time (0.25)
        posterior_compat = membrane_overlap * 0.4 + avg_semantic_sim * 0.35 + avg_time_consistency * 0.25

        # Combined overlap also includes semantic similarity
        combined_overlap = membrane_overlap * 0.6 + avg_semantic_sim * 0.4

        # Closure risk: would adding this incident create chains?
        # Check how many spine edges would result
        potential_spine_edges = 0
        for ei in existing_incidents:
            identity_overlap = incident.identity_overlap(ei)
            if identity_overlap > 0.5:
                potential_spine_edges += 1

        # More potential edges = higher closure risk
        closure_risk = min(potential_spine_edges * 0.2 + incident_metrics.closure_risk, 1.0)

        return compute_grouping_score(
            current_metrics,
            incident_metrics,
            combined_overlap,
            posterior_compat,
            closure_risk,
        )


class CaseAction(Enum):
    """Case-level dynamics actions."""
    GROW = auto()       # Admit incident to case
    PERIPHERY = auto()  # Link metabolically but not admit
    REJECT = auto()     # Not related
    DEFER = auto()      # Emit inquiry
    FISSION = auto()    # Split case into two


# =============================================================================
# Model Selection Weaver: The Organism
# =============================================================================

@dataclass
class WeaverEmission:
    """Output from the weaver - changes, inquiries, metrics."""

    class EmissionType(Enum):
        SURFACE_MERGED = auto()
        SURFACE_COUPLED = auto()
        SURFACE_SPLIT = auto()
        INCIDENT_ASSIMILATED = auto()
        INCIDENT_RELATED = auto()
        CASE_GREW = auto()
        CASE_FISSIONED = auto()
        INQUIRY_EMITTED = auto()
        DEFERRED = auto()

    emission_type: EmissionType
    object_id: str
    target_id: Optional[str] = None
    delta_score: float = 0.0
    rationale: str = ""
    inquiry: Optional[str] = None


class ModelSelectionWeaver:
    """
    Universal grouping weaver based on model selection.

    Objects compete as explanations. Grouping happens when it compresses
    evidence without increasing incoherence.
    """

    def __init__(
        self,
        merge_threshold: float = 0.5,
        closure_risk_max: float = 0.3,
        hub_entities: FrozenSet[str] = frozenset(),
    ):
        self.merge_threshold = merge_threshold
        self.closure_risk_max = closure_risk_max
        self.hub_entities = hub_entities

        # Storage
        self.surfaces: Dict[str, Surface] = {}
        self.incidents: Dict[str, Incident] = {}
        self.cases: Dict[str, Case] = {}

        # Indices for O(1) lookup
        self._surface_by_prop: Dict[str, Set[str]] = {}  # prop_key -> surface_ids
        self._incident_by_referent: Dict[str, Set[str]] = {}  # entity -> incident_ids
        self._case_by_entity: Dict[str, Set[str]] = {}  # entity -> case_ids

        # Emission log
        self.emissions: List[WeaverEmission] = []

        # Deferred decisions (need more evidence or adjudication)
        self.deferred: List[Tuple[str, str, str]] = []  # (type, id1, id2)

    def _generate_id(self, prefix: str, *args) -> str:
        """Generate deterministic ID."""
        content = "|".join(str(a) for a in args)
        h = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{prefix}_{h}"

    def _filter_hubs(self, entities: Set[str]) -> Set[str]:
        """Remove hub entities that shouldn't count for identity."""
        return entities - self.hub_entities

    # =========================================================================
    # L2: Surface Processing
    # =========================================================================

    def metabolize_claim(
        self,
        claim_id: str,
        text: str,
        proposition_key: str,
        source: str,
        entities: Set[str],
        anchor_entities: Set[str],
        event_time: Optional[datetime] = None,
        embedding: Optional[List[float]] = None,
    ) -> Tuple[str, List[WeaverEmission]]:
        """
        Process a claim into surface(s).

        Returns: (surface_id, emissions)
        """
        emissions = []

        # Filter hub entities
        anchor_entities = self._filter_hubs(anchor_entities)
        entities = self._filter_hubs(entities)

        # Look for existing surface with same proposition
        existing_ids = self._surface_by_prop.get(proposition_key, set())

        # Helper to compute semantic similarity
        def semantic_sim(emb_a: Optional[List[float]], emb_b: Optional[List[float]]) -> float:
            if emb_a is None or emb_b is None:
                return 0.5  # Unknown - moderate
            dot = sum(a * b for a, b in zip(emb_a, emb_b))
            norm_a = sum(x * x for x in emb_a) ** 0.5
            norm_b = sum(x * x for x in emb_b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.5
            return (dot / (norm_a * norm_b) + 1) / 2  # Normalize to 0-1

        best_target = None
        best_decision = None

        for sid in existing_ids:
            existing = self.surfaces[sid]

            # Compute compatibility
            source_overlap = 1.0 if source in existing.sources else 0.5
            claim_overlap = 0.0  # New claim, no overlap yet

            # Create temporary metrics for the new claim
            new_metrics = CoherenceMetrics(
                entropy=0.8,  # Unknown belief
                support_count=1,
                source_diversity=0.2,
            )

            decision = compute_grouping_score(
                existing.metrics(),
                new_metrics,
                overlap_strength=source_overlap * 0.5,  # Weak overlap for new claim
                posterior_compatibility=0.7 if source_overlap > 0.5 else 0.5,
                closure_risk=0.0,  # No closure risk at surface level
            )

            if best_decision is None or decision.delta_score > best_decision.delta_score:
                best_decision = decision
                best_target = sid

        # Apply decision
        if best_target and best_decision and best_decision.action == GroupingDecision.Action.MERGE:
            # Merge into existing surface
            surface = self.surfaces[best_target]
            surface.claim_ids.add(claim_id)
            surface.sources.add(source)
            surface.last_updated = event_time or datetime.now()

            # Store claim embedding for later aggregation
            if embedding:
                surface.claim_embeddings[claim_id] = embedding
                # Recompute surface embedding as average
                surface.embedding = surface.compute_aggregate_embedding()

            # Update belief entropy (more evidence = more certain)
            surface.belief_entropy *= 0.95

            emissions.append(WeaverEmission(
                emission_type=WeaverEmission.EmissionType.SURFACE_MERGED,
                object_id=best_target,
                delta_score=best_decision.delta_score,
                rationale=best_decision.rationale,
            ))

            return best_target, emissions

        # Create new surface
        surface_id = self._generate_id("sf", proposition_key, claim_id)
        surface = Surface(
            surface_id=surface_id,
            proposition_key=proposition_key,
            claim_ids={claim_id},
            sources={source},
            first_seen=event_time or datetime.now(),
            last_updated=event_time or datetime.now(),
            embedding=embedding,  # Set initial embedding
            claim_embeddings={claim_id: embedding} if embedding else {},
        )

        self.surfaces[surface_id] = surface

        # Update index
        if proposition_key not in self._surface_by_prop:
            self._surface_by_prop[proposition_key] = set()
        self._surface_by_prop[proposition_key].add(surface_id)

        # If there was a candidate but we didn't merge, add coupling
        if best_target and best_decision and best_decision.action == GroupingDecision.Action.COUPLE:
            surface.coupled_surfaces[best_target] = "CORROBORATE"
            self.surfaces[best_target].coupled_surfaces[surface_id] = "CORROBORATE"

            emissions.append(WeaverEmission(
                emission_type=WeaverEmission.EmissionType.SURFACE_COUPLED,
                object_id=surface_id,
                target_id=best_target,
                delta_score=best_decision.delta_score,
                rationale=best_decision.rationale,
            ))

        return surface_id, emissions

    # =========================================================================
    # L3: Incident Processing
    # =========================================================================

    def route_surface_to_incident(
        self,
        surface_id: str,
        referents: Set[str],
        event_time: Optional[datetime] = None,
    ) -> Tuple[str, List[WeaverEmission]]:
        """
        Route a surface to an incident.

        Returns: (incident_id, emissions)
        """
        emissions = []
        surface = self.surfaces.get(surface_id)
        if not surface:
            raise ValueError(f"Surface {surface_id} not found")

        # Filter hubs from referents
        referents = self._filter_hubs(referents)

        # Find candidate incidents by referent overlap
        candidate_ids = set()
        for ref in referents:
            if ref in self._incident_by_referent:
                candidate_ids.update(self._incident_by_referent[ref])

        # ALSO find candidates by semantic similarity (embedding-based routing)
        # This allows surfaces with "same happening, different wording/anchors" to merge
        if surface.embedding and len(candidate_ids) < len(self.incidents):
            for iid, incident in self.incidents.items():
                if iid not in candidate_ids:
                    sim = incident.semantic_similarity_to_surface(surface, self.surfaces)
                    if sim > 0.75:  # High semantic similarity threshold
                        candidate_ids.add(iid)

        best_incident = None
        best_decision = None

        membrane = IncidentMembrane()

        for iid in candidate_ids:
            incident = self.incidents[iid]

            # Check identity overlap - use intersection count as primary signal
            shared_refs = referents & incident.referents
            identity_overlap = len(shared_refs) / max(len(referents), 1)

            # Boost if key entity is shared
            if len(shared_refs) >= 1:
                identity_overlap = max(identity_overlap, 0.6)

            # Check time overlap
            time_compat = 0.5
            if event_time and incident.time_range:
                s, e = incident.time_range
                if s <= event_time <= e:
                    time_compat = 1.0
                elif abs((event_time - e).total_seconds()) < 86400 * 3:  # Within 3 days
                    time_compat = 0.7
            elif event_time is None or incident.time_range is None:
                # Unknown time - assume compatible
                time_compat = 0.6

            # Compute semantic similarity between surface and incident
            semantic_sim = incident.semantic_similarity_to_surface(surface, self.surfaces)

            # Compute decision - combine entity, time, and semantic signals
            # Weight: entities (0.5), semantic (0.3), time (0.2)
            posterior = identity_overlap * 0.5 + semantic_sim * 0.3 + time_compat * 0.2

            # Overlap strength also incorporates semantic similarity
            combined_overlap = identity_overlap * 0.7 + semantic_sim * 0.3

            decision = compute_grouping_score(
                incident.metrics(self.surfaces),
                surface.metrics(),
                overlap_strength=combined_overlap,
                posterior_compatibility=posterior,
                closure_risk=incident.metrics(self.surfaces).closure_risk * 0.5,  # Reduce closure penalty
            )

            if best_decision is None or decision.delta_score > best_decision.delta_score:
                best_decision = decision
                best_incident = iid

        # Apply decision
        if best_incident and best_decision:
            if best_decision.action == GroupingDecision.Action.MERGE:
                # Assimilate into existing incident
                incident = self.incidents[best_incident]
                incident.surface_ids.add(surface_id)

                # Update time range
                if event_time:
                    if incident.time_range:
                        s, e = incident.time_range
                        incident.time_range = (min(s, event_time), max(e, event_time))
                    else:
                        incident.time_range = (event_time, event_time)

                emissions.append(WeaverEmission(
                    emission_type=WeaverEmission.EmissionType.INCIDENT_ASSIMILATED,
                    object_id=best_incident,
                    target_id=surface_id,
                    delta_score=best_decision.delta_score,
                    rationale=best_decision.rationale,
                ))

                return best_incident, emissions

            elif best_decision.action == GroupingDecision.Action.COUPLE:
                # Related but not same - create new incident with metabolic edge
                pass  # Fall through to create new incident

            elif best_decision.action == GroupingDecision.Action.DEFER:
                # Emit inquiry and defer
                self.deferred.append(("incident", surface_id, best_incident))
                emissions.append(WeaverEmission(
                    emission_type=WeaverEmission.EmissionType.DEFERRED,
                    object_id=surface_id,
                    target_id=best_incident,
                    delta_score=best_decision.delta_score,
                    rationale=best_decision.rationale,
                    inquiry=best_decision.inquiry_seed,
                ))
                # Still create new incident for now

        # Create new incident
        incident_id = self._generate_id("inc", surface_id, *sorted(referents)[:3])
        incident = Incident(
            incident_id=incident_id,
            referents=frozenset(referents),
            surface_ids={surface_id},
            time_range=(event_time, event_time) if event_time else None,
        )

        self.incidents[incident_id] = incident

        # Update indices
        for ref in referents:
            if ref not in self._incident_by_referent:
                self._incident_by_referent[ref] = set()
            self._incident_by_referent[ref].add(incident_id)

        # Add metabolic edge if we had a candidate
        if best_incident and best_decision and best_decision.action == GroupingDecision.Action.COUPLE:
            incident.metabolic_edges[best_incident] = "RELATED_TO"
            self.incidents[best_incident].metabolic_edges[incident_id] = "RELATED_TO"

            emissions.append(WeaverEmission(
                emission_type=WeaverEmission.EmissionType.INCIDENT_RELATED,
                object_id=incident_id,
                target_id=best_incident,
                delta_score=best_decision.delta_score,
                rationale=best_decision.rationale,
            ))

        return incident_id, emissions

    # =========================================================================
    # L4: Case Processing
    # =========================================================================

    def route_incident_to_case(
        self,
        incident_id: str,
    ) -> Tuple[str, List[WeaverEmission]]:
        """
        Route an incident to a case.

        Returns: (case_id, emissions)
        """
        emissions = []
        incident = self.incidents.get(incident_id)
        if not incident:
            raise ValueError(f"Incident {incident_id} not found")

        # Find candidate cases by entity overlap
        candidate_ids = set()
        for ref in incident.referents:
            if ref in self._case_by_entity:
                candidate_ids.update(self._case_by_entity[ref])

        # ALSO find candidates by semantic similarity (embedding-based routing)
        # This allows incidents with "same story, different wording/anchors" to merge
        incident_emb = incident.compute_embedding(self.surfaces)
        if incident_emb and len(candidate_ids) < len(self.cases):
            for cid, case in self.cases.items():
                if cid not in candidate_ids:
                    sim = case.semantic_similarity_to_incident(incident, self.incidents, self.surfaces)
                    if sim > 0.75:  # High semantic similarity threshold
                        candidate_ids.add(cid)

        best_case = None
        best_decision = None

        for cid in candidate_ids:
            case = self.cases[cid]
            decision = case.should_admit_incident(incident, self.incidents, self.surfaces)

            if best_decision is None or decision.delta_score > best_decision.delta_score:
                best_decision = decision
                best_case = cid

        # Apply decision
        if best_case and best_decision:
            if best_decision.action == GroupingDecision.Action.MERGE:
                # Admit to case
                case = self.cases[best_case]
                case.incident_ids.add(incident_id)
                case.membrane.update(incident.referents)

                # Add spine edges to related incidents
                for existing_iid in case.incident_ids:
                    if existing_iid != incident_id:
                        existing = self.incidents[existing_iid]
                        overlap = incident.identity_overlap(existing)
                        if overlap > 0.5:
                            case.spine_edges.add((incident_id, existing_iid, "SAME_STORY"))

                emissions.append(WeaverEmission(
                    emission_type=WeaverEmission.EmissionType.CASE_GREW,
                    object_id=best_case,
                    target_id=incident_id,
                    delta_score=best_decision.delta_score,
                    rationale=best_decision.rationale,
                ))

                # Update indices
                for ref in incident.referents:
                    if ref not in self._case_by_entity:
                        self._case_by_entity[ref] = set()
                    self._case_by_entity[ref].add(best_case)

                return best_case, emissions

            elif best_decision.action == GroupingDecision.Action.COUPLE:
                # Metabolic relation only
                case = self.cases[best_case]
                case.metabolic_edges.add((incident_id, list(case.incident_ids)[0], "CONTEXT_FOR"))

                # Fall through to create new case

            elif best_decision.action == GroupingDecision.Action.DEFER:
                self.deferred.append(("case", incident_id, best_case))
                emissions.append(WeaverEmission(
                    emission_type=WeaverEmission.EmissionType.DEFERRED,
                    object_id=incident_id,
                    target_id=best_case,
                    delta_score=best_decision.delta_score,
                    rationale=best_decision.rationale,
                    inquiry=best_decision.inquiry_seed,
                ))

        # Create new case
        case_id = self._generate_id("case", incident_id)
        case = Case(
            case_id=case_id,
            incident_ids={incident_id},
            membrane=set(incident.referents),
        )

        self.cases[case_id] = case

        # Update indices
        for ref in incident.referents:
            if ref not in self._case_by_entity:
                self._case_by_entity[ref] = set()
            self._case_by_entity[ref].add(case_id)

        return case_id, emissions

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    def process_claim(
        self,
        claim_id: str,
        text: str,
        proposition_key: str,
        source: str,
        entities: Set[str],
        anchor_entities: Set[str],
        event_time: Optional[datetime] = None,
        embedding: Optional[List[float]] = None,
    ) -> Tuple[str, str, str, List[WeaverEmission]]:
        """
        Process a claim through all levels.

        Returns: (surface_id, incident_id, case_id, all_emissions)
        """
        all_emissions = []

        # L2: Metabolize into surface
        surface_id, surface_emissions = self.metabolize_claim(
            claim_id=claim_id,
            text=text,
            proposition_key=proposition_key,
            source=source,
            entities=entities,
            anchor_entities=anchor_entities,
            event_time=event_time,
            embedding=embedding,
        )
        all_emissions.extend(surface_emissions)

        # L3: Route to incident
        incident_id, incident_emissions = self.route_surface_to_incident(
            surface_id=surface_id,
            referents=anchor_entities,
            event_time=event_time,
        )
        all_emissions.extend(incident_emissions)

        # L4: Route to case
        case_id, case_emissions = self.route_incident_to_case(
            incident_id=incident_id,
        )
        all_emissions.extend(case_emissions)

        # Store emissions
        self.emissions.extend(all_emissions)

        return surface_id, incident_id, case_id, all_emissions

    # =========================================================================
    # Universal Triggers
    # =========================================================================

    def check_fission(self, case_id: str) -> Optional[Tuple[str, str]]:
        """
        Check if a case should fission (split into two).

        Returns: (new_case_id_1, new_case_id_2) if fission, None otherwise.
        """
        case = self.cases.get(case_id)
        if not case or len(case.incident_ids) < 3:
            return None

        metrics = case.metrics(self.incidents, self.surfaces)

        # Fission if conflict density is high and there's a natural split
        if metrics.conflict_density > 0.3 or metrics.health_score() < -0.5:
            # Find the cut point - incidents that are least connected
            # For now, simple heuristic: split by entity clustering
            # (In production, this would use graph clustering)
            pass

        return None

    def run_repair_cycle(self) -> List[WeaverEmission]:
        """
        Process deferred decisions with accumulated evidence.
        """
        emissions = []

        # Re-evaluate deferred decisions
        remaining = []
        for dtype, id1, id2 in self.deferred:
            # Try to resolve with current state
            # (In production, this might trigger LLM adjudication)
            remaining.append((dtype, id1, id2))

        self.deferred = remaining
        return emissions

    # =========================================================================
    # Summary
    # =========================================================================

    def summary(self) -> Dict:
        """Get summary metrics."""
        total_surfaces = len(self.surfaces)
        total_incidents = len(self.incidents)
        total_cases = len(self.cases)

        case_sizes = [len(c.incident_ids) for c in self.cases.values()]

        # Compute overall health
        case_health = []
        for c in self.cases.values():
            m = c.metrics(self.incidents, self.surfaces)
            case_health.append(m.health_score())

        return {
            "surfaces": total_surfaces,
            "incidents": total_incidents,
            "cases": total_cases,
            "largest_case": max(case_sizes) if case_sizes else 0,
            "multi_incident_cases": sum(1 for s in case_sizes if s > 1),
            "avg_case_health": sum(case_health) / len(case_health) if case_health else 0,
            "deferred_count": len(self.deferred),
            "total_emissions": len(self.emissions),
            "emission_types": {
                t.name: sum(1 for e in self.emissions if e.emission_type == t)
                for t in WeaverEmission.EmissionType
            },
        }
