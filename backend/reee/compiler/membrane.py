"""
Membrane Compiler: Deterministic topology decisions from artifacts.

This module is the SOLE authority for topology mutations. No other code
may create spine edges or merge cases. All decisions are deterministic
given the input artifacts.

Rules:
1. No referent overlap → context_for/unrelated + PERIPHERY/DEFER
2. Non-person referent overlap → spine (same_happening/update_to) + MERGE
3. Person-only overlap → require witness, else metabolic + PERIPHERY
4. Low confidence → DEFER regardless
"""

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Optional, Set


# =============================================================================
# Structural Language (finite, closed)
# =============================================================================

class Action(Enum):
    """Case membership actions - the only valid topology mutations."""
    MERGE = "merge"          # Join into same case (spine edge)
    PERIPHERY = "periphery"  # Link but don't merge (metabolic edge)
    DEFER = "defer"          # Insufficient signal, emit inquiry


class EdgeType(Enum):
    """Edge classifications - exhaustive."""
    # Spine (opens case membership)
    SAME_HAPPENING = "same_happening"
    UPDATE_TO = "update_to"
    # Metabolic (links without merging)
    CAUSES = "causes"
    RESPONSE_TO = "response_to"
    PHASE_OF = "phase_of"
    INVESTIGATES = "investigates"
    CHARGES = "charges"
    DISPUTES = "disputes"
    CONTEXT_FOR = "context_for"
    ANALOGY = "analogy"
    # No relationship
    UNRELATED = "unrelated"


class ReferentRole(Enum):
    """Referent categories - minimal set."""
    PLACE = "place"      # Specific location/facility (never demoted)
    PERSON = "person"    # Individual or named group
    OBJECT = "object"    # Vehicle, artifact, document, etc.


# =============================================================================
# Artifact Schema (input to membrane)
# =============================================================================

@dataclass(frozen=True)
class Referent:
    """A referent entity from artifact extraction."""
    entity_id: str
    role: ReferentRole


@dataclass(frozen=True)
class IncidentArtifact:
    """LLM-extracted artifact for a single incident.

    This is the ONLY input the membrane sees. No raw entity types,
    no database lookups, no heuristics.
    """
    incident_id: str
    referents: FrozenSet[Referent]
    contexts: FrozenSet[str]  # Entity IDs that are context-only
    time_start: Optional[float]  # Unix timestamp, if known
    confidence: float  # 0.0 to 1.0


# =============================================================================
# Membrane Decision (output)
# =============================================================================

@dataclass(frozen=True)
class MembraneDecision:
    """The compiler's decision for a pair of incidents."""
    edge_type: EdgeType
    action: Action
    confidence: float
    reason: str
    shared_referents: FrozenSet[Referent]


# =============================================================================
# Compiler Parameters (configurable per domain/orchestrator)
# =============================================================================

# Default thresholds
CONFIDENCE_THRESHOLD_MERGE = 0.7
CONFIDENCE_THRESHOLD_PERIPHERY = 0.4
TIME_WITNESS_WINDOW_SECONDS = 7 * 24 * 3600  # 7 days
UPDATE_THRESHOLD_SECONDS = 24 * 3600  # 24 hours


@dataclass(frozen=True)
class CompilerParams:
    """Parameters supplied by orchestrator - not hard-coded in compiler."""
    confidence_merge: float = CONFIDENCE_THRESHOLD_MERGE
    confidence_periphery: float = CONFIDENCE_THRESHOLD_PERIPHERY
    time_witness_window: float = TIME_WITNESS_WINDOW_SECONDS
    update_threshold: float = UPDATE_THRESHOLD_SECONDS


# Default params for backward compatibility
DEFAULT_PARAMS = CompilerParams()


def compile_pair(
    art1: IncidentArtifact,
    art2: IncidentArtifact,
    params: CompilerParams = DEFAULT_PARAMS,
) -> MembraneDecision:
    """
    Compile two incident artifacts into a membrane decision.

    This function embodies ALL topology rules. It is deterministic:
    same inputs always produce same outputs.
    """
    # Compute referent overlap
    refs1 = {(r.entity_id, r.role) for r in art1.referents}
    refs2 = {(r.entity_id, r.role) for r in art2.referents}
    shared_ids = {eid for eid, _ in refs1} & {eid for eid, _ in refs2}

    shared_referents = frozenset(
        r for r in art1.referents if r.entity_id in shared_ids
    )

    # Rule 4: Low confidence → DEFER
    min_confidence = min(art1.confidence, art2.confidence)
    if min_confidence < params.confidence_periphery:
        return MembraneDecision(
            edge_type=EdgeType.UNRELATED,
            action=Action.DEFER,
            confidence=min_confidence,
            reason="Low artifact confidence",
            shared_referents=shared_referents,
        )

    # Rule 1: No referent overlap → context_for or unrelated
    if not shared_referents:
        return MembraneDecision(
            edge_type=EdgeType.CONTEXT_FOR if _has_context_overlap(art1, art2) else EdgeType.UNRELATED,
            action=Action.PERIPHERY if _has_context_overlap(art1, art2) else Action.DEFER,
            confidence=min_confidence,
            reason="No shared referents",
            shared_referents=frozenset(),
        )

    # Categorize shared referents
    has_place = any(r.role == ReferentRole.PLACE for r in shared_referents)
    has_object = any(r.role == ReferentRole.OBJECT for r in shared_referents)
    has_person = any(r.role == ReferentRole.PERSON for r in shared_referents)
    person_only = has_person and not has_place and not has_object

    # Rule 3: Person-only overlap requires witness
    if person_only:
        has_witness = _has_time_witness(art1, art2, params)
        if not has_witness:
            return MembraneDecision(
                edge_type=EdgeType.PHASE_OF,
                action=Action.PERIPHERY,
                confidence=min_confidence * 0.8,
                reason="Person-only overlap without time witness",
                shared_referents=shared_referents,
            )

    # Rule 2: Non-person referent overlap → spine
    # Determine if update_to (temporal succession) or same_happening
    edge_type = _classify_spine_edge(art1, art2, params)

    # Confidence determines MERGE vs PERIPHERY
    if min_confidence >= params.confidence_merge:
        action = Action.MERGE
    else:
        action = Action.PERIPHERY

    return MembraneDecision(
        edge_type=edge_type,
        action=action,
        confidence=min_confidence,
        reason=f"Shared referents: {[r.entity_id for r in shared_referents]}",
        shared_referents=shared_referents,
    )


def _has_context_overlap(art1: IncidentArtifact, art2: IncidentArtifact) -> bool:
    """Check if incidents share any context entities."""
    return bool(art1.contexts & art2.contexts)


def _has_time_witness(
    art1: IncidentArtifact,
    art2: IncidentArtifact,
    params: CompilerParams,
) -> bool:
    """Check if incidents are within time witness window."""
    if art1.time_start is None or art2.time_start is None:
        return False
    return abs(art1.time_start - art2.time_start) <= params.time_witness_window


def _classify_spine_edge(
    art1: IncidentArtifact,
    art2: IncidentArtifact,
    params: CompilerParams,
) -> EdgeType:
    """Classify as same_happening or update_to based on temporal relationship."""
    if art1.time_start is None or art2.time_start is None:
        return EdgeType.SAME_HAPPENING

    # If more than update_threshold apart, likely an update
    if abs(art1.time_start - art2.time_start) > params.update_threshold:
        return EdgeType.UPDATE_TO

    return EdgeType.SAME_HAPPENING


# =============================================================================
# Invariants (for testing)
# =============================================================================

def assert_invariants(decision: MembraneDecision) -> None:
    """
    Validate membrane decision invariants.
    Raises AssertionError if any invariant is violated.
    """
    # Spine edges require MERGE or high-confidence PERIPHERY
    if decision.edge_type in (EdgeType.SAME_HAPPENING, EdgeType.UPDATE_TO):
        assert decision.action in (Action.MERGE, Action.PERIPHERY), \
            f"Spine edge {decision.edge_type} cannot have action {decision.action}"
        assert decision.shared_referents, \
            "Spine edge requires shared referents"

    # DEFER requires low confidence or no signal
    if decision.action == Action.DEFER:
        assert decision.confidence < CONFIDENCE_THRESHOLD_PERIPHERY or \
               not decision.shared_referents, \
            "DEFER should only occur with low confidence or no shared referents"

    # Metabolic edges should not MERGE
    metabolic_edges = {
        EdgeType.CAUSES, EdgeType.RESPONSE_TO, EdgeType.PHASE_OF,
        EdgeType.INVESTIGATES, EdgeType.CHARGES, EdgeType.DISPUTES,
        EdgeType.CONTEXT_FOR, EdgeType.ANALOGY,
    }
    if decision.edge_type in metabolic_edges:
        assert decision.action != Action.MERGE, \
            f"Metabolic edge {decision.edge_type} cannot MERGE"
