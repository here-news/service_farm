"""
Incident Routing - Pure functions for L2→L3 routing decisions.

Routes surfaces to incidents based on:
- Anchor overlap (MIN_SHARED_ANCHORS)
- Companion compatibility (COMPANION_OVERLAP_THRESHOLD)
- Underpowered allowance (sparse companions → benefit of doubt)

Emits BRIDGE_BLOCKED signals when anchors match but companions conflict.

Pure function - no DB, no LLM.
Candidate retrieval happens outside kernel (PartitionSnapshot).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, FrozenSet, List, Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from ..contracts.state import PartitionSnapshot

from ..contracts.evidence import ClaimEvidence
from ..contracts.state import (
    SurfaceState,
    IncidentState,
    PartitionSnapshot,
    compute_incident_signature,
)
from ..contracts.traces import DecisionTrace, FeatureVector, generate_trace_id
from ..contracts.signals import (
    EpistemicSignal,
    SignalType,
    Severity,
    generate_signal_id,
)


class RouteOutcome(Enum):
    """Outcome of routing decision."""
    JOINED_EXISTING = "joined_existing"
    JOINED_BY_SIGNATURE = "joined_by_signature"  # Exact signature match
    CREATED_NEW = "created_new"
    ALREADY_ROUTED = "already_routed"


@dataclass(frozen=True)
class RoutingParams:
    """Parameters for incident routing.

    Matches current weaver semantics.
    """
    min_shared_anchors: int = 2
    companion_overlap_threshold: float = 0.15
    kernel_version: str = "1.0.0"

    @property
    def params_hash(self) -> str:
        """Hash of parameters for trace reproducibility."""
        import hashlib
        content = f"{self.min_shared_anchors}|{self.companion_overlap_threshold}"
        return hashlib.sha256(content.encode()).hexdigest()[:8]


@dataclass(frozen=True)
class CandidateScore:
    """Scored candidate incident."""
    incident_id: str
    incident_signature: str
    shared_anchors: FrozenSet[str]
    anchor_count: int
    companion_overlap: float
    is_underpowered: bool  # True if companions too sparse to check


@dataclass(frozen=True)
class RoutingResult:
    """Result of routing decision."""
    outcome: RouteOutcome
    incident_id: str
    incident_signature: str
    is_new: bool
    trace: DecisionTrace
    signals: Tuple[EpistemicSignal, ...]


def find_candidates(
    surface: SurfaceState,
    snapshot: PartitionSnapshot,
    params: RoutingParams,
) -> List[CandidateScore]:
    """Find candidate incidents for a surface.

    Pure function - operates on PartitionSnapshot.

    Args:
        surface: The surface to route
        snapshot: Partition state containing candidate incidents
        params: Routing parameters

    Returns:
        List of scored candidates (may include rejected candidates)
    """
    candidates = []
    surface_anchors = surface.anchor_entities
    surface_companions = surface.entities - surface.anchor_entities

    for incident in snapshot.incidents:
        # Compute anchor overlap
        shared_anchors = surface_anchors & incident.anchor_entities
        anchor_count = len(shared_anchors)

        if anchor_count < params.min_shared_anchors:
            continue  # Not enough anchor overlap

        # Compute companion compatibility
        # IncidentState stores companion_entities separately
        incident_companions = incident.companion_entities

        if surface_companions and incident_companions:
            intersection = surface_companions & incident_companions
            union = surface_companions | incident_companions
            companion_overlap = len(intersection) / len(union) if union else 0.0
            is_underpowered = False
        else:
            # Sparse companions - mark as underpowered (benefit of doubt)
            companion_overlap = 0.5  # Default score for underpowered
            is_underpowered = True

        candidates.append(CandidateScore(
            incident_id=incident.id,
            incident_signature=incident.signature,
            shared_anchors=shared_anchors,
            anchor_count=anchor_count,
            companion_overlap=companion_overlap,
            is_underpowered=is_underpowered,
        ))

    return candidates


def decide_route(
    surface: SurfaceState,
    candidates: List[CandidateScore],
    params: RoutingParams,
    snapshot: Optional[PartitionSnapshot] = None,
) -> RoutingResult:
    """Decide which incident a surface should join.

    Pure function - emits traces and signals.

    Logic:
    0. Check for exact signature match in snapshot (deterministic join)
    1. Filter candidates by companion overlap threshold (unless underpowered)
    2. Emit BRIDGE_BLOCKED for rejected candidates with shared anchors
    3. Join best candidate or create new incident

    Args:
        surface: The surface to route
        candidates: Scored candidates from find_candidates
        params: Routing parameters
        snapshot: Optional snapshot for signature matching

    Returns:
        RoutingResult with outcome, traces, and signals
    """
    surface_companions = surface.entities - surface.anchor_entities

    # Step 0: Check for exact signature match (deterministic routing)
    # This ensures traces align with persisted reality when MERGE collapses
    computed_signature = compute_incident_signature(
        surface.anchor_entities,
        surface.time_start,
    )

    if snapshot:
        for incident in snapshot.incidents:
            if incident.signature == computed_signature:
                # Exact match - this is a deterministic join
                trace = DecisionTrace(
                    id=generate_trace_id(),
                    decision_type="incident_membership",
                    subject_id=surface.key.signature,
                    target_id=incident.signature,
                    candidate_ids=frozenset(c.incident_signature for c in candidates),
                    outcome="joined_by_signature",
                    features=FeatureVector(
                        anchor_overlap=1.0,  # Perfect match
                        companion_jaccard=1.0,
                    ),
                    rules_fired=frozenset({"SIGNATURE_MATCH"}),
                    params_hash=params.params_hash,
                    kernel_version=params.kernel_version,
                    timestamp=datetime.utcnow(),
                )
                return RoutingResult(
                    outcome=RouteOutcome.JOINED_BY_SIGNATURE,
                    incident_id=incident.id,
                    incident_signature=incident.signature,
                    is_new=False,
                    trace=trace,
                    signals=(),
                )

    # Partition into compatible vs blocked
    compatible = []
    blocked = []

    for candidate in candidates:
        if candidate.is_underpowered:
            # Underpowered → benefit of doubt → compatible
            compatible.append(candidate)
        elif candidate.companion_overlap >= params.companion_overlap_threshold:
            compatible.append(candidate)
        else:
            # Shared anchors but incompatible companions → bridge blocked
            blocked.append(candidate)

    # Emit BRIDGE_BLOCKED signals for blocked candidates
    signals: List[EpistemicSignal] = []
    for candidate in blocked:
        blocking_entity = next(iter(candidate.shared_anchors)) if candidate.shared_anchors else None
        signals.append(EpistemicSignal(
            id=generate_signal_id(),
            signal_type=SignalType.BRIDGE_BLOCKED,
            subject_id=surface.key.signature,
            subject_type="surface",
            severity=Severity.WARNING,
            evidence={
                "blocking_entity": blocking_entity,
                "shared_anchors": sorted(candidate.shared_anchors),
                "surface_companions": sorted(list(surface_companions)[:5]),
                "companion_overlap": candidate.companion_overlap,
                "threshold": params.companion_overlap_threshold,
                "incident_signature": candidate.incident_signature,
            },
            resolution_hint=f"Shared anchor '{blocking_entity}' does not bind because companion contexts are incompatible",
            timestamp=datetime.utcnow(),
        ))

    # Decide outcome
    if compatible:
        # Sort by companion overlap (descending), then anchor count
        compatible.sort(key=lambda c: (-c.companion_overlap, -c.anchor_count))
        best = compatible[0]

        trace = DecisionTrace(
            id=generate_trace_id(),
            decision_type="incident_membership",
            subject_id=surface.key.signature,
            target_id=best.incident_signature,
            candidate_ids=frozenset(c.incident_signature for c in candidates),
            outcome="joined_existing",
            features=FeatureVector(
                anchor_overlap=best.anchor_count / max(len(surface.anchor_entities), 1),
                companion_jaccard=best.companion_overlap,
            ),
            rules_fired=_compute_rules_fired(best, params),
            params_hash=params.params_hash,
            kernel_version=params.kernel_version,
            timestamp=datetime.utcnow(),
        )

        return RoutingResult(
            outcome=RouteOutcome.JOINED_EXISTING,
            incident_id=best.incident_id,
            incident_signature=best.incident_signature,
            is_new=False,
            trace=trace,
            signals=tuple(signals),
        )
    else:
        # Create new incident
        new_signature = compute_incident_signature(
            surface.anchor_entities,
            surface.time_start,
        )

        trace = DecisionTrace(
            id=generate_trace_id(),
            decision_type="incident_membership",
            subject_id=surface.key.signature,
            target_id=new_signature,
            candidate_ids=frozenset(c.incident_signature for c in candidates),
            outcome="created_new",
            features=FeatureVector(
                anchor_overlap=0.0,  # No match
                companion_jaccard=0.0,
            ),
            rules_fired=frozenset({"NO_COMPATIBLE_CANDIDATE"}),
            params_hash=params.params_hash,
            kernel_version=params.kernel_version,
            timestamp=datetime.utcnow(),
        )

        return RoutingResult(
            outcome=RouteOutcome.CREATED_NEW,
            incident_id="",  # Caller creates new incident
            incident_signature=new_signature,
            is_new=True,
            trace=trace,
            signals=tuple(signals),
        )


def _compute_rules_fired(
    candidate: CandidateScore,
    params: RoutingParams,
) -> FrozenSet[str]:
    """Compute which rules fired for this routing decision."""
    rules = set()

    # Anchor rule
    if candidate.anchor_count >= params.min_shared_anchors:
        rules.add("ANCHOR_OVERLAP_PASS")

    # Companion rule
    if candidate.is_underpowered:
        rules.add("COMPANION_UNDERPOWERED_ALLOW")
    elif candidate.companion_overlap >= params.companion_overlap_threshold:
        rules.add("COMPANION_OVERLAP_PASS")

    return frozenset(rules)


def compute_companion_overlap(
    set_a: FrozenSet[str],
    set_b: FrozenSet[str],
) -> Tuple[float, bool]:
    """Compute Jaccard overlap between companion sets.

    Returns:
        (overlap, is_underpowered)
    """
    if not set_a or not set_b:
        return 0.5, True  # Underpowered

    intersection = set_a & set_b
    union = set_a | set_b

    if not union:
        return 0.5, True

    return len(intersection) / len(union), False
