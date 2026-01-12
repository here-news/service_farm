"""
Membrane Decision Table: Pure Function Signatures
==================================================

This module defines the membrane contract as pure functions.
Types are derived from what these functions need, not the other way around.

The membrane decides:
1. Whether an incident belongs to a story (membership)
2. Why it belongs (core_reason)
3. What relationship it has (link_type)

These are orthogonal concerns, not a single enum.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Set, List, Optional, Tuple, Dict, Any, Literal
from datetime import datetime


# =============================================================================
# MINIMAL ENUMS (orthogonal concerns)
# =============================================================================

class Membership(Enum):
    """Membership level - kept small."""
    CORE = "core"           # Part of story
    PERIPHERY = "periphery" # Attached but not counted
    REJECT = "reject"       # Not attached


class CoreReason(Enum):
    """Why something is core - separate from membership."""
    ANCHOR = "anchor"       # Core-A: focal entity is anchor
    WARRANT = "warrant"     # Core-B: structural witness exists


class LinkType(Enum):
    """Relationship type - separate from membership."""
    MEMBER = "member"           # Inside story membrane
    RELATED_STORY = "related"   # Cross-story link (Wong Kwok-ngon saga)
    PERIPHERY_LINK = "periphery"  # Semantic-only attachment


# =============================================================================
# FOCAL SET (minimal structure for story identity)
# =============================================================================

@dataclass
class FocalSet:
    """
    The focal set defines story identity.

    Default: single-spine (kind="single", co_spines empty)
    Promotion to dyad/set requires explicit structural evidence.
    """
    primary: str                          # Main focal entity ID
    co_spines: Set[str] = field(default_factory=set)  # Usually empty
    kind: Literal["single", "dyad", "set"] = "single"

    # Evidence for multi-spine (empty for single)
    promotion_evidence: List[str] = field(default_factory=list)  # Constraint IDs

    def all_spines(self) -> Set[str]:
        """Return all focal entities."""
        return {self.primary} | self.co_spines

    def identity_key(self) -> str:
        """Canonical identity for hashing."""
        return "|".join(sorted(self.all_spines()))


# =============================================================================
# WITNESS = CONSTRAINT (not a parallel system)
# =============================================================================

# Witnesses ARE constraints in the ledger, not separate objects.
# We just define the kinds of constraints that count as witnesses.

WITNESS_KINDS = {"time", "geo", "event_type", "motif", "context"}
SEMANTIC_KINDS = {"embedding", "llm_proposal", "title_similarity"}

def is_structural_witness(constraint_kind: str) -> bool:
    """A constraint is a structural witness if it's in WITNESS_KINDS."""
    return constraint_kind in WITNESS_KINDS

def is_semantic_only(constraint_kind: str) -> bool:
    """A constraint is semantic-only if it's in SEMANTIC_KINDS."""
    return constraint_kind in SEMANTIC_KINDS


# =============================================================================
# MEMBRANE DECISION TABLE (pure functions)
# =============================================================================

@dataclass
class MembershipDecision:
    """Result of membrane decision."""
    membership: Membership
    core_reason: Optional[CoreReason] = None  # Only set if CORE
    link_type: LinkType = LinkType.MEMBER
    witnesses: List[str] = field(default_factory=list)  # Constraint IDs
    blocked_reason: Optional[str] = None  # Why not core
    constraint_source: Optional[str] = None  # "structural" or "semantic_proposal"


def classify_incident_membership(
    incident_anchors: Set[str],
    focal_set: FocalSet,
    constraints: List[Dict[str, Any]],  # From ledger, with 'kind' field
    hub_entities: Set[str],
) -> MembershipDecision:
    """
    MEMBRANE DECISION TABLE

    Inputs:
        incident_anchors: Entities that are anchors in this incident
        focal_set: The story's focal set (primary + optional co_spines)
        constraints: Relevant constraints from ledger
        hub_entities: Known hub entities (cannot provide witnesses)

    Returns:
        MembershipDecision with membership, reason, link_type, witnesses

    Decision logic:
        1. Core-A: focal.primary in incident_anchors (single)
                   OR all spines in anchors (dyad/set)
        2. Core-B: has ≥2 structural witnesses, ≥1 non-time
        3. Reject: only hub anchors shared
        4. Periphery: otherwise (semantic-only)
    """
    all_spines = focal_set.all_spines()

    # ======================
    # CORE-A CHECK (anchor)
    # ======================

    if focal_set.kind == "single":
        # Single spine: primary must be anchor
        if focal_set.primary in incident_anchors:
            return MembershipDecision(
                membership=Membership.CORE,
                core_reason=CoreReason.ANCHOR,
                link_type=LinkType.MEMBER,
                constraint_source="structural",  # Anchor match is structural
            )
    else:
        # Dyad/set: ALL spines must be anchors (or dyad motif present)
        if all_spines <= incident_anchors:
            return MembershipDecision(
                membership=Membership.CORE,
                core_reason=CoreReason.ANCHOR,
                link_type=LinkType.MEMBER,
                constraint_source="structural",  # Anchor match is structural
            )

    # ======================
    # REJECT CHECK (hub-only)
    # ======================

    # If the only shared anchors are hubs, reject
    shared_anchors = incident_anchors & all_spines
    non_hub_shared = shared_anchors - hub_entities

    # Also check: if incident anchors are ALL hubs, reject
    incident_non_hubs = incident_anchors - hub_entities
    if not incident_non_hubs:
        return MembershipDecision(
            membership=Membership.REJECT,
            link_type=LinkType.RELATED_STORY,  # Could be related but not member
            blocked_reason="incident has only hub anchors",
        )

    # ======================
    # CORE-B CHECK (warrant)
    # ======================

    # Count structural witnesses (excluding hub-sourced)
    structural_witnesses = []
    has_non_time_witness = False

    for c in constraints:
        kind = c.get("kind", "")
        source_entity = c.get("source_entity", "")

        # Skip if sourced from a hub
        if source_entity in hub_entities:
            continue

        if is_structural_witness(kind):
            structural_witnesses.append(c.get("id", ""))
            if kind != "time":
                has_non_time_witness = True

    # Core-B requires: ≥2 witnesses, at least one non-time
    if len(structural_witnesses) >= 2 and has_non_time_witness:
        return MembershipDecision(
            membership=Membership.CORE,
            core_reason=CoreReason.WARRANT,
            link_type=LinkType.MEMBER,
            witnesses=structural_witnesses,
            constraint_source="structural",  # Warrant is structural evidence
        )

    # ======================
    # PERIPHERY (default)
    # ======================

    # Check for semantic-only constraints
    semantic_constraints = [c for c in constraints if is_semantic_only(c.get("kind", ""))]

    # Has some evidence but not enough for core
    if structural_witnesses:
        blocked_reason = f"only {len(structural_witnesses)} witness(es), need ≥2 with non-time"
        constraint_source = "structural"  # Had structural, just not enough
    elif semantic_constraints:
        blocked_reason = "no structural witnesses"
        constraint_source = "semantic_proposal"  # Only semantic evidence
    else:
        blocked_reason = "no structural witnesses"
        constraint_source = None  # No evidence at all

    return MembershipDecision(
        membership=Membership.PERIPHERY,
        link_type=LinkType.PERIPHERY_LINK,
        witnesses=structural_witnesses,
        blocked_reason=blocked_reason,
        constraint_source=constraint_source,
    )


def can_promote_to_multi_spine(
    primary: str,
    candidate_co_spine: str,
    co_occurrence_count: int,
    total_incidents: int,
    pmi_score: float,
    hub_entities: Set[str],
) -> Tuple[bool, str]:
    """
    MULTI-SPINE PROMOTION RULE

    Promotion to dyad/set requires discriminative evidence:
    1. Neither entity is a hub
    2. Repeated co-anchor (≥3 incidents)
    3. High PMI/lift (≥2.0, meaning 2x more likely than chance)

    Returns:
        (can_promote, reason)
    """
    # Gate 1: No hubs
    if primary in hub_entities:
        return False, f"{primary} is a hub"
    if candidate_co_spine in hub_entities:
        return False, f"{candidate_co_spine} is a hub"

    # Gate 2: Repeated co-anchor
    if co_occurrence_count < 3:
        return False, f"only {co_occurrence_count} co-occurrences, need ≥3"

    # Gate 3: Discriminative (PMI ≥ 2.0)
    if pmi_score < 2.0:
        return False, f"PMI={pmi_score:.2f} < 2.0, not discriminative"

    return True, f"co-occurs {co_occurrence_count}x, PMI={pmi_score:.2f}"


def semantic_cannot_force_core(
    constraints: List[Dict[str, Any]],
) -> Tuple[bool, str]:
    """
    ANTI-TRAP RULE: Check if constraints would allow core merge.

    Core requires at least one non-semantic constraint.

    Returns:
        (has_non_semantic, reason)
    """
    for c in constraints:
        kind = c.get("kind", "")
        if is_structural_witness(kind):
            return True, f"has structural witness: {kind}"

    return False, "semantic-only: cannot force core"


# =============================================================================
# METRICS (from VOCABULARY.md)
# =============================================================================

def compute_core_leak_rate(
    core_incident_ids: Set[str],
    incidents_with_spine_anchor: Set[str],
) -> float:
    """
    core_leak_rate = (# core without spine anchor) / (# core)

    Target: 0.0 (all core incidents are Core-A or valid Core-B)
    """
    if not core_incident_ids:
        return 0.0

    without_anchor = core_incident_ids - incidents_with_spine_anchor
    return len(without_anchor) / len(core_incident_ids)


def compute_witness_scarcity(
    blocked_count: int,
    candidate_count: int,
) -> float:
    """
    witness_scarcity = blocked / candidates

    High value means extraction is underpowered.
    """
    if candidate_count == 0:
        return 0.0
    return blocked_count / candidate_count


# =============================================================================
# TESTS (inline, to validate contract)
# =============================================================================

def _test_membrane_contract():
    """Basic contract tests - run with: python -c 'from membrane import _test_membrane_contract; _test_membrane_contract()'"""

    hub_entities = {"Hong Kong", "John Lee"}

    # Test 1: Core-A (single spine, anchor present)
    focal = FocalSet(primary="Wang Fuk Court")
    result = classify_incident_membership(
        incident_anchors={"Wang Fuk Court", "Fire Services"},
        focal_set=focal,
        constraints=[],
        hub_entities=hub_entities,
    )
    assert result.membership == Membership.CORE
    assert result.core_reason == CoreReason.ANCHOR
    print("✓ Core-A: spine as anchor")

    # Test 2: Core-B (structural witnesses)
    result = classify_incident_membership(
        incident_anchors={"Tai Po", "Chris Tang"},
        focal_set=focal,
        constraints=[
            {"id": "c1", "kind": "geo", "source_entity": "Tai Po"},
            {"id": "c2", "kind": "time", "source_entity": ""},
        ],
        hub_entities=hub_entities,
    )
    assert result.membership == Membership.CORE
    assert result.core_reason == CoreReason.WARRANT
    print("✓ Core-B: geo + time witnesses")

    # Test 3: Reject (hub-only anchors)
    result = classify_incident_membership(
        incident_anchors={"Hong Kong", "John Lee"},
        focal_set=focal,
        constraints=[],
        hub_entities=hub_entities,
    )
    assert result.membership == Membership.REJECT
    print("✓ Reject: hub-only anchors")

    # Test 4: Periphery (time-only witness)
    result = classify_incident_membership(
        incident_anchors={"Tai Po", "Chris Tang"},
        focal_set=focal,
        constraints=[
            {"id": "c1", "kind": "time", "source_entity": ""},
        ],
        hub_entities=hub_entities,
    )
    assert result.membership == Membership.PERIPHERY
    print("✓ Periphery: time-only witness insufficient")

    # Test 5: Semantic cannot force core
    has_structural, reason = semantic_cannot_force_core([
        {"kind": "embedding"},
        {"kind": "llm_proposal"},
    ])
    assert not has_structural
    print("✓ Anti-trap: semantic-only cannot force core")

    # Test 6: Multi-spine promotion blocked for hub
    can_promote, reason = can_promote_to_multi_spine(
        primary="Wang Fuk Court",
        candidate_co_spine="Hong Kong",
        co_occurrence_count=10,
        total_incidents=100,
        pmi_score=3.0,
        hub_entities=hub_entities,
    )
    assert not can_promote
    print("✓ Multi-spine: hub blocked")

    # Test 7: Multi-spine promotion allowed for discriminative pair
    can_promote, reason = can_promote_to_multi_spine(
        primary="Do Kwon",
        candidate_co_spine="Terraform Labs",
        co_occurrence_count=15,
        total_incidents=100,
        pmi_score=4.5,
        hub_entities=hub_entities,
    )
    assert can_promote
    print("✓ Multi-spine: discriminative pair allowed")

    print("\n✅ All membrane contract tests pass")


if __name__ == "__main__":
    _test_membrane_contract()
