"""
Case Formation Contracts
========================

Stable spec for L4 case formation via spine + metabolic edges.

Key invariants:
1. Spine edges define case membership (same_happening, update_to)
2. Metabolic edges link without merging (causes, response_to, phase_of, ...)
3. Referent roles are never suppressed by df
4. Person-only referents require additional witness for spine
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Set, Dict, List, Optional, FrozenSet
from datetime import datetime, timedelta


# =============================================================================
# Edge Type Taxonomy
# =============================================================================

class EdgeType(Enum):
    """Typed edges between incidents."""

    # Spine edges (membership-defining for L4 Case)
    SAME_HAPPENING = "same_happening"
    UPDATE_TO = "update_to_same_happening"  # Standardized with relational_experiment.py

    # Metabolic edges (link without fusing)
    CAUSES = "causes_or_triggers"
    RESPONSE_TO = "response_to"
    PHASE_OF = "phase_of"
    INVESTIGATES = "investigates"
    CHARGES = "charges_or_prosecutes"
    DISPUTES = "denies_or_disputes"
    CONTEXT_FOR = "context_for"
    ANALOGY = "analogy_or_frame"

    # Rejection
    UNRELATED = "unrelated"

    @property
    def is_spine(self) -> bool:
        """Does this edge type define case membership?"""
        return self in {EdgeType.SAME_HAPPENING, EdgeType.UPDATE_TO}

    @property
    def is_directed(self) -> bool:
        """Does this edge have a direction (A -> B)?"""
        return self in {
            EdgeType.CAUSES, EdgeType.RESPONSE_TO, EdgeType.INVESTIGATES,
            EdgeType.CHARGES, EdgeType.DISPUTES,
        }


# =============================================================================
# Entity Role Taxonomy
# =============================================================================

class EntityRole(Enum):
    """Roles entities play in an incident."""

    # Referent roles (define identity - NEVER suppressed)
    REFERENT_FACILITY = "referent_facility"   # Specific building/structure
    REFERENT_PERSON = "referent_person"       # Person the incident is about
    REFERENT_LOCATION = "referent_location"   # Specific place (district)
    REFERENT_OBJECT = "referent_object"       # Specific object (vehicle, document)

    # Context roles (may be suppressed if globally broad)
    BROAD_LOCATION = "broad_location"         # Country/city level
    AUTHORITY = "authority"                   # Government official
    RESPONDER = "responder"                   # Emergency services
    PUBLISHER = "publisher"                   # News source
    COMMENTARY = "commentary"                 # Person mentioned for context
    CONTEXT = "context"                       # General context entity

    @property
    def is_referent(self) -> bool:
        """Is this a referent role (defines identity)?"""
        return self in {
            EntityRole.REFERENT_FACILITY,
            EntityRole.REFERENT_PERSON,
            EntityRole.REFERENT_LOCATION,
            EntityRole.REFERENT_OBJECT,
        }

    @property
    def is_person_referent(self) -> bool:
        """Is this a person-type referent (needs additional witness)?"""
        return self == EntityRole.REFERENT_PERSON

    @property
    def is_suppressible(self) -> bool:
        """Can this role be suppressed if entity is globally broad?"""
        return self in {
            EntityRole.BROAD_LOCATION,
            EntityRole.AUTHORITY,
            EntityRole.RESPONDER,
            EntityRole.PUBLISHER,
            EntityRole.COMMENTARY,
            EntityRole.CONTEXT,
        }


# =============================================================================
# Role Artifact (per-incident)
# =============================================================================

@dataclass
class IncidentRoleArtifact:
    """Per-incident role labeling artifact (O(n) extraction)."""

    incident_id: str
    referent_entity_ids: FrozenSet[str]
    role_map: Dict[str, EntityRole]
    time_start: Optional[datetime] = None

    def get_person_referents(self) -> Set[str]:
        """Get entity IDs that are person referents."""
        return {
            eid for eid, role in self.role_map.items()
            if role.is_person_referent and eid in self.referent_entity_ids
        }

    def get_non_person_referents(self) -> Set[str]:
        """Get entity IDs that are non-person referents (facility, location)."""
        return {
            eid for eid in self.referent_entity_ids
            if eid not in self.get_person_referents()
        }


# =============================================================================
# Spine Gate (deterministic)
# =============================================================================

@dataclass
class SpineGateResult:
    """Result of spine gate evaluation."""
    is_spine: bool
    edge_type: EdgeType
    confidence: float
    reason: str
    shared_referents: FrozenSet[str] = field(default_factory=frozenset)


def evaluate_spine_gate(
    art_a: IncidentRoleArtifact,
    art_b: IncidentRoleArtifact,
    time_closeness_days: float = 30.0,
) -> SpineGateResult:
    """
    Deterministic spine gate evaluation.

    Rules:
    1. Must have shared referent(s)
    2. If only shared referent is a person, require additional witness:
       - Time closeness (within time_closeness_days)
       - OR shared non-person referent (facility, location)
    3. Otherwise, spine edge

    Returns SpineGateResult with edge type and reason.
    """

    # Rule 1: Check for shared referents
    shared_refs = art_a.referent_entity_ids & art_b.referent_entity_ids

    if not shared_refs:
        return SpineGateResult(
            is_spine=False,
            edge_type=EdgeType.UNRELATED,
            confidence=0.3,
            reason="No shared referents",
        )

    # Rule 2: Person-only referent safety clause
    shared_persons = art_a.get_person_referents() & art_b.get_person_referents()
    shared_non_persons = art_a.get_non_person_referents() & art_b.get_non_person_referents()

    if shared_refs == shared_persons and not shared_non_persons:
        # Only person referent(s) shared - need additional witness

        # Witness 1: Time closeness
        time_close = False
        if art_a.time_start and art_b.time_start:
            delta = abs((art_a.time_start - art_b.time_start).total_seconds())
            time_close = delta <= time_closeness_days * 86400

        if time_close:
            return SpineGateResult(
                is_spine=True,
                edge_type=EdgeType.SAME_HAPPENING,
                confidence=0.75,
                reason=f"Person referent + time witness: {shared_refs}",
                shared_referents=frozenset(shared_refs),
            )
        else:
            # No witness - downgrade to phase_of (metabolic)
            return SpineGateResult(
                is_spine=False,
                edge_type=EdgeType.PHASE_OF,
                confidence=0.6,
                reason=f"Person-only referent, no time witness: {shared_refs}",
                shared_referents=frozenset(shared_refs),
            )

    # Rule 3: Non-person referent or multiple referents - strong spine
    if shared_non_persons:
        return SpineGateResult(
            is_spine=True,
            edge_type=EdgeType.SAME_HAPPENING,
            confidence=0.9,
            reason=f"Facility/location referent: {shared_non_persons}",
            shared_referents=frozenset(shared_refs),
        )

    # Multiple person referents - also spine
    if len(shared_persons) >= 2:
        return SpineGateResult(
            is_spine=True,
            edge_type=EdgeType.SAME_HAPPENING,
            confidence=0.85,
            reason=f"Multiple person referents: {shared_persons}",
            shared_referents=frozenset(shared_refs),
        )

    # Fallback - should not reach here
    return SpineGateResult(
        is_spine=True,
        edge_type=EdgeType.SAME_HAPPENING,
        confidence=0.7,
        reason=f"Shared referents: {shared_refs}",
        shared_referents=frozenset(shared_refs),
    )


# =============================================================================
# DF Shrinkage (principled hub estimation)
# =============================================================================

@dataclass
class EntityDFEstimate:
    """Entity document frequency with shrinkage."""

    entity_id: str
    df_global: int      # From full corpus
    df_local: int       # From current sample
    df_shrunk: float    # Shrunk estimate

    @classmethod
    def compute(
        cls,
        entity_id: str,
        df_global: int,
        df_local: int,
        alpha: float = 0.9,
    ) -> "EntityDFEstimate":
        """
        Compute shrunk df estimate.

        df_shrunk = α * df_global + (1-α) * df_local

        High α (default 0.9) means biased samples can't distort commonness.
        """
        df_shrunk = alpha * df_global + (1 - alpha) * df_local
        return cls(
            entity_id=entity_id,
            df_global=df_global,
            df_local=df_local,
            df_shrunk=df_shrunk,
        )


def should_suppress_entity(
    entity_id: str,
    role: EntityRole,
    df_shrunk: float,
    global_total_incidents: int,
    threshold_fraction: float = 0.10,
) -> bool:
    """
    Determine if entity should be suppressed from referent consideration.

    Args:
        entity_id: The entity being evaluated
        role: The role this entity plays in the incident
        df_shrunk: Shrunk DF estimate (alpha * df_global + (1-alpha) * df_local)
        global_total_incidents: Total incidents in GLOBAL corpus (not biased sample)
        threshold_fraction: Fraction of global corpus above which entity is "broad"

    Rules:
    1. Never suppress referent roles (facility, person, location, object)
    2. Only suppress context roles if df_shrunk > threshold_fraction * global_total
    """

    # Rule 1: Never suppress referent roles
    if role.is_referent:
        return False

    # Rule 2: Suppress context roles if globally broad
    if role.is_suppressible:
        threshold = threshold_fraction * global_total_incidents
        return df_shrunk > threshold

    return False


# =============================================================================
# Invariants (STUBS - not yet enforced, for documentation purposes)
# =============================================================================
#
# NOTE: These invariants document the intended behavior but are not yet
# enforced in the codebase. They serve as spec for future validation tests.
#

def invariant_spine_only_membership(edges: List[dict], cases: List[Set[str]]) -> bool:
    """
    Invariant: Cases are formed ONLY from spine edges.

    Metabolic edges must not affect case membership.

    STATUS: STUB - returns True, not actually enforced.
    TODO: Implement full graph traversal to verify spine connectivity.
    """
    # STUB: Full implementation needs graph traversal to verify
    # that every pair in same case is reachable via spine edges only
    return True


def invariant_referent_never_suppressed(
    artifacts: Dict[str, IncidentRoleArtifact],
    suppressed_entities: Set[str],
) -> bool:
    """
    Invariant: Referent-role entities are never suppressed.

    Only context-role entities can be suppressed based on df.

    STATUS: ENFORCED - this one is actually checked.
    """
    for art in artifacts.values():
        for eid in art.referent_entity_ids:
            role = art.role_map.get(eid)
            if role and role.is_referent and eid in suppressed_entities:
                return False
    return True


def invariant_person_referent_needs_witness(
    art_a: IncidentRoleArtifact,
    art_b: IncidentRoleArtifact,
    edge_type: EdgeType,
) -> bool:
    """
    Invariant: Person-only shared referent requires witness for spine edge.

    If only shared referent is a person and no time/location witness,
    edge must be metabolic (phase_of), not spine.

    STATUS: STUB - returns True, enforced via evaluate_spine_gate() instead.
    """
    # This invariant is enforced by evaluate_spine_gate(), not here.
    # This function exists for documentation/spec purposes.
    return True
