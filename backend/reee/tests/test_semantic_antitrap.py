"""
Semantic Anti-Trap Tests
========================

Tests that semantic-only constraints cannot create or merge structural cores.
This is the "percolation guard" that prevents pure embedding similarity from
destroying topology.

Key invariants tested:
1. Semantic-only constraints can attach to periphery but not create cores
2. No transitive closure through semantic-only edges
3. L3 membrane formation requires structural (entity overlap) evidence
4. High semantic similarity alone is insufficient for incident merge

The anti-trap principle:
- Semantic similarity is CONTEXTUAL evidence, not STRUCTURAL identity
- Two claims can be highly similar semantically but belong to different incidents
- Example: "Fire kills 10" and "Fire kills 12" are semantically similar
          but could be different fires in different places
"""

import pytest
from dataclasses import dataclass, field
from typing import Set, Optional, List
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reee.types import (
    Claim, Surface, Constraint, ConstraintType as REEEConstraintType, ConstraintLedger
)

# NOTE: This test uses its own ConstraintType to test anti-trap invariants.
# The REEE ConstraintType is {STRUCTURAL, SEMANTIC, TYPED, TEMPORAL, META}
# which describes the constraint source, not agreement/disagreement.
# For anti-trap tests, we care about semantic vs structural binding.


# ============================================================================
# LOCAL TEST TYPES (for testing anti-trap invariants)
# ============================================================================

class LocalConstraintType(Enum):
    """Local constraint type for anti-trap tests (relationship type)."""
    BINDING = "binding"      # Claims bind together (agree or relate)
    CONFLICT = "conflict"    # Claims conflict (disagree)


# Alias for cleaner test code
ConstraintType = LocalConstraintType


class ConstraintStrength(Enum):
    """Classification of constraint binding strength."""
    STRUCTURAL = "structural"  # Entity overlap, temporal co-occurrence
    SEMANTIC = "semantic"      # Embedding similarity only
    MIXED = "mixed"            # Both structural and semantic evidence


@dataclass
class ClassifiedConstraint:
    """Constraint with explicit strength classification."""
    source_id: str
    target_id: str
    constraint_type: ConstraintType
    strength: ConstraintStrength
    semantic_score: float = 0.0
    entity_overlap: Set[str] = field(default_factory=set)

    def is_structural(self) -> bool:
        """Returns True if constraint has structural grounding."""
        return self.strength in (ConstraintStrength.STRUCTURAL, ConstraintStrength.MIXED)

    def is_semantic_only(self) -> bool:
        """Returns True if constraint is purely semantic (no entity overlap)."""
        return self.strength == ConstraintStrength.SEMANTIC


# ============================================================================
# ANTI-TRAP L3 KERNEL
# ============================================================================

class AntiTrapL3Kernel:
    """
    L3 kernel that enforces the semantic anti-trap invariant.

    Key behavior:
    - Structural constraints can create/merge incidents
    - Semantic-only constraints can only add periphery (related surfaces)
    - No transitive closure through semantic-only edges
    """

    def __init__(self, semantic_threshold: float = 0.85):
        self.incidents = {}  # incident_id -> set of surface_ids
        self.surface_to_incident = {}  # surface_id -> incident_id
        self.periphery = {}  # incident_id -> set of (surface_id, semantic_score)
        self.constraints_processed = 0
        self.structural_merges = 0
        self.semantic_attachments = 0
        self.semantic_merge_blocked = 0
        self.semantic_threshold = semantic_threshold

    def process_constraint(self, constraint: ClassifiedConstraint) -> dict:
        """
        Process constraint respecting anti-trap rules.

        Returns dict with action taken and any warnings.
        """
        self.constraints_processed += 1

        if constraint.is_structural():
            return self._process_structural(constraint)
        else:
            return self._process_semantic_only(constraint)

    def _process_structural(self, constraint: ClassifiedConstraint) -> dict:
        """Process structural constraint - can create/merge incidents."""
        source_inc = self.surface_to_incident.get(constraint.source_id)
        target_inc = self.surface_to_incident.get(constraint.target_id)

        if source_inc is None and target_inc is None:
            # Create new incident with both surfaces
            inc_id = f"inc_{len(self.incidents):04d}"
            self.incidents[inc_id] = {constraint.source_id, constraint.target_id}
            self.surface_to_incident[constraint.source_id] = inc_id
            self.surface_to_incident[constraint.target_id] = inc_id
            self.periphery[inc_id] = set()
            return {"action": "create_incident", "incident_id": inc_id}

        elif source_inc is None:
            # Add source to target's incident
            self.incidents[target_inc].add(constraint.source_id)
            self.surface_to_incident[constraint.source_id] = target_inc
            return {"action": "expand_incident", "incident_id": target_inc}

        elif target_inc is None:
            # Add target to source's incident
            self.incidents[source_inc].add(constraint.target_id)
            self.surface_to_incident[constraint.target_id] = source_inc
            return {"action": "expand_incident", "incident_id": source_inc}

        elif source_inc != target_inc:
            # Merge incidents (structural evidence justifies this)
            self._merge_incidents(source_inc, target_inc)
            self.structural_merges += 1
            return {"action": "merge_incidents", "merged": target_inc, "into": source_inc}

        else:
            return {"action": "already_same_incident", "incident_id": source_inc}

    def _process_semantic_only(self, constraint: ClassifiedConstraint) -> dict:
        """
        Process semantic-only constraint - can ONLY attach periphery.

        ANTI-TRAP RULE: Semantic-only constraints cannot:
        1. Create new incidents
        2. Merge existing incidents
        3. Promote periphery to core membership
        """
        source_inc = self.surface_to_incident.get(constraint.source_id)
        target_inc = self.surface_to_incident.get(constraint.target_id)

        # Case 1: Neither surface is in an incident - DO NOTHING (no core creation)
        if source_inc is None and target_inc is None:
            self.semantic_merge_blocked += 1
            return {
                "action": "blocked",
                "reason": "semantic_only_cannot_create_incident",
                "warning": f"High similarity ({constraint.semantic_score:.2f}) but no structural evidence"
            }

        # Case 2: Source is in incident, target is not - add target as periphery
        if source_inc is not None and target_inc is None:
            self.periphery[source_inc].add((constraint.target_id, constraint.semantic_score))
            self.semantic_attachments += 1
            return {
                "action": "attach_periphery",
                "incident_id": source_inc,
                "peripheral_surface": constraint.target_id
            }

        # Case 3: Target is in incident, source is not - add source as periphery
        if target_inc is not None and source_inc is None:
            self.periphery[target_inc].add((constraint.source_id, constraint.semantic_score))
            self.semantic_attachments += 1
            return {
                "action": "attach_periphery",
                "incident_id": target_inc,
                "peripheral_surface": constraint.source_id
            }

        # Case 4: Both in different incidents - DO NOT MERGE (anti-trap!)
        if source_inc != target_inc:
            self.semantic_merge_blocked += 1
            return {
                "action": "blocked",
                "reason": "semantic_only_cannot_merge_incidents",
                "warning": f"High similarity ({constraint.semantic_score:.2f}) but incidents must stay separate",
                "incident_1": source_inc,
                "incident_2": target_inc
            }

        # Case 5: Already same incident - no-op
        return {"action": "already_same_incident", "incident_id": source_inc}

    def _merge_incidents(self, keep_id: str, merge_id: str):
        """Merge merge_id incident into keep_id."""
        surfaces = self.incidents.pop(merge_id)
        for sid in surfaces:
            self.surface_to_incident[sid] = keep_id
        self.incidents[keep_id].update(surfaces)

        # Merge periphery
        if merge_id in self.periphery:
            self.periphery[keep_id].update(self.periphery.pop(merge_id))


# ============================================================================
# TEST: SEMANTIC-ONLY CANNOT CREATE INCIDENTS
# ============================================================================

class TestSemanticOnlyCannotCreateIncidents:
    """Test that semantic-only constraints cannot create new incidents."""

    def test_semantic_only_blocked_when_no_existing_incidents(self):
        """Pure semantic similarity between unaffiliated surfaces → no incident."""
        kernel = AntiTrapL3Kernel()

        constraint = ClassifiedConstraint(
            source_id="sf_0001",
            target_id="sf_0002",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.SEMANTIC,
            semantic_score=0.95,  # Very high similarity!
            entity_overlap=set()  # But no entity overlap
        )

        result = kernel.process_constraint(constraint)

        assert result["action"] == "blocked"
        assert "cannot_create_incident" in result["reason"]
        assert len(kernel.incidents) == 0, "Semantic-only must not create incidents"

    def test_structural_can_create_incident(self):
        """Structural constraint (entity overlap) CAN create incident."""
        kernel = AntiTrapL3Kernel()

        constraint = ClassifiedConstraint(
            source_id="sf_0001",
            target_id="sf_0002",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.STRUCTURAL,
            semantic_score=0.60,  # Lower semantic score
            entity_overlap={"Hong Kong", "Tai Po"}  # But has entity overlap
        )

        result = kernel.process_constraint(constraint)

        assert result["action"] == "create_incident"
        assert len(kernel.incidents) == 1


# ============================================================================
# TEST: SEMANTIC-ONLY CANNOT MERGE INCIDENTS
# ============================================================================

class TestSemanticOnlyCannotMergeIncidents:
    """Test that semantic-only constraints cannot merge existing incidents."""

    def test_semantic_only_cannot_merge_separate_incidents(self):
        """Two incidents with high semantic similarity must NOT merge."""
        kernel = AntiTrapL3Kernel()

        # Create incident 1 via structural constraint
        kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_fire_a_1",
            target_id="sf_fire_a_2",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.STRUCTURAL,
            entity_overlap={"Hong Kong"}
        ))

        # Create incident 2 via structural constraint (different fire)
        kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_fire_b_1",
            target_id="sf_fire_b_2",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.STRUCTURAL,
            entity_overlap={"Kowloon"}  # Different location
        ))

        assert len(kernel.incidents) == 2, "Should have 2 separate incidents"

        # Now try to merge via semantic-only (both about "fire casualties")
        result = kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_fire_a_1",
            target_id="sf_fire_b_1",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.SEMANTIC,
            semantic_score=0.92,  # Very similar!
            entity_overlap=set()  # But no common entities
        ))

        assert result["action"] == "blocked"
        assert "cannot_merge" in result["reason"]
        assert len(kernel.incidents) == 2, "Incidents must remain separate"
        assert kernel.semantic_merge_blocked == 1

    def test_structural_can_merge_incidents(self):
        """Structural evidence CAN merge incidents."""
        kernel = AntiTrapL3Kernel()

        # Create two incidents
        kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_1",
            target_id="sf_2",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.STRUCTURAL,
            entity_overlap={"Entity A"}
        ))

        kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_3",
            target_id="sf_4",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.STRUCTURAL,
            entity_overlap={"Entity B"}
        ))

        assert len(kernel.incidents) == 2

        # Merge via structural (discovered shared entity)
        result = kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_1",
            target_id="sf_3",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.STRUCTURAL,
            entity_overlap={"Entity A", "Entity B"}  # Shared!
        ))

        assert result["action"] == "merge_incidents"
        assert len(kernel.incidents) == 1


# ============================================================================
# TEST: SEMANTIC-ONLY CAN ATTACH PERIPHERY
# ============================================================================

class TestSemanticOnlyCanAttachPeriphery:
    """Test that semantic-only constraints CAN attach periphery to existing incidents."""

    def test_semantic_attaches_unaffiliated_surface_as_periphery(self):
        """Semantic similarity to core surface → periphery attachment."""
        kernel = AntiTrapL3Kernel()

        # Create incident via structural
        kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_core_1",
            target_id="sf_core_2",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.STRUCTURAL,
            entity_overlap={"Hong Kong"}
        ))

        inc_id = list(kernel.incidents.keys())[0]

        # Semantic-only constraint from core to unaffiliated surface
        result = kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_core_1",
            target_id="sf_related",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.SEMANTIC,
            semantic_score=0.88,
            entity_overlap=set()
        ))

        assert result["action"] == "attach_periphery"
        assert result["incident_id"] == inc_id
        assert ("sf_related", 0.88) in kernel.periphery[inc_id]

        # Periphery is NOT part of incident core
        assert "sf_related" not in kernel.incidents[inc_id]
        assert "sf_related" not in kernel.surface_to_incident

    def test_periphery_remains_peripheral(self):
        """Peripheral surfaces cannot be promoted to core via more semantic evidence."""
        kernel = AntiTrapL3Kernel()

        # Create incident
        kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_core_1",
            target_id="sf_core_2",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.STRUCTURAL,
            entity_overlap={"Entity A"}
        ))

        # Attach periphery
        kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_core_1",
            target_id="sf_peripheral",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.SEMANTIC,
            semantic_score=0.85
        ))

        # More semantic evidence to same peripheral
        kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_core_2",
            target_id="sf_peripheral",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.SEMANTIC,
            semantic_score=0.90
        ))

        # Still not in core
        assert "sf_peripheral" not in kernel.surface_to_incident


# ============================================================================
# TEST: NO TRANSITIVE CLOSURE THROUGH SEMANTIC-ONLY
# ============================================================================

class TestNoTransitiveSemanticClosure:
    """Test that semantic-only edges don't create transitive connections."""

    def test_semantic_chain_does_not_connect_incidents(self):
        """
        Even if A→B (semantic) and B→C (semantic), A and C should NOT be linked
        through semantic-only transitive closure.
        """
        kernel = AntiTrapL3Kernel()

        # Create incident with surface A
        kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_A",
            target_id="sf_A2",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.STRUCTURAL,
            entity_overlap={"Entity A"}
        ))

        # Create separate incident with surface C
        kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_C",
            target_id="sf_C2",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.STRUCTURAL,
            entity_overlap={"Entity C"}
        ))

        assert len(kernel.incidents) == 2

        # Surface B is unaffiliated, semantically similar to both A and C

        # A → B (semantic) - B becomes periphery of incident A
        kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_A",
            target_id="sf_B",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.SEMANTIC,
            semantic_score=0.88
        ))

        # C → B (semantic) - B is already periphery of A, cannot become periphery of C
        # This should NOT merge the incidents!
        result = kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_C",
            target_id="sf_B",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.SEMANTIC,
            semantic_score=0.87
        ))

        # Incidents must remain separate
        assert len(kernel.incidents) == 2, "Transitive semantic should not merge incidents"

    def test_semantic_evidence_accumulates_but_doesnt_promote(self):
        """Multiple semantic constraints to same target don't promote to structural."""
        kernel = AntiTrapL3Kernel()

        # Create incident
        kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_core",
            target_id="sf_core2",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.STRUCTURAL,
            entity_overlap={"Entity"}
        ))

        # Multiple semantic constraints to unaffiliated surface
        for i in range(5):
            kernel.process_constraint(ClassifiedConstraint(
                source_id="sf_core",
                target_id="sf_target",
                constraint_type=ConstraintType.BINDING,
                strength=ConstraintStrength.SEMANTIC,
                semantic_score=0.85 + i * 0.02  # Increasing scores
            ))

        # Still not in incident core despite 5 semantic links
        assert "sf_target" not in kernel.surface_to_incident


# ============================================================================
# TEST: REAL-WORLD ANTI-TRAP SCENARIO
# ============================================================================

class TestRealWorldAntiTrap:
    """Test anti-trap with realistic scenarios."""

    def test_similar_fire_reports_stay_separate(self):
        """
        Two fires with similar casualty reports should not merge
        unless they share structural evidence (same location/entities).
        """
        kernel = AntiTrapL3Kernel()

        # Hong Kong fire incident
        kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_hk_fire_death_toll",
            target_id="sf_hk_fire_cause",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.STRUCTURAL,
            entity_overlap={"Hong Kong", "Tai Po", "Fire Department"}
        ))

        # Singapore fire incident (same day, similar scale)
        kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_sg_fire_death_toll",
            target_id="sf_sg_fire_cause",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.STRUCTURAL,
            entity_overlap={"Singapore", "SCDF"}
        ))

        # "Fire kills dozens" - semantically very similar!
        semantic_constraint = ClassifiedConstraint(
            source_id="sf_hk_fire_death_toll",  # "Fire kills 163 in Hong Kong"
            target_id="sf_sg_fire_death_toll",  # "Fire kills 45 in Singapore"
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.SEMANTIC,
            semantic_score=0.94,  # Very high - both about fire deaths
            entity_overlap=set()  # No shared entities
        )

        result = kernel.process_constraint(semantic_constraint)

        assert result["action"] == "blocked"
        assert len(kernel.incidents) == 2, "Different fires must remain separate incidents"

    def test_same_fire_different_sources_can_merge(self):
        """Same fire reported by different sources CAN merge (structural evidence)."""
        kernel = AntiTrapL3Kernel()

        # BBC report
        kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_bbc_death_toll",
            target_id="sf_bbc_cause",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.STRUCTURAL,
            entity_overlap={"Hong Kong", "Tai Po"}
        ))

        # SCMP report (same fire)
        kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_scmp_death_toll",
            target_id="sf_scmp_response",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.STRUCTURAL,
            entity_overlap={"Hong Kong", "Tai Po", "John Lee"}
        ))

        assert len(kernel.incidents) == 2

        # Merge via structural (same entities mentioned)
        result = kernel.process_constraint(ClassifiedConstraint(
            source_id="sf_bbc_death_toll",
            target_id="sf_scmp_death_toll",
            constraint_type=ConstraintType.BINDING,
            strength=ConstraintStrength.STRUCTURAL,  # STRUCTURAL - same entities
            semantic_score=0.91,
            entity_overlap={"Hong Kong", "Tai Po"}  # Same fire!
        ))

        assert result["action"] == "merge_incidents"
        assert len(kernel.incidents) == 1


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
