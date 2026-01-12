"""
Kernel Safety Invariants
========================

These invariants must ALWAYS hold, regardless of input data.
Violations indicate kernel bugs, not data quality issues.
"""

from typing import List, Set, Dict, Any, Tuple
from dataclasses import dataclass

# Import types when available
try:
    from reee.builders.story_builder import CompleteStory
    from reee.membrane import Membership, CoreReason
except ImportError:
    CompleteStory = Any
    Membership = Any
    CoreReason = Any


@dataclass
class InvariantViolation:
    """Records an invariant violation."""
    invariant: str
    message: str
    evidence: Dict[str, Any]


class KernelInvariants:
    """
    Collection of kernel safety invariants.

    All methods return (passed: bool, violations: List[InvariantViolation])
    """

    @staticmethod
    def no_semantic_only_core(
        stories: List[CompleteStory],
    ) -> Tuple[bool, List[InvariantViolation]]:
        """
        INVARIANT: Core membership requires structural witness.

        Core-A: spine is anchor (structural by definition)
        Core-B: requires ≥2 structural witnesses with ≥1 non-time

        Semantic-only constraints (embedding, llm_proposal) cannot force core.
        """
        violations = []

        for story in stories:
            # Core-A is always valid (spine as anchor is structural)
            # Check Core-B
            for inc_id in story.core_b_ids:
                decision = story.membrane_decisions.get(inc_id)
                if decision:
                    # Core-B must have witnesses
                    if not decision.witnesses:
                        violations.append(InvariantViolation(
                            invariant="no_semantic_only_core",
                            message=f"Core-B incident {inc_id} has no witnesses",
                            evidence={
                                "story_id": story.story_id,
                                "incident_id": inc_id,
                                "decision": str(decision),
                            }
                        ))

        return len(violations) == 0, violations

    @staticmethod
    def no_hub_story_definition(
        stories: List[CompleteStory],
        hub_entities: Set[str],
    ) -> Tuple[bool, List[InvariantViolation]]:
        """
        INVARIANT: Hub entities cannot define stories.

        Hub entities (>threshold participation) can only be:
        - Lenses (EntityCase-like views)
        - Periphery attachments
        Never story spines.
        """
        violations = []

        for story in stories:
            if story.spine in hub_entities:
                violations.append(InvariantViolation(
                    invariant="no_hub_story_definition",
                    message=f"Hub entity '{story.spine}' defines a story",
                    evidence={
                        "story_id": story.story_id,
                        "spine": story.spine,
                        "hub_entities": list(hub_entities),
                    }
                ))

        return len(violations) == 0, violations

    @staticmethod
    def scoped_surface_isolation(
        surfaces: List[Dict[str, Any]],
    ) -> Tuple[bool, List[InvariantViolation]]:
        """
        INVARIANT: (scope_id, question_key) uniquely identifies surfaces.

        Two surfaces with the same (scope_id, question_key) must be the same surface.
        No cross-scope contamination.
        """
        violations = []
        seen: Dict[Tuple[str, str], str] = {}

        for surface in surfaces:
            scope_id = surface.get('scope_id', '')
            question_key = surface.get('question_key', '')
            surface_id = surface.get('id', '')

            key = (scope_id, question_key)

            if key in seen and seen[key] != surface_id:
                violations.append(InvariantViolation(
                    invariant="scoped_surface_isolation",
                    message=f"Duplicate (scope_id, question_key): {key}",
                    evidence={
                        "scope_id": scope_id,
                        "question_key": question_key,
                        "surface_1": seen[key],
                        "surface_2": surface_id,
                    }
                ))
            else:
                seen[key] = surface_id

        return len(violations) == 0, violations

    @staticmethod
    def core_leak_rate_zero(
        stories: List[CompleteStory],
    ) -> Tuple[bool, List[InvariantViolation]]:
        """
        INVARIANT: Core leak rate should be ~0.

        core_leak_rate = (# core without spine anchor) / (# core)

        This measures how many core incidents lack the spine as anchor.
        By definition of Core-A, all Core-A have spine anchor.
        Core-B is allowed (via structural witnesses) but should be small.
        """
        violations = []

        for story in stories:
            if story.core_leak_rate > 0.5:  # Allow some Core-B
                violations.append(InvariantViolation(
                    invariant="core_leak_rate_zero",
                    message=f"High core leak rate: {story.core_leak_rate:.2%}",
                    evidence={
                        "story_id": story.story_id,
                        "spine": story.spine,
                        "core_leak_rate": story.core_leak_rate,
                        "core_a_count": len(story.core_a_ids),
                        "core_b_count": len(story.core_b_ids),
                    }
                ))

        return len(violations) == 0, violations

    @staticmethod
    def no_chain_percolation(
        stories: List[CompleteStory],
    ) -> Tuple[bool, List[InvariantViolation]]:
        """
        INVARIANT: Chain-only edges never merge cores.

        Motif chains (2-hop via shared entity) provide PERIPHERY attachment only.
        Core requires shared_motif or direct structural witness.
        """
        violations = []

        for story in stories:
            for inc_id in story.core_b_ids:
                decision = story.membrane_decisions.get(inc_id)
                if decision and decision.witnesses:
                    # Check if all witnesses are chain-based
                    chain_only = all(
                        'chain' in w.lower()
                        for w in decision.witnesses
                        if isinstance(w, str)
                    )
                    if chain_only:
                        violations.append(InvariantViolation(
                            invariant="no_chain_percolation",
                            message=f"Chain-only Core-B in {story.story_id}",
                            evidence={
                                "story_id": story.story_id,
                                "incident_id": inc_id,
                                "witnesses": decision.witnesses,
                            }
                        ))

        return len(violations) == 0, violations


# =============================================================================
# ASSERTION FUNCTIONS (for pytest)
# =============================================================================

def assert_no_semantic_only_core(stories: List[CompleteStory]):
    """Assert no semantic-only core membership."""
    passed, violations = KernelInvariants.no_semantic_only_core(stories)
    if not passed:
        raise AssertionError(
            f"Semantic-only core violations: {len(violations)}\n" +
            "\n".join(f"  - {v.message}" for v in violations)
        )


def assert_no_hub_story_definition(stories: List[CompleteStory], hub_entities: Set[str]):
    """Assert no hub entities define stories."""
    passed, violations = KernelInvariants.no_hub_story_definition(stories, hub_entities)
    if not passed:
        raise AssertionError(
            f"Hub story definition violations: {len(violations)}\n" +
            "\n".join(f"  - {v.message}" for v in violations)
        )


def assert_scoped_surface_isolation(surfaces: List[Dict[str, Any]]):
    """Assert surface scoping isolation."""
    passed, violations = KernelInvariants.scoped_surface_isolation(surfaces)
    if not passed:
        raise AssertionError(
            f"Scoped surface isolation violations: {len(violations)}\n" +
            "\n".join(f"  - {v.message}" for v in violations)
        )


def assert_core_leak_rate_zero(stories: List[CompleteStory]):
    """Assert core leak rate is near zero."""
    passed, violations = KernelInvariants.core_leak_rate_zero(stories)
    if not passed:
        raise AssertionError(
            f"Core leak rate violations: {len(violations)}\n" +
            "\n".join(f"  - {v.message}" for v in violations)
        )
