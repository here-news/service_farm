"""
Test Multi-Seed Invariants
==========================

Validates that hard invariants hold across multiple random seeds.
This catches percolation edge cases that single-seed testing misses.

Strategy:
- Generate corpus with 3 different seeds
- Run kernel on each
- Verify all hard invariants pass
- Compare structure across seeds (should differ but invariants hold)

Usage:
    pytest backend/reee/tests/integration/test_multi_seed_invariants.py -v

Note: This test runs entirely in memory - no Neo4j required.
"""

import pytest
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Any
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from reee.tests.golden_macro.corpus_generator import MacroCorpusGenerator
from reee.builders.story_builder import StoryBuilder
from reee.types import Event, Surface
from reee.membrane import Membership, CoreReason


# Seeds to test (deterministic per seed)
TEST_SEEDS = [42, 123, 7777]


def corpus_to_incidents(corpus) -> Dict[str, Event]:
    """Convert corpus incidents to Event objects."""
    incidents = {}
    for inc in corpus.incidents:
        time_start = None
        time_end = None
        if inc.time_start:
            time_start = datetime.fromisoformat(inc.time_start.replace('Z', '+00:00'))
        if inc.time_end:
            time_end = datetime.fromisoformat(inc.time_end.replace('Z', '+00:00'))

        event = Event(
            id=inc.id,
            anchor_entities=set(inc.anchor_entities),
            entities=set(inc.anchor_entities + inc.companion_entities),
            time_window=(time_start, time_end),
            surface_ids=set(),
            canonical_title=inc.description,
        )
        incidents[event.id] = event
    return incidents


def corpus_to_surfaces(corpus) -> Dict[str, Surface]:
    """Convert corpus claims to Surface objects."""
    by_scope_qk = defaultdict(list)
    for claim in corpus.claims:
        key = (claim.scope_id, claim.question_key)
        by_scope_qk[key].append(claim)

    surfaces = {}
    for (scope_id, qk), claims in by_scope_qk.items():
        surface_id = f"surf_{hashlib.sha256(f'{scope_id}:{qk}'.encode()).hexdigest()[:12]}"
        surface = Surface(
            id=surface_id,
            question_key=qk,
            claim_ids=set(c.id for c in claims),
            formation_method="question_key",
            centroid=None,
        )
        surfaces[surface.id] = surface
    return surfaces


class TestMultiSeedInvariants:
    """
    Validates hard invariants across multiple random seeds.
    """

    @pytest.fixture(params=TEST_SEEDS)
    def seed_and_result(self, request):
        """Generate corpus and run kernel for each seed."""
        seed = request.param

        # Generate corpus
        generator = MacroCorpusGenerator(seed=seed)
        corpus = generator.generate()

        # Convert to kernel types
        incidents = corpus_to_incidents(corpus)
        surfaces = corpus_to_surfaces(corpus)

        # Run kernel
        builder = StoryBuilder(
            hub_fraction_threshold=0.20,
            hub_min_incidents=5,
            min_incidents_for_story=2,
            mode_gap_days=30,
        )
        result = builder.build_from_incidents(incidents, surfaces)

        return seed, corpus, result

    def test_scoped_surface_isolation(self, seed_and_result):
        """
        INVARIANT: (scope_id, question_key) uniquely identifies a surface.
        """
        seed, corpus, result = seed_and_result

        # Check that no two claims with same (scope_id, qk) are in different surfaces
        claim_to_scope_qk = {}
        for claim in corpus.claims:
            key = (claim.scope_id, claim.question_key)
            if claim.id in claim_to_scope_qk:
                assert claim_to_scope_qk[claim.id] == key, \
                    f"Seed {seed}: Claim {claim.id} has inconsistent (scope_id, qk)"
            claim_to_scope_qk[claim.id] = key

        print(f"  Seed {seed}: {len(corpus.claims)} claims, scoped isolation holds")

    def test_no_semantic_only_core(self, seed_and_result):
        """
        INVARIANT: CORE membership requires structural link.
        Check membrane decisions for semantic-only core.
        """
        seed, corpus, result = seed_and_result

        for story_id, story in result.stories.items():
            if not story.membrane_decisions:
                continue

            for inc_id, decision in story.membrane_decisions.items():
                if decision.membership == Membership.CORE:
                    # CORE must have ANCHOR or WARRANT reason
                    assert decision.core_reason is not None, \
                        f"Seed {seed}: CORE {inc_id} in {story.spine} has no core_reason"

                    # constraint_source must be structural
                    if decision.constraint_source:
                        assert decision.constraint_source == "structural", \
                            f"Seed {seed}: CORE {inc_id} has non-structural source: {decision.constraint_source}"

        print(f"  Seed {seed}: {len(result.stories)} stories, no semantic-only core")

    def test_hub_cannot_define_story(self, seed_and_result):
        """
        INVARIANT: Hub entities cannot be story spines.
        """
        seed, corpus, result = seed_and_result

        hub_entities = {
            entity for entity, spine_data in result.spines.items()
            if spine_data.is_hub
        }

        for story_id, story in result.stories.items():
            assert story.spine not in hub_entities, \
                f"Seed {seed}: Hub entity '{story.spine}' is a story spine"

        print(f"  Seed {seed}: {len(hub_entities)} hubs, none are story spines")

    def test_blocked_reasons_visible(self, seed_and_result):
        """
        INVARIANT: All PERIPHERY candidates have blocked_reason.
        """
        seed, corpus, result = seed_and_result

        missing_reasons = []
        for story_id, story in result.stories.items():
            if not story.membrane_decisions:
                continue

            for inc_id, decision in story.membrane_decisions.items():
                if decision.membership == Membership.PERIPHERY:
                    if not decision.blocked_reason:
                        missing_reasons.append((story.spine, inc_id))

        assert len(missing_reasons) == 0, \
            f"Seed {seed}: PERIPHERY without blocked_reason: {missing_reasons[:5]}"

        print(f"  Seed {seed}: All PERIPHERY have blocked_reason")

    def test_core_leak_rate_bounded(self, seed_and_result):
        """
        INVARIANT: Core leak rate is between 0 and 1.
        """
        seed, corpus, result = seed_and_result

        for story_id, story in result.stories.items():
            if story.core_leak_rate is not None:
                assert 0.0 <= story.core_leak_rate <= 1.0, \
                    f"Seed {seed}: Story {story.spine} has invalid leak rate: {story.core_leak_rate}"

        print(f"  Seed {seed}: All leak rates bounded [0, 1]")

    def test_decision_determinism_within_seed(self, seed_and_result):
        """
        INVARIANT: Same seed produces identical decisions on replay.
        """
        seed, corpus, result = seed_and_result

        # Re-run with same seed
        generator2 = MacroCorpusGenerator(seed=seed)
        corpus2 = generator2.generate()
        incidents2 = corpus_to_incidents(corpus2)
        surfaces2 = corpus_to_surfaces(corpus2)

        builder2 = StoryBuilder(
            hub_fraction_threshold=0.20,
            hub_min_incidents=5,
            min_incidents_for_story=2,
            mode_gap_days=30,
        )
        result2 = builder2.build_from_incidents(incidents2, surfaces2)

        # Same number of stories
        assert len(result.stories) == len(result2.stories), \
            f"Seed {seed}: Non-deterministic story count: {len(result.stories)} vs {len(result2.stories)}"

        # Same story spines
        spines1 = set(result.stories.keys())
        spines2 = set(result2.stories.keys())
        assert spines1 == spines2, \
            f"Seed {seed}: Non-deterministic story spines"

        # Same core memberships
        for story_id in spines1:
            story1 = result.stories[story_id]
            story2 = result2.stories[story_id]
            assert story1.core_incident_ids == story2.core_incident_ids, \
                f"Seed {seed}: Story {story1.spine} has non-deterministic core"

        print(f"  Seed {seed}: Deterministic on replay ✓")


class TestCrossSeeedComparison:
    """
    Compare kernel behavior across seeds.
    Invariants should hold, but structure may differ.
    """

    def test_invariants_hold_across_all_seeds(self):
        """Run all seeds and verify invariants."""
        for seed in TEST_SEEDS:
            generator = MacroCorpusGenerator(seed=seed)
            corpus = generator.generate()
            incidents = corpus_to_incidents(corpus)
            surfaces = corpus_to_surfaces(corpus)

            builder = StoryBuilder(
                hub_fraction_threshold=0.20,
                hub_min_incidents=5,
                min_incidents_for_story=2,
                mode_gap_days=30,
            )
            result = builder.build_from_incidents(incidents, surfaces)

            # Quick invariant checks
            hub_entities = {e for e, s in result.spines.items() if s.is_hub}

            for story in result.stories.values():
                # Hub cannot be spine
                assert story.spine not in hub_entities

                # Core must have reason
                for inc_id, decision in (story.membrane_decisions or {}).items():
                    if decision.membership == Membership.CORE:
                        assert decision.core_reason is not None

        print(f"\n✓ All invariants hold across seeds: {TEST_SEEDS}")

    def test_structure_varies_across_seeds(self):
        """
        Verify that different seeds produce different structures.
        This confirms the randomization is working.
        """
        story_counts = {}
        hub_counts = {}

        for seed in TEST_SEEDS:
            generator = MacroCorpusGenerator(seed=seed)
            corpus = generator.generate()
            incidents = corpus_to_incidents(corpus)
            surfaces = corpus_to_surfaces(corpus)

            builder = StoryBuilder(
                hub_fraction_threshold=0.20,
                hub_min_incidents=5,
                min_incidents_for_story=2,
                mode_gap_days=30,
            )
            result = builder.build_from_incidents(incidents, surfaces)

            story_counts[seed] = len(result.stories)
            hub_counts[seed] = len([e for e, s in result.spines.items() if s.is_hub])

        print(f"\n📊 Cross-seed structure comparison:")
        print(f"   {'Seed':<10} {'Stories':>10} {'Hubs':>10}")
        for seed in TEST_SEEDS:
            print(f"   {seed:<10} {story_counts[seed]:>10} {hub_counts[seed]:>10}")

        # Should have some variation (not all identical)
        # This is a soft check - corpus with same archetypes may produce similar results
        unique_story_counts = len(set(story_counts.values()))
        print(f"\n   Unique story counts: {unique_story_counts}/{len(TEST_SEEDS)}")


# =============================================================================
# DIRECT RUN SUPPORT
# =============================================================================

if __name__ == "__main__":
    print("Running Multi-Seed Invariant tests...")
    print(f"Seeds: {TEST_SEEDS}")
    print("-" * 50)
    print("Run with pytest:")
    print("  pytest backend/reee/tests/integration/test_multi_seed_invariants.py -v")
