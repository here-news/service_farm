"""
Test Star Story Archetype (WFC Pattern)
=======================================

Milestone 2 Stop/Go Gate Test

This tests the case that broke k=2 motif recurrence:
- Star pattern: spine entity + rotating companions
- No pair recurs except (spine, X)
- k=2 motif recurrence would miss this entirely
- Spine-based StoryBuilder should form exactly 1 story

Acceptance Criteria:
- StoryBuilder produces exactly 1 story for WFC spine
- Core-A = incidents where WFC is anchor (all 5)
- Core-B = 0 (no structural witnesses without ledger)
- core_leak_rate == 0.0
- All invariants pass

Stop/Go Gate:
    pytest backend/reee/tests/integration/test_star_story_archetype.py -v
    Must pass: test_wfc_forms_one_story, test_core_leak_zero, test_invariants_hold
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Import story builder
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from reee.builders.story_builder import StoryBuilder, CompleteStory
from reee.types import Event, Surface
from reee.tests.invariants import (
    assert_no_semantic_only_core,
    assert_no_hub_story_definition,
    assert_scoped_surface_isolation,
    assert_core_leak_rate_zero,
)


# Fixture path
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
STAR_WFC_FIXTURE = FIXTURES_DIR / "golden_micro_star_wfc.json"


def load_fixture() -> Dict[str, Any]:
    """Load the star WFC fixture."""
    with open(STAR_WFC_FIXTURE) as f:
        return json.load(f)


def fixture_to_incidents(fixture: Dict[str, Any]) -> Dict[str, Event]:
    """Convert fixture incidents to Event objects dict (keyed by id)."""
    incidents = {}
    for inc_data in fixture.get('incidents', []):
        # Parse timestamps
        time_start = None
        time_end = None
        if inc_data.get('time_start'):
            time_start = datetime.fromisoformat(inc_data['time_start'].replace('Z', '+00:00'))
        if inc_data.get('time_end'):
            time_end = datetime.fromisoformat(inc_data['time_end'].replace('Z', '+00:00'))

        event = Event(
            id=inc_data['id'],
            anchor_entities=set(inc_data.get('anchor_entities', [])),
            entities=set(inc_data.get('anchor_entities', []) + inc_data.get('companion_entities', [])),
            time_window=(time_start, time_end),
            surface_ids=set(),  # Will be populated if needed
            canonical_title=inc_data.get('description', ''),
        )
        incidents[event.id] = event
    return incidents


def fixture_to_surfaces(fixture: Dict[str, Any]) -> Dict[str, Surface]:
    """Convert fixture claims to Surface objects dict (keyed by id)."""
    # Group claims by question_key to form surfaces
    by_question_key: Dict[str, List[Dict]] = {}
    for claim in fixture.get('claims', []):
        qk = claim.get('question_key', 'unknown')
        if qk not in by_question_key:
            by_question_key[qk] = []
        by_question_key[qk].append(claim)

    surfaces = {}
    for qk, claims in by_question_key.items():
        # Create deterministic surface ID
        import hashlib
        surface_id = hashlib.sha256(f"{qk}:{','.join(c['id'] for c in claims)}".encode()).hexdigest()[:12]

        surface = Surface(
            id=f"surf_{surface_id}",
            question_key=qk,
            claim_ids=set(c['id'] for c in claims),
            formation_method="question_key",
            centroid=None,  # Would be embedding in real system
        )
        surfaces[surface.id] = surface
    return surfaces


@pytest.fixture
def fixture_data():
    """Load fixture data."""
    return load_fixture()


@pytest.fixture
def incidents(fixture_data):
    """Convert fixture to incidents."""
    return fixture_to_incidents(fixture_data)


@pytest.fixture
def surfaces(fixture_data):
    """Convert fixture to surfaces."""
    return fixture_to_surfaces(fixture_data)


@pytest.fixture
def story_builder():
    """Create StoryBuilder with test configuration."""
    return StoryBuilder(
        hub_fraction_threshold=0.50,  # 50% threshold (WFC is in all 5 incidents but not a hub)
        hub_min_incidents=10,  # Need >10 incidents to be hub
        min_incidents_for_story=2,  # At least 2 incidents for a story
        mode_gap_days=30,  # 30 day gap for new temporal mode
    )


class TestStarStoryArchetype:
    """
    Milestone 2 acceptance tests for Star Story Archetype.

    Tests the WFC (World Food Council) star pattern:
    - WFC appears in ALL incidents as anchor
    - Each companion appears in exactly ONE incident
    - No (companion, companion) pairs recur
    - k=2 motif recurrence would fail
    - Spine-based approach should succeed
    """

    def test_fixture_loads_correctly(self, fixture_data):
        """Verify fixture structure is valid."""
        assert fixture_data['corpus_id'] == 'golden_star_wfc'
        assert fixture_data['archetype'] == 'star_story'
        assert len(fixture_data['claims']) == 15
        assert len(fixture_data['incidents']) == 5
        assert len(fixture_data['entities']) == 6

        # Verify expected values
        expected = fixture_data['expected']
        assert expected['story_count'] == 1
        assert expected['spine'] == 'World Food Council'
        assert expected['core_a_count'] == 5
        assert expected['core_b_count'] == 0

    def test_incidents_have_wfc_anchor(self, incidents):
        """Verify WFC is anchor in all incidents (star pattern)."""
        wfc_incidents = [
            inc for inc in incidents.values()
            if 'World Food Council' in inc.anchor_entities
        ]
        assert len(wfc_incidents) == 5, "WFC should be anchor in all 5 incidents"

    def test_no_recurring_companion_pairs(self, incidents):
        """Verify no (companion, companion) pairs recur (k=2 would fail)."""
        # Extract companion pairs (non-WFC entity pairs)
        pair_counts: Dict[tuple, int] = {}
        for inc in incidents.values():
            # Get non-WFC anchors
            companions = [e for e in inc.anchor_entities if e != 'World Food Council']
            # Count pairs
            for i, c1 in enumerate(companions):
                for c2 in companions[i+1:]:
                    pair = tuple(sorted([c1, c2]))
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # No companion pair should appear more than once
        recurring = {p: c for p, c in pair_counts.items() if c > 1}
        assert len(recurring) == 0, f"Found recurring companion pairs (k=2 false positives): {recurring}"

    def test_wfc_forms_one_story(self, story_builder, incidents, surfaces):
        """Test StoryBuilder produces exactly 1 story for WFC spine."""
        result = story_builder.build_from_incidents(incidents, surfaces)

        # Should produce exactly 1 story
        assert len(result.stories) == 1, f"Expected 1 story, got {len(result.stories)}"

        story = list(result.stories.values())[0]

        # Story spine should be WFC
        assert story.spine == 'World Food Council', f"Expected spine 'World Food Council', got '{story.spine}'"

        # All 5 incidents should be in core
        assert len(story.core_incident_ids) == 5, f"Expected 5 core incidents, got {len(story.core_incident_ids)}"

    def test_core_a_contains_all_incidents(self, story_builder, incidents, surfaces):
        """Test all incidents are Core-A (spine as anchor)."""
        result = story_builder.build_from_incidents(incidents, surfaces)
        story = list(result.stories.values())[0]

        # Core-A should contain all 5 incidents
        assert len(story.core_a_ids) == 5, f"Expected 5 Core-A incidents, got {len(story.core_a_ids)}"

        # Core-B should be empty (no structural witnesses without ledger)
        assert len(story.core_b_ids) == 0, f"Expected 0 Core-B incidents, got {len(story.core_b_ids)}"

    def test_core_leak_zero(self, story_builder, incidents, surfaces):
        """Test core_leak_rate is 0.0 (all core have spine anchor)."""
        result = story_builder.build_from_incidents(incidents, surfaces)
        story = list(result.stories.values())[0]

        assert story.core_leak_rate == 0.0, f"Expected core_leak_rate 0.0, got {story.core_leak_rate}"

    def test_no_hub_story_definition(self, story_builder, incidents, surfaces):
        """Test WFC is not classified as hub (appears in all incidents but count < threshold)."""
        result = story_builder.build_from_incidents(incidents, surfaces)

        # Derive hub_entities from spines
        hub_entities = {entity for entity, spine in result.spines.items() if spine.is_hub}

        # WFC should NOT be a hub (only 5 incidents, threshold is 10)
        assert 'World Food Council' not in hub_entities, \
            "WFC should not be classified as hub with only 5 incidents"

        # Invariant check
        assert_no_hub_story_definition(list(result.stories.values()), hub_entities)

    def test_invariants_hold(self, story_builder, incidents, surfaces):
        """Test all kernel safety invariants hold."""
        result = story_builder.build_from_incidents(incidents, surfaces)
        stories_list = list(result.stories.values())
        hub_entities = {entity for entity, spine in result.spines.items() if spine.is_hub}

        # 1. No semantic-only core
        assert_no_semantic_only_core(stories_list)

        # 2. No hub story definition
        assert_no_hub_story_definition(stories_list, hub_entities)

        # 3. Core leak rate near zero
        assert_core_leak_rate_zero(stories_list)

        # 4. Scoped surface isolation (if we have surfaces)
        if surfaces:
            surface_dicts = [
                {'id': s.id, 'scope_id': getattr(s, 'scope_id', s.id), 'question_key': s.question_key}
                for s in surfaces.values()
            ]
            assert_scoped_surface_isolation(surface_dicts)

    def test_periphery_tracking(self, story_builder, incidents, surfaces):
        """Test periphery candidates are tracked (observability)."""
        result = story_builder.build_from_incidents(incidents, surfaces)
        story = list(result.stories.values())[0]

        # In star pattern, all incidents are Core-A, so no periphery
        assert len(story.periphery_incident_ids) == 0, \
            f"Expected 0 periphery incidents, got {len(story.periphery_incident_ids)}"

        # Candidate pool should be tracked
        assert story.candidate_pool_size >= 5, \
            f"Expected candidate_pool_size >= 5, got {story.candidate_pool_size}"


# =============================================================================
# COMPARISON: What k=2 Motif Recurrence Would Produce
# =============================================================================

class TestK2MotifRecurrenceFailure:
    """
    Demonstrate why k=2 motif recurrence fails on star patterns.

    k=2 looks for pairs (A, B) that appear together in ≥2 incidents.
    In star pattern:
    - (WFC, GHI) appears in 1 incident
    - (WFC, UN) appears in 1 incident
    - (WFC, Asia) appears in 1 incident
    - etc.

    NO pair appears in ≥2 incidents, so k=2 would form 0 stories.
    """

    def test_k2_would_find_zero_motifs(self, incidents):
        """Demonstrate k=2 motif recurrence would fail."""
        # Count entity pairs across incidents
        pair_counts: Dict[tuple, int] = {}

        for inc in incidents.values():
            entities = list(inc.anchor_entities)
            for i, e1 in enumerate(entities):
                for e2 in entities[i+1:]:
                    pair = tuple(sorted([e1, e2]))
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # Find pairs appearing in ≥2 incidents (k=2 threshold)
        recurring_pairs = {p: c for p, c in pair_counts.items() if c >= 2}

        # k=2 would find ZERO recurring pairs in star pattern
        assert len(recurring_pairs) == 0, \
            f"k=2 would incorrectly find {len(recurring_pairs)} recurring pairs: {recurring_pairs}"

        # This proves k=2 motif recurrence fails on star-shaped news


# =============================================================================
# DIRECT RUN SUPPORT
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("Running Star Story Archetype tests...")
    print("-" * 50)

    fixture = load_fixture()
    incidents = fixture_to_incidents(fixture)
    surfaces = fixture_to_surfaces(fixture)

    builder = StoryBuilder(
        hub_fraction_threshold=0.50,
        hub_min_incidents=10,
        min_incidents_for_story=2,
        mode_gap_days=30,
    )

    print(f"Loaded: {len(incidents)} incidents, {len(surfaces)} surfaces")

    result = builder.build_from_incidents(incidents, surfaces)

    print(f"\nStoryBuilder Result:")
    print(f"  Stories formed: {len(result.stories)}")
    print(f"  Hub entities: {result.hub_entities}")

    if result.stories:
        story = result.stories[0]
        print(f"\nStory 1:")
        print(f"  Spine: {story.spine}")
        print(f"  Core-A: {len(story.core_a_ids)} incidents")
        print(f"  Core-B: {len(story.core_b_ids)} incidents")
        print(f"  Periphery: {len(story.periphery_incident_ids)} incidents")
        print(f"  Core leak rate: {story.core_leak_rate}")

    print("\n" + "-" * 50)
    print("Run with pytest for full validation:")
    print("  pytest backend/reee/tests/integration/test_star_story_archetype.py -v")
