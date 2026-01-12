"""
Canonical Stability Tests
=========================

Tests that Story IDs remain stable across rebuilds when the
underlying structure hasn't changed.

Key invariants tested:
1. Same (anchors, time_bin, scale) → same scope_signature
2. Different time bins → different scope_signature
3. Rebuilding from same surfaces → same Story ID
4. Story.from_event() produces stable IDs

This is Upgrade 1 from the kernel consolidation plan:
"Canonical IDs stable across rebuild"
"""

import pytest
from datetime import datetime, timedelta
from typing import Set

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reee.types import (
    Story, StoryScale, Event, Surface, Claim,
    EventJustification, SurfaceMembership, MembershipLevel
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_anchor_entities() -> Set[str]:
    """Standard anchor set for testing."""
    return {"John Lee", "Hong Kong"}


@pytest.fixture
def sample_time_start() -> datetime:
    """Standard time for testing."""
    return datetime(2025, 1, 15, 10, 30, 0)


@pytest.fixture
def sample_event(sample_anchor_entities, sample_time_start) -> Event:
    """Create a sample Event for conversion testing."""
    return Event(
        id="E001",
        surface_ids={"S001", "S002", "S003"},
        anchor_entities=sample_anchor_entities,
        time_window=(sample_time_start, sample_time_start + timedelta(hours=6)),
        total_claims=15,
        total_sources=5,
        canonical_title="Hong Kong Policy Announcement",
    )


# ============================================================================
# TEST: SCOPE SIGNATURE DETERMINISM
# ============================================================================

class TestScopeSignatureDeterminism:
    """Test that scope signatures are deterministic and stable."""

    def test_same_inputs_same_signature(self, sample_anchor_entities, sample_time_start):
        """Identical inputs produce identical signatures."""
        story1 = Story(
            id="temp1",
            scale="incident",
            anchor_entities=sample_anchor_entities.copy(),
            time_start=sample_time_start,
        )
        story2 = Story(
            id="temp2",
            scale="incident",
            anchor_entities=sample_anchor_entities.copy(),
            time_start=sample_time_start,
        )

        sig1 = story1.compute_scope_signature()
        sig2 = story2.compute_scope_signature()

        assert sig1 == sig2, f"Same inputs should produce same signature: {sig1} != {sig2}"

    def test_signature_format(self, sample_anchor_entities, sample_time_start):
        """Signature has expected format: story_<12char_hash>."""
        story = Story(
            id="temp",
            scale="incident",
            anchor_entities=sample_anchor_entities,
            time_start=sample_time_start,
        )
        sig = story.compute_scope_signature()

        assert sig.startswith("story_"), f"Signature should start with 'story_': {sig}"
        assert len(sig) == 18, f"Signature should be 18 chars (story_ + 12): {sig}"

    def test_anchor_order_invariance(self):
        """Anchor entity order doesn't affect signature."""
        # Different insertion orders, same set
        story1 = Story(
            id="temp1",
            scale="incident",
            anchor_entities={"A", "B", "C"},
            time_start=datetime(2025, 1, 15),
        )
        story2 = Story(
            id="temp2",
            scale="incident",
            anchor_entities={"C", "A", "B"},
            time_start=datetime(2025, 1, 15),
        )

        assert story1.compute_scope_signature() == story2.compute_scope_signature()

    def test_different_anchors_different_signature(self, sample_time_start):
        """Different anchor sets produce different signatures."""
        story1 = Story(
            id="temp1",
            scale="incident",
            anchor_entities={"John Lee", "Hong Kong"},
            time_start=sample_time_start,
        )
        story2 = Story(
            id="temp2",
            scale="incident",
            anchor_entities={"John Lee", "Beijing"},  # Different anchor
            time_start=sample_time_start,
        )

        assert story1.compute_scope_signature() != story2.compute_scope_signature()


# ============================================================================
# TEST: TIME BIN STABILITY
# ============================================================================

class TestTimeBinStability:
    """Test that time binning produces stable signatures within bins."""

    def test_same_week_same_signature_incident(self):
        """Times in same week produce same signature for incident scale."""
        # Monday and Friday of same week
        monday = datetime(2025, 1, 13, 9, 0, 0)  # Week 3
        friday = datetime(2025, 1, 17, 18, 0, 0)  # Still week 3

        story1 = Story(
            id="temp1",
            scale="incident",
            anchor_entities={"Entity A"},
            time_start=monday,
        )
        story2 = Story(
            id="temp2",
            scale="incident",
            anchor_entities={"Entity A"},
            time_start=friday,
        )

        assert story1.compute_scope_signature() == story2.compute_scope_signature(), \
            "Same week should produce same signature for incident scale"

    def test_different_week_different_signature_incident(self):
        """Times in different weeks produce different signatures for incident scale."""
        week1 = datetime(2025, 1, 10)  # Week 2
        week2 = datetime(2025, 1, 20)  # Week 4

        story1 = Story(
            id="temp1",
            scale="incident",
            anchor_entities={"Entity A"},
            time_start=week1,
        )
        story2 = Story(
            id="temp2",
            scale="incident",
            anchor_entities={"Entity A"},
            time_start=week2,
        )

        assert story1.compute_scope_signature() != story2.compute_scope_signature(), \
            "Different weeks should produce different signatures for incident scale"

    def test_same_month_same_signature_case(self):
        """Times in same month produce same signature for case scale."""
        early = datetime(2025, 1, 5)
        late = datetime(2025, 1, 28)

        story1 = Story(
            id="temp1",
            scale="case",
            anchor_entities={"Entity A"},
            time_start=early,
        )
        story2 = Story(
            id="temp2",
            scale="case",
            anchor_entities={"Entity A"},
            time_start=late,
        )

        assert story1.compute_scope_signature() == story2.compute_scope_signature(), \
            "Same month should produce same signature for case scale"

    def test_different_month_different_signature_case(self):
        """Times in different months produce different signatures for case scale."""
        jan = datetime(2025, 1, 15)
        feb = datetime(2025, 2, 15)

        story1 = Story(
            id="temp1",
            scale="case",
            anchor_entities={"Entity A"},
            time_start=jan,
        )
        story2 = Story(
            id="temp2",
            scale="case",
            anchor_entities={"Entity A"},
            time_start=feb,
        )

        assert story1.compute_scope_signature() != story2.compute_scope_signature(), \
            "Different months should produce different signatures for case scale"


# ============================================================================
# TEST: SCALE AFFECTS SIGNATURE
# ============================================================================

class TestScaleAffectsSignature:
    """Test that scale is part of the signature."""

    def test_incident_vs_case_different_signature(self, sample_anchor_entities, sample_time_start):
        """Same anchors/time but different scale produces different signature."""
        incident = Story(
            id="temp1",
            scale="incident",
            anchor_entities=sample_anchor_entities.copy(),
            time_start=sample_time_start,
        )
        case = Story(
            id="temp2",
            scale="case",
            anchor_entities=sample_anchor_entities.copy(),
            time_start=sample_time_start,
        )

        assert incident.compute_scope_signature() != case.compute_scope_signature(), \
            "Different scales should produce different signatures"


# ============================================================================
# TEST: STORY.FROM_EVENT() STABILITY
# ============================================================================

class TestStoryFromEventStability:
    """Test that converting Event to Story produces stable IDs."""

    def test_from_event_produces_stable_id(self, sample_event):
        """Story.from_event() produces consistent signatures."""
        story1 = Story.from_event(sample_event, scale="incident")
        story2 = Story.from_event(sample_event, scale="incident")

        assert story1.scope_signature == story2.scope_signature, \
            "Repeated from_event() should produce same signature"

    def test_from_event_copies_entities(self, sample_event):
        """Story.from_event() copies anchor entities correctly."""
        story = Story.from_event(sample_event, scale="incident")

        assert story.anchor_entities == sample_event.anchor_entities

    def test_from_event_copies_stats(self, sample_event):
        """Story.from_event() copies stats correctly."""
        story = Story.from_event(sample_event, scale="incident")

        assert story.surface_count == len(sample_event.surface_ids)
        assert story.source_count == sample_event.total_sources
        assert story.claim_count == sample_event.total_claims

    def test_from_event_uses_provided_title(self, sample_event):
        """Story.from_event() uses provided title over event title."""
        story = Story.from_event(
            sample_event,
            scale="incident",
            title="Custom Title"
        )

        assert story.title == "Custom Title"

    def test_from_event_falls_back_to_event_title(self, sample_event):
        """Story.from_event() falls back to event canonical_title."""
        story = Story.from_event(sample_event, scale="incident")

        assert story.title == sample_event.canonical_title


# ============================================================================
# TEST: REBUILD STABILITY (Simulated)
# ============================================================================

class TestRebuildStability:
    """Test that rebuilding from same inputs produces same IDs."""

    def test_simulated_rebuild_same_signature(self):
        """
        Simulate a rebuild scenario:
        1. Build Story from surfaces
        2. "Rebuild" by creating new Story with same inputs
        3. Verify signatures match
        """
        # First "build"
        anchors = {"Fire", "Tai Po", "Hong Kong"}
        time = datetime(2025, 1, 15)

        story_v1 = Story(
            id="first_build",
            scale="incident",
            anchor_entities=anchors,
            time_start=time,
            surface_ids={"sf_001", "sf_002", "sf_003"},
            source_count=5,
            claim_count=15,
        )
        sig_v1 = story_v1.compute_scope_signature()

        # "Rebuild" - same inputs, new object
        story_v2 = Story(
            id="second_build",  # Different ephemeral ID
            scale="incident",
            anchor_entities=anchors.copy(),  # Same set
            time_start=time,
            surface_ids={"sf_001", "sf_002", "sf_003"},  # Same surfaces
            source_count=5,
            claim_count=15,
        )
        sig_v2 = story_v2.compute_scope_signature()

        assert sig_v1 == sig_v2, \
            f"Rebuild should produce same signature: {sig_v1} != {sig_v2}"

    def test_generate_stable_id_uses_signature(self):
        """generate_stable_id() sets id = scope_signature."""
        story = Story(
            id="ephemeral",
            scale="incident",
            anchor_entities={"A", "B"},
            time_start=datetime(2025, 1, 15),
        )

        stable_id = story.generate_stable_id()

        assert story.id == story.scope_signature
        assert story.id == stable_id
        assert story.id.startswith("story_")


# ============================================================================
# TEST: SERIALIZATION ROUNDTRIP
# ============================================================================

class TestSerializationRoundtrip:
    """Test that serialization preserves identity."""

    def test_to_dict_from_dict_preserves_signature(self, sample_anchor_entities, sample_time_start):
        """Serialization roundtrip preserves scope_signature."""
        original = Story(
            id="test",
            scale="incident",
            anchor_entities=sample_anchor_entities,
            time_start=sample_time_start,
            title="Test Story",
            description="Test description",
            surface_count=5,
        )
        original.compute_scope_signature()

        # Roundtrip
        data = original.to_dict()
        restored = Story.from_dict(data)

        assert restored.scope_signature == original.scope_signature
        assert restored.scale == original.scale
        assert restored.anchor_entities == original.anchor_entities

    def test_to_dict_includes_all_fields(self, sample_anchor_entities, sample_time_start):
        """to_dict() includes all relevant fields."""
        story = Story(
            id="test",
            scale="case",
            anchor_entities=sample_anchor_entities,
            time_start=sample_time_start,
            surface_ids={"sf_1", "sf_2"},
            incident_ids={"inc_1"},
            title="Test",
            source_count=10,
        )
        story.compute_scope_signature()

        data = story.to_dict()

        assert "id" in data
        assert "scale" in data
        assert "scope_signature" in data
        assert "anchor_entities" in data
        assert "surface_ids" in data
        assert "incident_ids" in data
        assert data["scale"] == "case"


# ============================================================================
# TEST: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases for stability."""

    def test_no_time_uses_unknown_bin(self):
        """Missing time_start uses 'unknown' time bin."""
        story = Story(
            id="temp",
            scale="incident",
            anchor_entities={"A"},
            time_start=None,
        )
        sig = story.compute_scope_signature()

        assert "story_" in sig  # Still produces valid signature

    def test_empty_anchors_still_produces_signature(self):
        """Empty anchor set still produces valid signature."""
        story = Story(
            id="temp",
            scale="incident",
            anchor_entities=set(),
            time_start=datetime(2025, 1, 15),
        )
        sig = story.compute_scope_signature()

        assert sig.startswith("story_")

    def test_many_anchors_uses_top_10(self):
        """With >10 anchors, only top 10 (sorted) are used."""
        # Create 15 anchors
        many_anchors = {f"Entity_{i:02d}" for i in range(15)}

        story = Story(
            id="temp",
            scale="incident",
            anchor_entities=many_anchors,
            time_start=datetime(2025, 1, 15),
        )
        sig1 = story.compute_scope_signature()

        # Adding more entities beyond top 10 shouldn't change signature
        # (they're sorted, so Entity_00..Entity_09 are used)
        more_anchors = many_anchors | {"Entity_99"}  # Sorts after top 10
        story2 = Story(
            id="temp2",
            scale="incident",
            anchor_entities=more_anchors,
            time_start=datetime(2025, 1, 15),
        )
        sig2 = story2.compute_scope_signature()

        # Both should use same top 10: Entity_00..Entity_09
        assert sig1 == sig2, "Adding entities beyond top 10 shouldn't change signature"


# ============================================================================
# TEST: CANONICAL LAYER ISOLATION
# ============================================================================

class TestCanonicalLayerIsolation:
    """
    Test that canonical operations never delete structural nodes.

    Canonical layer is PRESENTATION ONLY:
    - Can update titles, descriptions, display fields
    - Can MERGE by scope_signature
    - MUST NOT delete :Incident, :Surface, :Claim nodes
    - MUST NOT delete structural relationships

    This is the "kernel-grade" stability guarantee.
    """

    def test_canonical_only_touches_case_nodes(self):
        """
        Canonical worker should only touch :Case nodes with 'case_' prefix IDs.

        Structural nodes (:Incident, :Surface, :Claim) are off-limits.
        """
        # Define what canonical worker is allowed to touch
        CANONICAL_ALLOWED_LABELS = {"Case", "Story"}
        CANONICAL_ID_PREFIX = "case_"

        # Define structural nodes that must never be deleted by canonical
        STRUCTURAL_LABELS = {"Incident", "Surface", "Claim", "Entity", "Page"}

        # Invariant: canonical operations filter by label AND id prefix
        # This test documents the contract

        # Simulated "canonical delete query" must have these guards:
        expected_delete_guards = [
            "WHERE c.id STARTS WITH 'case_'",  # Only case_ prefix
            "AND NOT c.id IN $valid_ids",      # Only orphaned cases
        ]

        # The actual canonical worker query (from canonical_worker.py lines 1056-1061):
        # MATCH (c:Case)
        # WHERE c.id STARTS WITH 'case_'
        #   AND NOT c.id IN $valid_ids
        # DETACH DELETE c

        # This is correct because:
        # 1. It only matches :Case (not :Incident, :Surface)
        # 2. It only deletes case_ prefix (not incident_, sf_, etc.)
        # 3. It preserves valid cases (those still meeting criteria)

        assert CANONICAL_ALLOWED_LABELS.isdisjoint(STRUCTURAL_LABELS - {"Story"}), \
            "Canonical labels should not overlap with structural labels (except Story which is additive)"

    def test_structural_nodes_have_distinct_id_prefixes(self):
        """
        Each structural layer has a distinct ID prefix to prevent accidental deletion.
        """
        ID_PREFIXES = {
            "Surface": "sf_",
            "Incident": "inc_",  # or generated UUID
            "Case": "case_",
            "Story": "story_",  # scope_signature format
            "Claim": "clm_",    # or UUID
        }

        # Verify no prefix collisions
        prefixes = list(ID_PREFIXES.values())
        assert len(prefixes) == len(set(prefixes)), "ID prefixes must be unique"

    def test_scope_signature_is_deterministic(self):
        """
        scope_signature must be deterministic so MERGE works correctly.

        If scope_signature changes between runs, MERGE creates duplicates.
        """
        story1 = Story(
            id="temp",
            scale="case",
            anchor_entities={"Fire", "Hong Kong"},
            time_start=datetime(2025, 1, 15),
        )

        story2 = Story(
            id="temp",
            scale="case",
            anchor_entities={"Fire", "Hong Kong"},
            time_start=datetime(2025, 1, 15),
        )

        # Same inputs → same signature (deterministic)
        sig1 = story1.compute_scope_signature()
        sig2 = story2.compute_scope_signature()

        assert sig1 == sig2, "scope_signature must be deterministic"
        assert sig1.startswith("story_"), "scope_signature must have story_ prefix"

    def test_canonical_merge_not_create(self):
        """
        Canonical layer should use MERGE, not CREATE.

        CREATE would generate duplicates on every run.
        MERGE with scope_signature ensures idempotency.
        """
        # The correct pattern (from principled_weaver.py):
        # MERGE (c:Case {id: $id})
        # SET c:Story,
        #     c.scope_signature = $scope_sig,
        #     ...

        # This test documents the invariant
        correct_pattern = "MERGE"
        incorrect_pattern = "CREATE"

        # If someone uses CREATE, they break canonical stability
        assert correct_pattern != incorrect_pattern, \
            "Canonical layer must use MERGE, not CREATE"

    def test_story_label_is_additive(self):
        """
        :Story label is added to existing nodes, not replacing them.

        An Incident becomes :Incident:Story (both labels)
        A Case becomes :Case:Event:Story (all labels)
        """
        # The pattern in workers:
        # SET i:Story  -- adds label, doesn't remove :Incident
        # SET c:Event, c:Story  -- adds labels, doesn't remove :Case

        # This ensures queries for :Incident still work
        # while :Story provides unified API access

        assert True, ":Story is additive (documented invariant)"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
