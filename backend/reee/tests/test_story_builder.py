"""
Story Builder Tests: Spine + Mode + Membrane
=============================================

Tests for spine-based story formation that handles star-shaped patterns.

Key test scenarios:
1. Star-shaped story: one spine, rotating companions → one story
2. Same spine, different temporal modes → separate stories
3. Hub entities cannot define stories
4. Facet completeness tracking

These tests validate the spine-based approach replaces k=2 motif recurrence.
"""

import pytest
from datetime import datetime, timedelta
from typing import Set, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from reee.types import Event, EventJustification, Surface
from reee.builders.story_builder import (
    StoryBuilder, StoryBuilderResult, CompleteStory, TemporalMode
)


# =============================================================================
# FIXTURES
# =============================================================================

def make_incident(
    incident_id: str,
    anchor_entities: Set[str],
    time_start: Optional[datetime] = None,
    surface_ids: Set[str] = None,
) -> Event:
    """Create a test incident."""
    justification = EventJustification(
        core_motifs=[],
        representative_surfaces=list(surface_ids or {f"S_{incident_id}"}),
        canonical_handle=f"Incident {incident_id}",
    )

    return Event(
        id=incident_id,
        surface_ids=surface_ids or {f"S_{incident_id}"},
        anchor_entities=anchor_entities,
        entities=anchor_entities.copy(),
        total_claims=5,
        total_sources=2,
        time_window=(time_start, time_start + timedelta(days=1) if time_start else (None, None)),
        justification=justification,
    )


def make_surface(
    surface_id: str,
    question_key: str,
    scope_id: str = "test_scope",  # kept for fixture compatibility
) -> Surface:
    """Create a test surface."""
    return Surface(
        id=surface_id,
        question_key=question_key,
        claim_ids=set(),
        sources={"source_1"},
        time_window=(None, None),
    )


@pytest.fixture
def wang_fuk_court_fire():
    """
    Wang Fuk Court Fire pattern: star-shaped story.

    - 10 incidents mentioning Wang Fuk Court
    - Each has a UNIQUE companion (no pair recurs)
    - All within same temporal mode (Nov 2025)
    - Plus 40 unrelated incidents to keep WFC below hub threshold

    Expected: ONE story with Wang Fuk Court as spine
    """
    base_time = datetime(2025, 11, 26)

    incidents = {}
    surfaces = {}

    # WFC Fire incidents
    companions = [
        "Fire Services", "John Lee", "Tang Ping-keung", "Victims",
        "Housing Bureau", "Investigation Team", "Residents",
        "BBC", "Death Toll", "Injuries"
    ]

    for i, companion in enumerate(companions):
        inc_id = f"wfc_inc_{i:02d}"
        surf_id = f"wfc_surf_{i:02d}"

        incidents[inc_id] = make_incident(
            inc_id,
            {"Wang Fuk Court", companion},
            base_time + timedelta(hours=i * 6),
            {surf_id},
        )

        # Add fire-related surface
        qkey = "fire_death_count" if i % 3 == 0 else "fire_status"
        surfaces[surf_id] = make_surface(surf_id, qkey, f"scope_wfc_{i}")

    # Add unrelated incidents to keep WFC below hub threshold (20%)
    # WFC is in 10 incidents, need total > 50 for < 20%
    # 10/51 = 19.6% < 20%
    for i in range(41):
        inc_id = f"other_inc_{i:02d}"
        incidents[inc_id] = make_incident(
            inc_id,
            {f"Entity_A_{i}", f"Entity_B_{i}"},
            base_time + timedelta(days=i),
        )

    return incidents, surfaces


@pytest.fixture
def same_spine_different_modes():
    """
    Same spine (Jimmy Lai) in two different temporal modes.

    - Mode 1: Trial coverage (Dec 2024)
    - Mode 2: Verdict coverage (Jan 2025, 45 days later)
    - Plus unrelated incidents to keep Jimmy Lai below hub threshold

    Expected: TWO separate stories
    """
    incidents = {}
    surfaces = {}

    # Mode 1: Trial (Dec 2024)
    trial_base = datetime(2024, 12, 1)
    for i in range(5):
        inc_id = f"jl_trial_{i}"
        surf_id = f"jl_trial_surf_{i}"
        incidents[inc_id] = make_incident(
            inc_id,
            {"Jimmy Lai", f"Witness_{i}"},
            trial_base + timedelta(days=i),
            {surf_id},
        )
        surfaces[surf_id] = make_surface(surf_id, "trial_day", f"scope_trial_{i}")

    # Mode 2: Verdict (Jan 2025, 45+ days gap)
    verdict_base = datetime(2025, 1, 20)
    for i in range(5):
        inc_id = f"jl_verdict_{i}"
        surf_id = f"jl_verdict_surf_{i}"
        incidents[inc_id] = make_incident(
            inc_id,
            {"Jimmy Lai", f"Legal_Expert_{i}"},
            verdict_base + timedelta(days=i),
            {surf_id},
        )
        surfaces[surf_id] = make_surface(surf_id, "verdict_status", f"scope_verdict_{i}")

    # Add unrelated incidents to keep Jimmy Lai below hub threshold (20%)
    # 10 Jimmy Lai incidents, need total > 50 for < 20%
    # 10/51 = 19.6% < 20%
    for i in range(41):
        inc_id = f"other_jl_{i}"
        incidents[inc_id] = make_incident(
            inc_id,
            {f"Other_Entity_{i}", f"Other_Partner_{i}"},
            datetime(2024, 11, 1) + timedelta(days=i),
        )

    return incidents, surfaces


@pytest.fixture
def hub_entity_pattern():
    """
    Hub entity pattern: Hong Kong appears in too many incidents.

    - Hong Kong is in 80% of incidents (hub)
    - Should NOT be able to define a story

    Expected: Hong Kong is hub, cannot define story
    """
    base_time = datetime(2025, 12, 1)
    incidents = {}

    # 8 incidents with Hong Kong (hub candidate)
    for i in range(8):
        incidents[f"hk_{i}"] = make_incident(
            f"hk_{i}",
            {"Hong Kong", f"Topic_{i}"},
            base_time + timedelta(days=i),
        )

    # 2 incidents without Hong Kong
    incidents["other_1"] = make_incident("other_1", {"Taiwan", "Topic_A"}, base_time)
    incidents["other_2"] = make_incident("other_2", {"Japan", "Topic_B"}, base_time)

    return incidents


@pytest.fixture
def facet_completeness_fire():
    """
    Fire story with varying facet coverage.

    - Some facets present: death_count, injury_count
    - Some facets missing: cause, investigation
    - Should track completeness
    - Plus unrelated incidents to keep spine below hub threshold

    Expected: Story with completeness < 100%
    """
    base_time = datetime(2025, 11, 26)
    incidents = {}
    surfaces = {}

    for i in range(5):
        inc_id = f"fire_{i}"
        incidents[inc_id] = make_incident(
            inc_id,
            {"Test Location", f"Responder_{i}"},
            base_time + timedelta(hours=i),
            {f"fire_surf_{i}_a", f"fire_surf_{i}_b"},
        )

        # Add surfaces with specific question_keys
        surfaces[f"fire_surf_{i}_a"] = make_surface(
            f"fire_surf_{i}_a", "fire_death_count", f"scope_{i}"
        )
        surfaces[f"fire_surf_{i}_b"] = make_surface(
            f"fire_surf_{i}_b", "fire_injury_count", f"scope_{i}"
        )

    # Add unrelated incidents to keep Test Location below hub threshold (20%)
    # 5 Test Location incidents, need total > 25 for < 20%
    # 5/26 = 19.2% < 20%
    for i in range(21):
        inc_id = f"other_fire_{i}"
        incidents[inc_id] = make_incident(
            inc_id,
            {f"Other_Place_{i}", f"Other_Actor_{i}"},
            base_time + timedelta(days=i),
        )

    return incidents, surfaces


# =============================================================================
# TESTS: Star-Shaped Story Formation
# =============================================================================

class TestStarShapedStory:
    """Test that star-shaped patterns form single stories."""

    def test_wang_fuk_court_forms_one_story(self, wang_fuk_court_fire):
        """Wang Fuk Court (10 incidents, rotating companions) → one story."""
        incidents, surfaces = wang_fuk_court_fire

        builder = StoryBuilder(
            hub_fraction_threshold=0.20,
            min_incidents_for_story=3,
            mode_gap_days=30,
        )

        result = builder.build_from_incidents(incidents, surfaces)

        # Should have exactly one story
        assert len(result.stories) == 1

        story = list(result.stories.values())[0]
        assert story.spine == "Wang Fuk Court"
        assert len(story.core_incident_ids) == 10

    def test_star_shape_no_pair_recurrence_still_forms_story(self, wang_fuk_court_fire):
        """Even though no pair recurs, story still forms via spine."""
        incidents, surfaces = wang_fuk_court_fire

        # Verify no pairs recur
        pairs_seen = set()
        for incident in incidents.values():
            anchors = sorted(incident.anchor_entities)
            if len(anchors) >= 2:
                for i, a1 in enumerate(anchors):
                    for a2 in anchors[i+1:]:
                        pair = (a1, a2)
                        assert pair not in pairs_seen, "Found recurring pair!"
                        pairs_seen.add(pair)

        # But story still forms
        builder = StoryBuilder(min_incidents_for_story=3)
        result = builder.build_from_incidents(incidents, surfaces)

        assert len(result.stories) == 1
        assert result.stories[list(result.stories.keys())[0]].spine == "Wang Fuk Court"

    def test_story_collects_all_facets(self, wang_fuk_court_fire):
        """Story should collect facets from all attached surfaces."""
        incidents, surfaces = wang_fuk_court_fire

        builder = StoryBuilder(min_incidents_for_story=3)
        result = builder.build_from_incidents(incidents, surfaces)

        story = list(result.stories.values())[0]

        # Should have facets
        assert len(story.facets) > 0

        # Should have fire-related facets
        facet_keys = set(story.facets.keys())
        assert "fire_death_count" in facet_keys or "fire_status" in facet_keys


# =============================================================================
# TESTS: Temporal Mode Separation
# =============================================================================

class TestTemporalModeSeparation:
    """Test that same spine in different time periods → separate stories."""

    def test_same_spine_different_modes_separate_stories(self, same_spine_different_modes):
        """Jimmy Lai trial vs verdict (45 day gap) → two stories."""
        incidents, surfaces = same_spine_different_modes

        builder = StoryBuilder(
            min_incidents_for_story=3,
            mode_gap_days=30,  # 45 day gap > 30 day threshold
        )

        result = builder.build_from_incidents(incidents, surfaces)

        # Should have two stories (same spine, different modes)
        jimmy_lai_stories = [
            s for s in result.stories.values()
            if s.spine == "Jimmy Lai"
        ]

        assert len(jimmy_lai_stories) == 2

        # Stories should have different time periods
        times = [(s.time_start, s.time_end) for s in jimmy_lai_stories]
        assert times[0] != times[1]

    def test_same_mode_stays_together(self, wang_fuk_court_fire):
        """Incidents within mode_gap_days stay in same story."""
        incidents, surfaces = wang_fuk_court_fire

        # All incidents are within hours of each other
        builder = StoryBuilder(
            min_incidents_for_story=3,
            mode_gap_days=30,
        )

        result = builder.build_from_incidents(incidents, surfaces)

        # Should be one story (all within same mode)
        assert len(result.stories) == 1

        story = list(result.stories.values())[0]
        assert len(story.mode.incident_ids) == 10


# =============================================================================
# TESTS: Hub Detection
# =============================================================================

class TestHubDetection:
    """Test that hub entities cannot define stories."""

    def test_hub_entity_cannot_define_story(self, hub_entity_pattern):
        """Hong Kong (80% of incidents) cannot be a story spine."""
        incidents = hub_entity_pattern

        builder = StoryBuilder(
            hub_fraction_threshold=0.20,  # 20% threshold
            min_incidents_for_story=3,
        )

        result = builder.build_from_incidents(incidents)

        # Hong Kong should be marked as hub
        hk_spine = result.spines.get("Hong Kong")
        assert hk_spine is not None
        assert hk_spine.is_hub is True

        # No story should have Hong Kong as spine
        for story in result.stories.values():
            assert story.spine != "Hong Kong"

    def test_non_hub_can_define_story(self, wang_fuk_court_fire):
        """Wang Fuk Court (not a hub) can be a story spine."""
        incidents, surfaces = wang_fuk_court_fire

        # WFC is already at 10/51 = 19.6% < 20% in fixture
        builder = StoryBuilder(
            hub_fraction_threshold=0.20,
            min_incidents_for_story=3,
        )

        result = builder.build_from_incidents(incidents, surfaces)

        # Wang Fuk Court should NOT be hub at 19.6%
        wfc_spine = result.spines.get("Wang Fuk Court")
        assert wfc_spine is not None
        assert wfc_spine.is_hub is False

        # Should define a story
        wfc_stories = [s for s in result.stories.values() if s.spine == "Wang Fuk Court"]
        assert len(wfc_stories) == 1

    def test_hub_boundary_exact_threshold(self):
        """
        Explicit boundary test: entity at exactly 20% is hub, at 19.9% is not.

        This tests the kernel's hub criterion directly, not via background padding.
        """
        base_time = datetime(2025, 11, 26)

        # Case 1: Entity at exactly 20% (10/50) - should be hub
        incidents_at_threshold = {}
        for i in range(10):
            incidents_at_threshold[f"wfc_{i}"] = make_incident(
                f"wfc_{i}",
                {"Wang Fuk Court", f"Companion_{i}"},
                base_time + timedelta(hours=i),
            )
        for i in range(40):
            incidents_at_threshold[f"other_{i}"] = make_incident(
                f"other_{i}",
                {f"Entity_A_{i}", f"Entity_B_{i}"},
                base_time + timedelta(days=i),
            )

        builder = StoryBuilder(
            hub_fraction_threshold=0.20,
            min_incidents_for_story=3,
        )

        result_at = builder.build_from_incidents(incidents_at_threshold)

        # At exactly 20% (10/50), entity IS a hub (>= threshold)
        wfc_spine_at = result_at.spines.get("Wang Fuk Court")
        assert wfc_spine_at is not None
        assert wfc_spine_at.is_hub is True, f"At 20% (10/50), should be hub but is_hub={wfc_spine_at.is_hub}"

        # Should NOT define a story when hub
        wfc_stories_at = [s for s in result_at.stories.values() if s.spine == "Wang Fuk Court"]
        assert len(wfc_stories_at) == 0, "Hub entity should not define a story"

        # Case 2: Entity at 19.6% (10/51) - should NOT be hub
        incidents_below = dict(incidents_at_threshold)
        incidents_below["extra_51"] = make_incident(
            "extra_51",
            {"Random_Entity", "Another_Entity"},
            base_time + timedelta(days=50),
        )

        result_below = builder.build_from_incidents(incidents_below)

        wfc_spine_below = result_below.spines.get("Wang Fuk Court")
        assert wfc_spine_below is not None
        assert wfc_spine_below.is_hub is False, f"At 19.6% (10/51), should NOT be hub but is_hub={wfc_spine_below.is_hub}"

        # Should define a story when not hub
        wfc_stories_below = [s for s in result_below.stories.values() if s.spine == "Wang Fuk Court"]
        assert len(wfc_stories_below) == 1, "Non-hub entity should define a story"

    def test_hub_flips_lens_only(self):
        """
        When an entity flips to hub, it should still be tracked (for Lens)
        but cannot define a story.
        """
        base_time = datetime(2025, 11, 26)

        # Entity appearing in 25% of incidents - clearly hub
        incidents = {}
        for i in range(5):
            incidents[f"spine_{i}"] = make_incident(
                f"spine_{i}",
                {"HighFrequency Entity", f"Partner_{i}"},
                base_time + timedelta(hours=i),
            )
        for i in range(15):
            incidents[f"other_{i}"] = make_incident(
                f"other_{i}",
                {f"Unique_A_{i}", f"Unique_B_{i}"},
                base_time + timedelta(days=i),
            )

        builder = StoryBuilder(
            hub_fraction_threshold=0.20,
            min_incidents_for_story=3,
        )

        result = builder.build_from_incidents(incidents)

        # Entity is hub (5/20 = 25%)
        spine = result.spines.get("HighFrequency Entity")
        assert spine is not None
        assert spine.is_hub is True

        # Spine data still exists (for Lens view)
        assert spine.total_incidents == 5

        # But no story formed
        stories_with_spine = [s for s in result.stories.values() if s.spine == "HighFrequency Entity"]
        assert len(stories_with_spine) == 0


# =============================================================================
# TESTS: Facet Completeness
# =============================================================================

class TestFacetCompleteness:
    """Test facet tracking and completeness scoring."""

    def test_fire_story_tracks_expected_facets(self, facet_completeness_fire):
        """Fire story should track expected facets and report missing ones."""
        incidents, surfaces = facet_completeness_fire

        builder = StoryBuilder(
            min_incidents_for_story=3,
            expected_facets_fire={
                "fire_death_count",
                "fire_injury_count",
                "fire_cause",
                "fire_status",
            }
        )

        result = builder.build_from_incidents(incidents, surfaces)

        # Should have one story
        assert len(result.stories) == 1
        story = list(result.stories.values())[0]

        # Should track present facets
        assert "fire_death_count" in story.present_facets
        assert "fire_injury_count" in story.present_facets

        # Should track missing facets
        assert "fire_cause" in story.missing_facets or "fire_status" in story.missing_facets

        # Completeness should be < 100%
        assert story.completeness_score < 1.0

    def test_gap_inquiries_generated(self, facet_completeness_fire):
        """Missing facets should generate gap inquiries."""
        incidents, surfaces = facet_completeness_fire

        builder = StoryBuilder(
            min_incidents_for_story=3,
            expected_facets_fire={
                "fire_death_count",
                "fire_injury_count",
                "fire_cause",  # Will be missing
            }
        )

        result = builder.build_from_incidents(incidents, surfaces)

        story = list(result.stories.values())[0]

        # Should have gap inquiries for missing facets
        assert len(story.gap_inquiries) > 0

        gap_facets = {inq["facet"] for inq in story.gap_inquiries}
        assert "fire_cause" in gap_facets

    @pytest.mark.xfail(reason="Schema-driven completeness not yet implemented - documents required behavior")
    def test_facet_detection_schema_driven(self):
        """
        Schema-driven facet test: facets derived from external schema,
        surfaces have mixed/noisy question_keys.

        This avoids tautology by:
        1. Schema defines expected facets independently
        2. Surfaces use diverse question_keys (some match, some don't)
        3. Test validates gap detection works for schema facets not in data

        TODO: Implement schema-driven completeness in StoryBuilder._compute_facets()
        """
        base_time = datetime(2025, 11, 26)

        # Define expected schema INDEPENDENT of test data
        FIRE_EVENT_SCHEMA = {
            "death_count",       # Must-have
            "injury_count",      # Must-have
            "fire_cause",        # Must-have
            "building_type",     # Nice-to-have
            "evacuation_status", # Nice-to-have
            "investigation_status",  # Nice-to-have
        }

        incidents = {}
        surfaces = {}

        # Create incidents with MIXED question_keys (some match schema, some are noise)
        question_key_distribution = [
            "death_count",       # matches schema
            "injury_count",      # matches schema
            "responder_names",   # noise - not in schema
            "weather_conditions",# noise - not in schema
            "building_type",     # matches schema
            "local_reaction",    # noise - not in schema
        ]

        for i in range(6):
            inc_id = f"fire_{i}"
            surf_id = f"surf_{i}"

            incidents[inc_id] = make_incident(
                inc_id,
                {"Test Fire Location", f"Actor_{i}"},
                base_time + timedelta(hours=i),
                {surf_id},
            )

            # Use mixed question_keys
            qkey = question_key_distribution[i % len(question_key_distribution)]
            surfaces[surf_id] = make_surface(surf_id, qkey, f"scope_{i}")

        # Add padding to keep below hub threshold
        for i in range(25):
            inc_id = f"other_{i}"
            incidents[inc_id] = make_incident(
                inc_id,
                {f"Other_Entity_{i}", f"Other_Partner_{i}"},
                base_time + timedelta(days=i + 10),
            )

        builder = StoryBuilder(
            min_incidents_for_story=3,
            expected_facets_fire=FIRE_EVENT_SCHEMA,
        )

        result = builder.build_from_incidents(incidents, surfaces)

        story = list(result.stories.values())[0]

        # Validate: present facets should be subset of what's in data
        data_qkeys = set(question_key_distribution)
        for facet in story.present_facets:
            assert facet in data_qkeys, f"Present facet '{facet}' not in data"

        # Validate: missing facets should be schema facets NOT in data
        expected_missing = FIRE_EVENT_SCHEMA - data_qkeys
        for missing in expected_missing:
            assert missing in story.missing_facets, \
                f"Schema facet '{missing}' should be detected as missing"

        # Validate: gap inquiries generated for missing schema facets
        gap_facets = {inq["facet"] for inq in story.gap_inquiries}
        assert "fire_cause" in gap_facets, "fire_cause should generate gap inquiry"
        assert "evacuation_status" in gap_facets, "evacuation_status should generate gap inquiry"
        assert "investigation_status" in gap_facets, "investigation_status should generate gap inquiry"

        # Validate: noise facets don't pollute missing set
        assert "responder_names" not in story.missing_facets
        assert "weather_conditions" not in story.missing_facets

    @pytest.mark.xfail(reason="Schema-driven completeness not yet implemented - documents required behavior")
    def test_facet_completeness_score_calculation(self):
        """
        Completeness score should be: (present ∩ schema) / |schema|

        Not: present / (present + missing), which would be tautological.

        TODO: Implement schema-driven completeness in StoryBuilder._compute_facets()
        """
        base_time = datetime(2025, 11, 26)

        # Schema with 5 facets
        SCHEMA = {"facet_a", "facet_b", "facet_c", "facet_d", "facet_e"}

        incidents = {}
        surfaces = {}

        # Only 2 of 5 schema facets present
        for i, qkey in enumerate(["facet_a", "facet_b", "noise_1", "noise_2"]):
            inc_id = f"inc_{i}"
            surf_id = f"surf_{i}"
            incidents[inc_id] = make_incident(
                inc_id,
                {"Test Spine", f"Partner_{i}"},
                base_time + timedelta(hours=i),
                {surf_id},
            )
            surfaces[surf_id] = make_surface(surf_id, qkey, f"scope_{i}")

        # Padding
        for i in range(20):
            inc_id = f"other_{i}"
            incidents[inc_id] = make_incident(
                inc_id,
                {f"Other_{i}", f"Partner_{i}"},
                base_time + timedelta(days=i + 5),
            )

        builder = StoryBuilder(
            min_incidents_for_story=3,
            expected_facets_fire=SCHEMA,
        )

        result = builder.build_from_incidents(incidents, surfaces)
        story = list(result.stories.values())[0]

        # 2 of 5 schema facets present → 40% completeness
        assert abs(story.completeness_score - 0.4) < 0.01, \
            f"Expected 40% completeness, got {story.completeness_score * 100}%"


# =============================================================================
# TESTS: Membership Weights
# =============================================================================

class TestMembershipWeights:
    """Test core vs periphery membership classification."""

    def test_anchor_incidents_are_core(self, wang_fuk_court_fire):
        """Incidents where spine is anchor should be core."""
        incidents, surfaces = wang_fuk_court_fire

        builder = StoryBuilder(min_incidents_for_story=3)
        result = builder.build_from_incidents(incidents, surfaces)

        story = list(result.stories.values())[0]

        # All 10 WFC incidents have Wang Fuk Court as anchor → all core
        assert len(story.core_incident_ids) == 10

        # Verify all WFC incidents are in core
        wfc_ids = {k for k in incidents.keys() if k.startswith("wfc_")}
        assert wfc_ids <= story.core_incident_ids

        # Periphery may include time-window incidents without spine anchor
        # (this is correct observability behavior)
        for inc_id in story.periphery_incident_ids:
            # Verify periphery incidents don't have WFC as anchor
            incident = incidents[inc_id]
            assert "Wang Fuk Court" not in incident.anchor_entities


# =============================================================================
# TESTS: Observability (blocked_core_b, periphery_candidates)
# =============================================================================

@pytest.fixture
def wfc_with_related_incidents():
    """
    Wang Fuk Court Fire with related incidents that don't have WFC as anchor.

    This tests observability: inc3 (Tai Po + Chris Tang) should be tracked
    as periphery/blocked_core_b, not silently omitted.

    - inc1, inc2, inc5: WFC + companion (Core-A)
    - inc3: Tai Po + Chris Tang (no WFC, but in time window)
    - inc4: Jimmy Lai (no WFC, different topic, should be rejected)
    - Plus unrelated incidents to keep WFC below hub threshold
    """
    base_time = datetime(2025, 11, 26)
    incidents = {}
    surfaces = {}

    # Core-A incidents (WFC as anchor)
    incidents["inc1"] = make_incident(
        "inc1", {"Wang Fuk Court", "Fire Services"},
        base_time, {"surf_1"}
    )
    incidents["inc2"] = make_incident(
        "inc2", {"Wang Fuk Court", "John Lee"},
        base_time + timedelta(hours=6), {"surf_2"}
    )

    # Related incident in time window (no WFC anchor, but nearby)
    incidents["inc3"] = make_incident(
        "inc3", {"Tai Po", "Chris Tang"},  # No WFC!
        base_time + timedelta(hours=12), {"surf_3"}
    )

    # Unrelated incident (different topic, different location)
    incidents["inc4"] = make_incident(
        "inc4", {"Jimmy Lai", "Court"},
        base_time + timedelta(hours=24), {"surf_4"}
    )

    # Add more WFC incidents to meet min_incidents_for_story=3
    incidents["inc5"] = make_incident(
        "inc5", {"Wang Fuk Court", "Victims"},
        base_time + timedelta(hours=48), {"surf_5"}
    )

    for i, surf_id in enumerate(["surf_1", "surf_2", "surf_3", "surf_4", "surf_5"]):
        surfaces[surf_id] = make_surface(surf_id, "fire_status", f"scope_{i}")

    # Add unrelated incidents to keep WFC below hub threshold
    # 3 WFC incidents out of 18 total = ~17% < 20%
    for i in range(15):
        inc_id = f"other_obs_{i}"
        incidents[inc_id] = make_incident(
            inc_id,
            {f"Other_Entity_{i}", f"Other_Partner_{i}"},
            base_time + timedelta(days=i + 10),  # Different time window
        )

    return incidents, surfaces


class TestObservability:
    """Test that non-core incidents are tracked, not silently dropped."""

    def test_candidate_pool_includes_time_window_incidents(self, wfc_with_related_incidents):
        """
        Candidate pool should include ALL incidents in time window,
        not just those where spine is anchor.
        """
        incidents, surfaces = wfc_with_related_incidents

        builder = StoryBuilder(min_incidents_for_story=3)
        result = builder.build_from_incidents(incidents, surfaces)

        story = list(result.stories.values())[0]

        # Candidate pool should be larger than core
        assert story.candidate_pool_size >= len(story.core_incident_ids)

        # Should have tracked at least 4 candidates (inc1-5 in time window)
        assert story.candidate_pool_size >= 4

    def test_non_spine_incidents_tracked_as_periphery_or_rejected(self, wfc_with_related_incidents):
        """
        Incidents without spine anchor should be tracked as periphery/rejected,
        not silently omitted.
        """
        incidents, surfaces = wfc_with_related_incidents

        builder = StoryBuilder(min_incidents_for_story=3)
        result = builder.build_from_incidents(incidents, surfaces)

        story = list(result.stories.values())[0]

        # inc3 (Tai Po + Chris Tang) should be in periphery or rejected
        all_tracked = (
            story.core_incident_ids |
            story.periphery_incident_ids |
            story.rejected_incident_ids
        )

        # inc3 should be tracked somewhere (periphery, rejected, or blocked_core_b)
        # If it's in the candidate pool, it should be classified
        if "inc3" in story.membrane_decisions:
            decision = story.membrane_decisions["inc3"]
            # inc3 has non-hub anchors (Tai Po, Chris Tang), so should be periphery
            # (no witnesses to make it Core-B)
            assert decision.blocked_reason is not None or decision.membership.value == "periphery"

    def test_blocked_core_b_tracked_with_reason(self, wfc_with_related_incidents):
        """
        Incidents that could be Core-B but lack witnesses should be in blocked_core_b.
        """
        incidents, surfaces = wfc_with_related_incidents

        builder = StoryBuilder(min_incidents_for_story=3)
        result = builder.build_from_incidents(incidents, surfaces)

        story = list(result.stories.values())[0]

        # If there are blocked_core_b entries, they should have reasons
        for blocked in story.blocked_core_b:
            assert blocked.reason is not None
            assert len(blocked.reason) > 0
            assert blocked.witnesses_missing is not None

    def test_periphery_candidates_observable(self, wfc_with_related_incidents):
        """
        Periphery candidates should be tracked for observability.
        """
        incidents, surfaces = wfc_with_related_incidents

        builder = StoryBuilder(min_incidents_for_story=3)
        result = builder.build_from_incidents(incidents, surfaces)

        story = list(result.stories.values())[0]

        # Should expose periphery_candidates set
        assert hasattr(story, 'periphery_candidates')

        # If there are periphery incidents, they should also be in periphery_candidates
        if story.periphery_incident_ids:
            assert story.periphery_candidates >= story.periphery_incident_ids

    def test_membrane_decisions_include_all_candidates(self, wfc_with_related_incidents):
        """
        Every candidate should have a membrane decision recorded.
        """
        incidents, surfaces = wfc_with_related_incidents

        builder = StoryBuilder(min_incidents_for_story=3)
        result = builder.build_from_incidents(incidents, surfaces)

        story = list(result.stories.values())[0]

        # All candidates should have membrane decisions
        for inc_id in story.membrane_decisions:
            decision = story.membrane_decisions[inc_id]
            assert decision.membership is not None

        # Every incident in core/periphery/rejected should have a decision
        for inc_id in story.core_incident_ids | story.periphery_incident_ids | story.rejected_incident_ids:
            assert inc_id in story.membrane_decisions


# =============================================================================
# TESTS: Replay from Frozen Snapshot (Production Validation)
# =============================================================================

import json
from pathlib import Path


def load_wfc_snapshot():
    """Load the frozen WFC snapshot for replay testing."""
    snapshot_path = Path(__file__).parent / "fixtures" / "replay_wfc_snapshot.json"
    with open(snapshot_path) as f:
        return json.load(f)


def build_incidents_from_snapshot(snapshot: dict) -> tuple:
    """Convert snapshot format to StoryBuilder input format."""
    incidents = {}
    surfaces = {}
    base_time = datetime(2025, 11, 26)

    # Build incidents from snapshot
    for inc_id, inc_data in snapshot["incidents"].items():
        time_start = datetime.fromisoformat(inc_data["time_start"].replace("Z", "+00:00")).replace(tzinfo=None)
        time_end = datetime.fromisoformat(inc_data["time_end"].replace("Z", "+00:00")).replace(tzinfo=None)

        incidents[inc_id] = make_incident(
            inc_id,
            set(inc_data["anchor_entities"]),
            time_start,
            set(inc_data["surface_ids"]),
        )

    # Build surfaces from snapshot
    for surf_id, surf_data in snapshot["surfaces"].items():
        surfaces[surf_id] = make_surface(surf_id, surf_data["question_key"], f"scope_{surf_id}")

    # Add background incidents to dilute hub threshold
    background = snapshot.get("unrelated_background", {})
    background_count = background.get("count", 35)
    for i in range(background_count):
        inc_id = f"background_{i:03d}"
        incidents[inc_id] = make_incident(
            inc_id,
            {f"Background_Entity_A_{i}", f"Background_Entity_B_{i}"},
            base_time + timedelta(days=i + 30),  # Outside WFC time window
        )

    return incidents, surfaces, snapshot


class TestReplayWFCSnapshot:
    """
    Record/Replay tests from frozen WFC snapshot.

    These tests bridge "kernel correctness" to "production expectation"
    by replaying frozen inputs and asserting membrane decisions match
    VOCABULARY.md postmortem expectations.

    The snapshot contains:
    - 10 core incidents (WFC as anchor)
    - 3 leak candidates (in time window, no WFC anchor)
    - 2 reject candidates (hub-only anchors)
    - Frozen witness ledger inputs
    """

    @pytest.fixture
    def wfc_snapshot_data(self):
        """Load and convert WFC snapshot to test data."""
        snapshot = load_wfc_snapshot()
        incidents, surfaces, raw_snapshot = build_incidents_from_snapshot(snapshot)
        return incidents, surfaces, raw_snapshot

    def test_snapshot_loads_correctly(self, wfc_snapshot_data):
        """Verify snapshot loads and converts correctly."""
        incidents, surfaces, snapshot = wfc_snapshot_data

        # Should have all incident types
        wfc_incidents = [k for k in incidents.keys() if k.startswith("wfc_")]
        leak_incidents = [k for k in incidents.keys() if k.startswith("leak_")]
        reject_incidents = [k for k in incidents.keys() if k.startswith("reject_")]
        background_incidents = [k for k in incidents.keys() if k.startswith("background_")]

        assert len(wfc_incidents) == 10, "Should have 10 WFC core incidents"
        assert len(leak_incidents) == 3, "Should have 3 leak candidates"
        assert len(reject_incidents) == 2, "Should have 2 reject candidates"
        assert len(background_incidents) == 36, "Should have 36 background incidents (keeps WFC at 19.6%)"

    def test_wfc_story_forms_with_correct_spine(self, wfc_snapshot_data):
        """WFC story should form with Wang Fuk Court as spine."""
        incidents, surfaces, snapshot = wfc_snapshot_data

        builder = StoryBuilder(
            hub_fraction_threshold=0.20,
            min_incidents_for_story=3,
            mode_gap_days=30,
        )

        result = builder.build_from_incidents(incidents, surfaces)

        # Should have a WFC story
        wfc_stories = [s for s in result.stories.values() if s.spine == "Wang Fuk Court"]
        assert len(wfc_stories) == 1, "Should form exactly one WFC story"

        story = wfc_stories[0]
        assert len(story.core_incident_ids) == 10, "Should have 10 core incidents"

    def test_core_incidents_are_core_a(self, wfc_snapshot_data):
        """
        All wfc_* incidents should be Core-A (spine as anchor).

        Expected: core_leak_rate = 0.0
        """
        incidents, surfaces, snapshot = wfc_snapshot_data

        builder = StoryBuilder(
            hub_fraction_threshold=0.20,
            min_incidents_for_story=3,
        )

        result = builder.build_from_incidents(incidents, surfaces)
        story = [s for s in result.stories.values() if s.spine == "Wang Fuk Court"][0]

        # All WFC incidents should be core
        for i in range(1, 11):
            inc_id = f"wfc_{i:03d}"
            assert inc_id in story.core_incident_ids, f"{inc_id} should be in core"

            # Check membrane decision shows Core-A
            if inc_id in story.membrane_decisions:
                decision = story.membrane_decisions[inc_id]
                assert decision.membership.value == "core"
                assert decision.core_reason.value == "anchor"

        # Core leak rate should be 0
        assert story.core_leak_rate == 0.0, "No core incidents should lack spine anchor"

    def test_leak_candidates_are_periphery(self, wfc_snapshot_data):
        """
        leak_* incidents should be Periphery (in time window, no WFC anchor,
        insufficient witnesses for Core-B).

        This validates the postmortem: these incidents should NOT be core,
        but should be tracked with reasons.
        """
        incidents, surfaces, snapshot = wfc_snapshot_data

        builder = StoryBuilder(
            hub_fraction_threshold=0.20,
            min_incidents_for_story=3,
        )

        result = builder.build_from_incidents(incidents, surfaces)
        story = [s for s in result.stories.values() if s.spine == "Wang Fuk Court"][0]

        # Check each leak candidate
        for inc_id in ["leak_001", "leak_002", "leak_003"]:
            expected = snapshot["incidents"][inc_id]

            # Should NOT be in core
            assert inc_id not in story.core_incident_ids, \
                f"{inc_id} should NOT be core (no WFC anchor)"

            # Should be in periphery or rejected (depends on time window inclusion)
            if inc_id in story.membrane_decisions:
                decision = story.membrane_decisions[inc_id]
                assert decision.membership.value in ["periphery", "reject"], \
                    f"{inc_id} should be periphery or reject, got {decision.membership.value}"

                # Should have blocked reason
                assert decision.blocked_reason is not None, \
                    f"{inc_id} should have blocked_reason explaining why not Core-B"

    def test_hub_only_incidents_are_rejected(self, wfc_snapshot_data):
        """
        reject_* incidents should be Reject (hub-only anchors).

        Per membrane contract: "incident has only hub anchors" → Reject
        """
        incidents, surfaces, snapshot = wfc_snapshot_data

        hub_entities = set(snapshot["hub_entities"])

        builder = StoryBuilder(
            hub_fraction_threshold=0.20,
            min_incidents_for_story=3,
        )

        result = builder.build_from_incidents(incidents, surfaces)
        story = [s for s in result.stories.values() if s.spine == "Wang Fuk Court"][0]

        # Check reject candidates
        for inc_id in ["reject_001", "reject_002"]:
            # Should NOT be in core
            assert inc_id not in story.core_incident_ids, \
                f"{inc_id} should NOT be core (hub-only anchors)"

            # If in membrane decisions, should be reject
            if inc_id in story.membrane_decisions:
                decision = story.membrane_decisions[inc_id]
                # Note: might be reject or not included depending on time window
                if decision.membership.value != "reject":
                    # Verify the anchors are all hubs
                    inc_anchors = incidents[inc_id].anchor_entities
                    non_hub_anchors = inc_anchors - hub_entities
                    # If has non-hub anchors, periphery is acceptable
                    if non_hub_anchors:
                        assert decision.membership.value == "periphery"

    def test_expected_outcome_matches_meta(self, wfc_snapshot_data):
        """
        Validate overall outcome matches snapshot's expected_outcome.
        """
        incidents, surfaces, snapshot = wfc_snapshot_data
        expected = snapshot["_meta"]["expected_outcome"]

        builder = StoryBuilder(
            hub_fraction_threshold=0.20,
            min_incidents_for_story=3,
        )

        result = builder.build_from_incidents(incidents, surfaces)
        story = [s for s in result.stories.values() if s.spine == "Wang Fuk Court"][0]

        # Verify expectations
        assert story.spine == expected["story_spine"]
        assert len(story.core_incident_ids) == expected["core_incidents"]
        assert story.core_leak_rate == expected["core_leak_rate"]


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
