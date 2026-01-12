"""
Tests for REEE Kernel modules.

These tests verify:
- Scope computation purity and determinism
- Question key extraction fallback chain
- Hub entity suppression
- Pattern matching for typed questions
"""

import pytest
from typing import FrozenSet

from reee.kernel.scope import (
    compute_scope_id,
    is_hub_entity,
    extract_primary_anchors,
    DEFAULT_HUB_ENTITIES,
)
from reee.kernel.question_key import (
    extract_question_key,
    FallbackLevel,
    is_typed_question,
    extract_question_type,
)


class TestScopeComputation:
    """Tests for scope_id computation."""

    def test_deterministic_scope(self):
        """Same anchors should produce same scope_id."""
        anchors1 = frozenset({"Hong Kong", "John Lee"})
        anchors2 = frozenset({"John Lee", "Hong Kong"})  # Different order

        scope1 = compute_scope_id(anchors1)
        scope2 = compute_scope_id(anchors2)

        assert scope1 == scope2

    def test_scope_format(self):
        """Scope should have correct format."""
        anchors = frozenset({"Hong Kong", "John Lee"})
        scope = compute_scope_id(anchors)

        assert scope.startswith("scope_")
        # Two anchors, sorted and joined
        assert "hongkong" in scope or "johnlee" in scope

    def test_hub_suppression(self):
        """Hub entities should be suppressed from scope."""
        anchors = frozenset({"Hong Kong", "United States"})
        scope = compute_scope_id(anchors)

        # "United States" is a hub, should be suppressed
        assert "hongkong" in scope
        assert "unitedstates" not in scope

    def test_all_hubs_fallback(self):
        """If all anchors are hubs, use them anyway."""
        anchors = frozenset({"United States", "China"})
        scope = compute_scope_id(anchors)

        # All are hubs, so use them
        assert scope.startswith("scope_")
        assert len(scope) > len("scope_")

    def test_empty_anchors(self):
        """Empty anchors should produce unscoped."""
        scope = compute_scope_id(frozenset())
        assert scope == "scope_unscoped"

    def test_single_anchor(self):
        """Single anchor should work."""
        scope = compute_scope_id(frozenset({"California"}))
        assert scope == "scope_california"

    def test_max_two_anchors(self):
        """Only top 2 anchors used for scope drift tolerance."""
        anchors = frozenset({"A", "B", "C", "D"})
        scope = compute_scope_id(anchors)

        # Normalized and sorted: a, b, c, d -> uses a, b
        assert scope == "scope_a_b"

    def test_normalization(self):
        """Names should be normalized (lowercase, no spaces/apostrophes)."""
        anchors = frozenset({"John O'Brien", "Mary-Anne Smith"})
        scope = compute_scope_id(anchors)

        assert "'" not in scope
        assert "-" not in scope
        assert " " not in scope


class TestHubEntity:
    """Tests for hub entity detection."""

    def test_known_hubs(self):
        """Known hub entities should be detected."""
        assert is_hub_entity("United States")
        assert is_hub_entity("China")
        assert is_hub_entity("European Union")

    def test_non_hubs(self):
        """Non-hub entities should not be detected."""
        assert not is_hub_entity("Hong Kong")
        assert not is_hub_entity("John Lee")
        assert not is_hub_entity("California")


class TestExtractPrimaryAnchors:
    """Tests for primary anchor extraction."""

    def test_filters_hubs(self):
        """Should filter out hub entities."""
        entities = frozenset({"Hong Kong", "United States", "John Lee"})
        anchors, all_hubs = extract_primary_anchors(entities)

        assert "Hong Kong" in anchors
        assert "John Lee" in anchors
        assert "United States" not in anchors
        assert not all_hubs

    def test_all_hubs_fallback(self):
        """If all are hubs, use them anyway."""
        entities = frozenset({"United States", "China"})
        anchors, all_hubs = extract_primary_anchors(entities)

        assert len(anchors) > 0
        assert all_hubs

    def test_max_anchors(self):
        """Should respect max_anchors limit."""
        entities = frozenset({"A", "B", "C", "D", "E", "F"})
        anchors, _ = extract_primary_anchors(entities, max_anchors=3)

        assert len(anchors) == 3


class TestQuestionKeyExtraction:
    """Tests for question_key extraction."""

    def test_explicit_key(self):
        """Explicit key should be used when provided."""
        result = extract_question_key(
            text="Some claim text",
            entities=frozenset({"A"}),
            anchors=frozenset({"A"}),
            page_id="page_1",
            claim_id="c1",
            explicit_key="custom_key",
            explicit_confidence=0.9,
        )

        assert result.question_key == "custom_key"
        assert result.fallback_level == FallbackLevel.EXPLICIT
        assert result.confidence == 0.9

    def test_explicit_key_low_confidence_ignored(self):
        """Explicit key with low confidence should fall through."""
        result = extract_question_key(
            text="10 people died in the fire",
            entities=frozenset({"A"}),
            anchors=frozenset({"A"}),
            page_id="page_1",
            claim_id="c1",
            explicit_key="low_conf_key",
            explicit_confidence=0.5,  # Below 0.7 threshold
        )

        # Should fall through to pattern matching
        assert result.question_key != "low_conf_key"
        assert result.fallback_level == FallbackLevel.PATTERN

    def test_pattern_death_count(self):
        """Death patterns should extract death_count key."""
        result = extract_question_key(
            text="10 people killed in the fire",
            entities=frozenset(),
            anchors=frozenset(),
            page_id="page_1",
            claim_id="c1",
        )

        assert "death_count" in result.question_key
        assert result.fallback_level == FallbackLevel.PATTERN

    def test_pattern_injury_count(self):
        """Injury patterns should extract injury_count key."""
        result = extract_question_key(
            text="30 people were injured",
            entities=frozenset(),
            anchors=frozenset(),
            page_id="page_1",
            claim_id="c1",
        )

        assert "injury_count" in result.question_key
        assert result.fallback_level == FallbackLevel.PATTERN

    def test_pattern_with_event_type(self):
        """Event type should prefix the question key."""
        result = extract_question_key(
            text="The fire killed 10 people",
            entities=frozenset(),
            anchors=frozenset(),
            page_id="page_1",
            claim_id="c1",
        )

        assert result.question_key == "fire_death_count"

    def test_entity_fallback(self):
        """Entity-derived key when no pattern matches."""
        result = extract_question_key(
            text="John spoke about the economy",
            entities=frozenset({"John Lee", "Hong Kong"}),
            anchors=frozenset({"John Lee", "Hong Kong"}),
            page_id="page_1",
            claim_id="c1",
        )

        assert result.fallback_level == FallbackLevel.ENTITY
        assert result.question_key.startswith("about_")
        assert result.confidence == 0.6

    def test_page_scope_fallback(self):
        """Page scope fallback when no entities."""
        result = extract_question_key(
            text="Something happened",
            entities=frozenset(),
            anchors=frozenset(),
            page_id="page_123",
            claim_id="c1",
        )

        assert result.fallback_level == FallbackLevel.PAGE_SCOPE
        assert "page_123" in result.question_key

    def test_singleton_fallback(self):
        """Singleton fallback when nothing else works."""
        result = extract_question_key(
            text="Something happened",
            entities=frozenset(),
            anchors=frozenset(),
            page_id=None,
            claim_id="claim_abc",
        )

        assert result.fallback_level == FallbackLevel.SINGLETON
        assert "claim_abc" in result.question_key
        assert result.confidence == 0.1


class TestTypedQuestion:
    """Tests for typed question utilities."""

    def test_is_typed_question_count(self):
        """Count questions should be typed."""
        assert is_typed_question("fire_death_count")
        assert is_typed_question("incident_injury_count")

    def test_is_typed_question_not_typed(self):
        """Non-count questions should not be typed."""
        assert not is_typed_question("about_hongkong_johnlee")
        assert not is_typed_question("page_scope_123")
        assert not is_typed_question("policy_announcement")

    def test_extract_question_type(self):
        """Should extract type from question_key."""
        assert extract_question_type("fire_death_count") == "count"
        assert extract_question_type("incident_status") == "status"
        assert extract_question_type("person_location") == "location"

    def test_extract_question_type_none(self):
        """Should return None for untyped questions."""
        assert extract_question_type("about_hongkong") is None
        assert extract_question_type("singleton_123") is None


class TestFallbackChainOrdering:
    """Tests verifying fallback chain priority."""

    def test_explicit_beats_pattern(self):
        """Explicit key beats pattern matching."""
        result = extract_question_key(
            text="10 people died",  # Would match death pattern
            entities=frozenset(),
            anchors=frozenset(),
            page_id="page_1",
            claim_id="c1",
            explicit_key="explicit_key",
            explicit_confidence=0.9,
        )

        assert result.question_key == "explicit_key"
        assert result.fallback_level == FallbackLevel.EXPLICIT

    def test_pattern_beats_entity(self):
        """Pattern matching beats entity-derived."""
        result = extract_question_key(
            text="10 casualties reported",
            entities=frozenset({"Hospital X"}),
            anchors=frozenset({"Hospital X"}),
            page_id="page_1",
            claim_id="c1",
        )

        assert "death_count" in result.question_key
        assert result.fallback_level == FallbackLevel.PATTERN

    def test_entity_beats_page(self):
        """Entity-derived beats page scope."""
        result = extract_question_key(
            text="Some claim",  # No pattern match
            entities=frozenset({"Entity A"}),
            anchors=frozenset({"Entity A"}),
            page_id="page_1",
            claim_id="c1",
        )

        assert result.fallback_level == FallbackLevel.ENTITY

    def test_page_beats_singleton(self):
        """Page scope beats singleton."""
        result = extract_question_key(
            text="Some claim",
            entities=frozenset(),
            anchors=frozenset(),
            page_id="page_xyz",
            claim_id="c1",
        )

        assert result.fallback_level == FallbackLevel.PAGE_SCOPE
