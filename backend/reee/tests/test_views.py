"""
Test: Views Module
==================

Tests for multi-scale event views (IncidentEventView, CaseView).

Key behaviors tested:
- Dispersion-based hubness detection
- Hub filtering on all binding signals
- Bridge-resistant clustering (core/periphery)
- Relation backbone
"""

import pytest
from datetime import datetime, timedelta

from reee.types import Surface, Event
from reee.views import (
    CaseViewParams,
    CaseView,
    build_case_clusters,
    analyze_hubness,
    compute_co_anchor_dispersion,
    build_relation_backbone_from_incidents,
    ViewScale,
)


@pytest.fixture
def base_time():
    return datetime(2024, 1, 1)


@pytest.fixture
def simple_incidents(base_time):
    """Two incidents that should cluster together."""
    return {
        'inc1': Event(
            id='inc1',
            surface_ids={'s1', 's2'},
            total_claims=3,
            entities={'A', 'B', 'C'},
            anchor_entities={'A', 'B'},
            time_window=(base_time, base_time + timedelta(days=1)),
        ),
        'inc2': Event(
            id='inc2',
            surface_ids={'s3', 's4'},
            total_claims=2,
            entities={'A', 'B', 'D'},
            anchor_entities={'A', 'B'},
            time_window=(base_time + timedelta(days=10), base_time + timedelta(days=11)),
        ),
    }


@pytest.fixture
def hub_scenario_incidents(base_time):
    """
    Scenario where 'Hong Kong' is a hub (appears in disjoint contexts).

    - HK Democracy: Jimmy Lai + Hong Kong
    - Fire: Tai Po + Hong Kong
    - Market: Hang Seng + Hong Kong

    Hong Kong's co-anchors (Jimmy Lai, Tai Po, Hang Seng) never co-occur,
    so Hong Kong should be classified as a hub.
    """
    return {
        'inc_hk': Event(
            id='inc_hk',
            surface_ids={'s1', 's2'},
            total_claims=5,
            entities={'Jimmy Lai', 'Hong Kong', 'democracy'},
            anchor_entities={'Jimmy Lai', 'Hong Kong'},
            time_window=(base_time, base_time + timedelta(days=1)),
        ),
        'inc_fire': Event(
            id='inc_fire',
            surface_ids={'s3', 's4'},
            total_claims=3,
            entities={'Tai Po', 'Hong Kong', 'fire'},
            anchor_entities={'Tai Po', 'Hong Kong'},
            time_window=(base_time + timedelta(days=30), base_time + timedelta(days=31)),
        ),
        'inc_market': Event(
            id='inc_market',
            surface_ids={'s5', 's6'},
            total_claims=2,
            entities={'Hang Seng', 'Hong Kong', 'stock'},
            anchor_entities={'Hang Seng', 'Hong Kong'},
            time_window=(base_time + timedelta(days=60), base_time + timedelta(days=61)),
        ),
    }


def make_surfaces(n):
    """Helper to create minimal surfaces."""
    return {f's{i}': Surface(id=f's{i}', claim_ids=set()) for i in range(1, n + 1)}


class TestHubnessAnalysis:
    """Tests for dispersion-based hubness detection."""

    def test_hub_detection_high_dispersion(self, hub_scenario_incidents):
        """Hub entity has high dispersion (co-anchors don't co-occur)."""
        result = analyze_hubness(
            hub_scenario_incidents,
            frequency_threshold=3,
            dispersion_threshold=0.7,
        )

        assert 'Hong Kong' in result.hubs
        hk = result.anchors['Hong Kong']
        assert hk.frequency == 3
        assert hk.co_anchor_dispersion == 1.0  # No co-anchors ever co-occur
        assert hk.is_hub is True
        assert hk.is_backbone is False

    def test_backbone_detection_low_dispersion(self, simple_incidents):
        """Backbone entity has low dispersion (co-anchors co-occur)."""
        result = analyze_hubness(
            simple_incidents,
            frequency_threshold=2,
            dispersion_threshold=0.7,
        )

        # A and B appear together in both incidents
        assert 'A' in result.anchors
        assert 'B' in result.anchors

        # They should be backbones (low dispersion) since their co-anchor co-occurs
        a = result.anchors['A']
        b = result.anchors['B']
        assert a.co_anchor_dispersion == 0.0  # B always co-occurs
        assert b.co_anchor_dispersion == 0.0  # A always co-occurs

    def test_neutral_low_frequency(self, hub_scenario_incidents):
        """Low frequency entities are neutral (neither hub nor backbone)."""
        result = analyze_hubness(
            hub_scenario_incidents,
            frequency_threshold=3,  # Only Hong Kong meets this
            dispersion_threshold=0.7,
        )

        # Jimmy Lai appears in only 1 incident
        assert 'Jimmy Lai' in result.anchors
        jl = result.anchors['Jimmy Lai']
        assert jl.frequency == 1
        assert jl.is_hub is False
        assert jl.is_backbone is False  # Neutral


class TestCoAnchorDispersion:
    """Tests for dispersion computation."""

    def test_perfect_dispersion_disjoint_contexts(self, hub_scenario_incidents):
        """Co-anchors that never co-occur give dispersion=1.0."""
        from collections import defaultdict

        anchor_to_incidents = defaultdict(set)
        for inc_id, inc in hub_scenario_incidents.items():
            for anchor in inc.anchor_entities:
                anchor_to_incidents[anchor].add(inc_id)

        dispersion, co_dist = compute_co_anchor_dispersion(
            'Hong Kong',
            hub_scenario_incidents,
            anchor_to_incidents,
        )

        assert dispersion == 1.0
        assert 'Jimmy Lai' in co_dist
        assert 'Tai Po' in co_dist
        assert 'Hang Seng' in co_dist

    def test_zero_dispersion_cohesive_context(self, simple_incidents):
        """Co-anchors that always co-occur give dispersion=0.0."""
        from collections import defaultdict

        anchor_to_incidents = defaultdict(set)
        for inc_id, inc in simple_incidents.items():
            for anchor in inc.anchor_entities:
                anchor_to_incidents[anchor].add(inc_id)

        dispersion, co_dist = compute_co_anchor_dispersion(
            'A',
            simple_incidents,
            anchor_to_incidents,
        )

        assert dispersion == 0.0
        assert 'B' in co_dist


class TestCaseViewClustering:
    """Tests for case-level clustering."""

    def test_shared_anchors_cluster_together(self, simple_incidents, base_time):
        """Incidents sharing anchors should cluster into one case."""
        surfaces = make_surfaces(4)
        params = CaseViewParams()

        view = CaseView(surfaces, simple_incidents, params)
        result = view.build()

        # Both incidents should be in the same case
        assert result.scale == ViewScale.CASE
        non_singletons = [e for e in result.events.values()
                         if not e.id.startswith('case_singleton')]
        assert len(non_singletons) == 1

        case = non_singletons[0]
        assert 's1' in case.surface_ids or 's2' in case.surface_ids
        assert 's3' in case.surface_ids or 's4' in case.surface_ids

    def test_hub_suppression_prevents_bridging(self, hub_scenario_incidents):
        """Hub entities should not bridge unrelated incidents."""
        surfaces = make_surfaces(6)
        params = CaseViewParams(
            use_local_hubness=True,
            hubness_frequency_threshold=3,
            hubness_dispersion_threshold=0.7,
        )

        view = CaseView(surfaces, hub_scenario_incidents, params)
        result = view.build()

        # All should be singletons - Hong Kong is suppressed
        singletons = [e for e in result.events.values()
                      if e.id.startswith('case_singleton')]
        assert len(singletons) == 3

    def test_no_hubness_allows_bridging(self, hub_scenario_incidents):
        """Without hubness, hub entities bridge everything."""
        surfaces = make_surfaces(6)
        params = CaseViewParams(
            use_local_hubness=False,  # Disable hubness
        )

        view = CaseView(surfaces, hub_scenario_incidents, params)
        result = view.build()

        # All should merge into one case via Hong Kong
        non_singletons = [e for e in result.events.values()
                         if not e.id.startswith('case_singleton')]
        assert len(non_singletons) == 1


class TestCorePeripheryClustering:
    """Tests for bridge-resistant core/periphery clustering."""

    def test_core_requires_multiple_signals(self, base_time):
        """Core edges require score >= threshold AND signals >= 2."""
        incidents = {
            'inc1': Event(
                id='inc1',
                surface_ids={'s1'},
                total_claims=2,
                entities={'A', 'B', 'C'},
                anchor_entities={'A', 'B'},
                time_window=(base_time, base_time),
            ),
            'inc2': Event(
                id='inc2',
                surface_ids={'s2'},
                total_claims=2,
                entities={'A', 'B', 'D'},
                anchor_entities={'A', 'B'},
                time_window=(base_time, base_time),
            ),
            'inc3': Event(
                id='inc3',
                surface_ids={'s3'},
                total_claims=1,
                entities={'X'},  # Weak connection only
                anchor_entities={'X'},
                time_window=(base_time, base_time),
            ),
        }
        surfaces = make_surfaces(3)
        params = CaseViewParams(
            case_core_threshold=0.4,
            case_core_min_signals=2,
        )

        view = CaseView(surfaces, incidents, params)
        result = view.build()

        # inc1 and inc2 should form a core, inc3 should be singleton
        cases = list(result.events.values())
        assert any(
            ('s1' in c.surface_ids and 's2' in c.surface_ids)
            for c in cases
        )


class TestRelationBackbone:
    """Tests for relation backbone from anchor co-occurrence."""

    def test_backbone_from_co_occurrence(self, simple_incidents):
        """Anchors co-occurring in multiple incidents form backbone."""
        backbone = build_relation_backbone_from_incidents(
            simple_incidents,
            min_co_occurrence=2,
        )

        # A and B co-occur in both incidents
        assert backbone.are_related('A', 'B', min_corroboration=2)
        assert ('A', 'B') in backbone.edges or ('B', 'A') in backbone.edges

    def test_backbone_requires_threshold(self, simple_incidents):
        """Backbone relations require min co-occurrence threshold."""
        backbone = build_relation_backbone_from_incidents(
            simple_incidents,
            min_co_occurrence=3,  # Higher than actual
        )

        # Should have no relations at this threshold
        assert backbone.total_relations == 0


class TestViewTrace:
    """Tests for view provenance/trace."""

    def test_trace_includes_params(self, simple_incidents):
        """View trace should include parameters used."""
        surfaces = make_surfaces(4)
        params = CaseViewParams(
            temporal_window_days=180,
            hubness_dispersion_threshold=0.6,
        )

        view = CaseView(surfaces, simple_incidents, params)
        result = view.build()

        assert result.trace.view_scale == ViewScale.CASE
        assert result.trace.params_snapshot['temporal_window_days'] == 180
        assert result.trace.params_snapshot['hubness_dispersion_threshold'] == 0.6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
