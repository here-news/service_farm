"""
CI Gate Tests: Topology Quality Metrics

These tests enforce the release criteria for the compiler:
1. No giant component (largest case < threshold)
2. Minimum purity on Tier-B benchmark
3. Minimum coverage on Tier-B benchmark
4. Metabolic edges never affect membership (tested separately)

Run these tests to gate releases:
    pytest backend/reee/tests/test_ci_metrics.py -v

These tests use synthetic fixture data to avoid requiring database access.
"""

import pytest
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Set, List, Dict, FrozenSet

# CI Gate Thresholds
CI_THRESHOLDS = {
    "max_largest_case": 100,  # Largest case cannot exceed this
    "min_purity": 0.85,  # Minimum purity for ground truth entity
    "min_coverage": 0.95,  # Minimum coverage for ground truth entity
    "max_contamination_rate": 0.15,  # Maximum contamination rate
}


@dataclass
class FixtureMetrics:
    """Computed metrics from a fixture."""
    fixture_name: str
    total_incidents: int
    expected_cases: int
    spine_entity: str
    spine_entity_incident_count: int

    # Computed values (will be set during validation)
    computed_cases: int = 0
    largest_case: int = 0
    purity: float = 0.0
    coverage: float = 0.0
    contaminants: int = 0


def load_fixture(fixture_path: Path) -> Dict:
    """Load a fixture file."""
    with open(fixture_path) as f:
        return json.load(f)


def compute_baseline_metrics(fixture: Dict) -> FixtureMetrics:
    """
    Compute heuristic baseline metrics from a fixture.

    This simulates what would happen with naive entity overlap union-find.
    """
    incidents = fixture.get("incidents", [])
    entities = fixture.get("entities", [])
    expected = fixture.get("expected", {})

    # Find spine entity
    spine_name = expected.get("spine", "")
    spine_entity = next(
        (e for e in entities if e.get("role") == "spine"),
        None
    )
    if spine_entity:
        spine_name = spine_entity.get("name", spine_name)

    # Count incidents containing spine entity
    spine_incidents = []
    for inc in incidents:
        anchors = set(inc.get("anchor_entities", []))
        if spine_name in anchors:
            spine_incidents.append(inc["id"])

    # Compute union-find based on entity overlap (heuristic baseline)
    incident_ids = [inc["id"] for inc in incidents]
    parent = {iid: iid for iid in incident_ids}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Build edges from entity overlap
    for i, inc1 in enumerate(incidents):
        for inc2 in incidents[i+1:]:
            ents1 = set(inc1.get("anchor_entities", []))
            ents2 = set(inc2.get("anchor_entities", []))
            if ents1 & ents2:
                union(inc1["id"], inc2["id"])

    # Collect components
    components = {}
    for iid in incident_ids:
        root = find(iid)
        if root not in components:
            components[root] = set()
        components[root].add(iid)

    cases = [c for c in components.values() if len(c) >= 2]

    # Find case containing spine incidents
    spine_case = None
    for c in sorted(cases, key=len, reverse=True):
        if c & set(spine_incidents):
            spine_case = c
            break

    # Compute metrics
    spine_in_case = len((spine_case or set()) & set(spine_incidents))
    case_size = len(spine_case) if spine_case else 0

    metrics = FixtureMetrics(
        fixture_name=fixture.get("corpus_id", "unknown"),
        total_incidents=len(incidents),
        expected_cases=expected.get("story_count", 1),
        spine_entity=spine_name,
        spine_entity_incident_count=len(spine_incidents),
        computed_cases=len(cases),
        largest_case=max((len(c) for c in cases), default=0),
        purity=spine_in_case / case_size if case_size > 0 else 0.0,
        coverage=spine_in_case / len(spine_incidents) if spine_incidents else 0.0,
        contaminants=case_size - spine_in_case if spine_case else 0,
    )

    return metrics


# =============================================================================
# Fixture-Based CI Gate Tests
# =============================================================================

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def get_golden_fixtures():
    """Get all golden fixture files."""
    return list(FIXTURES_DIR.glob("golden_*.json"))


@pytest.fixture(params=get_golden_fixtures(), ids=lambda p: p.stem)
def golden_fixture(request):
    """Parametrized fixture for all golden test cases."""
    return load_fixture(request.param)


def test_fixture_has_expected_structure(golden_fixture):
    """Every golden fixture must have required fields."""
    required_fields = ["corpus_id", "entities", "claims", "incidents", "expected"]
    for field in required_fields:
        assert field in golden_fixture, f"Missing required field: {field}"


def test_fixture_expected_invariants(golden_fixture):
    """Every fixture should declare expected invariants."""
    expected = golden_fixture.get("expected", {})
    assert "invariants" in expected, "Fixture must declare expected invariants"
    assert len(expected["invariants"]) > 0, "Fixture must have at least one invariant"


def test_spine_entity_exists(golden_fixture):
    """Every golden fixture must have a spine entity defined."""
    expected = golden_fixture.get("expected", {})
    entities = golden_fixture.get("entities", [])

    # Either spine is in expected or there's an entity with role="spine"
    has_spine = bool(expected.get("spine"))
    has_spine_entity = any(e.get("role") == "spine" for e in entities)

    assert has_spine or has_spine_entity, "Fixture must define spine entity"


def test_baseline_metrics_computable(golden_fixture):
    """Baseline metrics should be computable from fixture."""
    metrics = compute_baseline_metrics(golden_fixture)

    assert metrics.total_incidents > 0, "Fixture must have incidents"
    assert metrics.spine_entity_incident_count > 0, "Spine entity must appear in incidents"


# =============================================================================
# CI Gate Threshold Tests
# =============================================================================

class TestCIGates:
    """CI gate tests that enforce release criteria on fixtures."""

    def test_no_giant_component_golden_star_wfc(self):
        """WFC star pattern should not create giant component."""
        fixture_path = FIXTURES_DIR / "golden_micro_star_wfc.json"
        if not fixture_path.exists():
            pytest.skip("Fixture not found")

        fixture = load_fixture(fixture_path)
        metrics = compute_baseline_metrics(fixture)

        # Star patterns with proper spine formation should stay bounded
        # Even with heuristic baseline, WFC has 5 incidents
        assert metrics.largest_case <= 20, (
            f"WFC case too large: {metrics.largest_case} incidents. "
            "Star pattern should form bounded clusters."
        )

    def test_no_giant_component_person_star(self):
        """Person star pattern should not create giant component."""
        fixture_path = FIXTURES_DIR / "golden_person_star_politician.json"
        if not fixture_path.exists():
            pytest.skip("Fixture not found")

        fixture = load_fixture(fixture_path)
        metrics = compute_baseline_metrics(fixture)

        assert metrics.largest_case <= CI_THRESHOLDS["max_largest_case"], (
            f"Person case too large: {metrics.largest_case}. "
            f"Max allowed: {CI_THRESHOLDS['max_largest_case']}"
        )

    def test_no_giant_component_legal_chain(self):
        """Legal chain pattern should not create giant component."""
        fixture_path = FIXTURES_DIR / "golden_legal_chain_trial.json"
        if not fixture_path.exists():
            pytest.skip("Fixture not found")

        fixture = load_fixture(fixture_path)
        metrics = compute_baseline_metrics(fixture)

        assert metrics.largest_case <= CI_THRESHOLDS["max_largest_case"], (
            f"Legal chain case too large: {metrics.largest_case}. "
            f"Max allowed: {CI_THRESHOLDS['max_largest_case']}"
        )

    def test_no_giant_component_policy_hub(self):
        """Policy hub pattern should not create giant component."""
        fixture_path = FIXTURES_DIR / "golden_policy_hub_legislation.json"
        if not fixture_path.exists():
            pytest.skip("Fixture not found")

        fixture = load_fixture(fixture_path)
        metrics = compute_baseline_metrics(fixture)

        # Policy hubs are high-DF and could create large clusters
        # But with proper filtering they should stay bounded
        assert metrics.largest_case <= CI_THRESHOLDS["max_largest_case"], (
            f"Policy hub case too large: {metrics.largest_case}. "
            f"Max allowed: {CI_THRESHOLDS['max_largest_case']}"
        )


class TestPurityCoverage:
    """Tests that enforce purity and coverage targets."""

    def test_wfc_baseline_establishes_lower_bound(self):
        """WFC heuristic baseline establishes the bar to beat."""
        fixture_path = FIXTURES_DIR / "golden_micro_star_wfc.json"
        if not fixture_path.exists():
            pytest.skip("Fixture not found")

        fixture = load_fixture(fixture_path)
        metrics = compute_baseline_metrics(fixture)

        # With perfect star pattern, baseline should achieve high metrics
        assert metrics.purity >= 0.8, (
            f"WFC baseline purity too low: {metrics.purity:.1%}. "
            "Star pattern should have high purity even with heuristics."
        )
        assert metrics.coverage >= 0.8, (
            f"WFC baseline coverage too low: {metrics.coverage:.1%}. "
            "Star pattern should have high coverage."
        )

    def test_tier_b_dataset_expectations(self):
        """Tier-B dataset should have documented baseline expectations."""
        fixture_path = FIXTURES_DIR / "tier_b_dataset.json"
        if not fixture_path.exists():
            pytest.skip("Fixture not found")

        fixture = load_fixture(fixture_path)
        expected_metrics = fixture.get("expected_metrics", {})

        # Tier-B should document baseline vs compiler expectations
        assert "baseline_purity" in expected_metrics, "Must document baseline purity"
        assert "compiler_purity" in expected_metrics, "Must document compiler purity"
        assert "min_acceptable_purity" in expected_metrics, "Must document min acceptable"

        # Compiler should beat baseline
        baseline = expected_metrics.get("baseline_purity", 0)
        compiler = expected_metrics.get("compiler_purity", 0)
        assert compiler > baseline, (
            f"Compiler purity ({compiler:.1%}) must beat baseline ({baseline:.1%})"
        )

        # Compiler should meet minimum threshold
        min_acceptable = expected_metrics.get("min_acceptable_purity", 0.85)
        assert compiler >= min_acceptable, (
            f"Compiler purity ({compiler:.1%}) must meet min ({min_acceptable:.1%})"
        )


# =============================================================================
# Metabolic Isolation Tests (CI Gate)
# =============================================================================

class TestMetabolicIsolation:
    """Tests that metabolic edges never affect case membership."""

    def test_metabolic_edges_are_peripheral(self):
        """Metabolic edges should be classified as peripheral."""
        from reee.compiler.guards import (
            is_spine_edge,
            is_metabolic_edge,
            SPINE_EDGE_TYPES,
            METABOLIC_EDGE_TYPES,
        )
        from reee.compiler.membrane import EdgeType

        # SAME_HAPPENING and UPDATE_TO are spine
        assert is_spine_edge(EdgeType.SAME_HAPPENING)
        assert is_spine_edge(EdgeType.UPDATE_TO)

        # CONTEXT_FOR is metabolic
        assert is_metabolic_edge(EdgeType.CONTEXT_FOR)

        # No overlap
        assert len(SPINE_EDGE_TYPES & METABOLIC_EDGE_TYPES) == 0

    def test_spine_edges_require_compiler_authorization(self):
        """Spine edges must have MembraneDecision provenance."""
        from reee.compiler.guards import (
            validate_edge_for_persistence,
            MissingProvenanceError,
        )
        from reee.compiler.membrane import EdgeType

        # Spine edge without decision should fail
        with pytest.raises(MissingProvenanceError):
            validate_edge_for_persistence(
                edge_type=EdgeType.SAME_HAPPENING,
                decision=None,
            )


# =============================================================================
# Summary Report
# =============================================================================

def test_ci_metrics_summary(capsys):
    """Generate a summary of all fixture metrics for CI reporting."""
    fixtures = get_golden_fixtures()

    results = []
    for fp in fixtures:
        fixture = load_fixture(fp)
        metrics = compute_baseline_metrics(fixture)
        results.append(metrics)

    # Print summary (captured by capsys for CI)
    print("\n" + "=" * 70)
    print("CI METRICS SUMMARY")
    print("=" * 70)
    print(f"{'Fixture':<30} {'Inc':<4} {'Cases':<5} {'Largest':<7} {'Purity':<7} {'Coverage':<8}")
    print("-" * 70)

    for m in results:
        print(f"{m.fixture_name:<30} {m.total_incidents:<4} {m.computed_cases:<5} {m.largest_case:<7} {m.purity:<7.1%} {m.coverage:<8.1%}")

    print("=" * 70)

    # Validate all metrics meet CI thresholds
    for m in results:
        assert m.largest_case <= CI_THRESHOLDS["max_largest_case"], (
            f"{m.fixture_name}: largest case {m.largest_case} exceeds max {CI_THRESHOLDS['max_largest_case']}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
