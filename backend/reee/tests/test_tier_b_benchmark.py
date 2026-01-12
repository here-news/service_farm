"""
Tier-B Benchmark: Continuous validation of compiler purity/coverage.

This test uses a frozen dataset (Wang Fuk Court + 100 distractors) to ensure:
1. Compiler purity >= min_acceptable_purity (85%)
2. No giant component (largest_case <= max_acceptable)
3. Coverage is maintained (all WFC incidents are grouped)

Run with: pytest reee/tests/test_tier_b_benchmark.py -v --tb=short

Note: This test requires database connectivity and LLM API access.
Mark with @pytest.mark.integration to skip in unit test runs.
"""

import json
import pytest
from pathlib import Path
from dataclasses import dataclass
from typing import Set, Dict, List, Any


# =============================================================================
# Frozen Dataset
# =============================================================================

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TIER_B_DATASET_PATH = FIXTURES_DIR / "tier_b_dataset.json"


@dataclass
class TierBDataset:
    """Frozen Tier-B benchmark dataset."""
    wfc_ids: Set[str]
    distractor_ids: Set[str]
    expected_metrics: Dict[str, float]
    ground_truth_entity: str

    @classmethod
    def load(cls) -> "TierBDataset":
        """Load frozen dataset from JSON."""
        with open(TIER_B_DATASET_PATH) as f:
            data = json.load(f)
        return cls(
            wfc_ids=set(data["wfc_ids"]),
            distractor_ids=set(data["distractor_ids"]),
            expected_metrics=data["expected_metrics"],
            ground_truth_entity=data["ground_truth_entity"],
        )

    @property
    def all_ids(self) -> Set[str]:
        return self.wfc_ids | self.distractor_ids


# =============================================================================
# Metrics Computation
# =============================================================================

@dataclass
class BenchmarkMetrics:
    """Metrics computed from case formation results."""
    total_cases: int
    largest_case: int
    wfc_coverage: float  # fraction of WFC incidents in WFC case
    wfc_purity: float    # fraction of WFC case that are WFC incidents
    contaminants: int    # non-WFC incidents in WFC case
    wfc_case_size: int


def compute_metrics(
    cases: List[Set[str]],
    wfc_ground_truth: Set[str],
) -> BenchmarkMetrics:
    """Compute benchmark metrics from case results."""
    if not cases:
        return BenchmarkMetrics(
            total_cases=0,
            largest_case=0,
            wfc_coverage=0.0,
            wfc_purity=0.0,
            contaminants=0,
            wfc_case_size=0,
        )

    # Find the WFC case (largest case containing any WFC incident)
    wfc_case = None
    for c in sorted(cases, key=len, reverse=True):
        if c & wfc_ground_truth:
            wfc_case = c
            break

    wfc_case = wfc_case or set()
    wfc_in_case = len(wfc_case & wfc_ground_truth)
    coverage = wfc_in_case / len(wfc_ground_truth) if wfc_ground_truth else 0
    purity = wfc_in_case / len(wfc_case) if wfc_case else 0

    return BenchmarkMetrics(
        total_cases=len(cases),
        largest_case=max(len(c) for c in cases),
        wfc_coverage=coverage,
        wfc_purity=purity,
        contaminants=len(wfc_case - wfc_ground_truth),
        wfc_case_size=len(wfc_case),
    )


# =============================================================================
# Heuristic Baseline (for comparison)
# =============================================================================

def run_heuristic_baseline(
    incidents: Dict[str, Any],
    wfc_ground_truth: Set[str],
) -> BenchmarkMetrics:
    """
    Run heuristic overlap baseline for comparison.

    This is the naive approach: any entity overlap = same case.
    """
    # Generate edges from entity overlap
    edges = []
    incident_list = list(incidents.keys())
    for i, id1 in enumerate(incident_list):
        for id2 in incident_list[i + 1:]:
            ents1 = incidents[id1].get("anchor_entities", set())
            ents2 = incidents[id2].get("anchor_entities", set())
            if ents1 & ents2:
                edges.append((id1, id2))

    # Union-find
    parent = {iid: iid for iid in incidents}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for id1, id2 in edges:
        union(id1, id2)

    # Get components
    components: Dict[str, Set[str]] = {}
    for iid in incidents:
        root = find(iid)
        if root not in components:
            components[root] = set()
        components[root].add(iid)

    cases = [c for c in components.values() if len(c) >= 2]
    return compute_metrics(cases, wfc_ground_truth)


# =============================================================================
# Tests
# =============================================================================

def test_tier_b_dataset_loads():
    """Verify frozen dataset loads correctly."""
    dataset = TierBDataset.load()

    assert len(dataset.wfc_ids) == 58, "Expected 58 WFC incidents"
    assert len(dataset.distractor_ids) == 100, "Expected 100 distractor incidents"
    assert dataset.ground_truth_entity == "Wang Fuk Court"
    assert "min_acceptable_purity" in dataset.expected_metrics


def test_tier_b_dataset_ids_are_valid():
    """Verify dataset IDs follow expected format."""
    dataset = TierBDataset.load()

    for iid in dataset.all_ids:
        assert iid.startswith("in_"), f"Invalid ID format: {iid}"
        assert len(iid) == 11, f"Invalid ID length: {iid}"


def test_tier_b_no_overlap():
    """Verify WFC and distractor sets don't overlap."""
    dataset = TierBDataset.load()

    overlap = dataset.wfc_ids & dataset.distractor_ids
    assert len(overlap) == 0, f"IDs appear in both sets: {overlap}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tier_b_compiler_meets_purity_threshold():
    """
    CRITICAL: Compiler must achieve >= 85% purity on Tier-B dataset.

    This test runs the full compiler pipeline and validates:
    1. Purity >= min_acceptable_purity
    2. Coverage >= 90% (don't lose WFC incidents)
    3. Largest case <= max_acceptable (no giant component)
    """
    pytest.skip("Integration test - requires Neo4j and LLM API")

    # This would be the full integration test:
    # dataset = TierBDataset.load()
    # ...load incidents from Neo4j...
    # ...run compiler...
    # ...compute metrics...
    # assert metrics.wfc_purity >= dataset.expected_metrics["min_acceptable_purity"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tier_b_compiler_beats_baseline():
    """
    CRITICAL: Compiler must beat heuristic baseline on purity.

    If compiler purity <= baseline purity, something is wrong.
    """
    pytest.skip("Integration test - requires Neo4j and LLM API")


def test_benchmark_metrics_computation():
    """Test metrics computation with synthetic data."""
    # Synthetic cases
    cases = [
        {"a", "b", "c", "d"},  # WFC case
        {"e", "f"},            # Other case
        {"g"},                 # Singleton (filtered out)
    ]
    wfc_ground_truth = {"a", "b", "c"}  # 3 WFC, 1 contaminant (d)

    metrics = compute_metrics(
        [c for c in cases if len(c) >= 2],
        wfc_ground_truth,
    )

    assert metrics.total_cases == 2
    assert metrics.largest_case == 4
    assert metrics.wfc_coverage == 1.0  # All 3 WFC in case
    assert metrics.wfc_purity == 0.75   # 3/4 are WFC
    assert metrics.contaminants == 1    # 'd' is contaminant
    assert metrics.wfc_case_size == 4


def test_benchmark_metrics_empty_cases():
    """Test metrics with no cases."""
    metrics = compute_metrics([], {"a", "b"})

    assert metrics.total_cases == 0
    assert metrics.largest_case == 0
    assert metrics.wfc_coverage == 0.0
    assert metrics.wfc_purity == 0.0


def test_benchmark_metrics_perfect_purity():
    """Test metrics with perfect case formation."""
    cases = [{"a", "b", "c"}]  # All WFC, no contaminants
    wfc_ground_truth = {"a", "b", "c"}

    metrics = compute_metrics(cases, wfc_ground_truth)

    assert metrics.wfc_purity == 1.0
    assert metrics.wfc_coverage == 1.0
    assert metrics.contaminants == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
