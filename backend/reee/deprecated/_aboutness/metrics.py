"""
Clustering Metrics for L3 Event Emergence
==========================================

Proper clustering evaluation metrics for comparing emerged events to legacy events.

B³ (B-cubed) Metrics:
- Precision: For each claim, what fraction of its cluster-mates belong to the same GT event?
- Recall: For each claim, what fraction of its GT event-mates are in the same cluster?
- F1: Harmonic mean

Purity/Completeness (NMI-style):
- Purity: For each emerged event, fraction belonging to dominant legacy event
- Completeness: For each legacy event, fraction captured by dominant emerged event

These are the correct metrics for clustering evaluation, unlike exact-match
precision/recall which will always look bad when #emerged != #legacy.
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

from reee.types import Surface, Event


@dataclass
class B3Metrics:
    """B³ clustering metrics."""
    precision: float
    recall: float
    f1: float

    # Per-event breakdown
    event_precision: Dict[str, float] = field(default_factory=dict)
    event_recall: Dict[str, float] = field(default_factory=dict)


@dataclass
class PurityMetrics:
    """Purity and completeness metrics."""
    purity: float  # Average purity of emerged events
    completeness: float  # Average completeness of legacy events

    # Per-event breakdown
    event_purity: Dict[str, Tuple[float, str]] = field(default_factory=dict)  # (purity, dominant_legacy)
    legacy_completeness: Dict[str, Tuple[float, str]] = field(default_factory=dict)  # (completeness, dominant_emerged)


@dataclass
class ClusteringEvaluation:
    """Complete clustering evaluation."""
    b3: B3Metrics
    purity: PurityMetrics

    # Summary stats
    num_emerged: int
    num_legacy: int
    num_claims: int

    # Distribution
    emerged_sizes: List[int]
    legacy_sizes: List[int]


def compute_b3_metrics(
    emerged_events: Dict[str, Event],
    surfaces: Dict[str, Surface],
    claim_to_legacy: Dict[str, str],
) -> B3Metrics:
    """
    Compute B³ precision, recall, F1 for emerged vs legacy events.

    B³ treats each claim as a data point and measures:
    - Precision: avg fraction of cluster-mates that share GT label
    - Recall: avg fraction of GT-mates that share cluster

    Args:
        emerged_events: Emerged events from clustering
        surfaces: Surface dict for claim lookup
        claim_to_legacy: claim_id -> legacy_event_name mapping (ground truth)

    Returns:
        B3Metrics with precision, recall, F1
    """
    # Build claim -> emerged_event mapping
    claim_to_emerged = {}
    for event_id, event in emerged_events.items():
        for surface_id in event.surface_ids:
            if surface_id in surfaces:
                for claim_id in surfaces[surface_id].claim_ids:
                    claim_to_emerged[claim_id] = event_id

    # Only evaluate claims that have both emerged and legacy assignments
    eval_claims = set(claim_to_emerged.keys()) & set(claim_to_legacy.keys())

    if not eval_claims:
        return B3Metrics(precision=0.0, recall=0.0, f1=0.0)

    # Build cluster membership sets
    emerged_clusters = defaultdict(set)  # emerged_event_id -> set of claim_ids
    legacy_clusters = defaultdict(set)   # legacy_event_name -> set of claim_ids

    for claim_id in eval_claims:
        emerged_clusters[claim_to_emerged[claim_id]].add(claim_id)
        legacy_clusters[claim_to_legacy[claim_id]].add(claim_id)

    # Compute B³ per claim, then average
    precision_sum = 0.0
    recall_sum = 0.0
    n = len(eval_claims)

    # Track per-legacy-event metrics for breakdown
    legacy_precision_sum = defaultdict(float)
    legacy_recall_sum = defaultdict(float)
    legacy_count = defaultdict(int)

    for claim_id in eval_claims:
        emerged_id = claim_to_emerged[claim_id]
        legacy_name = claim_to_legacy[claim_id]

        # Claims in same emerged cluster
        emerged_mates = emerged_clusters[emerged_id]
        # Claims in same legacy event
        legacy_mates = legacy_clusters[legacy_name]

        # Intersection: claims that are both cluster-mates AND GT-mates
        correct = emerged_mates & legacy_mates

        # B³ Precision: |correct| / |emerged_mates|
        precision = len(correct) / len(emerged_mates) if emerged_mates else 0.0
        # B³ Recall: |correct| / |legacy_mates|
        recall = len(correct) / len(legacy_mates) if legacy_mates else 0.0

        precision_sum += precision
        recall_sum += recall

        # Track per-event
        legacy_precision_sum[legacy_name] += precision
        legacy_recall_sum[legacy_name] += recall
        legacy_count[legacy_name] += 1

    b3_precision = precision_sum / n
    b3_recall = recall_sum / n
    b3_f1 = 2 * b3_precision * b3_recall / (b3_precision + b3_recall) if (b3_precision + b3_recall) > 0 else 0.0

    # Per-event breakdown
    event_precision = {
        name: legacy_precision_sum[name] / legacy_count[name]
        for name in legacy_count
    }
    event_recall = {
        name: legacy_recall_sum[name] / legacy_count[name]
        for name in legacy_count
    }

    return B3Metrics(
        precision=b3_precision,
        recall=b3_recall,
        f1=b3_f1,
        event_precision=event_precision,
        event_recall=event_recall,
    )


def compute_purity_metrics(
    emerged_events: Dict[str, Event],
    surfaces: Dict[str, Surface],
    claim_to_legacy: Dict[str, str],
) -> PurityMetrics:
    """
    Compute purity and completeness metrics.

    Purity: For each emerged event, what fraction belongs to dominant legacy event?
    Completeness: For each legacy event, what fraction is captured by dominant emerged event?

    Args:
        emerged_events: Emerged events from clustering
        surfaces: Surface dict for claim lookup
        claim_to_legacy: claim_id -> legacy_event_name mapping (ground truth)

    Returns:
        PurityMetrics with purity, completeness
    """
    # Build claim -> emerged_event mapping
    emerged_to_claims = defaultdict(set)
    for event_id, event in emerged_events.items():
        for surface_id in event.surface_ids:
            if surface_id in surfaces:
                for claim_id in surfaces[surface_id].claim_ids:
                    emerged_to_claims[event_id].add(claim_id)

    # Build legacy -> claims mapping
    legacy_to_claims = defaultdict(set)
    for claim_id, legacy_name in claim_to_legacy.items():
        legacy_to_claims[legacy_name].add(claim_id)

    # ===== PURITY =====
    # For each emerged event, find dominant legacy event
    event_purity = {}
    total_purity = 0.0
    total_emerged_claims = 0

    for event_id, event_claims in emerged_to_claims.items():
        # Count claims per legacy event
        legacy_dist = defaultdict(int)
        labeled_claims = 0
        for claim_id in event_claims:
            if claim_id in claim_to_legacy:
                legacy_dist[claim_to_legacy[claim_id]] += 1
                labeled_claims += 1

        if labeled_claims == 0:
            continue

        # Purity = max fraction
        dominant_legacy = max(legacy_dist.items(), key=lambda x: x[1])
        purity = dominant_legacy[1] / labeled_claims

        event_purity[event_id] = (purity, dominant_legacy[0])
        total_purity += purity * labeled_claims
        total_emerged_claims += labeled_claims

    avg_purity = total_purity / total_emerged_claims if total_emerged_claims > 0 else 0.0

    # ===== COMPLETENESS =====
    # For each legacy event, find dominant emerged event
    legacy_completeness = {}
    total_completeness = 0.0
    total_legacy_claims = 0

    for legacy_name, legacy_claims in legacy_to_claims.items():
        # Count claims per emerged event
        emerged_dist = defaultdict(int)
        for claim_id in legacy_claims:
            for event_id, event_claims in emerged_to_claims.items():
                if claim_id in event_claims:
                    emerged_dist[event_id] += 1
                    break  # claim can only be in one emerged event

        if not emerged_dist:
            continue

        # Completeness = max fraction captured
        dominant_emerged = max(emerged_dist.items(), key=lambda x: x[1])
        completeness = dominant_emerged[1] / len(legacy_claims)

        legacy_completeness[legacy_name] = (completeness, dominant_emerged[0])
        total_completeness += completeness * len(legacy_claims)
        total_legacy_claims += len(legacy_claims)

    avg_completeness = total_completeness / total_legacy_claims if total_legacy_claims > 0 else 0.0

    return PurityMetrics(
        purity=avg_purity,
        completeness=avg_completeness,
        event_purity=event_purity,
        legacy_completeness=legacy_completeness,
    )


def evaluate_clustering(
    emerged_events: Dict[str, Event],
    surfaces: Dict[str, Surface],
    claim_to_legacy: Dict[str, str],
) -> ClusteringEvaluation:
    """
    Complete clustering evaluation with B³ and purity/completeness.

    Args:
        emerged_events: Emerged events from clustering
        surfaces: Surface dict for claim lookup
        claim_to_legacy: claim_id -> legacy_event_name mapping (ground truth)

    Returns:
        ClusteringEvaluation with all metrics
    """
    b3 = compute_b3_metrics(emerged_events, surfaces, claim_to_legacy)
    purity = compute_purity_metrics(emerged_events, surfaces, claim_to_legacy)

    # Count claims in emerged events
    emerged_claims = set()
    for event in emerged_events.values():
        for sid in event.surface_ids:
            if sid in surfaces:
                emerged_claims.update(surfaces[sid].claim_ids)

    # Sizes
    emerged_sizes = sorted([e.total_claims for e in emerged_events.values()], reverse=True)

    legacy_sizes_dict = defaultdict(int)
    for legacy_name in claim_to_legacy.values():
        legacy_sizes_dict[legacy_name] += 1
    legacy_sizes = sorted(legacy_sizes_dict.values(), reverse=True)

    return ClusteringEvaluation(
        b3=b3,
        purity=purity,
        num_emerged=len(emerged_events),
        num_legacy=len(set(claim_to_legacy.values())),
        num_claims=len(emerged_claims & set(claim_to_legacy.keys())),
        emerged_sizes=emerged_sizes,
        legacy_sizes=legacy_sizes,
    )


def print_evaluation_report(eval_result: ClusteringEvaluation, verbose: bool = False):
    """Print a formatted evaluation report."""
    print("=" * 70)
    print("CLUSTERING EVALUATION (B³ + Purity/Completeness)")
    print("=" * 70)
    print()

    print("SUMMARY:")
    print(f"  Emerged events: {eval_result.num_emerged}")
    print(f"  Legacy events:  {eval_result.num_legacy}")
    print(f"  Evaluated claims: {eval_result.num_claims}")
    print()

    print("B³ METRICS (claim-level):")
    print(f"  Precision: {eval_result.b3.precision:.1%}")
    print(f"  Recall:    {eval_result.b3.recall:.1%}")
    print(f"  F1:        {eval_result.b3.f1:.1%}")
    print()

    print("PURITY / COMPLETENESS (event-level):")
    print(f"  Purity:      {eval_result.purity.purity:.1%} (avg purity of emerged events)")
    print(f"  Completeness:{eval_result.purity.completeness:.1%} (avg completeness of legacy events)")
    print()

    if verbose:
        print("SIZE DISTRIBUTION:")
        print(f"  Emerged: {eval_result.emerged_sizes[:10]}...")
        print(f"  Legacy:  {eval_result.legacy_sizes[:10]}...")
        print()

        print("PER-LEGACY-EVENT B³:")
        for name, prec in sorted(eval_result.b3.event_precision.items(), key=lambda x: -x[1]):
            rec = eval_result.b3.event_recall.get(name, 0.0)
            print(f"  {name[:30]:30s}: P={prec:.0%} R={rec:.0%}")
        print()

        print("PER-EMERGED PURITY:")
        for event_id, (purity, dominant) in sorted(
            eval_result.purity.event_purity.items(),
            key=lambda x: -x[1][0]
        )[:10]:
            print(f"  {event_id}: {purity:.0%} → {dominant[:25]}")
        print()

        print("PER-LEGACY COMPLETENESS:")
        for legacy, (comp, dominant) in sorted(
            eval_result.purity.legacy_completeness.items(),
            key=lambda x: -x[1][0]
        ):
            print(f"  {legacy[:30]:30s}: {comp:.0%} in {dominant}")
