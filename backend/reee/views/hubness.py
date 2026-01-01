"""
Local Hubness Detection
=======================

Dispersion-based hubness for CaseView.

Key insight: Hubs vs Backbones
- High frequency + low dispersion = BACKBONE (binds related incidents)
- High frequency + high dispersion = HUB (suppressed, bridges unrelated incidents)

Dispersion Measures:
1. Co-anchor entropy: For incidents containing anchor (a), what's the distribution
   of other co-occurring anchors? High entropy = hub (appears with everything)
2. Centroid variance: How spread are the incident embeddings? High variance = hub

This differs from global IDF:
- IDF penalizes any frequently occurring entity
- Dispersion-based hubness penalizes entities that appear across unrelated contexts
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import math

from ..types import Event


@dataclass
class AnchorHubness:
    """Hubness analysis for a single anchor entity."""
    anchor: str
    frequency: int  # Number of incidents containing this anchor

    # Dispersion measures
    # NOTE: This is NOT Shannon entropy. It's a [0,1] dispersion score:
    #   dispersion = 1 - cohesion
    #   cohesion = fraction of co-anchor pairs that co-occur in some incident
    # Low dispersion (< 0.7) = backbone, high dispersion (>= 0.7) = hub
    co_anchor_dispersion: float
    centroid_variance: Optional[float]  # Variance of incident embeddings (if available)

    # Classification
    is_backbone: bool  # High freq + low dispersion = binds
    is_hub: bool  # High freq + high dispersion = suppressed

    # Evidence for audit
    incident_ids: Set[str] = field(default_factory=set)
    co_anchor_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class HubnessResult:
    """Result of hubness analysis over a set of incidents."""
    anchors: Dict[str, AnchorHubness]  # anchor -> hubness info

    # Summary statistics
    total_anchors: int = 0
    backbone_count: int = 0
    hub_count: int = 0
    neutral_count: int = 0

    # For trace
    params_used: Dict[str, Any] = field(default_factory=dict)

    @property
    def backbones(self) -> Set[str]:
        """Anchors classified as backbones (bind incidents)."""
        return {a for a, h in self.anchors.items() if h.is_backbone}

    @property
    def hubs(self) -> Set[str]:
        """Anchors classified as hubs (suppressed)."""
        return {a for a, h in self.anchors.items() if h.is_hub}


def compute_co_anchor_dispersion(
    anchor: str,
    incidents: Dict[str, Event],
    anchor_to_incidents: Dict[str, Set[str]],
) -> Tuple[float, Dict[str, int]]:
    """
    Compute dispersion of co-occurring anchors for a given anchor.

    The key insight: we want to detect whether an anchor appears in
    *different contexts* vs *similar contexts*.

    A BACKBONE anchor (like "Do Kwon") will have:
    - High co-occurrence with a stable core set (Terraform Labs, Luna)
    - These core co-anchors also frequently co-occur with each other

    A HUB anchor (like "Hong Kong") will have:
    - Co-occurrences spread across unrelated anchors
    - The co-anchors do NOT co-occur with each other

    We measure this via:
    1. Cluster cohesion: How much do the co-anchors co-occur with each other?
       High cohesion = backbone, low cohesion = hub

    Returns:
        (dispersion_score, co_anchor_counts)
        - dispersion_score: 0 = backbone (low dispersion), 1 = hub (high dispersion)
    """
    incident_ids = anchor_to_incidents.get(anchor, set())
    if not incident_ids:
        return 0.0, {}

    # Count co-occurring anchors across incidents containing this anchor
    co_anchor_counts: Dict[str, int] = defaultdict(int)

    for inc_id in incident_ids:
        inc = incidents.get(inc_id)
        if not inc:
            continue
        for other_anchor in inc.anchor_entities:
            if other_anchor != anchor:
                co_anchor_counts[other_anchor] += 1

    if not co_anchor_counts:
        return 0.0, {}

    # Compute cluster cohesion: how often do co-anchors appear together?
    # For each pair of co-anchors, check if they ever appear in same incident
    co_anchors = list(co_anchor_counts.keys())
    if len(co_anchors) < 2:
        # Only one co-anchor: can't compute cohesion, assume backbone
        return 0.0, dict(co_anchor_counts)

    # Count co-occurrences between the co-anchors themselves
    pair_cooccurrence = 0
    pair_total = 0

    for i, a1 in enumerate(co_anchors):
        for a2 in co_anchors[i+1:]:
            pair_total += 1
            # Check if a1 and a2 ever appear in same incident
            a1_incidents = anchor_to_incidents.get(a1, set())
            a2_incidents = anchor_to_incidents.get(a2, set())
            if a1_incidents & a2_incidents:
                pair_cooccurrence += 1

    # Cohesion = fraction of co-anchor pairs that co-occur
    cohesion = pair_cooccurrence / pair_total if pair_total > 0 else 0.0

    # Dispersion = 1 - cohesion
    # High cohesion = low dispersion = backbone
    # Low cohesion = high dispersion = hub
    dispersion = 1.0 - cohesion

    return dispersion, dict(co_anchor_counts)


def compute_centroid_variance(
    anchor: str,
    incidents: Dict[str, Event],
    anchor_to_incidents: Dict[str, Set[str]],
    incident_embeddings: Optional[Dict[str, List[float]]] = None,
) -> Optional[float]:
    """
    Compute variance of incident embeddings for incidents containing this anchor.

    High variance = incidents are semantically diverse = hub-like.
    Low variance = incidents are semantically similar = backbone-like.

    Returns None if embeddings not available.
    """
    if not incident_embeddings:
        return None

    incident_ids = anchor_to_incidents.get(anchor, set())
    embeddings = []

    for inc_id in incident_ids:
        if inc_id in incident_embeddings:
            embeddings.append(incident_embeddings[inc_id])

    if len(embeddings) < 2:
        return None

    # Compute centroid
    dim = len(embeddings[0])
    centroid = [sum(e[d] for e in embeddings) / len(embeddings) for d in range(dim)]

    # Compute average squared distance from centroid
    variance = 0.0
    for emb in embeddings:
        dist_sq = sum((emb[d] - centroid[d]) ** 2 for d in range(dim))
        variance += dist_sq
    variance /= len(embeddings)

    return variance


def analyze_hubness(
    incidents: Dict[str, Event],
    frequency_threshold: int = 3,
    dispersion_threshold: float = 0.7,
    variance_threshold: float = 0.5,
    incident_embeddings: Optional[Dict[str, List[float]]] = None,
) -> HubnessResult:
    """
    Analyze hubness of anchors across incidents.

    Classification:
    - Backbone: freq >= threshold AND dispersion < threshold
    - Hub: freq >= threshold AND dispersion >= threshold
    - Neutral: freq < threshold (not enough signal)

    Args:
        incidents: Dict of incident_id -> Event
        frequency_threshold: Min incidents for freq to be "high"
        dispersion_threshold: Above this = high dispersion (hub)
            Note: This is a [0,1] score, not Shannon entropy.
            dispersion = 1 - cohesion, where cohesion is the fraction
            of co-anchor pairs that co-occur in some incident.
        variance_threshold: Above this = high dispersion (if embeddings available)
        incident_embeddings: Optional embeddings for centroid variance

    Returns:
        HubnessResult with anchor classifications
    """
    # Build anchor -> incidents mapping
    anchor_to_incidents: Dict[str, Set[str]] = defaultdict(set)
    for inc_id, inc in incidents.items():
        for anchor in inc.anchor_entities:
            anchor_to_incidents[anchor].add(inc_id)

    # Analyze each anchor
    anchors: Dict[str, AnchorHubness] = {}
    backbone_count = 0
    hub_count = 0
    neutral_count = 0

    for anchor, incident_ids in anchor_to_incidents.items():
        frequency = len(incident_ids)

        # Compute dispersion measures
        dispersion, co_anchor_dist = compute_co_anchor_dispersion(
            anchor, incidents, anchor_to_incidents
        )

        variance = compute_centroid_variance(
            anchor, incidents, anchor_to_incidents, incident_embeddings
        )

        # Classification
        if frequency < frequency_threshold:
            # Low frequency = neutral (not enough signal)
            is_backbone = False
            is_hub = False
            neutral_count += 1
        else:
            # High frequency: check dispersion
            # Use co-anchor dispersion as primary, embedding variance as secondary
            high_dispersion = dispersion >= dispersion_threshold
            if variance is not None:
                high_dispersion = high_dispersion or variance >= variance_threshold

            is_backbone = not high_dispersion
            is_hub = high_dispersion

            if is_backbone:
                backbone_count += 1
            else:
                hub_count += 1

        anchors[anchor] = AnchorHubness(
            anchor=anchor,
            frequency=frequency,
            co_anchor_dispersion=dispersion,
            centroid_variance=variance,
            is_backbone=is_backbone,
            is_hub=is_hub,
            incident_ids=incident_ids,
            co_anchor_distribution=co_anchor_dist,
        )

    return HubnessResult(
        anchors=anchors,
        total_anchors=len(anchors),
        backbone_count=backbone_count,
        hub_count=hub_count,
        neutral_count=neutral_count,
        params_used={
            'frequency_threshold': frequency_threshold,
            'dispersion_threshold': dispersion_threshold,
            'variance_threshold': variance_threshold,
        },
    )


def print_hubness_report(result: HubnessResult, top_k: int = 10):
    """Print a human-readable hubness analysis report."""
    print("=" * 70)
    print("LOCAL HUBNESS ANALYSIS")
    print("=" * 70)
    print(f"Total anchors: {result.total_anchors}")
    print(f"Backbones: {result.backbone_count}")
    print(f"Hubs: {result.hub_count}")
    print(f"Neutral: {result.neutral_count}")
    print()

    # Sort by frequency
    sorted_anchors = sorted(
        result.anchors.values(),
        key=lambda h: h.frequency,
        reverse=True
    )

    print(f"Top {top_k} by frequency:")
    print("-" * 70)
    for h in sorted_anchors[:top_k]:
        classification = "BACKBONE" if h.is_backbone else ("HUB" if h.is_hub else "neutral")
        variance_str = f", var={h.centroid_variance:.3f}" if h.centroid_variance else ""
        dispersion_str = f"disp={h.co_anchor_dispersion:.3f}"
        print(
            f"  {h.anchor}: freq={h.frequency}, {dispersion_str}"
            f"{variance_str} → {classification}"
        )
        if h.co_anchor_distribution:
            top_coanchors = sorted(
                h.co_anchor_distribution.items(),
                key=lambda x: -x[1]
            )[:3]
            coanchor_str = ", ".join(f"{a}({c})" for a, c in top_coanchors)
            print(f"    Co-anchors: {coanchor_str}")

    print()
    print("Backbones (bind incidents):")
    for anchor in sorted(result.backbones)[:5]:
        h = result.anchors[anchor]
        print(f"  {anchor}: freq={h.frequency}, disp={h.co_anchor_dispersion:.3f}")

    print()
    print("Hubs (suppressed, bridge unrelated):")
    for anchor in sorted(result.hubs)[:5]:
        h = result.anchors[anchor]
        print(f"  {anchor}: freq={h.frequency}, disp={h.co_anchor_dispersion:.3f}")
