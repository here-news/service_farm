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

HIGHER-ORDER UPGRADE (2025-01):
- Added context_compatible() for pairwise checking
- Entity can bind S1↔S2 only if its companion sets in both surfaces overlap
- This prevents "John Lee" from bridging fire↔Lai when companions don't overlap
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, TYPE_CHECKING
from collections import defaultdict
from datetime import datetime, timedelta
import math

from ..types import Event

if TYPE_CHECKING:
    from ..types import Surface


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


def analyze_surface_hubness(
    surfaces: Dict[str, "Surface"],
    frequency_threshold: int = 3,
    dispersion_threshold: float = 0.7,
) -> HubnessResult:
    """
    Analyze hubness of anchors across surfaces (for IncidentEventView).

    This is the surface-level variant of analyze_hubness, used before
    incident events are formed. Surfaces have anchor_entities just like
    Events, so the dispersion computation is identical.

    Classification:
    - Backbone: freq >= threshold AND dispersion < threshold
    - Hub: freq >= threshold AND dispersion >= threshold
    - Neutral: freq < threshold (not enough signal)

    Args:
        surfaces: Dict of surface_id -> Surface
        frequency_threshold: Min surfaces for freq to be "high"
        dispersion_threshold: Above this = high dispersion (hub)

    Returns:
        HubnessResult with anchor classifications
    """
    # Build anchor -> surfaces mapping
    anchor_to_surfaces: Dict[str, Set[str]] = defaultdict(set)
    for sid, surface in surfaces.items():
        for anchor in surface.anchor_entities:
            anchor_to_surfaces[anchor].add(sid)

    # Analyze each anchor using surface-level dispersion
    anchors: Dict[str, AnchorHubness] = {}
    backbone_count = 0
    hub_count = 0
    neutral_count = 0

    for anchor, surface_ids in anchor_to_surfaces.items():
        frequency = len(surface_ids)

        # Compute co-anchor dispersion at surface level
        dispersion, co_anchor_dist = _compute_surface_co_anchor_dispersion(
            anchor, surfaces, anchor_to_surfaces
        )

        # Classification
        if frequency < frequency_threshold:
            is_backbone = False
            is_hub = False
            neutral_count += 1
        else:
            high_dispersion = dispersion >= dispersion_threshold
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
            centroid_variance=None,  # Not computed at surface level
            is_backbone=is_backbone,
            is_hub=is_hub,
            incident_ids=surface_ids,  # Reusing field for surface_ids
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
            'level': 'surface',
        },
    )


def _compute_surface_co_anchor_dispersion(
    anchor: str,
    surfaces: Dict[str, "Surface"],
    anchor_to_surfaces: Dict[str, Set[str]],
) -> Tuple[float, Dict[str, int]]:
    """
    Compute dispersion of co-occurring anchors at surface level.

    Same algorithm as compute_co_anchor_dispersion but works with Surfaces.
    """
    surface_ids = anchor_to_surfaces.get(anchor, set())
    if not surface_ids:
        return 0.0, {}

    # Count co-occurring anchors
    co_anchor_counts: Dict[str, int] = defaultdict(int)
    for sid in surface_ids:
        surface = surfaces.get(sid)
        if not surface:
            continue
        for other_anchor in surface.anchor_entities:
            if other_anchor != anchor:
                co_anchor_counts[other_anchor] += 1

    if not co_anchor_counts:
        return 0.0, {}

    # Compute cluster cohesion
    co_anchors = list(co_anchor_counts.keys())
    if len(co_anchors) < 2:
        return 0.0, dict(co_anchor_counts)

    pair_cooccurrence = 0
    pair_total = 0

    for i, a1 in enumerate(co_anchors):
        for a2 in co_anchors[i+1:]:
            pair_total += 1
            a1_surfaces = anchor_to_surfaces.get(a1, set())
            a2_surfaces = anchor_to_surfaces.get(a2, set())
            if a1_surfaces & a2_surfaces:
                pair_cooccurrence += 1

    cohesion = pair_cooccurrence / pair_total if pair_total > 0 else 0.0
    dispersion = 1.0 - cohesion

    return dispersion, dict(co_anchor_counts)


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


# =============================================================================
# TIME-MODE HUBNESS FOR L2→L3 (SURFACE → INCIDENT)
# =============================================================================

@dataclass
class TimeModeInfo:
    """Time-based mode analysis for a single anchor."""
    anchor: str
    frequency: int  # Number of surfaces containing this anchor

    # Time mode analysis
    num_clusters: int  # Number of temporal clusters (k)
    time_span_days: float  # Total span from min to max time
    largest_cluster_fraction: float  # Fraction of surfaces in largest cluster
    cluster_sizes: List[int]  # Size of each temporal cluster

    # Classification (4-way: backbone, hub_time, hub_coincident, splittable)
    # Plus INDETERMINATE when data is too sparse for reliable classification
    is_backbone: bool  # Single tight mode + low dispersion → binds incident
    is_hub: bool  # Multi-modal spread → bridges distinct time periods (HUB_TIME)
    is_splittable: bool  # Multi-modal but each cluster tight → per-cluster backbone
    is_hub_coincident: bool = False  # Single time mode but high dispersion → bridges concurrent incidents
    is_indeterminate: bool = False  # Data too sparse to classify reliably

    # Within-cluster dispersion (for HUB_COINCIDENT detection)
    within_cluster_dispersion: float = 0.0

    # Co-anchor graph metrics (for rigor tracking)
    coanchor_edge_density: float = 0.0  # edges / possible_edges in co-anchor graph
    largest_component_fraction: float = 0.0  # fraction of core co-anchors in largest component

    # Evidence
    surface_ids: Set[str] = field(default_factory=set)
    cluster_time_ranges: List[Tuple[datetime, datetime]] = field(default_factory=list)

    # Mode-scoped splittable support: which cluster each surface belongs to
    # For splittable anchors, surfaces in same cluster can bind; different clusters = hub-like
    surface_to_cluster: Dict[str, int] = field(default_factory=dict)

    def same_mode(self, surface_id_1: str, surface_id_2: str) -> bool:
        """Check if two surfaces are in the same temporal mode for this anchor.

        For mode-scoped splittables: returns True if both surfaces are in the same
        cluster, allowing the anchor to act as discriminative glue within-mode.
        Returns False if surfaces are in different clusters (hub-like across modes).

        For backbones: always returns True (single mode).
        For hubs (time or coincident): always returns False (no binding allowed).
        """
        if self.is_hub or self.is_hub_coincident:
            return False
        if self.is_backbone:
            return True
        # Splittable: check cluster membership
        c1 = self.surface_to_cluster.get(surface_id_1)
        c2 = self.surface_to_cluster.get(surface_id_2)
        if c1 is None or c2 is None:
            # One or both surfaces don't have time → conservative: no binding
            return False
        return c1 == c2


@dataclass
class TimeModeHubnessResult:
    """Result of time-mode hubness analysis."""
    anchors: Dict[str, TimeModeInfo]

    # Summary
    total_anchors: int = 0
    backbone_count: int = 0
    hub_count: int = 0
    hub_coincident_count: int = 0
    splittable_count: int = 0
    neutral_count: int = 0  # Below frequency threshold

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

    @property
    def splittable(self) -> Set[str]:
        """Anchors that are multi-modal but coherent per-cluster."""
        return {a for a, h in self.anchors.items() if h.is_splittable}

    @property
    def hub_coincident(self) -> Set[str]:
        """Anchors that are single time-mode but bridge concurrent incidents."""
        return {a for a, h in self.anchors.items() if h.is_hub_coincident}

    @property
    def indeterminate(self) -> Set[str]:
        """Anchors where HUB_COINCIDENT status couldn't be reliably determined."""
        return {a for a, h in self.anchors.items() if h.is_indeterminate}

    @property
    def all_hubs(self) -> Set[str]:
        """All hub anchors (time-based OR coincident) that should be suppressed."""
        return self.hubs | self.hub_coincident

    def can_bind(self, anchor: str, surface_id_1: str, surface_id_2: str) -> bool:
        """Check if anchor can bind two surfaces (mode-scoped).

        For incident-level clustering, this implements the anti-percolation law:
        - Backbones: always can bind
        - Hubs: never can bind
        - Splittables: can bind only if both surfaces are in the same temporal mode

        Args:
            anchor: The anchor entity to check
            surface_id_1: First surface ID
            surface_id_2: Second surface ID

        Returns:
            True if anchor can act as discriminative glue for this pair
        """
        info = self.anchors.get(anchor)
        if not info:
            # Unknown anchor - treat as neutral (can bind)
            return True
        return info.same_mode(surface_id_1, surface_id_2)

    def get_binding_anchors(
        self,
        shared_anchors: Set[str],
        surface_id_1: str,
        surface_id_2: str,
    ) -> Set[str]:
        """Filter shared anchors to those that can bind this surface pair.

        Returns the subset of shared_anchors that pass mode-scoped filtering:
        - Backbones pass
        - Hubs are filtered out
        - Splittables pass only if surfaces are in same mode
        """
        return {
            anchor for anchor in shared_anchors
            if self.can_bind(anchor, surface_id_1, surface_id_2)
        }


@dataclass
class DispersionResult:
    """Result of within-cluster dispersion computation with rigor metrics."""
    dispersion: float  # 0 = cohesive, 1 = bridging, 0.5 = indeterminate
    is_indeterminate: bool  # True if data too sparse for reliable classification
    edge_density: float  # edges / possible_edges in co-anchor graph
    largest_component_fraction: float  # fraction of core co-anchors in largest component
    num_core_coanchors: int  # number of core co-anchors analyzed
    num_components: int  # number of connected components


def _compute_within_cluster_dispersion(
    anchor: str,
    cluster_surface_ids: List[str],
    surfaces: Dict[str, "Surface"],
    min_coanchor_freq: int = 2,
    min_edge_density: float = 0.1,  # Below this = indeterminate
) -> DispersionResult:
    """Compute co-anchor dispersion within a time cluster.

    Measures whether the anchor bridges distinct topic clusters vs
    covering different facets of the same topic.

    Strategy: Build a co-anchor co-occurrence graph and check connectivity.
    - Connected graph → same topic (different facets) → low dispersion
    - Disconnected components → bridging topics → high dispersion

    Returns DispersionResult with:
    - dispersion in [0, 1]: 0 = cohesive, 1 = bridging
    - is_indeterminate: True if data too sparse for reliable classification
    - rigor metrics: edge_density, largest_component_fraction
    """
    empty_result = DispersionResult(
        dispersion=0.0, is_indeterminate=True,
        edge_density=0.0, largest_component_fraction=0.0,
        num_core_coanchors=0, num_components=0
    )

    if len(cluster_surface_ids) < 2:
        return empty_result

    # Collect co-anchors for each surface (excluding the anchor itself)
    coanchor_counts: Dict[str, int] = defaultdict(int)
    surface_coanchors: List[Set[str]] = []

    for sid in cluster_surface_ids:
        s = surfaces.get(sid)
        if s:
            coanchors = s.anchor_entities - {anchor}
            if coanchors:
                surface_coanchors.append(coanchors)
                for ca in coanchors:
                    coanchor_counts[ca] += 1

    # Need at least 2 surfaces with co-anchors
    if len(surface_coanchors) < 2:
        return empty_result

    # Focus on "core" co-anchors that appear multiple times
    # (single-occurrence co-anchors are noise)
    core_coanchors = {ca for ca, cnt in coanchor_counts.items() if cnt >= min_coanchor_freq}

    # If fewer than 2 core co-anchors, can't determine connectivity
    if len(core_coanchors) < 2:
        return empty_result

    # Build co-occurrence graph: which core co-anchors appear together?
    adj: Dict[str, Set[str]] = defaultdict(set)
    edge_count = 0

    for surface_cas in surface_coanchors:
        # Only consider core co-anchors
        core_in_surface = surface_cas & core_coanchors
        if len(core_in_surface) >= 2:
            # Add edges between all pairs
            core_list = list(core_in_surface)
            for i, ca1 in enumerate(core_list):
                for ca2 in core_list[i+1:]:
                    if ca2 not in adj[ca1]:
                        edge_count += 1
                    adj[ca1].add(ca2)
                    adj[ca2].add(ca1)

    # Compute edge density
    n = len(core_coanchors)
    possible_edges = n * (n - 1) // 2
    edge_density = edge_count / possible_edges if possible_edges > 0 else 0.0

    # If no edges or density too low, mark as indeterminate
    if not adj or edge_density < min_edge_density:
        return DispersionResult(
            dispersion=0.5,  # Uncertain
            is_indeterminate=True,
            edge_density=edge_density,
            largest_component_fraction=0.0,
            num_core_coanchors=n,
            num_components=n  # All isolated
        )

    # BFS to find connected components and their sizes
    visited = set()
    component_sizes = []

    for start in core_coanchors:
        if start in visited:
            continue
        if start not in adj:
            # Isolated node
            visited.add(start)
            component_sizes.append(1)
            continue

        # BFS from this node
        component_size = 0
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component_size += 1
            queue.extend(adj[node] - visited)
        component_sizes.append(component_size)

    num_components = len(component_sizes)
    largest_component = max(component_sizes) if component_sizes else 0
    largest_component_fraction = largest_component / n if n > 0 else 0.0

    # Dispersion based on component fragmentation
    # 1 component = cohesive (dispersion 0)
    # Many components = bridging (dispersion approaches 1)
    if num_components == 1:
        dispersion = 0.0
    else:
        # Scale: 2 components = 0.5, 3+ = higher
        dispersion = 1.0 - (1.0 / num_components)
        dispersion = min(dispersion, 1.0)

    return DispersionResult(
        dispersion=dispersion,
        is_indeterminate=False,
        edge_density=edge_density,
        largest_component_fraction=largest_component_fraction,
        num_core_coanchors=n,
        num_components=num_components
    )


def analyze_surface_time_mode_hubness(
    surfaces: Dict[str, "Surface"],
    frequency_threshold: int = 3,
    gap_days: float = 3.0,
    max_span_days: float = 14.0,
    min_cluster_fraction: float = 0.5,
    coincident_dispersion_threshold: float = 0.7,
) -> TimeModeHubnessResult:
    """
    Analyze hubness of anchors using time-based mode detection.

    For L2→L3 clustering, "hubness" means an anchor bridges multiple
    temporal modes (distinct incidents). This differs from co-anchor
    dispersion which fails at surface granularity.

    Algorithm:
    1. For each anchor with freq >= threshold:
       - Collect surface timepoints (midpoint of time window)
       - Sort by time
       - Partition into clusters where consecutive gap > gap_days
       - Compute: k=#clusters, largest_cluster_fraction, span

    2. Classification (4-way):
       - Backbone: k=1 + low dispersion (or dominant cluster + low dispersion)
       - Hub (time): k>=2 AND no dominant cluster (bridges time periods)
       - Hub (coincident): k=1 (or dominant) BUT high within-cluster dispersion
       - Splittable: k>=2, has dominant cluster, low within-cluster dispersion

    Args:
        surfaces: Dict of surface_id -> Surface
        frequency_threshold: Min surfaces for analysis
        gap_days: Days gap that starts a new cluster
        max_span_days: Max span for single-mode backbone
        min_cluster_fraction: Min fraction in largest cluster for backbone
        coincident_dispersion_threshold: Above this = hub_coincident

    Returns:
        TimeModeHubnessResult with anchor classifications
    """
    # Build anchor -> surfaces mapping
    anchor_to_surfaces: Dict[str, Set[str]] = defaultdict(set)
    for sid, surface in surfaces.items():
        for anchor in surface.anchor_entities:
            anchor_to_surfaces[anchor].add(sid)

    anchors: Dict[str, TimeModeInfo] = {}
    backbone_count = 0
    hub_coincident_count = 0
    hub_count = 0
    splittable_count = 0
    neutral_count = 0

    for anchor, surface_ids in anchor_to_surfaces.items():
        frequency = len(surface_ids)

        if frequency < frequency_threshold:
            # Below threshold - neutral
            anchors[anchor] = TimeModeInfo(
                anchor=anchor,
                frequency=frequency,
                num_clusters=0,
                time_span_days=0.0,
                largest_cluster_fraction=0.0,
                cluster_sizes=[],
                is_backbone=False,
                is_hub=False,
                is_splittable=False,
                surface_ids=surface_ids,
            )
            neutral_count += 1
            continue

        # Collect timepoints
        timepoints = []
        for sid in surface_ids:
            surface = surfaces.get(sid)
            if not surface:
                continue

            # Get time window
            time_start = surface.time_window[0] if surface.time_window else None
            time_end = surface.time_window[1] if surface.time_window else None

            # Parse if string
            if isinstance(time_start, str):
                try:
                    time_start = datetime.fromisoformat(time_start.replace('Z', '+00:00'))
                except:
                    time_start = None
            if isinstance(time_end, str):
                try:
                    time_end = datetime.fromisoformat(time_end.replace('Z', '+00:00'))
                except:
                    time_end = None

            # Use midpoint if both available, otherwise use what we have
            if time_start and time_end:
                midpoint = time_start + (time_end - time_start) / 2
            elif time_start:
                midpoint = time_start
            elif time_end:
                midpoint = time_end
            else:
                continue

            timepoints.append((midpoint, sid))

        if not timepoints:
            # No time data - can't classify by time mode
            anchors[anchor] = TimeModeInfo(
                anchor=anchor,
                frequency=frequency,
                num_clusters=1,  # Assume single mode
                time_span_days=0.0,
                largest_cluster_fraction=1.0,
                cluster_sizes=[frequency],
                is_backbone=True,  # Default to backbone if no time data
                is_hub=False,
                is_splittable=False,
                surface_ids=surface_ids,
            )
            backbone_count += 1
            continue

        # Sort by time
        timepoints.sort(key=lambda x: x[0])

        # Partition into clusters by time gaps
        clusters: List[List[Tuple[datetime, str]]] = [[timepoints[0]]]
        gap_threshold = timedelta(days=gap_days)

        for i in range(1, len(timepoints)):
            prev_time = timepoints[i-1][0]
            curr_time = timepoints[i][0]
            gap = curr_time - prev_time

            if gap > gap_threshold:
                # Start new cluster
                clusters.append([timepoints[i]])
            else:
                clusters[-1].append(timepoints[i])

        # Compute metrics
        num_clusters = len(clusters)
        cluster_sizes = [len(c) for c in clusters]
        largest_cluster_fraction = max(cluster_sizes) / len(timepoints)

        min_time = timepoints[0][0]
        max_time = timepoints[-1][0]
        time_span_days = (max_time - min_time).total_seconds() / 86400.0

        # Compute cluster time ranges for evidence and surface->cluster mapping
        cluster_time_ranges = []
        surface_to_cluster: Dict[str, int] = {}
        for cluster_idx, cluster in enumerate(clusters):
            c_times = [t for t, _ in cluster]
            cluster_time_ranges.append((min(c_times), max(c_times)))
            # Map each surface to its cluster index
            for _, sid in cluster:
                surface_to_cluster[sid] = cluster_idx

        # Find dominant cluster for dispersion check
        dominant_cluster_idx = cluster_sizes.index(max(cluster_sizes))
        dominant_cluster_sids = [sid for sid, cidx in surface_to_cluster.items()
                                  if cidx == dominant_cluster_idx]

        # Compute within-cluster dispersion for dominant cluster
        disp_result = _compute_within_cluster_dispersion(
            anchor, dominant_cluster_sids, surfaces
        )
        within_cluster_dispersion = disp_result.dispersion

        # Classification logic (4-way + INDETERMINATE)
        is_hub_coincident = False
        is_indeterminate = disp_result.is_indeterminate

        if num_clusters == 1:
            # Single cluster - check dispersion to determine backbone vs hub_coincident
            if disp_result.is_indeterminate:
                # Not enough data to reliably detect HUB_COINCIDENT
                # Default to BACKBONE (conservative - allows binding)
                is_backbone = True
                is_hub = False
                is_splittable = False
            elif within_cluster_dispersion >= coincident_dispersion_threshold:
                # High dispersion = bridges concurrent incidents
                is_backbone = False
                is_hub = False
                is_splittable = False
                is_hub_coincident = True
            else:
                is_backbone = True
                is_hub = False
                is_splittable = False
        elif largest_cluster_fraction >= min_cluster_fraction and time_span_days <= max_span_days:
            # Dominant cluster in tight span - check dispersion
            if disp_result.is_indeterminate:
                is_backbone = True
                is_hub = False
                is_splittable = False
            elif within_cluster_dispersion >= coincident_dispersion_threshold:
                is_backbone = False
                is_hub = False
                is_splittable = False
                is_hub_coincident = True
            else:
                is_backbone = True
                is_hub = False
                is_splittable = False
        elif largest_cluster_fraction < min_cluster_fraction:
            # Multi-modal spread - hub (time)
            is_backbone = False
            is_hub = True
            is_splittable = False
        else:
            # Multi-modal with dominant cluster - check dispersion for splittable vs hub_coincident
            if disp_result.is_indeterminate:
                # Not enough data - default to SPLITTABLE (allows within-mode binding)
                is_backbone = False
                is_hub = False
                is_splittable = True
            elif within_cluster_dispersion >= coincident_dispersion_threshold:
                # High dispersion even in dominant cluster = hub_coincident
                is_backbone = False
                is_hub = False
                is_splittable = False
                is_hub_coincident = True
            else:
                # Low dispersion = safe to use within-mode
                is_backbone = False
                is_hub = False
                is_splittable = True

        # Update counts
        if is_backbone:
            backbone_count += 1
        elif is_hub:
            hub_count += 1
        elif is_splittable:
            splittable_count += 1
        elif is_hub_coincident:
            hub_coincident_count += 1

        anchors[anchor] = TimeModeInfo(
            anchor=anchor,
            frequency=frequency,
            num_clusters=num_clusters,
            time_span_days=time_span_days,
            largest_cluster_fraction=largest_cluster_fraction,
            cluster_sizes=cluster_sizes,
            is_backbone=is_backbone,
            is_hub=is_hub,
            is_splittable=is_splittable,
            is_hub_coincident=is_hub_coincident,
            is_indeterminate=is_indeterminate,
            within_cluster_dispersion=within_cluster_dispersion,
            coanchor_edge_density=disp_result.edge_density,
            largest_component_fraction=disp_result.largest_component_fraction,
            surface_ids=surface_ids,
            cluster_time_ranges=cluster_time_ranges,
            surface_to_cluster=surface_to_cluster,
        )

    return TimeModeHubnessResult(
        anchors=anchors,
        total_anchors=len(anchors),
        backbone_count=backbone_count,
        hub_count=hub_count,
        hub_coincident_count=hub_coincident_count,
        splittable_count=splittable_count,
        neutral_count=neutral_count,
        params_used={
            'frequency_threshold': frequency_threshold,
            'gap_days': gap_days,
            'max_span_days': max_span_days,
            'min_cluster_fraction': min_cluster_fraction,
            'coincident_dispersion_threshold': coincident_dispersion_threshold,
        },
    )


# =============================================================================
# HIGHER-ORDER CONTEXT COMPATIBILITY (2025-01 upgrade)
# =============================================================================

def context_compatible(
    entity: str,
    surface1: "Surface",
    surface2: "Surface",
    min_overlap: float = 0.15,
) -> bool:
    """
    Check if entity's companion context is compatible between two surfaces.

    This is the KEY higher-order check that prevents bridge entities
    from merging unrelated topics.

    Algorithm:
    - Get entity's companions in S1 (other anchors in S1)
    - Get entity's companions in S2 (other anchors in S2)
    - Check Jaccard overlap

    If overlap >= min_overlap → contexts are compatible → entity can bind
    If overlap < min_overlap → contexts are incompatible → entity bridges topics

    Example:
    - S1: John Lee + Wang Fuk Court + Tai Po (fire context)
    - S2: John Lee + Jimmy Lai + Esther Toh (trial context)
    - John Lee's companions in S1: {Wang Fuk Court, Tai Po}
    - John Lee's companions in S2: {Jimmy Lai, Esther Toh}
    - Overlap = 0 → INCOMPATIBLE → John Lee should NOT bind S1↔S2

    Args:
        entity: The anchor entity to check
        surface1: First surface
        surface2: Second surface
        min_overlap: Minimum Jaccard overlap for compatibility

    Returns:
        True if entity can bind these surfaces (compatible context)
        False if entity bridges different topics (incompatible)
    """
    # Get companions (other anchors) in each surface
    companions1 = surface1.anchor_entities - {entity}
    companions2 = surface2.anchor_entities - {entity}

    # If either surface has no other anchors, can't determine context
    # Be conservative: allow binding (this is the existing behavior)
    if not companions1 or not companions2:
        return True

    # Jaccard overlap
    intersection = len(companions1 & companions2)
    union = len(companions1 | companions2)

    if union == 0:
        return True

    overlap = intersection / union
    return overlap >= min_overlap


def filter_binding_anchors_by_context(
    shared_anchors: Set[str],
    surface1: "Surface",
    surface2: "Surface",
    min_overlap: float = 0.15,
) -> Set[str]:
    """
    Filter shared anchors to those with compatible context between surfaces.

    This is the higher-order upgrade to get_binding_anchors().
    Instead of just checking hub status, we check if each anchor's
    companion context is compatible between the two surfaces.

    Args:
        shared_anchors: Anchors that appear in both surfaces
        surface1: First surface
        surface2: Second surface
        min_overlap: Minimum Jaccard overlap for compatibility

    Returns:
        Subset of shared_anchors that pass context compatibility check
    """
    return {
        anchor for anchor in shared_anchors
        if context_compatible(anchor, surface1, surface2, min_overlap)
    }
