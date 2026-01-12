"""
Aboutness Scoring for L2 -> L3 Event Clustering
================================================

Computes soft aboutness edges between surfaces.
These edges represent "same event, different aspect" associations.
They are NOT identity edges and should NOT be used to merge surfaces.

Uses 2-of-3 signal constraint with IDF hub penalty.
"""

import math
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

from reee.types import Surface, AboutnessLink, Event, Parameters


# =============================================================================
# HIGHER-ORDER CONTEXT COMPATIBILITY (inlined to avoid circular import)
# =============================================================================
#
# LOCKED RULES (2025-01):
#
# 1. Singleton anchors never form cores (they can only attach periphery)
# 2. A binding anchor must pass context-compatibility test between scopes
# 3. Entropy is a prior on trust, NOT the decision itself
# 4. Emit meta-claims when test is underpowered: "context_insufficient_support"
#
# Key insight: Use entropy for SCREENING, use compatibility for DECISIONS.
#
# Why this matters:
# - High entropy doesn't mean hub (Wang Fuk Court has many companions in fire
#   story but is still a legitimate backbone for that incident family)
# - Entropy conflates "bridges multiple topics" with "rich incident with aspects"
# - Context compatibility is the principled check: are P(C|e,scope_A) and
#   P(C|e,scope_B) compatible?

@dataclass
class ContextCompatibilityResult:
    """Result of context compatibility check with audit trail."""
    compatible: bool
    overlap: float  # Jaccard overlap of companion sets
    companions1: Set[str]
    companions2: Set[str]
    underpowered: bool  # True if sample too small for reliable decision
    reason: str  # Human-readable explanation


def context_compatible(
    entity: str,
    surface1: Surface,
    surface2: Surface,
    min_overlap: float = 0.15,
    min_companions: int = 2,  # Minimum companions for reliable test
) -> ContextCompatibilityResult:
    """
    Check if entity's companion context is compatible between two surfaces.

    This is the DECISION function (not entropy-based screening).

    Algorithm:
    - Build P(C|e, scope=S1) and P(C|e, scope=S2) from companion sets
    - Compute Jaccard similarity (TODO: Jensen-Shannon with smoothing)
    - If contexts are incompatible → entity cannot bind these surfaces

    LOCKED RULES:
    - If either surface has < min_companions, mark as underpowered
    - Underpowered tests return compatible=True but underpowered=True
    - This allows binding but requires meta-claim: "context_insufficient_support"

    Example:
    - S1: John Lee + Wang Fuk Court + Tai Po (fire context)
    - S2: John Lee + Jimmy Lai + Esther Toh (trial context)
    - companions1 = {Wang Fuk Court, Tai Po}
    - companions2 = {Jimmy Lai, Esther Toh}
    - Overlap = 0 → INCOMPATIBLE → John Lee cannot bind S1↔S2

    Returns:
        ContextCompatibilityResult with full audit trail
    """
    companions1 = surface1.anchor_entities - {entity}
    companions2 = surface2.anchor_entities - {entity}

    # Check for underpowered test (insufficient support)
    if len(companions1) < min_companions or len(companions2) < min_companions:
        return ContextCompatibilityResult(
            compatible=True,  # Allow binding but flag as underpowered
            overlap=0.0,
            companions1=companions1,
            companions2=companions2,
            underpowered=True,
            reason=f"context_insufficient_support: {len(companions1)}+{len(companions2)} companions < {min_companions}*2"
        )

    # Jaccard overlap of companion sets
    intersection = companions1 & companions2
    union = companions1 | companions2

    if not union:
        return ContextCompatibilityResult(
            compatible=True,
            overlap=0.0,
            companions1=companions1,
            companions2=companions2,
            underpowered=True,
            reason="empty_union"
        )

    overlap = len(intersection) / len(union)
    compatible = overlap >= min_overlap

    if compatible:
        reason = f"compatible: Jaccard={overlap:.3f} >= {min_overlap}"
    else:
        reason = f"incompatible_context: Jaccard={overlap:.3f} < {min_overlap}, disjoint={companions1} vs {companions2}"

    return ContextCompatibilityResult(
        compatible=compatible,
        overlap=overlap,
        companions1=companions1,
        companions2=companions2,
        underpowered=False,
        reason=reason
    )


def filter_binding_anchors_by_context(
    shared_anchors: Set[str],
    surface1: Surface,
    surface2: Surface,
    min_overlap: float = 0.15,
) -> Tuple[Set[str], List[Dict]]:
    """
    Filter shared anchors to those with compatible context between surfaces.

    This is the higher-order upgrade that prevents bridge entities from
    merging unrelated topics.

    Returns:
        Tuple of:
        - Set of anchors that pass compatibility check
        - List of meta-claims for underpowered/incompatible tests (for audit)
    """
    binding = set()
    meta_claims = []

    for anchor in shared_anchors:
        result = context_compatible(anchor, surface1, surface2, min_overlap)

        if result.compatible:
            binding.add(anchor)

        # Emit meta-claim for audit trail
        if result.underpowered:
            meta_claims.append({
                'type': 'context_insufficient_support',
                'anchor': anchor,
                'reason': result.reason,
                'surfaces': [surface1.id, surface2.id]
            })
        elif not result.compatible:
            meta_claims.append({
                'type': 'context_incompatible',
                'anchor': anchor,
                'overlap': result.overlap,
                'companions1': list(result.companions1),
                'companions2': list(result.companions2),
                'surfaces': [surface1.id, surface2.id]
            })

    return binding, meta_claims


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between embeddings."""
    if not a or not b:
        return 0.0
    a_arr, b_arr = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


# Hub detection is now purely dispersion-based (see Parameters.hub_max_df)
# Previously hardcoded entities were removed to allow domain-agnostic operation

# Publishers/sources that should not act as binding anchors
# These often appear as entities but are really metadata, not incident identifiers
PUBLISHER_ENTITIES = {
    'Time', 'TIME', 'New York Times', 'Washington Post', 'BBC',
    'CNN', 'Fox News', 'Reuters', 'AP', 'Associated Press',
    'The Guardian', 'Daily Mail', 'SCMP', 'South China Morning Post',
}


class AboutnessScorer:
    """
    Computes aboutness scores between surfaces for event clustering.

    Uses parameters:
        - hub_max_df: Anchors in more surfaces than this get zero weight
        - aboutness_min_signals: Require N of 3 signals for edge
        - aboutness_threshold: Min score to link surfaces into events

    Supports mode-scoped splittables:
        - If time_mode_hubness is provided, splittable anchors are allowed
          to bind surfaces that are in the same temporal cluster.
        - This prevents cross-mode bridges while preserving within-mode glue.
    """

    def __init__(self, surfaces: Dict[str, Surface], params: Parameters = None):
        self.surfaces = surfaces
        self.params = params or Parameters()

        # Get time-mode hubness result for mode-scoped filtering (optional)
        # If provided, enables mode-scoped splittables instead of global filtering
        self.time_mode_hubness = getattr(self.params, 'time_mode_hubness', None)

        # Get global hub anchors (pure hubs that never bind)
        # This is only used if time_mode_hubness is not available
        self.global_hub_anchors = getattr(self.params, 'hub_anchors', set())

        # Compute IDF weights
        self._compute_idf()

    def _compute_idf(self):
        """Compute IDF weights for anchors and entities."""
        surfaces = list(self.surfaces.values())
        n_surfaces = len(surfaces)
        hub_max_df = self.params.hub_max_df

        # Get pure hubs (never bind) - NOT including splittables when mode-scoped
        if self.time_mode_hubness:
            # Mode-scoped: only pure hubs get global IDF=0
            # Splittables get full IDF; filtering is done per-pair in score_pair
            pure_hubs = self.time_mode_hubness.hubs
        else:
            # Legacy: use global hub_anchors (may include splittables)
            pure_hubs = self.global_hub_anchors

        # Compute anchor document frequency
        anchor_df = defaultdict(int)
        for s in surfaces:
            for anchor in s.anchor_entities:
                anchor_df[anchor] += 1

        # Compute anchor IDF with hub penalty
        # Only PURE HUBS get IDF=0 globally; splittables get full IDF for mode-scoped check
        self.anchor_idf = {}
        for anchor, df in anchor_df.items():
            if anchor in pure_hubs:
                # Pure hub: never binds anything
                self.anchor_idf[anchor] = 0.0
            elif hub_max_df is not None and df > hub_max_df:
                # Frequency-based hub (legacy fallback)
                self.anchor_idf[anchor] = 0.0
            else:
                self.anchor_idf[anchor] = math.log(1 + n_surfaces / df)

        # Compute entity IDF with hub penalty
        entity_df = defaultdict(int)
        for s in surfaces:
            for e in s.entities:
                entity_df[e] += 1

        self.entity_idf = {}
        for e, df in entity_df.items():
            if e in pure_hubs:
                # If entity is also a pure hub, suppress it
                self.entity_idf[e] = 0.0
            elif hub_max_df is not None and df > hub_max_df:
                self.entity_idf[e] = 0.0
            else:
                self.entity_idf[e] = math.log(1 + n_surfaces / df)

    def score_pair(self, s1: Surface, s2: Surface) -> Tuple[float, Dict]:
        """
        Compute aboutness score between two surfaces.

        For incident-level events, applies:
        1. Temporal gate: surfaces must be within Δ days
        2. Discriminative anchor: at least one high-IDF shared anchor
        3. Multi-signal requirement: 2+ of anchor/semantic/entity

        Returns: (score, evidence_dict)
        """
        min_signals = self.params.aboutness_min_signals
        evidence = {}
        signals_met = 0

        # =================================================================
        # TEMPORAL GATE (for incident-level events)
        # =================================================================
        temporal_compatible = True
        time_penalty = 1.0

        # Safely unpack time windows (may be None)
        if s1.time_window:
            t1_start, t1_end = s1.time_window
        else:
            t1_start, t1_end = None, None

        if s2.time_window:
            t2_start, t2_end = s2.time_window
        else:
            t2_start, t2_end = None, None

        # Check if time is known
        t1_known = t1_start is not None or t1_end is not None
        t2_known = t2_start is not None or t2_end is not None

        if t1_known and t2_known:
            # Both have time - check overlap within window
            delta_days = self.params.temporal_window_days

            # Use midpoint or available time
            from datetime import timedelta
            t1 = t1_start or t1_end
            t2 = t2_start or t2_end

            days_apart = abs((t1 - t2).days)
            if days_apart > delta_days:
                temporal_compatible = False
            evidence['days_apart'] = days_apart
        elif not t1_known or not t2_known:
            # At least one has unknown time
            # If require_time_for_clustering is True (strict mode), reject the edge
            # Otherwise apply penalty
            if getattr(self.params, 'require_time_for_clustering', False):
                temporal_compatible = False
                evidence['gate'] = 'time_required'
            else:
                time_penalty = self.params.temporal_unknown_penalty
            evidence['time_unknown'] = True

        evidence['temporal_compatible'] = temporal_compatible

        # Reject if temporally incompatible
        if not temporal_compatible:
            evidence['gate'] = 'temporal'
            evidence['signals_met'] = 0
            return 0.0, evidence

        # =================================================================
        # Signal 1: Anchor overlap (IDF-weighted, mode-scoped for splittables)
        # =================================================================
        shared_anchors = s1.anchor_entities & s2.anchor_entities
        anchor_score = 0.0
        has_discriminative = False
        discriminative_anchors = []

        # Apply mode-scoped filtering for splittable anchors
        # Splittables can bind only if both surfaces are in the same temporal cluster
        if self.time_mode_hubness:
            binding_anchors = self.time_mode_hubness.get_binding_anchors(
                shared_anchors, s1.id, s2.id
            )
        else:
            # Legacy: no mode-scoping, just use shared anchors
            binding_anchors = shared_anchors

        # =================================================================
        # HIGHER-ORDER CONTEXT COMPATIBILITY CHECK
        # =================================================================
        # Filter out anchors that bridge incompatible contexts.
        # An anchor is context-compatible if its companions in S1 overlap
        # with its companions in S2 (i.e., it appears in the same topic).
        #
        # Example: John Lee appears in both Hong Kong Fire and Jimmy Lai trial.
        # - In fire surfaces: companions = {Wang Fuk Court, Tai Po, Joe Chow}
        # - In Lai surfaces: companions = {Jimmy Lai, Esther Toh, Apple Daily}
        # - Companion overlap = 0 → John Lee bridges different topics
        # - Therefore John Lee should NOT bind fire to Lai surfaces
        #
        # This prevents percolation through bridge entities that appear
        # in multiple unrelated contexts.
        #
        # LOCKED RULE: Emit meta-claims for underpowered tests so REEE is
        # honest about constraint scarcity.
        binding_anchors, context_meta_claims = filter_binding_anchors_by_context(
            binding_anchors, s1, s2, min_overlap=0.15
        )
        evidence['context_filtered_anchors'] = list(shared_anchors - binding_anchors)
        evidence['context_meta_claims'] = context_meta_claims

        if binding_anchors:
            anchor_score = sum(self.anchor_idf.get(a, 0) for a in binding_anchors)
            max_anchor = max(
                sum(self.anchor_idf.get(a, 0) for a in s1.anchor_entities),
                sum(self.anchor_idf.get(a, 0) for a in s2.anchor_entities),
                1.0
            )
            anchor_score = min(anchor_score / max_anchor, 1.0)
            if anchor_score > 0.3:
                signals_met += 1

            # Check for discriminative anchors (high IDF, excluding publishers)
            # Only anchors that pass mode-scoping can be discriminative
            idf_threshold = self.params.discriminative_idf_threshold
            for a in binding_anchors:
                # Skip publisher entities - they act as metadata, not incident identifiers
                if a in PUBLISHER_ENTITIES:
                    continue
                if self.anchor_idf.get(a, 0) >= idf_threshold:
                    has_discriminative = True
                    discriminative_anchors.append(a)

        evidence['anchor_score'] = anchor_score
        evidence['shared_anchors'] = list(shared_anchors)
        evidence['discriminative_anchors'] = discriminative_anchors
        evidence['has_discriminative'] = has_discriminative

        # =================================================================
        # Signal 2: Semantic similarity (centroid)
        # =================================================================
        semantic_score = 0.0
        if s1.centroid and s2.centroid:
            semantic_score = cosine_similarity(s1.centroid, s2.centroid)
            if semantic_score > 0.45:
                signals_met += 1
        evidence['semantic_score'] = semantic_score

        # =================================================================
        # Signal 3: Entity overlap (IDF-weighted, EXCLUDING anchors)
        # =================================================================
        non_anchor_entities_1 = s1.entities - s1.anchor_entities
        non_anchor_entities_2 = s2.entities - s2.anchor_entities
        shared_entities = non_anchor_entities_1 & non_anchor_entities_2
        entity_score = 0.0
        if shared_entities:
            entity_weight = sum(self.entity_idf.get(e, 0) for e in shared_entities)
            max_entity = max(
                sum(self.entity_idf.get(e, 0) for e in non_anchor_entities_1),
                sum(self.entity_idf.get(e, 0) for e in non_anchor_entities_2),
                1.0
            )
            entity_score = min(entity_weight / max_entity, 1.0)
            if entity_score > 0.3:
                signals_met += 1
        evidence['entity_score'] = entity_score
        evidence['shared_entities'] = list(shared_entities)[:5]

        # Source diversity
        source_overlap = len(s1.sources & s2.sources)
        source_diversity = 1.0 if source_overlap == 0 else 0.5
        evidence['source_diversity'] = source_diversity

        # =================================================================
        # GATES
        # =================================================================
        evidence['signals_met'] = signals_met

        # Gate 1: Minimum signals
        # Exception: If we have a discriminative anchor AND high anchor_score,
        # allow 1 signal (when embeddings are unavailable)
        #
        # SAFETY: Only allow bypass if discriminative anchor is safe for binding:
        # - BACKBONE with low dispersion (confirmed cohesive)
        # - SPLITTABLE in same mode (already enforced by binding_anchors)
        # - NOT if anchor has indeterminate HUB_COINCIDENT status
        has_strong_discriminative = has_discriminative and anchor_score > 0.7

        # Additional safety check: verify anchors are truly safe for anchor-only binding
        if has_strong_discriminative and self.time_mode_hubness:
            safe_for_bypass = False
            for anchor in discriminative_anchors:
                info = self.time_mode_hubness.anchors.get(anchor)
                if not info:
                    continue

                # HUB_TIME or HUB_COINCIDENT = never safe
                if info.is_hub or info.is_hub_coincident:
                    continue

                # For INDETERMINATE anchors: trust time-mode but require same-cluster
                # (we can't detect HUB_COINCIDENT, so only bind within same time cluster)
                if info.is_indeterminate:
                    # BACKBONE + indeterminate: allow within same time window
                    # (surfaces must be temporally compatible - already checked by temporal gate)
                    if info.is_backbone:
                        safe_for_bypass = True
                        break
                    # SPLITTABLE + indeterminate: require same cluster
                    if info.is_splittable and info.same_mode(s1.id, s2.id):
                        safe_for_bypass = True
                        break
                else:
                    # Confirmed (non-indeterminate): use original logic
                    if info.is_backbone and info.within_cluster_dispersion < 0.5:
                        safe_for_bypass = True
                        break
                    if info.is_splittable and info.same_mode(s1.id, s2.id):
                        safe_for_bypass = True
                        break

            # If no anchor is confirmed safe, don't allow bypass
            if not safe_for_bypass:
                has_strong_discriminative = False

        evidence['has_strong_discriminative'] = has_strong_discriminative
        if signals_met < min_signals and not has_strong_discriminative:
            return 0.0, evidence

        # Gate 2: Discriminative anchor requirement
        if self.params.require_discriminative_anchor and not has_discriminative:
            # Allow if semantic is very high (same topic, different angle)
            if semantic_score < 0.75:
                evidence['gate'] = 'no_discriminative_anchor'
                return 0.0, evidence

        # Gate 3: Single-signal contamination (anchor-only without semantics)
        # Skip this gate if we have a strong discriminative anchor (backbone that binds)
        anchor_only = (anchor_score > 0.3 and semantic_score < 0.5 and entity_score < 0.3)
        if anchor_only and semantic_score < 0.4 and not has_strong_discriminative:
            evidence['gate'] = 'anchor_only'
            return 0.0, evidence

        # Gate 4: Semantic-only without any anchor (topic drift)
        semantic_only = (semantic_score > 0.45 and anchor_score < 0.3 and entity_score < 0.3)
        if semantic_only and semantic_score < 0.85:
            evidence['gate'] = 'semantic_only'
            return 0.0, evidence

        # =================================================================
        # FINAL SCORE
        # =================================================================
        score = (
            0.45 * anchor_score +
            0.25 * semantic_score +
            0.20 * entity_score +
            0.10 * source_diversity
        )

        # Apply time penalty if unknown
        score *= time_penalty

        return score, evidence

    def compute_all_edges(self) -> List[Tuple[str, str, float, Dict]]:
        """
        Compute aboutness edges for all surface pairs.

        Returns: List of (surface_id_1, surface_id_2, score, evidence_dict)
        """
        surfaces = list(self.surfaces.values())
        aboutness_edges = []

        for i, s1 in enumerate(surfaces):
            for s2 in surfaces[i+1:]:
                score, evidence = self.score_pair(s1, s2)
                if score > 0:
                    aboutness_edges.append((s1.id, s2.id, score, evidence))

                    # Populate Surface.about_links (bidirectional)
                    s1.about_links.append(AboutnessLink(
                        target_id=s2.id,
                        score=score,
                        evidence=evidence.copy()
                    ))
                    s2.about_links.append(AboutnessLink(
                        target_id=s1.id,
                        score=score,
                        evidence=evidence.copy()
                    ))

        return aboutness_edges


def compute_aboutness_edges(
    surfaces: Dict[str, Surface],
    params: Parameters = None
) -> List[Tuple[str, str, float, Dict]]:
    """
    Convenience function to compute aboutness edges.

    Returns: List of (surface_id_1, surface_id_2, score, evidence_dict)
    """
    scorer = AboutnessScorer(surfaces, params)
    return scorer.compute_all_edges()


def compute_events_from_aboutness(
    surfaces: Dict[str, Surface],
    aboutness_edges: List[Tuple[str, str, float, Dict]],
    params: Parameters = None
) -> Dict[str, Event]:
    """
    Cluster surfaces into events using core/periphery model.

    Instead of connected components (which percolates through weak bridges),
    uses a two-phase approach:

    1. CORE FORMATION: Build cores from strong edges only (2x threshold)
       - Connected components on strong edges form event cores
       - Cores are dense, high-confidence clusters

    2. PERIPHERY ATTACHMENT: Attach remaining surfaces to best core
       - Each unassigned surface joins the core with highest affinity
       - Periphery attachments NEVER merge cores
       - Surfaces with no strong affinity stay singleton

    This prevents the "one weak bridge merges everything" pathology.
    """
    params = params or Parameters()
    base_threshold = params.aboutness_threshold

    # Thresholds for core vs periphery
    core_threshold = base_threshold * 2.0  # Strong edges form cores
    periphery_threshold = base_threshold  # Weak edges attach to cores

    # Build edge lookup
    edge_map = {}  # (s1, s2) -> (score, evidence)
    for s1_id, s2_id, score, evidence in aboutness_edges:
        edge_map[(s1_id, s2_id)] = (score, evidence)
        edge_map[(s2_id, s1_id)] = (score, evidence)

    # =================================================================
    # PHASE 1: Core formation (strong edges only)
    # =================================================================
    strong_adj = defaultdict(set)
    for s1_id, s2_id, score, evidence in aboutness_edges:
        if score >= core_threshold:
            # Additional check: require discriminative anchor for core edges
            has_discrim = evidence.get('has_discriminative', False)
            if has_discrim:
                strong_adj[s1_id].add(s2_id)
                strong_adj[s2_id].add(s1_id)

    # Find connected components on strong graph → cores
    visited = set()
    cores = []  # List of sets of surface IDs

    for surface_id in surfaces:
        if surface_id in visited:
            continue

        # BFS on strong edges only
        core = set()
        stack = [surface_id]
        while stack:
            curr = stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            core.add(curr)
            stack.extend(strong_adj[curr] - visited)

        # Only keep cores with >1 surface (singletons handled in periphery phase)
        if len(core) > 1:
            cores.append(core)

    # =================================================================
    # PHASE 2: Periphery attachment (weak edges)
    # =================================================================
    # Surfaces already in a core
    core_assigned = set()
    surface_to_core = {}  # surface_id -> core_index
    for i, core in enumerate(cores):
        for sid in core:
            core_assigned.add(sid)
            surface_to_core[sid] = i

    # Assign remaining surfaces to best core (or leave as singleton)
    for surface_id in surfaces:
        if surface_id in core_assigned:
            continue

        # Find best core by aggregate affinity
        best_core = None
        best_affinity = 0.0

        for core_idx, core in enumerate(cores):
            # Sum affinity to all surfaces in this core
            total_affinity = 0.0
            edge_count = 0
            for core_sid in core:
                key = (surface_id, core_sid)
                if key in edge_map:
                    score, _ = edge_map[key]
                    if score >= periphery_threshold:
                        total_affinity += score
                        edge_count += 1

            # Require at least one edge above threshold
            if edge_count > 0 and total_affinity > best_affinity:
                best_affinity = total_affinity
                best_core = core_idx

        if best_core is not None:
            cores[best_core].add(surface_id)
            surface_to_core[surface_id] = best_core
        else:
            # No good core match - create singleton event
            cores.append({surface_id})
            surface_to_core[surface_id] = len(cores) - 1

    # =================================================================
    # PHASE 3: Create events from cores
    # =================================================================
    events = {}
    for i, core in enumerate(cores):
        event_id = f"E{i+1:03d}"
        surfaces_in_event = [surfaces[sid] for sid in core if sid in surfaces]
        event = _event_from_surfaces(event_id, surfaces_in_event)
        events[event_id] = event

    return events


def _event_from_surfaces(event_id: str, surfaces: List[Surface]) -> Event:
    """Compute event properties from surfaces."""
    if not surfaces:
        return Event(id=event_id)

    # Centroid (average of surface centroids)
    centroids = [s.centroid for s in surfaces if s.centroid]
    centroid = np.mean(centroids, axis=0).tolist() if centroids else None

    # Aggregate
    total_claims = sum(len(s.claim_ids) for s in surfaces)
    all_sources = set()
    all_entities = set()
    all_anchors = set()

    for s in surfaces:
        all_sources.update(s.sources)
        all_entities.update(s.entities)
        all_anchors.update(s.anchor_entities)

    # Time window
    starts = [s.time_window[0] for s in surfaces if s.time_window[0]]
    ends = [s.time_window[1] for s in surfaces if s.time_window[1]]
    time_window = (
        min(starts) if starts else None,
        max(ends) if ends else None
    )

    return Event(
        id=event_id,
        surface_ids={s.id for s in surfaces},
        centroid=centroid,
        total_claims=total_claims,
        total_sources=len(all_sources),
        entities=all_entities,
        anchor_entities=all_anchors,
        time_window=time_window
    )
