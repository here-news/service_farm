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
from typing import Dict, List, Tuple, Set

from ..types import Surface, AboutnessLink, Event, Parameters


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between embeddings."""
    if not a or not b:
        return 0.0
    a_arr, b_arr = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a_arr), np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


# Known hub entities that should always be penalized
HUB_ENTITIES = {
    'Hong Kong', 'China', 'United States', 'US', 'UK',
    'United Kingdom', 'Beijing', 'Taiwan', 'Europe'
}


class AboutnessScorer:
    """
    Computes aboutness scores between surfaces for event clustering.

    Uses parameters:
        - hub_max_df: Anchors in more surfaces than this get zero weight
        - aboutness_min_signals: Require N of 3 signals for edge
        - aboutness_threshold: Min score to link surfaces into events
    """

    def __init__(self, surfaces: Dict[str, Surface], params: Parameters = None):
        self.surfaces = surfaces
        self.params = params or Parameters()

        # Compute IDF weights
        self._compute_idf()

    def _compute_idf(self):
        """Compute IDF weights for anchors and entities."""
        surfaces = list(self.surfaces.values())
        n_surfaces = len(surfaces)
        hub_max_df = self.params.hub_max_df

        # Compute anchor document frequency
        anchor_df = defaultdict(int)
        for s in surfaces:
            for anchor in s.anchor_entities:
                anchor_df[anchor] += 1

        # Compute anchor IDF with hub penalty
        self.anchor_idf = {}
        for anchor, df in anchor_df.items():
            if df > hub_max_df:
                self.anchor_idf[anchor] = 0.0  # Hub penalty
            else:
                self.anchor_idf[anchor] = math.log(1 + n_surfaces / df)

        # Compute entity IDF with hub penalty
        entity_df = defaultdict(int)
        for s in surfaces:
            for e in s.entities:
                entity_df[e] += 1

        self.entity_idf = {}
        for e, df in entity_df.items():
            if e in HUB_ENTITIES:
                self.entity_idf[e] = 0.0  # Known hub penalty
            elif df > hub_max_df:
                self.entity_idf[e] = 0.0  # Frequency-based hub penalty
            else:
                self.entity_idf[e] = math.log(1 + n_surfaces / df)

    def score_pair(self, s1: Surface, s2: Surface) -> Tuple[float, Dict]:
        """
        Compute aboutness score between two surfaces.

        Returns: (score, evidence_dict)
        """
        min_signals = self.params.aboutness_min_signals
        evidence = {}
        signals_met = 0

        # Signal 1: Strong anchor overlap (IDF-weighted)
        shared_anchors = s1.anchor_entities & s2.anchor_entities
        anchor_score = 0.0
        if shared_anchors:
            anchor_score = sum(self.anchor_idf.get(a, 0) for a in shared_anchors)
            max_anchor = max(
                sum(self.anchor_idf.get(a, 0) for a in s1.anchor_entities),
                sum(self.anchor_idf.get(a, 0) for a in s2.anchor_entities),
                1.0
            )
            anchor_score = min(anchor_score / max_anchor, 1.0)
            if anchor_score > 0.3:
                signals_met += 1
        evidence['anchor_score'] = anchor_score
        evidence['shared_anchors'] = list(shared_anchors)

        # Signal 2: Semantic similarity (centroid)
        semantic_score = 0.0
        if s1.centroid and s2.centroid:
            semantic_score = cosine_similarity(s1.centroid, s2.centroid)
            if semantic_score > 0.45:
                signals_met += 1
        evidence['semantic_score'] = semantic_score

        # Signal 3: Entity overlap (IDF-weighted, EXCLUDING anchors)
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

        # Signal 4: Source diversity
        source_overlap = len(s1.sources & s2.sources)
        source_diversity = 1.0 if source_overlap == 0 else 0.5
        evidence['source_diversity'] = source_diversity

        # 2-of-3 constraint
        if signals_met < min_signals:
            evidence['signals_met'] = signals_met
            return 0.0, evidence

        # Additional gates for single-signal contamination
        anchor_only = (anchor_score > 0.3 and semantic_score < 0.5 and entity_score < 0.3)
        if anchor_only and semantic_score < 0.4:
            evidence['signals_met'] = signals_met
            evidence['gate'] = 'anchor_only'
            return 0.0, evidence

        semantic_only = (semantic_score > 0.45 and anchor_score < 0.3 and entity_score < 0.3)
        if semantic_only and semantic_score < 0.85:
            evidence['signals_met'] = signals_met
            evidence['gate'] = 'semantic_only'
            return 0.0, evidence

        # Compute overall score
        score = (
            0.45 * anchor_score +
            0.25 * semantic_score +
            0.20 * entity_score +
            0.10 * source_diversity
        )

        evidence['signals_met'] = signals_met
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
    Cluster surfaces into events based on aboutness edges.

    Uses connected components on edges above threshold.
    """
    params = params or Parameters()
    aboutness_threshold = params.aboutness_threshold

    # Build aboutness adjacency (above threshold only)
    adj = defaultdict(set)
    for s1_id, s2_id, score, _ in aboutness_edges:
        if score >= aboutness_threshold:
            adj[s1_id].add(s2_id)
            adj[s2_id].add(s1_id)

    # Find connected components
    visited = set()
    event_groups = []

    for surface_id in surfaces:
        if surface_id in visited:
            continue

        group = set()
        stack = [surface_id]
        while stack:
            curr = stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            group.add(curr)
            stack.extend(adj[curr] - visited)

        event_groups.append(group)

    # Create events
    events = {}
    for i, group in enumerate(event_groups):
        event_id = f"E{i+1:03d}"
        surfaces_in_event = [surfaces[sid] for sid in group]
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
