"""
Case Event View
===============

Narrative operator: "These happenings are part of the same story"

Semantics:
- Loose temporal window (months to years)
- Entity relation backbone as primary signal
- Local hubness: dispersion-based (not global IDF)
  - Backbone = high freq + low dispersion → binds incidents
  - Hub = high freq + high dispersion → suppressed
- Shared topic/domain as secondary signal
- Bridge-resistant (don't merge unrelated cases)

This view clusters incidents into cases/storylines.
It operates on incidents (from IncidentEventView) or directly on surfaces.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict

from ..types import Surface, Event, Parameters
from .types import ViewScale, ViewTrace, ViewResult
from .hubness import analyze_hubness, HubnessResult
from .relations import RelationBackbone, build_relation_backbone_from_incidents


@dataclass
class CaseViewParams:
    """
    Parameters specific to case-level clustering.

    These control the narrative operator behavior.
    """
    # Temporal window (loose)
    temporal_window_days: int = 365  # 1 year default

    # Relation backbone
    use_relation_backbone: bool = True
    relation_corroboration_threshold: int = 2  # Min co-occurrences to trust relation
    relation_signal_weight: float = 0.3  # Weight of relation signal in scoring

    # Local hubness (dispersion-based, not IDF)
    use_local_hubness: bool = True
    # Frequency threshold: can be absolute (int) or relative (float < 1.0)
    # Relative: fraction of total incidents (e.g., 0.1 = top 10% by freq)
    hubness_frequency_threshold: float = 3  # Absolute count, or fraction if < 1.0
    hubness_dispersion_threshold: float = 0.7  # Above this = hub (suppressed)
    hubness_variance_threshold: float = 0.5  # Secondary: embedding variance threshold

    # Topic/domain signal
    use_topic_signal: bool = True
    topic_similarity_threshold: float = 0.3

    # Clustering (core/periphery, bridge-resistant)
    case_core_threshold: float = 0.4  # Min score for core edges
    case_core_min_signals: int = 2  # Min signals for core edges
    case_periphery_threshold: float = 0.2  # Min score for periphery attachment


class CaseView:
    """
    Builder for case-level clusters.

    Uses narrative semantics: loose time, relation backbone,
    local hubness, topic similarity.

    Can operate on:
    - Surfaces directly (flat clustering)
    - Incidents from IncidentEventView (hierarchical: incidents -> cases)
    """

    def __init__(
        self,
        surfaces: Dict[str, Surface],
        incidents: Optional[Dict[str, Event]] = None,
        params: CaseViewParams = None,
        relation_graph: Optional[Dict[str, Set[str]]] = None,
        incident_embeddings: Optional[Dict[str, List[float]]] = None,
    ):
        self.surfaces = surfaces
        self.incidents = incidents  # Optional: cluster incidents into cases
        self.params = params or CaseViewParams()
        self.relation_graph = relation_graph or {}  # entity -> related entities (legacy)
        self.incident_embeddings = incident_embeddings  # For variance-based hubness

        self._edges: List[Tuple[str, str, float, Dict]] = []
        self._hubness: Optional[HubnessResult] = None
        self._backbone: Optional[RelationBackbone] = None

    def compute_hubness(self) -> HubnessResult:
        """
        Compute local hubness for anchors using dispersion-based approach.

        Must be called with incidents (not surfaces) for proper hubness analysis.
        """
        if not self.incidents:
            raise ValueError("Hubness analysis requires incidents, not surfaces")

        # Compute effective frequency threshold
        # If < 1.0, treat as fraction of total incidents
        freq_threshold = self.params.hubness_frequency_threshold
        if freq_threshold < 1.0:
            # Relative threshold: fraction of incidents
            freq_threshold = max(2, int(len(self.incidents) * freq_threshold))

        self._hubness = analyze_hubness(
            incidents=self.incidents,
            frequency_threshold=int(freq_threshold),
            dispersion_threshold=self.params.hubness_dispersion_threshold,
            variance_threshold=self.params.hubness_variance_threshold,
            incident_embeddings=self.incident_embeddings,
        )
        return self._hubness

    def compute_backbone(self) -> RelationBackbone:
        """
        Build relation backbone from incident anchor co-occurrence.

        Relations are entity pairs that co-occur in multiple incidents.
        Only corroborated relations (count >= threshold) contribute to scoring.
        """
        if not self.incidents:
            raise ValueError("Backbone analysis requires incidents, not surfaces")

        self._backbone = build_relation_backbone_from_incidents(
            incidents=self.incidents,
            min_co_occurrence=self.params.relation_corroboration_threshold,
        )
        return self._backbone

    def compute_edges(self) -> List[Tuple[str, str, float, Dict]]:
        """
        Compute case-level edges between incidents (or surfaces).

        Uses relation backbone + topic similarity + loose temporal.
        Hub anchors are suppressed (don't contribute to binding).
        """
        # Compute hubness first if using incidents
        if self.incidents and self.params.use_local_hubness:
            self.compute_hubness()

        # Compute relation backbone if using incidents
        if self.incidents and self.params.use_relation_backbone:
            self.compute_backbone()

        # If we have incidents, cluster incidents into cases
        # Otherwise, cluster surfaces directly (with looser params)
        if self.incidents:
            return self._compute_incident_edges()
        else:
            return self._compute_surface_edges()

    def _compute_incident_edges(self) -> List[Tuple[str, str, float, Dict]]:
        """Compute edges between incidents for case formation."""
        edges = []
        incident_list = list(self.incidents.values())

        for i, inc1 in enumerate(incident_list):
            for inc2 in incident_list[i+1:]:
                score, evidence = self._score_incident_pair(inc1, inc2)
                if score > 0:
                    edges.append((inc1.id, inc2.id, score, evidence))

        self._edges = edges
        return edges

    def _filter_hubs(self, anchors: Set[str]) -> Set[str]:
        """
        Filter out hub anchors (high freq + high dispersion).

        Hubs don't contribute to case binding - they bridge unrelated contexts.
        Backbones (high freq + low dispersion) are kept - they bind related incidents.
        """
        if not self._hubness or not self.params.use_local_hubness:
            return anchors

        return {a for a in anchors if a not in self._hubness.hubs}

    def _score_incident_pair(
        self,
        inc1: Event,
        inc2: Event
    ) -> Tuple[float, Dict]:
        """
        Score affinity between two incidents for case formation.

        Signals (need 2+ for core, 1+ for periphery):
        1. Shared anchors (after hub filtering)
        2. Related anchors (via relation backbone, corroborated)
        3. Entity overlap (broader than anchors)
        4. Temporal proximity (within case window)
        """
        evidence = {}
        signals = 0

        # Filter out hubs - they don't contribute to binding
        anchors1 = self._filter_hubs(inc1.anchor_entities)
        anchors2 = self._filter_hubs(inc2.anchor_entities)

        # Track which anchors were suppressed
        suppressed = set()
        if self._hubness:
            suppressed = (inc1.anchor_entities | inc2.anchor_entities) & self._hubness.hubs

        # Signal 1: Shared anchors (direct overlap)
        shared_anchors = anchors1 & anchors2
        anchor_score = 0.0
        if shared_anchors:
            anchor_score = len(shared_anchors) / max(len(anchors1), len(anchors2), 1)
            signals += 1

        evidence['anchor_score'] = anchor_score
        evidence['shared_anchors'] = list(shared_anchors)
        evidence['suppressed_hubs'] = list(suppressed)

        # Signal 2: Related anchors (via corroborated backbone)
        # Backbone pairs must not involve hub entities - hubs are exactly the
        # entities that bridge unrelated contexts, so using them in backbone
        # defeats the purpose of hub filtering.
        backbone_pairs = set()
        backbone_score = 0.0
        if self._backbone and self.params.use_relation_backbone:
            raw_pairs = self._backbone.find_related_pairs(
                inc1.anchor_entities, inc2.anchor_entities,
                min_corroboration=self.params.relation_corroboration_threshold
            )
            # Filter: NEITHER entity in the pair can be a hub
            # (Do Kwon, Terraform Labs) is valid only if both are non-hubs
            # (Jimmy Lai, Hong Kong) is blocked because Hong Kong is a hub
            hubs = self._hubness.hubs if self._hubness else set()
            for a1, a2 in raw_pairs:
                if a1 not in hubs and a2 not in hubs:
                    backbone_pairs.add((a1, a2))

            if backbone_pairs:
                backbone_score = min(1.0, len(backbone_pairs) * 0.3)
                signals += 1

        evidence['backbone_pairs'] = [list(p) for p in backbone_pairs]
        evidence['backbone_score'] = backbone_score

        # Signal 3: Entity overlap (broader than anchors, but still hub-filtered)
        # Hub entities (like "Hong Kong") shouldn't contribute to binding via entity
        # overlap either - they bridge unrelated contexts just as much as anchors.
        hubs = self._hubness.hubs if self._hubness else set()
        entities1_filtered = inc1.entities - hubs
        entities2_filtered = inc2.entities - hubs
        shared_entities = entities1_filtered & entities2_filtered
        entity_score = len(shared_entities) / max(
            len(entities1_filtered), len(entities2_filtered), 1
        )
        if entity_score > 0.3:
            signals += 1
        evidence['entity_score'] = entity_score
        evidence['shared_entities'] = list(shared_entities)[:10]
        evidence['entity_hubs_suppressed'] = list((inc1.entities | inc2.entities) & hubs)

        # Signal 4: Temporal proximity (loose for cases)
        temporal_ok = True
        t1_start, t1_end = inc1.time_window
        t2_start, t2_end = inc2.time_window

        if t1_start and t2_start:
            days_apart = abs((t1_start - t2_start).days)
            temporal_ok = days_apart <= self.params.temporal_window_days
            evidence['days_apart'] = days_apart

        evidence['temporal_ok'] = temporal_ok

        if not temporal_ok:
            return 0.0, evidence

        # Require at least 1 signal for case binding
        evidence['signals'] = signals
        if signals < 1:
            return 0.0, evidence

        # Final score: anchor + backbone + entity + temporal
        # Backbone adds to score but doesn't dominate
        relation_weight = self.params.relation_signal_weight
        score = (
            0.4 * anchor_score +
            relation_weight * backbone_score +
            (0.4 - relation_weight) * entity_score +
            0.2 * (1.0 if temporal_ok else 0.0)
        )

        return score, evidence

    def _find_related_anchors(
        self,
        anchors1: Set[str],
        anchors2: Set[str]
    ) -> Set[Tuple[str, str]]:
        """
        Find related anchor pairs via relation graph.

        Returns pairs of (anchor1, anchor2) where anchor1 is in anchors1,
        anchor2 is in anchors2, and they're related via the relation graph.
        """
        if not self.relation_graph:
            return set()

        related = set()
        for a1 in anchors1:
            if a1 in self.relation_graph:
                for a2 in anchors2:
                    if a2 in self.relation_graph[a1]:
                        related.add((a1, a2))
        return related

    def _compute_surface_edges(self) -> List[Tuple[str, str, float, Dict]]:
        """
        Compute edges between surfaces for case formation.

        Uses looser params than incident view.
        """
        # Stub: use incident view with relaxed params
        # Full implementation: relation backbone + topic signals
        edges = []
        surface_list = list(self.surfaces.values())

        for i, s1 in enumerate(surface_list):
            for s2 in surface_list[i+1:]:
                score, evidence = self._score_surface_pair_for_case(s1, s2)
                if score > 0:
                    edges.append((s1.id, s2.id, score, evidence))

        self._edges = edges
        return edges

    def _score_surface_pair_for_case(
        self,
        s1: Surface,
        s2: Surface
    ) -> Tuple[float, Dict]:
        """Score surface pair for case-level binding (looser than incident)."""
        evidence = {}

        # Check relation backbone
        related = self._find_related_anchors(s1.anchor_entities, s2.anchor_entities)
        shared = s1.anchor_entities & s2.anchor_entities

        if not related and not shared:
            return 0.0, {'reason': 'no_anchor_connection'}

        evidence['shared_anchors'] = list(shared)
        evidence['related_anchors'] = list(related)

        # Loose temporal check
        t1, _ = s1.time_window
        t2, _ = s2.time_window
        if t1 and t2:
            days = abs((t1 - t2).days)
            if days > self.params.temporal_window_days:
                return 0.0, {'reason': 'temporal_too_far', 'days': days}

        # Score based on connection strength
        score = 0.4 if shared else 0.2  # Shared > related
        if related:
            score += 0.2

        return score, evidence

    def cluster_cases(self) -> Dict[str, Event]:
        """
        Cluster incidents/surfaces into cases using core/periphery model.

        Bridge-resistant: cores form only from strong multi-signal edges,
        periphery attaches to best core but never merges cores.

        Phase 1: Build cores from strong edges (score >= core_threshold, signals >= 2)
        Phase 2: Attach periphery incidents to best core (but don't merge cores)
        """
        if not self._edges:
            self.compute_edges()

        if self.incidents:
            items = set(self.incidents.keys())
        else:
            items = set(self.surfaces.keys())

        # Build edge lookup
        edge_map = {}
        for id1, id2, score, evidence in self._edges:
            edge_map[(id1, id2)] = (score, evidence)
            edge_map[(id2, id1)] = (score, evidence)

        # Phase 1: Core formation
        # Strong edges: high score AND multiple signals (not just one weak connection)
        core_threshold = self.params.case_core_threshold
        min_signals_for_core = self.params.case_core_min_signals

        strong_adj = defaultdict(set)
        for id1, id2, score, evidence in self._edges:
            if score >= core_threshold:
                signals = evidence.get('signals', 0)
                if signals >= min_signals_for_core:
                    strong_adj[id1].add(id2)
                    strong_adj[id2].add(id1)

        # Find connected components on strong graph only
        visited = set()
        cores = []

        for item_id in items:
            if item_id in visited:
                continue

            core = set()
            stack = [item_id]
            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                # Only traverse strong edges
                if curr in strong_adj or curr == item_id:
                    core.add(curr)
                    stack.extend(strong_adj[curr] - visited)

            # Only keep cores with multiple members
            if len(core) > 1:
                cores.append(core)

        # Phase 2: Periphery attachment
        # Incidents not in any core get attached to their best core
        core_items = set().union(*cores) if cores else set()
        unassigned = items - core_items

        periphery_assignments = {}
        for item_id in unassigned:
            best_core_idx = None
            best_score = 0.0

            for core_idx, core in enumerate(cores):
                for core_item_id in core:
                    edge_key = (item_id, core_item_id)
                    if edge_key in edge_map:
                        score, _ = edge_map[edge_key]
                        if score > best_score:
                            best_score = score
                            best_core_idx = core_idx

            # Attach if score exceeds periphery threshold
            if best_core_idx is not None and best_score >= self.params.case_periphery_threshold:
                periphery_assignments[item_id] = best_core_idx

        # Build Event objects for cases
        events = {}
        for case_idx, core in enumerate(cores):
            event_id = f"case_{case_idx:03d}"

            # Collect all items (core + periphery)
            all_items = set(core)
            for item_id, cidx in periphery_assignments.items():
                if cidx == case_idx:
                    all_items.add(item_id)

            if self.incidents:
                # Aggregate from incidents
                all_surfaces = set()
                all_entities = set()
                all_anchors = set()
                total_claims = 0

                for inc_id in all_items:
                    inc = self.incidents[inc_id]
                    all_surfaces.update(inc.surface_ids)
                    all_entities.update(inc.entities)
                    all_anchors.update(inc.anchor_entities)
                    total_claims += inc.total_claims

                events[event_id] = Event(
                    id=event_id,
                    surface_ids=all_surfaces,
                    total_claims=total_claims,
                    entities=all_entities,
                    anchor_entities=all_anchors,
                )
            else:
                # Aggregate from surfaces
                all_entities = set()
                all_anchors = set()
                total_claims = 0

                for sid in all_items:
                    s = self.surfaces[sid]
                    all_entities.update(s.entities)
                    all_anchors.update(s.anchor_entities)
                    total_claims += len(s.claim_ids)

                events[event_id] = Event(
                    id=event_id,
                    surface_ids=all_items,
                    total_claims=total_claims,
                    entities=all_entities,
                    anchor_entities=all_anchors,
                )

        # Add singleton cases for unassigned items (no core, no periphery attachment)
        remaining = items - core_items - set(periphery_assignments.keys())
        for i, item_id in enumerate(remaining):
            event_id = f"case_singleton_{i:03d}"

            if self.incidents:
                inc = self.incidents[item_id]
                events[event_id] = Event(
                    id=event_id,
                    surface_ids=inc.surface_ids,
                    total_claims=inc.total_claims,
                    entities=inc.entities,
                    anchor_entities=inc.anchor_entities,
                )
            else:
                s = self.surfaces[item_id]
                events[event_id] = Event(
                    id=event_id,
                    surface_ids={item_id},
                    total_claims=len(s.claim_ids),
                    entities=s.entities,
                    anchor_entities=s.anchor_entities,
                )

        return events

    def build(self) -> ViewResult:
        """Build the case view with full trace."""
        events = self.cluster_cases()

        # Build hubness summary for trace
        hubness_summary = {}
        if self._hubness:
            hubness_summary = {
                'total_anchors': self._hubness.total_anchors,
                'backbones': self._hubness.backbone_count,
                'hubs': self._hubness.hub_count,
                'neutral': self._hubness.neutral_count,
                'hub_list': list(self._hubness.hubs)[:10],
                'backbone_list': list(self._hubness.backbones)[:10],
            }

        # Build backbone summary for trace
        backbone_summary = {}
        if self._backbone:
            backbone_summary = {
                'total_relations': self._backbone.total_relations,
                'corroborated_relations': self._backbone.corroborated_relations,
            }

        trace = ViewTrace(
            view_scale=ViewScale.CASE,
            surface_ids=set(self.surfaces.keys()),
            params_version=1,
            params_snapshot={
                'temporal_window_days': self.params.temporal_window_days,
                'use_relation_backbone': self.params.use_relation_backbone,
                'relation_corroboration_threshold': self.params.relation_corroboration_threshold,
                'use_local_hubness': self.params.use_local_hubness,
                'hubness_dispersion_threshold': self.params.hubness_dispersion_threshold,
                'hubness_frequency_threshold': self.params.hubness_frequency_threshold,
                'case_core_threshold': self.params.case_core_threshold,
                'case_core_min_signals': self.params.case_core_min_signals,
                'case_periphery_threshold': self.params.case_periphery_threshold,
                'hubness_summary': hubness_summary,
                'backbone_summary': backbone_summary,
            },
            edges_computed=len(self._edges),
            events_formed=len([e for e in events.values() if not e.id.startswith('case_singleton')]),
            singletons=len([e for e in events.values() if e.id.startswith('case_singleton')]),
        )

        return ViewResult(
            scale=ViewScale.CASE,
            events=events,
            trace=trace,
        )


def build_case_clusters(
    surfaces: Dict[str, Surface],
    incidents: Optional[Dict[str, Event]] = None,
    params: CaseViewParams = None,
    relation_graph: Optional[Dict[str, Set[str]]] = None,
    incident_embeddings: Optional[Dict[str, List[float]]] = None,
) -> ViewResult:
    """
    Convenience function to build case clusters.

    Can operate on surfaces directly or on incidents.
    When using incidents, hubness analysis will be performed to
    identify and suppress hub anchors (high freq + high dispersion).
    """
    view = CaseView(surfaces, incidents, params, relation_graph, incident_embeddings)
    return view.build()
