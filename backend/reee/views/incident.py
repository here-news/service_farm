"""
Incident Event View
===================

Cohesion operator: "These facts are about the same happening"

Semantics:
- Tight temporal window (default: 7 days)
- Requires discriminative anchor (high-IDF shared entity)
- 2-of-3 signal gate: anchor overlap + semantic similarity + entity overlap
- Bridge-resistant clustering (core/periphery)
- Global hub penalty (entities in >N surfaces get IDF=0)

This is the current L3 implementation, now explicitly named and parameterized.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict

from ..types import Surface, Event, Parameters, MembershipLevel, SurfaceMembership
from ..aboutness.scorer import AboutnessScorer, cosine_similarity
from .types import ViewScale, ViewTrace, ViewResult


@dataclass
class IncidentViewParams:
    """
    Parameters specific to incident-level event formation.

    These control the cohesion operator behavior.
    """
    # Temporal gate
    temporal_window_days: int = 7
    temporal_unknown_penalty: float = 0.5

    # Hub penalty (global DF-based)
    hub_max_df: int = 5  # Entities in more surfaces than this get IDF=0

    # Signal thresholds
    anchor_signal_threshold: float = 0.3
    semantic_signal_threshold: float = 0.45
    entity_signal_threshold: float = 0.3
    min_signals: int = 2  # Require N of 3 signals

    # Discriminative anchor requirement
    require_discriminative_anchor: bool = True
    discriminative_idf_threshold: float = 1.5

    # Clustering
    core_threshold_multiplier: float = 2.0  # Strong edges = base * multiplier
    base_threshold: float = 0.15

    def to_parameters(self) -> Parameters:
        """Convert to legacy Parameters object for compatibility."""
        return Parameters(
            temporal_window_days=self.temporal_window_days,
            temporal_unknown_penalty=self.temporal_unknown_penalty,
            hub_max_df=self.hub_max_df,
            aboutness_min_signals=self.min_signals,
            aboutness_threshold=self.base_threshold,
            require_discriminative_anchor=self.require_discriminative_anchor,
            discriminative_idf_threshold=self.discriminative_idf_threshold,
        )


class IncidentEventView:
    """
    Builder for incident-level events.

    Uses cohesion semantics: tight time, discriminative anchors,
    multi-signal gates, bridge-resistant clustering.
    """

    def __init__(
        self,
        surfaces: Dict[str, Surface],
        params: IncidentViewParams = None,
    ):
        self.surfaces = surfaces
        self.params = params or IncidentViewParams()
        self._legacy_params = self.params.to_parameters()

        # Trace data
        self._gates_hit: Dict[str, int] = defaultdict(int)
        self._edges: List[Tuple[str, str, float, Dict]] = []

    def compute_edges(self) -> List[Tuple[str, str, float, Dict]]:
        """
        Compute aboutness edges between surfaces.

        Uses the existing AboutnessScorer with incident-level params.
        """
        scorer = AboutnessScorer(self.surfaces, self._legacy_params)
        self._edges = scorer.compute_all_edges()

        # Track gate statistics
        surface_list = list(self.surfaces.values())
        for i, s1 in enumerate(surface_list):
            for s2 in surface_list[i+1:]:
                _, evidence = scorer.score_pair(s1, s2)
                gate = evidence.get('gate')
                if gate:
                    self._gates_hit[gate] += 1
                elif evidence.get('signals_met', 0) < self.params.min_signals:
                    self._gates_hit['signals_met < min'] += 1

        return self._edges

    def cluster_events(self) -> Dict[str, Event]:
        """
        Cluster surfaces into events using core/periphery model.

        Phase 1: Build cores from strong edges (2x threshold)
        Phase 2: Attach periphery surfaces to best core
        """
        if not self._edges:
            self.compute_edges()

        base_threshold = self.params.base_threshold
        core_threshold = base_threshold * self.params.core_threshold_multiplier

        # Build edge lookup
        edge_map = {}
        for s1_id, s2_id, score, evidence in self._edges:
            edge_map[(s1_id, s2_id)] = (score, evidence)
            edge_map[(s2_id, s1_id)] = (score, evidence)

        # Phase 1: Core formation (strong edges + discriminative anchor)
        strong_adj = defaultdict(set)
        for s1_id, s2_id, score, evidence in self._edges:
            if score >= core_threshold:
                has_discrim = evidence.get('has_discriminative', False)
                if has_discrim:
                    strong_adj[s1_id].add(s2_id)
                    strong_adj[s2_id].add(s1_id)

        # Find connected components on strong graph
        visited = set()
        cores = []

        for surface_id in self.surfaces:
            if surface_id in visited:
                continue

            core = set()
            stack = [surface_id]
            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                core.add(curr)
                stack.extend(strong_adj[curr] - visited)

            if len(core) > 1:
                cores.append(core)

        # Phase 2: Periphery attachment
        core_surfaces = set().union(*cores) if cores else set()
        unassigned = set(self.surfaces.keys()) - core_surfaces

        # For each unassigned surface, find best core
        periphery_assignments = {}
        for surface_id in unassigned:
            best_core_idx = None
            best_score = 0.0

            for core_idx, core in enumerate(cores):
                for core_surface_id in core:
                    edge_key = (surface_id, core_surface_id)
                    if edge_key in edge_map:
                        score, _ = edge_map[edge_key]
                        if score > best_score:
                            best_score = score
                            best_core_idx = core_idx

            if best_core_idx is not None and best_score >= base_threshold:
                periphery_assignments[surface_id] = best_core_idx

        # Build Event objects
        events = {}
        for core_idx, core in enumerate(cores):
            event_id = f"incident_{core_idx:03d}"

            # Collect all surfaces (core + periphery)
            all_surface_ids = set(core)
            for sid, cidx in periphery_assignments.items():
                if cidx == core_idx:
                    all_surface_ids.add(sid)

            # Build memberships
            memberships = {}
            for sid in core:
                memberships[sid] = SurfaceMembership(
                    surface_id=sid,
                    level=MembershipLevel.CORE,
                    score=1.0,
                    evidence={'phase': 'core_formation'},
                )
            for sid, cidx in periphery_assignments.items():
                if cidx == core_idx:
                    edge_key = next(
                        (k for k in edge_map if k[0] == sid and k[1] in core),
                        None
                    )
                    score = edge_map[edge_key][0] if edge_key else 0.0
                    memberships[sid] = SurfaceMembership(
                        surface_id=sid,
                        level=MembershipLevel.PERIPHERY,
                        score=score,
                        evidence={'phase': 'periphery_attachment'},
                    )

            # Aggregate properties from surfaces
            all_entities = set()
            all_anchors = set()
            all_sources = set()
            total_claims = 0
            min_time, max_time = None, None

            for sid in all_surface_ids:
                s = self.surfaces[sid]
                all_entities.update(s.entities)
                all_anchors.update(s.anchor_entities)
                all_sources.update(s.sources)
                total_claims += len(s.claim_ids)

                t_start, t_end = s.time_window
                if t_start:
                    min_time = min(min_time, t_start) if min_time else t_start
                if t_end:
                    max_time = max(max_time, t_end) if max_time else t_end

            events[event_id] = Event(
                id=event_id,
                surface_ids=all_surface_ids,
                total_claims=total_claims,
                total_sources=len(all_sources),
                entities=all_entities,
                anchor_entities=all_anchors,
                time_window=(min_time, max_time),
                memberships=memberships,
            )

        # Add singletons as single-surface events
        singleton_surfaces = unassigned - set(periphery_assignments.keys())
        for i, sid in enumerate(singleton_surfaces):
            event_id = f"incident_singleton_{i:03d}"
            s = self.surfaces[sid]

            events[event_id] = Event(
                id=event_id,
                surface_ids={sid},
                total_claims=len(s.claim_ids),
                total_sources=len(s.sources),
                entities=s.entities,
                anchor_entities=s.anchor_entities,
                time_window=s.time_window,
                memberships={
                    sid: SurfaceMembership(
                        surface_id=sid,
                        level=MembershipLevel.CORE,
                        score=1.0,
                        evidence={'phase': 'singleton'},
                    )
                },
            )

        return events

    def build(self) -> ViewResult:
        """
        Build the incident event view with full trace.
        """
        events = self.cluster_events()

        # Build trace
        trace = ViewTrace(
            view_scale=ViewScale.INCIDENT,
            surface_ids=set(self.surfaces.keys()),
            params_version=1,
            params_snapshot={
                'temporal_window_days': self.params.temporal_window_days,
                'hub_max_df': self.params.hub_max_df,
                'min_signals': self.params.min_signals,
                'require_discriminative_anchor': self.params.require_discriminative_anchor,
            },
            edges_computed=len(self._edges),
            events_formed=len([e for e in events.values() if len(e.surface_ids) > 1]),
            singletons=len([e for e in events.values() if len(e.surface_ids) == 1]),
            gates_hit=dict(self._gates_hit),
            avg_signals_per_edge=sum(
                e[3].get('signals_met', 0) for e in self._edges
            ) / len(self._edges) if self._edges else 0.0,
        )

        return ViewResult(
            scale=ViewScale.INCIDENT,
            events=events,
            trace=trace,
        )


def build_incident_events(
    surfaces: Dict[str, Surface],
    params: IncidentViewParams = None,
) -> ViewResult:
    """
    Convenience function to build incident events.

    Returns ViewResult with events and trace.
    """
    view = IncidentEventView(surfaces, params)
    return view.build()
