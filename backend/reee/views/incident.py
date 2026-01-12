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
# NOTE: AboutnessScorer is deprecated but still used here for edge scoring.
# Import is deferred to compute_edges() to avoid warning on module load.
# This will be refactored when views/ migrates to membrane-based scoring.
# See deprecated/RELIC.md for migration timeline (removal: 2026-02-01).
from .types import ViewScale, ViewTrace, ViewResult
from .hubness import (
    analyze_surface_hubness, HubnessResult,
    analyze_surface_time_mode_hubness, TimeModeHubnessResult,
)


@dataclass
class IncidentViewParams:
    """
    Parameters specific to incident-level event formation.

    These control the cohesion operator behavior.
    """
    # Temporal gate
    temporal_window_days: int = 7
    temporal_unknown_penalty: float = 0.5
    require_time_for_clustering: bool = False  # If True, reject edges where either surface lacks time

    # Hub detection strategy: "time_mode" (recommended), "dispersion", or "frequency"
    #
    # time_mode (L2→L3 principled):
    #   - Backbone: anchor's surfaces form single temporal mode → binds incident
    #   - Hub: anchor's surfaces span multiple separated time clusters → suppressed
    #   - This is the correct criterion for surface→incident clustering
    #
    # dispersion (works for L3→L4, not L2→L3):
    #   - Backbone: high freq + low co-anchor dispersion
    #   - Hub: high freq + high co-anchor dispersion
    #   - Fails at surface level because surfaces are too granular
    #
    # frequency (legacy fallback):
    #   - Hub: anchor in >N% of surfaces → suppressed
    hub_detection: str = "time_mode"  # "time_mode", "dispersion", or "frequency"

    # Time-mode hub detection params (primary for L2→L3)
    time_mode_frequency_threshold: int = 3  # Min surfaces for analysis
    time_mode_gap_days: float = 3.0  # Gap that starts new temporal cluster
    time_mode_max_span_days: float = 14.0  # Max span for single-mode backbone
    time_mode_min_cluster_fraction: float = 0.5  # Min fraction in largest cluster

    # Dispersion-based hub detection params (for L3→L4 / CaseView)
    hub_frequency_threshold: int = 3  # Min surfaces for an anchor to be evaluated
    hub_dispersion_threshold: float = 0.7  # Above this = hub (bridges unrelated)

    # Frequency-based hub detection params (legacy fallback)
    hub_max_pct: float = 0.07  # Percentage of corpus
    hub_max_df: int = None  # Computed dynamically from corpus size

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
        params = Parameters(
            temporal_window_days=self.temporal_window_days,
            temporal_unknown_penalty=self.temporal_unknown_penalty,
            hub_max_df=self.hub_max_df,
            aboutness_min_signals=self.min_signals,
            aboutness_threshold=self.base_threshold,
            require_discriminative_anchor=self.require_discriminative_anchor,
            discriminative_idf_threshold=self.discriminative_idf_threshold,
        )
        # Add dynamic attribute for time-gating
        params.require_time_for_clustering = self.require_time_for_clustering
        return params


class IncidentEventView:
    """
    Builder for incident-level events.

    Uses cohesion semantics: tight time, discriminative anchors,
    multi-signal gates, bridge-resistant clustering.

    Hub detection uses time-mode analysis (L2→L3 principled):
    - Backbone: anchor's surfaces form single temporal mode → binds incident
    - Hub: anchor's surfaces span multiple time clusters → suppressed
    """

    def __init__(
        self,
        surfaces: Dict[str, Surface],
        params: IncidentViewParams = None,
    ):
        self.surfaces = surfaces
        self.params = params or IncidentViewParams()

        # Compute hubness based on strategy
        self._hubness: Optional[HubnessResult] = None
        self._time_mode_hubness: Optional[TimeModeHubnessResult] = None
        self._hub_anchors: Set[str] = set()

        if self.params.hub_detection == "time_mode":
            # Time-mode hubness: correct criterion for L2→L3
            self._time_mode_hubness = analyze_surface_time_mode_hubness(
                surfaces,
                frequency_threshold=self.params.time_mode_frequency_threshold,
                gap_days=self.params.time_mode_gap_days,
                max_span_days=self.params.time_mode_max_span_days,
                min_cluster_fraction=self.params.time_mode_min_cluster_fraction,
            )
            # All hubs (time-mode AND coincident) get global suppression
            # - HUB_TIME: bridges multiple time periods
            # - HUB_COINCIDENT: single time mode but bridges concurrent incidents
            # Splittable anchors are handled per-pair in AboutnessScorer via mode-scoping:
            # - Can bind surfaces in same temporal cluster (within-mode glue)
            # - Cannot bind surfaces in different clusters (prevents cross-mode bridges)
            self._hub_anchors = self._time_mode_hubness.all_hubs
        elif self.params.hub_detection == "dispersion":
            # Dispersion-based (works for L3→L4, not L2→L3)
            self._hubness = analyze_surface_hubness(
                surfaces,
                frequency_threshold=self.params.hub_frequency_threshold,
                dispersion_threshold=self.params.hub_dispersion_threshold,
            )
            self._hub_anchors = self._hubness.hubs
        else:
            # Legacy frequency-based fallback
            if self.params.hub_max_df is None:
                n_surfaces = len(surfaces)
                self.params.hub_max_df = max(10, int(n_surfaces * self.params.hub_max_pct) + 1)

        self._legacy_params = self.params.to_parameters()
        # Pass hub set and time_mode_hubness to legacy params for AboutnessScorer
        self._legacy_params.hub_anchors = self._hub_anchors
        # Pass full hubness result for mode-scoped splittable filtering
        if self._time_mode_hubness:
            self._legacy_params.time_mode_hubness = self._time_mode_hubness

        # Trace data
        self._gates_hit: Dict[str, int] = defaultdict(int)
        self._edges: List[Tuple[str, str, float, Dict]] = []

    def compute_edges(self) -> List[Tuple[str, str, float, Dict]]:
        """
        Compute aboutness edges between surfaces.

        Uses the existing AboutnessScorer with incident-level params.
        """
        # Lazy import to defer deprecation warning until actual use
        from ..aboutness.scorer import AboutnessScorer
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

        # Phase 3: Anchor-based attachment for no-time surfaces (QUARANTINE)
        # These surfaces couldn't form edges due to missing time, but can join
        # events via shared backbone anchors. They don't bridge cores.
        still_unassigned = unassigned - set(periphery_assignments.keys())
        quarantine_assignments = {}

        # Only use TRUE backbone anchors for quarantine attachment
        # Splittable anchors span multiple contexts and can bridge unrelated incidents
        backbone_anchors = set()
        if self._time_mode_hubness:
            backbone_anchors = self._time_mode_hubness.backbones

        for surface_id in still_unassigned:
            s = self.surfaces[surface_id]
            surface_anchors = s.anchor_entities & backbone_anchors

            if not surface_anchors:
                continue

            # Find best core by shared backbone anchors
            best_core_idx = None
            best_overlap = 0

            for core_idx, core in enumerate(cores):
                # Get all backbone anchors in this core
                core_anchors = set()
                for core_sid in core:
                    core_s = self.surfaces[core_sid]
                    core_anchors.update(core_s.anchor_entities & backbone_anchors)

                overlap = len(surface_anchors & core_anchors)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_core_idx = core_idx

            if best_core_idx is not None and best_overlap >= 1:
                quarantine_assignments[surface_id] = best_core_idx

        # Build Event objects
        events = {}
        for core_idx, core in enumerate(cores):
            event_id = f"incident_{core_idx:03d}"

            # Collect all surfaces (core + periphery + quarantine)
            all_surface_ids = set(core)
            for sid, cidx in periphery_assignments.items():
                if cidx == core_idx:
                    all_surface_ids.add(sid)
            for sid, cidx in quarantine_assignments.items():
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
            for sid, cidx in quarantine_assignments.items():
                if cidx == core_idx:
                    s = self.surfaces[sid]
                    shared_anchors = list(s.anchor_entities & backbone_anchors)[:3]
                    memberships[sid] = SurfaceMembership(
                        surface_id=sid,
                        level=MembershipLevel.QUARANTINE,
                        score=0.3,  # Lower confidence for anchor-only attachment
                        evidence={
                            'phase': 'quarantine_anchor_attachment',
                            'shared_backbones': shared_anchors,
                            'no_time': True,
                        },
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

                if s.time_window:
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
        singleton_surfaces = unassigned - set(periphery_assignments.keys()) - set(quarantine_assignments.keys())
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

        # Build trace with hubness info
        hubness_info = {}
        if self._time_mode_hubness:
            hubness_info = {
                'hub_detection': 'time_mode',
                'backbone_count': self._time_mode_hubness.backbone_count,
                'hub_time_count': self._time_mode_hubness.hub_count,
                'hub_coincident_count': self._time_mode_hubness.hub_coincident_count,
                'splittable_count': self._time_mode_hubness.splittable_count,
                'hubs': list(self._hub_anchors)[:10],  # Top 10 for trace
                'hub_coincident': list(self._time_mode_hubness.hub_coincident)[:5],
            }
        elif self._hubness:
            hubness_info = {
                'hub_detection': 'dispersion',
                'hub_count': self._hubness.hub_count,
                'backbone_count': self._hubness.backbone_count,
                'hubs': list(self._hub_anchors)[:10],  # Top 10 for trace
            }
        else:
            hubness_info = {
                'hub_detection': 'frequency',
                'hub_max_pct': self.params.hub_max_pct,
                'hub_max_df': self.params.hub_max_df,
            }

        trace = ViewTrace(
            view_scale=ViewScale.INCIDENT,
            surface_ids=set(self.surfaces.keys()),
            params_version=1,
            params_snapshot={
                'temporal_window_days': self.params.temporal_window_days,
                'n_surfaces': len(self.surfaces),
                'min_signals': self.params.min_signals,
                'require_discriminative_anchor': self.params.require_discriminative_anchor,
                **hubness_info,
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
