"""
View Types
==========

Common types for event views.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Set, List, Optional, Any


class ViewScale(Enum):
    """Scale at which events are computed."""
    INCIDENT = "incident"  # Tight time, high precision, cohesion
    CASE = "case"          # Loose time, relation backbone, narrative
    SAGA = "saga"          # Long horizon, causal chains (future)


@dataclass
class ViewTrace:
    """
    Provenance record for a view computation.

    Ensures epistemic replayability: given the same inputs and params,
    we can reproduce the same view.
    """
    view_scale: ViewScale
    computed_at: datetime = field(default_factory=datetime.utcnow)

    # Input snapshot
    surface_ids: Set[str] = field(default_factory=set)
    params_version: int = 1
    params_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Computation details
    edges_computed: int = 0
    events_formed: int = 0
    singletons: int = 0

    # For debugging/analysis
    gates_hit: Dict[str, int] = field(default_factory=dict)
    avg_signals_per_edge: float = 0.0


@dataclass
class ViewResult:
    """
    Result of a view computation.

    Contains:
    - incidents: Multi-surface event clusters (the real "emerged incidents")
    - isolated_surfaces: Single-surface clusters (propositions without event context)
    - trace: Provenance for reproducibility

    The distinction matters:
    - Incidents = multiple surfaces bound by aboutness = true event emergence
    - Isolated surfaces = just propositions, no event-level binding yet
    """
    scale: ViewScale
    events: Dict[str, Any]  # event_id -> Event (all, for backwards compat)
    trace: ViewTrace

    @property
    def incidents(self) -> Dict[str, Any]:
        """Multi-surface events (true incident emergence)."""
        return {
            eid: e for eid, e in self.events.items()
            if len(e.surface_ids) > 1
        }

    @property
    def isolated_surfaces(self) -> Dict[str, Any]:
        """Single-surface clusters (propositions without event binding)."""
        return {
            eid: e for eid, e in self.events.items()
            if len(e.surface_ids) == 1
        }

    @property
    def total_incidents(self) -> int:
        """Count of true multi-surface incidents."""
        return len(self.incidents)

    @property
    def total_isolated(self) -> int:
        """Count of isolated surfaces."""
        return len(self.isolated_surfaces)

    @property
    def total_events(self) -> int:
        """Total events (incidents + isolated). For backwards compat."""
        return len(self.events)

    @property
    def total_surfaces(self) -> int:
        return sum(len(e.surface_ids) for e in self.events.values())

    @property
    def avg_surfaces_per_incident(self) -> float:
        """Average surfaces per true incident (excludes isolated)."""
        incidents = self.incidents
        if not incidents:
            return 0.0
        return sum(len(e.surface_ids) for e in incidents.values()) / len(incidents)

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"ViewResult({self.scale.value}): "
            f"{self.total_incidents} incidents, "
            f"{self.total_isolated} isolated surfaces, "
            f"avg {self.avg_surfaces_per_incident:.1f} surfaces/incident"
        )
