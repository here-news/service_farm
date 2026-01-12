"""
Delta Contract - Kernel output.

TopologyDelta describes all changes from a kernel.step() call:
- Structural changes (surfaces, incidents, links)
- Traces (decision explanations)
- Signals (quality indicators)
- Inquiries (actionable requests)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, FrozenSet

from .state import SurfaceState, IncidentState
from .traces import DecisionTrace, BeliefUpdateTrace
from .signals import EpistemicSignal, InquirySeed


@dataclass(frozen=True)
class Link:
    """Edge to create/update in the graph.

    Used for CONTAINS, MEMBER_OF, and other relationships.
    """

    from_id: str  # Source node ID or signature
    relation: str  # Relationship type (e.g., "CONTAINS", "MEMBER_OF")
    to_id: str  # Target node ID or signature
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "from_id": self.from_id,
            "relation": self.relation,
            "to_id": self.to_id,
            "properties": self.properties,
        }


@dataclass
class TopologyDelta:
    """Complete kernel output from a step() call.

    This is the ONLY output from the kernel.
    Worker applies this delta to persistence.

    Design principles:
    - All changes are explicit (no side effects)
    - Traces explain every decision
    - Signals indicate quality issues
    - Inquiries are actionable
    """

    # Structural changes (kernel-owned)
    surface_upserts: List[SurfaceState] = field(default_factory=list)
    incident_upserts: List[IncidentState] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)
    stale_ids: List[str] = field(default_factory=list)  # IDs to mark stale

    # Traces (immutable history)
    decision_traces: List[DecisionTrace] = field(default_factory=list)
    belief_traces: List[BeliefUpdateTrace] = field(default_factory=list)

    # Signals (quality indicators, replaces MetaClaim)
    signals: List[EpistemicSignal] = field(default_factory=list)

    # Inquiry seeds (actionable)
    inquiries: List[InquirySeed] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Check if delta has any structural changes."""
        return bool(
            self.surface_upserts
            or self.incident_upserts
            or self.links
            or self.stale_ids
        )

    @property
    def all_traces(self) -> List:
        """Get all traces (decision + belief)."""
        return list(self.decision_traces) + list(self.belief_traces)

    def merge(self, other: "TopologyDelta") -> "TopologyDelta":
        """Merge two deltas (for batch processing).

        Note: This creates a new delta, doesn't modify in place.
        """
        return TopologyDelta(
            surface_upserts=self.surface_upserts + other.surface_upserts,
            incident_upserts=self.incident_upserts + other.incident_upserts,
            links=self.links + other.links,
            stale_ids=self.stale_ids + other.stale_ids,
            decision_traces=self.decision_traces + other.decision_traces,
            belief_traces=self.belief_traces + other.belief_traces,
            signals=self.signals + other.signals,
            inquiries=self.inquiries + other.inquiries,
        )

    def to_summary(self) -> Dict[str, Any]:
        """Generate summary for logging."""
        return {
            "surfaces_upserted": len(self.surface_upserts),
            "incidents_upserted": len(self.incident_upserts),
            "links_created": len(self.links),
            "stale_marked": len(self.stale_ids),
            "decision_traces": len(self.decision_traces),
            "belief_traces": len(self.belief_traces),
            "signals": len(self.signals),
            "inquiries": len(self.inquiries),
            "signal_types": list(set(s.signal_type.value for s in self.signals)),
        }
