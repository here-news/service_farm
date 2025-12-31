"""
Event Weaver (Tier 2) - Built on Kernel Output
===============================================

The Weaver doesn't cluster claims directly - it builds on Kernel's semantic analysis.

Architecture:
    Claims → Kernel → Nodes/Surfaces → Weaver → Events

Kernel provides:
    - Nodes: distinct facts
    - Surfaces: clusters of related nodes
    - Edges: CONFIRMS/REFINES/SUPERSEDES/CONFLICTS

Weaver uses:
    - Surfaces as proto-events (semantic clusters)
    - Entity overlap to merge related surfaces
    - Temporal proximity for narrative chains

This avoids duplicating semantic work and leverages the Kernel's LLM-backed classification.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from models.domain.claim import Claim
    from test_eu.core.kernel import EpistemicKernel
    from test_eu.core.topology import Topology, Surface, Node


class EventRelationType(Enum):
    """Relations between events."""
    FOLLOWS = "follows"       # E1 happened after E2
    CAUSES = "causes"         # E1 caused E2
    ENABLES = "enables"       # E1 enabled E2
    PART_OF = "part_of"       # E1 is sub-event of E2
    RELATED_TO = "related_to" # Generic relation


@dataclass
class WovenEvent:
    """
    Event built from Kernel surfaces.

    An event is a semantic cluster (Surface) with:
    - Distinct facts (Nodes)
    - Source attestations
    - Temporal bounds
    - Entity mentions
    """
    id: str
    surface_id: int                      # Source surface from Kernel

    # From Surface
    facts: List[str] = field(default_factory=list)  # Node texts
    claim_ids: Set[str] = field(default_factory=set)
    source_count: int = 0
    entropy: float = 0.0

    # Extracted
    entities: Set[str] = field(default_factory=set)
    entity_names: Set[str] = field(default_factory=set)
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None

    # Computed
    label: str = ""
    confidence: float = 0.0


@dataclass
class EventEdge:
    """Relationship between events."""
    source_id: str
    target_id: str
    relation: EventRelationType
    confidence: float = 0.5
    reasoning: str = ""


class SemanticWeaver:
    """
    Builds events from Kernel's semantic output.

    Instead of clustering claims by entity overlap,
    uses Kernel's Surfaces (semantic clusters) as event foundations.
    """

    def __init__(self):
        self.events: List[WovenEvent] = []
        self.edges: List[EventEdge] = []
        self._event_counter = 0

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"ev_{self._event_counter:03d}"

    def weave_from_kernel(
        self,
        kernel: 'EpistemicKernel',
        claims_by_id: Dict[str, 'Claim'] = None
    ) -> Dict:
        """
        Build events from Kernel's topology.

        Each Surface with 2+ sources becomes an event.
        Single-source surfaces remain as unconfirmed claims.

        Args:
            kernel: EpistemicKernel with processed claims
            claims_by_id: Optional claim lookup for entity extraction

        Returns:
            {
                'events': List[WovenEvent],
                'unconfirmed_facts': int,
                'total_facts': int
            }
        """
        claims_by_id = claims_by_id or {}

        # Ensure topology is computed
        kernel.topo.compute()

        self.events = []
        unconfirmed = 0

        for surface in kernel.topo.surfaces:
            # Get all claim IDs from nodes in this surface
            claim_ids = set()
            facts = []
            total_sources = 0

            # Surface has node_indices, look up actual nodes from topology
            for node_idx in surface.node_indices:
                node = kernel.topo.nodes[node_idx]
                claim_ids.update(node.claim_ids)
                facts.append(node.text)
                total_sources += node.source_count

            # Only surfaces with 2+ sources become events
            if total_sources < 2:
                unconfirmed += 1
                continue

            # Extract entities from claims
            entities = set()
            entity_names = set()
            time_start = None
            time_end = None

            for cid in claim_ids:
                claim = claims_by_id.get(cid)
                if claim:
                    # Entities
                    if hasattr(claim, 'entities') and claim.entities:
                        for e in claim.entities:
                            if hasattr(e, 'id'):
                                entities.add(e.id)
                            if hasattr(e, 'canonical_name'):
                                entity_names.add(e.canonical_name)

                    # Time bounds
                    if claim.event_time:
                        ct = claim.event_time
                        if isinstance(ct, str):
                            try:
                                ct = datetime.fromisoformat(ct[:19])
                            except:
                                ct = None

                        if ct:
                            if not time_start or ct < time_start:
                                time_start = ct
                            if not time_end or ct > time_end:
                                time_end = ct

            # Create event
            event = WovenEvent(
                id=self._next_event_id(),
                surface_id=surface.id,
                facts=facts,
                claim_ids=claim_ids,
                source_count=total_sources,
                entropy=surface.entropy(),
                entities=entities,
                entity_names=entity_names,
                time_start=time_start,
                time_end=time_end,
                label=surface.label or facts[0][:50] if facts else "",
                confidence=1.0 - surface.entropy()
            )

            self.events.append(event)

        # Build event-event relations based on temporal ordering
        self._build_temporal_edges()

        return {
            'events': self.events,
            'edges': self.edges,
            'unconfirmed_facts': unconfirmed,
            'total_facts': len(kernel.topo.surfaces)
        }

    def _build_temporal_edges(self):
        """Build FOLLOWS edges between temporally ordered events."""
        self.edges = []

        # Sort events by start time
        timed_events = [e for e in self.events if e.time_start]
        timed_events.sort(key=lambda e: e.time_start)

        # Create FOLLOWS chain
        for i in range(len(timed_events) - 1):
            e1 = timed_events[i]
            e2 = timed_events[i + 1]

            # Check entity overlap for relevance
            shared = e1.entities & e2.entities
            if shared:
                self.edges.append(EventEdge(
                    source_id=e1.id,
                    target_id=e2.id,
                    relation=EventRelationType.FOLLOWS,
                    confidence=len(shared) / max(len(e1.entities | e2.entities), 1),
                    reasoning=f"Shared entities: {len(shared)}"
                ))

    def summary(self) -> Dict:
        """Get weaver state summary."""
        return {
            'total_events': len(self.events),
            'total_edges': len(self.edges),
            'events': [
                {
                    'id': e.id,
                    'label': e.label[:50],
                    'facts': len(e.facts),
                    'claims': len(e.claim_ids),
                    'sources': e.source_count,
                    'entropy': round(e.entropy, 2),
                    'entities': list(e.entity_names)[:3]
                }
                for e in sorted(self.events, key=lambda x: -x.source_count)
            ]
        }


# =============================================================================
# LEGACY WEAVER (for backwards compatibility with TUI)
# =============================================================================

@dataclass
class EventCandidate:
    """Legacy event candidate for backwards compatibility."""
    id: str
    entities: Set[str] = field(default_factory=set)
    entity_names: Set[str] = field(default_factory=set)
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None
    claim_ids: Set[str] = field(default_factory=set)
    embedding: Optional[List[float]] = None
    confidence: float = 0.0


class EventWeaver:
    """
    Legacy weaver - wraps SemanticWeaver for TUI compatibility.

    NOTE: This is deprecated. Use SemanticWeaver directly.
    """

    def __init__(self, event_repository=None, claim_repository=None):
        self.event_repo = event_repository
        self.claim_repo = claim_repository
        self.event_candidates: List[EventCandidate] = []
        self._claim_to_event: Dict[str, str] = {}
        self._event_counter = 0

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"ev_candidate_{self._event_counter:03d}"

    async def weave_claim(
        self,
        claim,
        embedding: Optional[List[float]] = None
    ) -> Dict:
        """
        Legacy method - creates events per-claim.

        For proper semantic clustering, use SemanticWeaver.weave_from_kernel()
        """
        # Extract entities
        claim_entities = set()
        entity_names = set()

        if hasattr(claim, 'entities') and claim.entities:
            for e in claim.entities:
                if hasattr(e, 'id'):
                    claim_entities.add(e.id)
                if hasattr(e, 'canonical_name'):
                    entity_names.add(e.canonical_name)

        # Parse time
        claim_time = None
        if hasattr(claim, 'event_time') and claim.event_time:
            if hasattr(claim.event_time, 'date'):
                claim_time = claim.event_time
            else:
                try:
                    claim_time = datetime.fromisoformat(str(claim.event_time)[:19])
                except:
                    pass

        # Find matching event by entity overlap
        best_event = None
        best_overlap = 0

        for event in self.event_candidates:
            if not claim_entities or not event.entities:
                continue

            shared = len(claim_entities & event.entities)
            if shared > best_overlap:
                best_overlap = shared
                best_event = event

        # Require at least 2 shared entities (not just 1 like Hong Kong)
        if best_event and best_overlap >= 2:
            best_event.claim_ids.add(claim.id)
            best_event.entities.update(claim_entities)
            best_event.entity_names.update(entity_names)

            if claim_time:
                if not best_event.time_start or claim_time < best_event.time_start:
                    best_event.time_start = claim_time
                if not best_event.time_end or claim_time > best_event.time_end:
                    best_event.time_end = claim_time

            self._claim_to_event[claim.id] = best_event.id

            return {
                'action': 'linked',
                'event_id': best_event.id,
                'entity_overlap': best_overlap
            }
        else:
            # Create new event
            new_event = EventCandidate(
                id=self._next_event_id(),
                entities=claim_entities,
                entity_names=entity_names,
                time_start=claim_time,
                time_end=claim_time,
                claim_ids={claim.id},
                embedding=embedding
            )
            self.event_candidates.append(new_event)
            self._claim_to_event[claim.id] = new_event.id

            return {
                'action': 'created',
                'event_id': new_event.id,
                'entity_overlap': 0
            }

    def merge_events(self, min_shared_entities: int = 2) -> int:
        """Merge events with shared entities."""
        merges = 0
        i = 0

        while i < len(self.event_candidates):
            j = i + 1
            while j < len(self.event_candidates):
                e1 = self.event_candidates[i]
                e2 = self.event_candidates[j]

                shared = len(e1.entities & e2.entities)

                if shared >= min_shared_entities:
                    # Merge e2 into e1
                    e1.entities.update(e2.entities)
                    e1.entity_names.update(e2.entity_names)
                    e1.claim_ids.update(e2.claim_ids)

                    if e2.time_start:
                        if not e1.time_start or e2.time_start < e1.time_start:
                            e1.time_start = e2.time_start
                    if e2.time_end:
                        if not e1.time_end or e2.time_end > e1.time_end:
                            e1.time_end = e2.time_end

                    # Update mappings
                    for cid in e2.claim_ids:
                        self._claim_to_event[cid] = e1.id

                    self.event_candidates.pop(j)
                    merges += 1
                else:
                    j += 1
            i += 1

        return merges

    def summary(self) -> Dict:
        return {
            'total_events': len(self.event_candidates),
            'total_claims_linked': len(self._claim_to_event)
        }


# =============================================================================
# RECURSIVE WEAVER (Meta-events from events)
# =============================================================================

@dataclass
class MetaEvent:
    """Meta-event grouping related events."""
    id: str
    event_ids: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    label: str = ""
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None


class RecursiveWeaver:
    """
    Phase 3: Weave events into meta-events.

    Uses same pattern as claim→event, but at event level.
    """

    META_ENTITY_OVERLAP = 0.2
    TEMPORAL_CHAIN_DAYS = 30

    def __init__(self):
        self.meta_events: List[MetaEvent] = []
        self.event_edges: List[EventEdge] = []
        self._meta_counter = 0

    async def weave_events(self, events: List[EventCandidate]) -> Dict:
        """Weave events into meta-events."""
        self.meta_events = []
        self.event_edges = []

        for event in events:
            await self._weave_event(event)

        return self.summary()

    async def _weave_event(self, event: EventCandidate):
        """Add event to meta-event structure."""
        best_meta = None
        best_overlap = 0.0

        for meta in self.meta_events:
            if not event.entities or not meta.entities:
                continue

            shared = len(event.entities & meta.entities)
            total = len(event.entities | meta.entities)
            overlap = shared / total if total > 0 else 0

            if overlap > best_overlap and overlap >= self.META_ENTITY_OVERLAP:
                best_overlap = overlap
                best_meta = meta

        if best_meta:
            best_meta.event_ids.add(event.id)
            best_meta.entities.update(event.entities)

            if event.time_start:
                if not best_meta.time_start or event.time_start < best_meta.time_start:
                    best_meta.time_start = event.time_start
            if event.time_end:
                if not best_meta.time_end or event.time_end > best_meta.time_end:
                    best_meta.time_end = event.time_end
        else:
            self._meta_counter += 1
            new_meta = MetaEvent(
                id=f"meta_{self._meta_counter:03d}",
                event_ids={event.id},
                entities=set(event.entities),
                label=', '.join(list(event.entity_names)[:3]) if event.entity_names else event.id,
                time_start=event.time_start,
                time_end=event.time_end
            )
            self.meta_events.append(new_meta)

    def get_narrative_chains(self) -> List[List[str]]:
        """Extract temporal chains."""
        chains = []
        for meta in self.meta_events:
            if len(meta.event_ids) > 1:
                chains.append(sorted(meta.event_ids))
        return chains

    def summary(self) -> Dict:
        return {
            'total_meta_events': len(self.meta_events),
            'total_event_edges': len(self.event_edges),
            'meta_events': [
                {
                    'id': m.id,
                    'label': m.label,
                    'events': len(m.event_ids),
                    'entities': list(m.entities)[:5],
                    'time_start': str(m.time_start)[:10] if m.time_start else None,
                    'time_end': str(m.time_end)[:10] if m.time_end else None
                }
                for m in self.meta_events
            ],
            'narrative_chains': self.get_narrative_chains()
        }
