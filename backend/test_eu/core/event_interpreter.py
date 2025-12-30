"""
Event Interpreter
=================

Higher-level interpretation layer that groups surfaces into real-world events.

This is NOT part of the kernel. The kernel computes pure epistemic topology
(embeddings, source overlap, entropy). Event detection is an interpretation
that sits ON TOP of the topology.

Philosophy:
- Surfaces are connected components based on semantic similarity
- Events are human-meaningful groupings (claims about same real-world incident)
- One event may span multiple surfaces (different aspects of same incident)
- One surface may contain only one event (semantically coherent claims)

Usage:
    from test_eu.core.kernel import EpistemicKernel
    from test_eu.core.event_interpreter import EventInterpreter

    kernel = EpistemicKernel(llm_client=llm)
    # ... process claims ...
    kernel.topo.compute()

    interpreter = EventInterpreter(llm_client=llm)
    events = await interpreter.detect_events(kernel.topo)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from .topology import Topology, Surface, Node, cosine_similarity


@dataclass
class Event:
    """
    A detected real-world event.

    An event groups surfaces that describe the same real-world incident,
    even if they discuss different aspects (death toll, investigation, response).
    """
    id: str
    name: str
    surface_ids: List[int] = field(default_factory=list)
    total_nodes: int = 0
    total_sources: int = 0
    keywords: List[str] = field(default_factory=list)
    confidence: float = 0.0  # How confident we are this is a coherent event

    @property
    def mass(self) -> float:
        """Epistemic weight of this event."""
        return self.total_nodes * self.total_sources

    def __repr__(self):
        return f"Event({self.id}: {self.name}, {len(self.surface_ids)} surfaces, mass={self.mass})"


class EventInterpreter:
    """
    Interprets topology surfaces as real-world events.

    Approaches:
    1. LLM-based: Ask LLM which surfaces belong to same event
    2. Centroid-based: Cluster surfaces by centroid similarity
    3. Keyword-based: Group by shared named entities/keywords

    The interpreter sets node.event_id on topology nodes for downstream use.
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self._event_counter = 0

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"event_{self._event_counter:03d}"

    async def detect_events_llm(self, topo: Topology, min_mass: int = 5) -> List[Event]:
        """
        Use LLM to group surfaces into events.

        Only considers surfaces with mass >= min_mass to reduce noise.
        """
        if not self.llm:
            raise ValueError("LLM client required for LLM-based event detection")

        # Get significant surfaces
        surfaces = sorted(
            [s for s in topo.surfaces if s.size * s.total_sources >= min_mass],
            key=lambda s: -(s.size * s.total_sources)
        )

        if not surfaces:
            return []

        # Build surface descriptions for LLM
        descriptions = []
        for s in surfaces[:20]:  # Limit to top 20 by mass
            nodes = [topo.nodes[i] for i in s.node_indices[:3]]  # Sample claims
            sample_claims = "; ".join(n.text[:100] for n in nodes)
            descriptions.append(f"S{s.id}: {s.label} (claims: {sample_claims})")

        prompt = f"""Analyze these news surfaces and group them by real-world event.
Each surface contains related claims. Some surfaces may be about the SAME event
but discussing different aspects (e.g., death toll vs investigation vs response).

Surfaces:
{chr(10).join(descriptions)}

Return JSON with event groupings:
{{
  "events": [
    {{"name": "Hong Kong Tai Po Fire", "surfaces": [0, 5, 11, 16], "keywords": ["fire", "hong kong", "tai po"]}},
    {{"name": "Jimmy Lai Trial", "surfaces": [3], "keywords": ["jimmy lai", "trial"]}}
  ]
}}

Group surfaces by the SAME REAL-WORLD INCIDENT, not by topic similarity.
Only include surfaces that clearly belong to a specific event."""

        response = await self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=500
        )

        import json
        result = json.loads(response.choices[0].message.content)

        # Build Event objects
        events = []
        for ev in result.get("events", []):
            event_id = self._next_event_id()
            surface_ids = ev.get("surfaces", [])

            # Calculate totals
            total_nodes = 0
            total_sources = set()
            for sid in surface_ids:
                for s in surfaces:
                    if s.id == sid:
                        total_nodes += s.size
                        for ni in s.node_indices:
                            total_sources.update(topo.nodes[ni].sources)
                        # Set event_id on nodes
                        for ni in s.node_indices:
                            topo.nodes[ni].event_id = event_id
                        break

            events.append(Event(
                id=event_id,
                name=ev.get("name", f"Event {event_id}"),
                surface_ids=surface_ids,
                total_nodes=total_nodes,
                total_sources=len(total_sources),
                keywords=ev.get("keywords", []),
                confidence=0.8  # LLM-based detection has decent confidence
            ))

        return events

    def detect_events_centroid(
        self,
        topo: Topology,
        threshold: float = 0.45,
        min_mass: int = 5
    ) -> List[Event]:
        """
        Group surfaces into events by centroid similarity.

        This is a fallback when LLM is not available.
        Less accurate than LLM but faster and cheaper.
        """
        # Get significant surfaces
        surfaces = sorted(
            [s for s in topo.surfaces if s.size * s.total_sources >= min_mass],
            key=lambda s: -(s.size * s.total_sources)
        )

        if not surfaces:
            return []

        # Cluster surfaces by centroid similarity
        # Using simple greedy clustering
        assigned = set()
        events = []

        for s in surfaces:
            if s.id in assigned:
                continue

            # Start new event with this surface
            cluster = [s.id]
            assigned.add(s.id)

            # Find similar surfaces
            for s2 in surfaces:
                if s2.id in assigned:
                    continue
                if s.centroid and s2.centroid:
                    sim = cosine_similarity(s.centroid, s2.centroid)
                    if sim >= (1 - threshold):  # threshold is distance, so convert
                        cluster.append(s2.id)
                        assigned.add(s2.id)

            # Create event
            event_id = self._next_event_id()
            total_nodes = 0
            total_sources = set()

            for sid in cluster:
                for surf in surfaces:
                    if surf.id == sid:
                        total_nodes += surf.size
                        for ni in surf.node_indices:
                            total_sources.update(topo.nodes[ni].sources)
                            topo.nodes[ni].event_id = event_id
                        break

            # Use first surface label as event name
            first_surface = next(s for s in surfaces if s.id == cluster[0])

            events.append(Event(
                id=event_id,
                name=first_surface.label or f"Event {event_id}",
                surface_ids=cluster,
                total_nodes=total_nodes,
                total_sources=len(total_sources),
                confidence=0.5 if len(cluster) == 1 else 0.6
            ))

        return events

    async def detect_events(
        self,
        topo: Topology,
        method: str = "auto",
        **kwargs
    ) -> List[Event]:
        """
        Detect events using specified method.

        Args:
            topo: Computed topology
            method: "llm", "centroid", or "auto" (llm if available, else centroid)
            **kwargs: Passed to specific method

        Returns:
            List of detected events
        """
        if method == "auto":
            method = "llm" if self.llm else "centroid"

        if method == "llm":
            return await self.detect_events_llm(topo, **kwargs)
        elif method == "centroid":
            return self.detect_events_centroid(topo, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

    def summarize_events(self, events: List[Event]) -> str:
        """Generate human-readable summary of detected events."""
        if not events:
            return "No significant events detected."

        lines = ["Detected Events:", "-" * 40]

        for ev in sorted(events, key=lambda e: -e.mass):
            lines.append(f"\n{ev.name}")
            lines.append(f"  Surfaces: {len(ev.surface_ids)}")
            lines.append(f"  Claims: {ev.total_nodes}")
            lines.append(f"  Sources: {ev.total_sources}")
            lines.append(f"  Mass: {ev.mass}")
            if ev.keywords:
                lines.append(f"  Keywords: {', '.join(ev.keywords)}")

        return "\n".join(lines)
