"""
Principled Event Builder
========================

Builds events from surfaces using context compatibility and anti-trap rule.

This implements the principled emergence approach for L3 event formation:
1. Check context compatibility for shared entities
2. Apply anti-trap rule: cores require ≥2 constraints, ≥1 non-semantic
3. Use asymmetric membrane: block only when sure, allow periphery when underpowered
4. Motif sharing is positive structural evidence

THEORY:
- Context compatibility prevents percolation (mega-merges)
- Anti-trap rule prevents LLM-only similarity from forming cores
- Asymmetric membrane is honest about uncertainty

Usage:
    builder = PrincipledEventBuilder()
    events, ledger = await builder.build_from_surfaces(surfaces)
"""

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any

from ..types import (
    Surface, Event, Constraint, ConstraintType, ConstraintLedger,
    MembershipLevel, SurfaceMembership, EventSignature, EventJustification
)
from .surface_builder import context_compatible, ContextResult


@dataclass
class EventBuilderResult:
    """Result of event building."""
    events: Dict[str, Event]
    ledger: ConstraintLedger
    stats: Dict[str, Any]


class PrincipledEventBuilder:
    """
    Builds events from surfaces using context compatibility.

    This replaces the binary multi-signal gates in AboutnessScorer
    with a principled approach based on context compatibility.
    """

    def __init__(
        self,
        min_companions: int = 1,  # Lowered from 2 - surfaces are typically small
        overlap_threshold: float = 0.15
    ):
        self.min_companions = min_companions
        self.overlap_threshold = overlap_threshold
        self.ledger = ConstraintLedger()

    async def build_from_surfaces(
        self,
        surfaces: Dict[str, Surface],
        surface_ledger: ConstraintLedger = None,
        min_claims_for_event: int = 2,  # Only surfaces with 2+ claims become events
    ) -> EventBuilderResult:
        """
        Build events from surfaces using anchor-based grouping.

        Strategy:
        1. Filter to meaningful surfaces (multi-claim)
        2. Group surfaces by shared anchor entities
        3. Use context compatibility to validate merges
        4. Singleton surfaces don't become events (they're just surfaces)

        Args:
            surfaces: Dict of surface_id -> Surface
            surface_ledger: Optional ledger from surface building
            min_claims_for_event: Minimum claims for a surface to become an event

        Returns:
            EventBuilderResult with events and ledger
        """
        self.ledger = ConstraintLedger()

        # Import constraints from surface ledger
        if surface_ledger:
            for c in surface_ledger.constraints:
                self.ledger.add(c, scope=c.scope)

        # Step 1: Filter to meaningful surfaces
        meaningful_surfaces = {
            sid: s for sid, s in surfaces.items()
            if len(s.claim_ids) >= min_claims_for_event
        }

        # Step 2: Group by shared anchor entities
        events = self._form_anchor_based_events(meaningful_surfaces)

        # Step 3: Compute event properties
        for event in events.values():
            self._compute_event_properties(event, surfaces)

        stats = {
            "surfaces": len(surfaces),
            "meaningful_surfaces": len(meaningful_surfaces),
            "events": len(events),
            "core_edges": sum(1 for e in events.values() if len(e.core_surfaces()) > 1),
            "largest_core": max((len(e.core_surfaces()) for e in events.values()), default=0),
            "constraints": len(self.ledger.constraints),
        }

        return EventBuilderResult(
            events=events,
            ledger=self.ledger,
            stats=stats
        )

    def _form_anchor_based_events(
        self,
        surfaces: Dict[str, Surface]
    ) -> Dict[str, Event]:
        """
        Form events by grouping surfaces that share anchor entities.

        An event is a connected component of surfaces linked by shared anchors.
        """
        # Build entity -> surfaces mapping
        entity_to_surfaces: Dict[str, Set[str]] = {}
        for sid, s in surfaces.items():
            for entity in s.entities:
                if entity not in entity_to_surfaces:
                    entity_to_surfaces[entity] = set()
                entity_to_surfaces[entity].add(sid)

        # Build surface adjacency based on shared entities
        surface_edges: Dict[str, Set[str]] = {sid: set() for sid in surfaces}
        for entity, sids in entity_to_surfaces.items():
            if len(sids) >= 2:
                sids_list = list(sids)
                for i, s1 in enumerate(sids_list):
                    for s2 in sids_list[i+1:]:
                        # Check context compatibility before linking
                        result = context_compatible(
                            entity, surfaces[s1], surfaces[s2],
                            min_companions=self.min_companions,
                            overlap_threshold=self.overlap_threshold
                        )
                        # Allow link if compatible OR underpowered (benefit of doubt)
                        if result.compatible or result.underpowered:
                            surface_edges[s1].add(s2)
                            surface_edges[s2].add(s1)

                            # Record constraint
                            pair_key = f"{min(s1, s2)}:{max(s1, s2)}"
                            self.ledger.add(Constraint(
                                constraint_type=ConstraintType.STRUCTURAL,
                                assertion=f"Surfaces share anchor '{entity}'",
                                evidence={
                                    "entity": entity,
                                    "compatible": result.compatible,
                                    "underpowered": result.underpowered,
                                    "overlap": result.overlap,
                                },
                                provenance="anchor_based_event"
                            ), scope=pair_key)

        # Find connected components
        visited = set()
        events = {}
        event_idx = 0

        for surface_id in surfaces:
            if surface_id in visited:
                continue

            # BFS to find component
            component = set()
            queue = [surface_id]

            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                component.add(curr)
                queue.extend(surface_edges[curr] - visited)

            # Create event
            event_id = f"E{event_idx:03d}"
            event = Event(
                id=event_id,
                surface_ids=component
            )

            # All surfaces in anchor-based events are core
            for sid in component:
                event.memberships[sid] = SurfaceMembership(
                    surface_id=sid,
                    level=MembershipLevel.CORE,
                    score=1.0,
                    evidence={"formation": "anchor_based"}
                )

            events[event_id] = event
            event_idx += 1

        return events

    def _compute_compatibility(
        self,
        surfaces: Dict[str, Surface]
    ) -> Dict[Tuple[str, str], Tuple[bool, bool, List[ContextResult]]]:
        """
        Compute context compatibility for all surface pairs.

        Returns:
            Dict[(s1, s2)] -> (is_compatible, is_underpowered, [ContextResult])
        """
        compatibility = {}

        surface_ids = list(surfaces.keys())

        for i, s1_id in enumerate(surface_ids):
            for s2_id in surface_ids[i+1:]:
                s1 = surfaces[s1_id]
                s2 = surfaces[s2_id]

                # Find shared entities
                shared = s1.entities & s2.entities
                if not shared:
                    continue

                results = []
                any_compatible = False
                all_underpowered = True

                for entity in shared:
                    result = context_compatible(
                        entity, s1, s2,
                        min_companions=self.min_companions,
                        overlap_threshold=self.overlap_threshold
                    )
                    results.append(result)

                    if result.underpowered:
                        # Emit meta-claim for audit
                        pair_key = f"{s1_id}:{s2_id}"
                        self.ledger.add(Constraint(
                            constraint_type=ConstraintType.META,
                            assertion=f"Context test underpowered for '{entity}'",
                            evidence={
                                "entity": entity,
                                "companions1_size": result.companions1_size,
                                "companions2_size": result.companions2_size,
                                "min_required": self.min_companions
                            },
                            provenance="context_compatibility_underpowered"
                        ), scope=pair_key)
                    else:
                        all_underpowered = False

                        if result.compatible:
                            any_compatible = True
                            # Add structural constraint
                            pair_key = f"{s1_id}:{s2_id}"
                            self.ledger.add(Constraint(
                                constraint_type=ConstraintType.STRUCTURAL,
                                assertion=f"Entity '{entity}' has compatible context",
                                evidence={
                                    "entity": entity,
                                    "overlap": result.overlap,
                                    "companions1_size": result.companions1_size,
                                    "companions2_size": result.companions2_size,
                                },
                                provenance="context_compatibility"
                            ), scope=pair_key)

                compatibility[(s1_id, s2_id)] = (any_compatible, all_underpowered, results)

        return compatibility

    def _form_events(
        self,
        surfaces: Dict[str, Surface],
        compatibility: Dict[Tuple[str, str], Tuple[bool, bool, List[ContextResult]]]
    ) -> Dict[str, Event]:
        """
        Form events using asymmetric membrane rule.

        ASYMMETRIC MEMBRANE RULE:
        - Incompatible (overlap < threshold with sufficient data) → BLOCK
        - Compatible (overlap >= threshold) → allow core if anti-trap passes
        - Underpowered BUT has motif constraints → can form core
        - Underpowered with no motif → periphery only
        """
        core_edges = []
        periphery_edges = []
        blocked_edges = []

        for (s1_id, s2_id), (is_compatible, is_underpowered, results) in compatibility.items():
            pair_key = f"{s1_id}:{s2_id}"
            constraints = self.ledger.for_scope(pair_key)

            # Count constraint types
            structural = [c for c in constraints if c.constraint_type == ConstraintType.STRUCTURAL]
            motif_constraints = [c for c in structural if "motif" in c.provenance]

            # ASYMMETRIC RULE:

            # 1. If incompatible (powered + overlap < threshold) → BLOCK
            if not is_compatible and not is_underpowered:
                blocked_edges.append((s1_id, s2_id))
                continue

            # 2. If compatible (powered + overlap >= threshold) → core candidate
            if is_compatible:
                can_core, reason = self.ledger.can_form_core(s1_id, s2_id)
                if can_core:
                    core_edges.append((s1_id, s2_id))
                else:
                    periphery_edges.append((s1_id, s2_id))
                continue

            # 3. If underpowered → check for motif evidence
            if is_underpowered and motif_constraints:
                # Have motif evidence despite sparse context
                core_edges.append((s1_id, s2_id))
                continue

            # 4. Underpowered with no motif evidence → periphery only
            if is_underpowered:
                periphery_edges.append((s1_id, s2_id))

        # Form events from core edges (connected components)
        visited = set()
        events = {}
        event_idx = 0

        # Build adjacency from core edges only
        adj = defaultdict(set)
        for s1, s2 in core_edges:
            adj[s1].add(s2)
            adj[s2].add(s1)

        for surface_id in surfaces:
            if surface_id in visited:
                continue

            # BFS on core edges
            component = set()
            queue = [surface_id]

            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                component.add(curr)
                queue.extend(adj[curr] - visited)

            event_id = f"E{event_idx:03d}"
            event = Event(
                id=event_id,
                surface_ids=component
            )

            # Create memberships
            for sid in component:
                # Determine if core or not
                is_core = any(
                    (sid == s1 and s2 in component) or (sid == s2 and s1 in component)
                    for s1, s2 in core_edges
                )
                level = MembershipLevel.CORE if is_core else MembershipLevel.PERIPHERY

                event.memberships[sid] = SurfaceMembership(
                    surface_id=sid,
                    level=level,
                    score=1.0 if level == MembershipLevel.CORE else 0.5,
                    evidence={"formation": "principled_emergence"}
                )

            events[event_id] = event
            event_idx += 1

        # Attach periphery surfaces to events
        for s1, s2 in periphery_edges:
            e1 = self._find_event(events, s1)
            e2 = self._find_event(events, s2)

            if e1 and e2 and e1 != e2:
                # Cross-event periphery connection - don't merge
                pass

        return events

    def _find_event(self, events: Dict[str, Event], surface_id: str) -> Optional[str]:
        """Find which event contains a surface."""
        for eid, event in events.items():
            if surface_id in event.surface_ids:
                return eid
        return None

    def _compute_event_properties(
        self,
        event: Event,
        surfaces: Dict[str, Surface]
    ):
        """
        Compute derived properties for an event.
        """
        all_entities = set()
        all_anchors = set()
        all_sources = set()
        times = []
        total_claims = 0

        for sid in event.surface_ids:
            surface = surfaces.get(sid)
            if surface:
                all_entities.update(surface.entities)
                all_anchors.update(surface.anchor_entities)
                all_sources.update(surface.sources)
                total_claims += len(surface.claim_ids)

                if surface.time_window[0]:
                    times.append(surface.time_window[0])
                if surface.time_window[1]:
                    times.append(surface.time_window[1])

        event.entities = all_entities
        event.anchor_entities = all_anchors
        event.total_claims = total_claims
        event.total_sources = len(all_sources)

        if times:
            event.time_window = (min(times), max(times))

        # Create signature
        event.signature = EventSignature(
            anchor_weights={a: 1.0 for a in all_anchors},
            entity_weights={e: 1.0 for e in all_entities},
            source_count=len(all_sources),
            time_window=event.time_window
        )

        # Create justification bundle
        event.justification = self._compute_justification(event, surfaces)

    def _compute_justification(
        self,
        event: Event,
        surfaces: Dict[str, Surface]
    ) -> EventJustification:
        """
        Compute the justification bundle for an event.

        This provides full explainability:
        1. Membrane proof (why grouped)
        2. What happened (representative surfaces)
        3. Why rejected (blocked bridges)
        """
        justification = EventJustification()

        # === MEMBRANE PROOF ===

        # Collect core motifs from constraints
        core_motifs = []
        context_passes = []
        blocked_bridges = []
        underpowered_edges = []

        for sid in event.surface_ids:
            for other_sid in event.surface_ids:
                if sid >= other_sid:
                    continue

                pair_key = f"{min(sid, other_sid)}:{max(sid, other_sid)}"
                constraints = self.ledger.for_scope(pair_key)

                for c in constraints:
                    if c.constraint_type == ConstraintType.STRUCTURAL:
                        if "motif" in c.provenance.lower():
                            # Core motif evidence
                            motif_ents = c.evidence.get("motif_entities", [])
                            support = c.evidence.get("support", 0)
                            if motif_ents:
                                core_motifs.append({
                                    "entities": motif_ents,
                                    "support": support,
                                })
                        elif "context" in c.provenance.lower() or "anchor" in c.provenance.lower():
                            # Context compatibility pass
                            context_passes.append({
                                "surface1": sid,
                                "surface2": other_sid,
                                "overlap": c.evidence.get("overlap", 0),
                                "status": "compatible" if c.evidence.get("compatible") else "underpowered",
                                "entity": c.evidence.get("entity", "")
                            })

                    elif c.constraint_type == ConstraintType.META:
                        if "underpowered" in c.provenance.lower():
                            underpowered_edges.append({
                                "entity": c.evidence.get("entity", ""),
                                "reason": f"companions too sparse ({c.evidence.get('companions1_size', 0)}, {c.evidence.get('companions2_size', 0)})",
                                "treated_as": "periphery"
                            })

        # Deduplicate motifs
        seen_motifs = set()
        unique_motifs = []
        for m in core_motifs:
            key = frozenset(m["entities"])
            if key not in seen_motifs:
                seen_motifs.add(key)
                unique_motifs.append(m)

        justification.core_motifs = unique_motifs[:10]  # Top 10
        justification.context_passes = context_passes[:20]
        justification.underpowered_edges = underpowered_edges[:10]

        # === WHAT HAPPENED ===

        # Representative surfaces: pick by claim count (mass)
        surface_list = [(sid, surfaces.get(sid)) for sid in event.surface_ids]
        surface_list = [(sid, s) for sid, s in surface_list if s]
        surface_list.sort(key=lambda x: -len(x[1].claim_ids))

        rep_surfaces = surface_list[:3]  # Top 3 by mass
        justification.representative_surfaces = [sid for sid, _ in rep_surfaces]

        # Get titles and key facts from surfaces
        for sid, s in rep_surfaces:
            if s.canonical_title:
                justification.representative_titles.append(s.canonical_title)
            elif s.description:
                justification.representative_titles.append(s.description[:100])
            else:
                # Fallback: entities + claim count
                ents = list(s.entities)[:3]
                justification.representative_titles.append(
                    f"{', '.join(ents)} ({len(s.claim_ids)} claims)"
                )

            if s.key_facts:
                justification.representative_facts.extend(s.key_facts[:2])

        # === CANONICAL PROPOSITION HANDLE ===

        # Create handle from representative surface + time
        if rep_surfaces:
            primary_sid, primary = rep_surfaces[0]

            # Try to build: action + place + time
            entities = sorted(primary.entities)[:3]
            time_str = ""
            if event.time_window[0]:
                time_str = f" ({event.time_window[0].strftime('%Y-%m-%d')})"

            if primary.canonical_title:
                justification.canonical_handle = primary.canonical_title + time_str
            else:
                # Fallback to entities
                justification.canonical_handle = f"{', '.join(entities)}{time_str}"

            justification.handle_citations = [primary_sid]

        return justification
