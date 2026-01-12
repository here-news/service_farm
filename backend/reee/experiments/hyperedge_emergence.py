"""
Hyperedge Emergence Experiment
==============================

Validates the higher-order theory: events emerge from repeated motifs (k≥2),
not singleton entity overlap.

Theory:
- Each claim defines a hyperedge over its entity set
- A motif is a k-set (k≥2) that appears in multiple claims
- Event cores form around supported motifs
- Singletons (k=1) can attach but never merge cores

This directly addresses:
- Co-incident hubs ("United States" can't merge unrelated events)
- Percolation resistance (bridges require motif support, not singleton)
- Principled event membranes (coherent multi-way scenes)

Usage:
    docker exec herenews-app python -m reee.experiments.hyperedge_emergence
"""

import asyncio
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Set, Tuple, Optional, Any
import math

# For running in container
import sys
sys.path.insert(0, '/app')


@dataclass
class Hyperedge:
    """
    A claim viewed as a hyperedge over entities.

    The claim text is evidence; the entity set is structure.
    """
    claim_id: str
    entities: frozenset  # The hyperedge connects these nodes
    source: str
    time: Optional[datetime]
    text: str

    # Typed extraction (if available)
    question_key: Optional[str] = None
    extracted_value: Optional[Any] = None

    def k(self) -> int:
        """Hyperedge degree (number of entities)."""
        return len(self.entities)

    def subsets(self, min_k: int = 2) -> List[frozenset]:
        """All k-subsets with k >= min_k."""
        result = []
        for k in range(min_k, len(self.entities) + 1):
            for combo in combinations(self.entities, k):
                result.append(frozenset(combo))
        return result


@dataclass
class Motif:
    """
    A supported k-set that appears in multiple hyperedges.

    Motifs are the binding signal for event formation.
    Higher support = stronger evidence of coherent scene.
    """
    entities: frozenset
    support: int  # Number of hyperedges containing this motif
    hyperedge_ids: Set[str] = field(default_factory=set)
    sources: Set[str] = field(default_factory=set)

    def k(self) -> int:
        return len(self.entities)

    def is_discriminative(self, total_hyperedges: int) -> bool:
        """A motif is discriminative if it doesn't appear everywhere."""
        # IDF-like: rare motifs are more discriminative
        idf = math.log(total_hyperedges / (self.support + 1))
        return idf > 1.0

    def binding_strength(self, total_hyperedges: int) -> float:
        """
        Combined signal: support * discriminativeness.

        High support + high IDF = strong binding evidence.
        """
        idf = math.log(total_hyperedges / (self.support + 1))
        # Normalize support contribution
        support_factor = min(self.support / 5.0, 1.0)  # Saturates at 5
        return support_factor * max(idf, 0.1)


@dataclass
class EmergentEvent:
    """
    An event discovered through motif clustering.

    Core = hyperedges connected by supported motifs
    The event "membrane" is the boundary of motif coherence.
    """
    id: str
    core_hyperedges: Set[str] = field(default_factory=set)
    periphery_hyperedges: Set[str] = field(default_factory=set)

    # What binds this event
    binding_motifs: List[Motif] = field(default_factory=list)

    # Aggregated properties
    all_entities: Set[str] = field(default_factory=set)
    sources: Set[str] = field(default_factory=set)
    time_range: Tuple[Optional[datetime], Optional[datetime]] = (None, None)

    # Meta-observations
    membrane_stability: float = 0.0  # How stable across threshold sweep
    hub_exposure: Set[str] = field(default_factory=set)  # Singleton-only entities

    def claim_count(self) -> int:
        return len(self.core_hyperedges) + len(self.periphery_hyperedges)


class HyperedgeEmergence:
    """
    Discovers events from claims using higher-order structure.

    Algorithm:
    1. Load claims as hyperedges
    2. Extract all k-subsets (k≥2)
    3. Count motif support
    4. Build motif adjacency graph
    5. Cluster into events via connected components on motif graph
    6. Attach periphery via weaker signals
    """

    def __init__(
        self,
        min_motif_support: int = 2,
        min_motif_k: int = 2,
        min_binding_strength: float = 0.3,
    ):
        self.min_motif_support = min_motif_support
        self.min_motif_k = min_motif_k
        self.min_binding_strength = min_binding_strength

        self.hyperedges: Dict[str, Hyperedge] = {}
        self.motifs: Dict[frozenset, Motif] = {}
        self.events: Dict[str, EmergentEvent] = {}

        # Diagnostics
        self.singleton_entities: Set[str] = set()  # Appear but never in motifs
        self.hub_entities: Set[str] = set()  # High-freq but low motif participation

    def add_hyperedge(self, he: Hyperedge):
        """Add a claim-as-hyperedge."""
        self.hyperedges[he.claim_id] = he

    def extract_motifs(self) -> Dict[frozenset, Motif]:
        """
        Extract all supported motifs from hyperedges.

        A motif must appear in >= min_support hyperedges.
        """
        # Count k-subset occurrences
        subset_counts: Dict[frozenset, Set[str]] = defaultdict(set)
        subset_sources: Dict[frozenset, Set[str]] = defaultdict(set)

        for he in self.hyperedges.values():
            for subset in he.subsets(min_k=self.min_motif_k):
                subset_counts[subset].add(he.claim_id)
                subset_sources[subset].add(he.source)

        # Filter to supported motifs
        self.motifs = {}
        for subset, claim_ids in subset_counts.items():
            if len(claim_ids) >= self.min_motif_support:
                self.motifs[subset] = Motif(
                    entities=subset,
                    support=len(claim_ids),
                    hyperedge_ids=claim_ids,
                    sources=subset_sources[subset],
                )

        # Identify singleton-only entities (appear but never in motifs)
        all_motif_entities = set()
        for m in self.motifs.values():
            all_motif_entities.update(m.entities)

        all_entities = set()
        for he in self.hyperedges.values():
            all_entities.update(he.entities)

        self.singleton_entities = all_entities - all_motif_entities

        return self.motifs

    def build_motif_adjacency(self) -> Dict[str, Set[str]]:
        """
        Build adjacency between hyperedges via shared motifs.

        Two hyperedges are adjacent if they share a supported motif.
        This is the key higher-order constraint: singleton overlap doesn't count.
        """
        adj: Dict[str, Set[str]] = defaultdict(set)

        n_total = len(self.hyperedges)

        for motif in self.motifs.values():
            # Check binding strength
            strength = motif.binding_strength(n_total)
            if strength < self.min_binding_strength:
                continue

            # All hyperedges sharing this motif are adjacent
            he_list = list(motif.hyperedge_ids)
            for i, he1 in enumerate(he_list):
                for he2 in he_list[i+1:]:
                    adj[he1].add(he2)
                    adj[he2].add(he1)

        return adj

    def _merge_by_discriminative_anchor(
        self,
        components: List[Set[str]],
        min_shared_motifs: int = 1,
        max_anchor_components: int = 6,  # Anchor in >6 components = hub
    ) -> List[Set[str]]:
        """
        Two-phase merge:

        Phase 1: Merge components that share a motif (strict)
        Phase 2: Merge components that share a discriminative anchor
                 (anchor appears in ≤max_anchor_components components)

        This handles the Hong Kong Fire case:
        - {Joe Chow, Wang Fuk Court} and {Hong Kong, Wang Fuk Court} don't share a motif
        - But they both have "Wang Fuk Court" as a discriminative anchor
        - Wang Fuk Court only appears in ~5 fire-related components (not a hub)
        - So they should merge

        While preventing:
        - "Hong Kong" appears in many components across topics → hub, no merge
        - "John Lee" appears in fire AND Jimmy Lai → bridges topics, no merge
        """
        if len(components) <= 1:
            return components

        # Phase 1: Get motifs per component
        component_motifs: List[Set[frozenset]] = []
        component_entities: List[Set[str]] = []

        for comp in components:
            motifs_in_comp = set()
            entities_in_comp = set()

            for motif_entities, motif in self.motifs.items():
                if motif.hyperedge_ids & comp:
                    motifs_in_comp.add(motif_entities)

            for he_id in comp:
                entities_in_comp.update(self.hyperedges[he_id].entities)

            component_motifs.append(motifs_in_comp)
            component_entities.append(entities_in_comp)

        # Phase 1: Build merge graph based on shared motifs
        n_comp = len(components)
        merge_adj: Dict[int, Set[int]] = defaultdict(set)

        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                shared_motifs = component_motifs[i] & component_motifs[j]
                if len(shared_motifs) >= min_shared_motifs:
                    merge_adj[i].add(j)
                    merge_adj[j].add(i)

        # Phase 2: Count component membership for each entity
        entity_component_count: Dict[str, int] = defaultdict(int)
        for ents in component_entities:
            for e in ents:
                entity_component_count[e] += 1

        # Phase 2: Add edges for discriminative anchor sharing
        # Only if anchor is NOT a hub (appears in limited components)
        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                if j in merge_adj[i]:
                    continue  # Already connected by motif

                shared_entities = component_entities[i] & component_entities[j]

                for e in shared_entities:
                    # Is this entity discriminative (not a hub)?
                    if entity_component_count[e] <= max_anchor_components:
                        # Additional check: entity must be in a motif in BOTH components
                        # (not just floating as a singleton)
                        in_motif_i = any(e in m for m in component_motifs[i])
                        in_motif_j = any(e in m for m in component_motifs[j])

                        if in_motif_i and in_motif_j:
                            merge_adj[i].add(j)
                            merge_adj[j].add(i)
                            break

        # Phase 3: Find connected components in merge graph
        visited = set()
        merged_components = []

        for i in range(n_comp):
            if i in visited:
                continue

            merged = set()
            queue = [i]
            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                merged.update(components[curr])
                queue.extend(merge_adj[curr] - visited)

            merged_components.append(merged)

        # Store for diagnostics
        self._component_motifs = component_motifs
        self._entity_component_count = entity_component_count

        return merged_components

    def cluster_events(self, merge_by_anchor: bool = True) -> Dict[str, EmergentEvent]:
        """
        Cluster hyperedges into events via motif adjacency.

        Phase 1: Connected components on motif graph (cores)
        Phase 2: Merge components that share a discriminative anchor
        Phase 3: Attach periphery via weaker signals

        The anchor merge (Phase 2) is key for the Hong Kong Fire case:
        - Multiple small motif-clusters all mention "Wang Fuk Court"
        - Wang Fuk Court has low motif ratio (hub-like) BUT
        - It's actually discriminative (only appears in fire context)
        - So we merge components that share it
        """
        if not self.motifs:
            self.extract_motifs()

        adj = self.build_motif_adjacency()

        # Phase 1: Find connected components on motif graph
        visited = set()
        components = []

        for he_id in self.hyperedges:
            if he_id in visited:
                continue

            # BFS from this hyperedge
            component = set()
            queue = [he_id]

            while queue:
                curr = queue.pop(0)
                if curr in visited:
                    continue
                visited.add(curr)
                component.add(curr)
                queue.extend(adj[curr] - visited)

            if len(component) >= 2:  # Only events with 2+ hyperedges
                components.append(component)

        # Phase 2: Merge components by shared discriminative anchor
        # An anchor is discriminative if IDF > threshold (not a global hub)
        if merge_by_anchor and len(components) > 1:
            components = self._merge_by_discriminative_anchor(components)

        # Phase 3: Build event objects
        self.events = {}
        for i, component in enumerate(components):
            event_id = f"ev_{i:03d}"

            # Find binding motifs for this event
            binding_motifs = []
            for motif in self.motifs.values():
                overlap = motif.hyperedge_ids & component
                if len(overlap) >= 2:  # Motif binds at least 2 hyperedges in event
                    binding_motifs.append(motif)

            # Sort by binding strength
            n_total = len(self.hyperedges)
            binding_motifs.sort(
                key=lambda m: m.binding_strength(n_total),
                reverse=True
            )

            # Aggregate properties
            all_entities = set()
            sources = set()
            min_time, max_time = None, None

            for he_id in component:
                he = self.hyperedges[he_id]
                all_entities.update(he.entities)
                sources.add(he.source)
                if he.time:
                    if min_time is None or he.time < min_time:
                        min_time = he.time
                    if max_time is None or he.time > max_time:
                        max_time = he.time

            # Identify hub exposure (entities that appear but aren't in binding motifs)
            motif_entities = set()
            for m in binding_motifs:
                motif_entities.update(m.entities)
            hub_exposure = all_entities - motif_entities

            self.events[event_id] = EmergentEvent(
                id=event_id,
                core_hyperedges=component,
                binding_motifs=binding_motifs[:10],  # Top 10
                all_entities=all_entities,
                sources=sources,
                time_range=(min_time, max_time),
                hub_exposure=hub_exposure,
            )

        return self.events

    def analyze_hub_prevention(self) -> Dict[str, Any]:
        """
        Analyze how the motif rule prevents hub-driven mega-merges.

        Returns evidence that singletons don't collapse structure.
        """
        # Find entities that appear in many hyperedges but few motifs
        entity_hyperedge_count: Dict[str, int] = defaultdict(int)
        entity_motif_count: Dict[str, int] = defaultdict(int)

        for he in self.hyperedges.values():
            for e in he.entities:
                entity_hyperedge_count[e] += 1

        for motif in self.motifs.values():
            for e in motif.entities:
                entity_motif_count[e] += 1

        # Hub ratio: high hyperedge count / low motif participation
        hub_candidates = []
        for entity, he_count in entity_hyperedge_count.items():
            motif_count = entity_motif_count.get(entity, 0)
            if he_count >= 5:  # Frequent entity
                ratio = motif_count / he_count if he_count > 0 else 0
                hub_candidates.append({
                    'entity': entity,
                    'hyperedge_count': he_count,
                    'motif_count': motif_count,
                    'motif_ratio': ratio,
                    'is_hub': ratio < 0.3,  # Low motif participation = hub
                })

        hub_candidates.sort(key=lambda x: x['hyperedge_count'], reverse=True)

        return {
            'total_entities': len(entity_hyperedge_count),
            'singleton_only': len(self.singleton_entities),
            'hub_candidates': hub_candidates[:15],
            'events_formed': len(self.events),
        }

    def print_report(self):
        """Print emergence analysis report."""
        print("=" * 80)
        print("HYPEREDGE EMERGENCE REPORT")
        print("=" * 80)

        print(f"\n[L0] Hyperedges (claims): {len(self.hyperedges)}")
        print(f"[L1] Motifs (k≥{self.min_motif_k}, support≥{self.min_motif_support}): {len(self.motifs)}")
        print(f"[L2] Emergent Events: {len(self.events)}")

        # Top motifs
        print("\n" + "-" * 40)
        print("TOP BINDING MOTIFS")
        print("-" * 40)

        n_total = len(self.hyperedges)
        sorted_motifs = sorted(
            self.motifs.values(),
            key=lambda m: m.binding_strength(n_total),
            reverse=True
        )

        for m in sorted_motifs[:10]:
            strength = m.binding_strength(n_total)
            print(f"  {set(m.entities)}")
            print(f"    support={m.support}, k={m.k()}, strength={strength:.2f}")

        # Events
        print("\n" + "-" * 40)
        print("EMERGENT EVENTS")
        print("-" * 40)

        sorted_events = sorted(
            self.events.values(),
            key=lambda e: len(e.core_hyperedges),
            reverse=True
        )

        for ev in sorted_events[:10]:
            print(f"\n[{ev.id}] {len(ev.core_hyperedges)} claims, {len(ev.sources)} sources")
            print(f"  Binding motifs: {len(ev.binding_motifs)}")
            if ev.binding_motifs:
                top_motif = ev.binding_motifs[0]
                print(f"  Top: {set(top_motif.entities)} (support={top_motif.support})")
            print(f"  Hub exposure: {ev.hub_exposure}")

        # Hub analysis
        print("\n" + "-" * 40)
        print("HUB PREVENTION ANALYSIS")
        print("-" * 40)

        analysis = self.analyze_hub_prevention()
        print(f"  Total entities: {analysis['total_entities']}")
        print(f"  Singleton-only (no motif participation): {analysis['singleton_only']}")
        print(f"  Events formed: {analysis['events_formed']}")

        print("\n  Top frequent entities:")
        for h in analysis['hub_candidates'][:10]:
            status = "HUB" if h['is_hub'] else "backbone"
            print(f"    {h['entity']}: {h['hyperedge_count']} claims, "
                  f"{h['motif_count']} motifs, ratio={h['motif_ratio']:.2f} [{status}]")


async def load_hyperedges_from_neo4j(neo4j, limit: int = 500) -> List[Hyperedge]:
    """Load claims as hyperedges from Neo4j."""

    claims = await neo4j._execute_read('''
        MATCH (c:Claim)
        WHERE c.text IS NOT NULL
        OPTIONAL MATCH (p:Page)-[:EMITS]->(c)
        OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
        WITH c, p, collect(DISTINCT e.canonical_name) as entity_names
        WHERE size(entity_names) >= 2
        RETURN c.id as id,
               c.text as text,
               p.domain as source,
               p.published_at as time,
               entity_names
        ORDER BY p.published_at DESC
        LIMIT $limit
    ''', {'limit': limit})

    hyperedges = []
    for row in claims:
        entities = frozenset(e for e in row['entity_names'] if e)
        if len(entities) >= 2:
            hyperedges.append(Hyperedge(
                claim_id=row['id'],
                entities=entities,
                source=row['source'] or 'unknown',
                time=row['time'],
                text=row['text'],
            ))

    return hyperedges


async def main():
    from services.neo4j_service import Neo4jService

    print("Loading hyperedges from Neo4j...")
    neo4j = Neo4jService()
    await neo4j.connect()

    hyperedges = await load_hyperedges_from_neo4j(neo4j, limit=500)
    print(f"Loaded {len(hyperedges)} hyperedges")

    await neo4j.close()

    # Run emergence
    emergence = HyperedgeEmergence(
        min_motif_support=2,
        min_motif_k=2,
        min_binding_strength=0.2,
    )

    for he in hyperedges:
        emergence.add_hyperedge(he)

    emergence.extract_motifs()
    emergence.cluster_events()
    emergence.print_report()

    # Show sample claims from top event
    if emergence.events:
        top_event = max(emergence.events.values(), key=lambda e: len(e.core_hyperedges))
        print("\n" + "=" * 80)
        print(f"SAMPLE CLAIMS FROM TOP EVENT [{top_event.id}]")
        print("=" * 80)

        for he_id in list(top_event.core_hyperedges)[:5]:
            he = emergence.hyperedges[he_id]
            print(f"\n[{he.source}]")
            print(f"  Entities: {set(he.entities)}")
            print(f"  Text: {he.text[:150]}...")


if __name__ == "__main__":
    asyncio.run(main())
