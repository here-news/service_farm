"""
Progressive Recursive Emergence v2

Fixed: Allow clusters to GROW by absorbing, not just pair-merge.

At each layer:
1. Find clusters with enough energy to grow
2. Find unmerged EUs that can join existing clusters
3. Let clusters absorb compatible EUs
4. Also allow new clusters to form from high-energy pairs

Run inside container:
    docker exec herenews-app python /app/test_eu/progressive_emergence_v2.py
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import heapq

from load_graph import load_snapshot, GraphSnapshot, ClaimData


@dataclass
class EU:
    """EventfulUnit - recursive structure."""
    id: str
    depth: int

    # If leaf
    claim_id: Optional[str] = None
    text: Optional[str] = None

    # If composite
    child_ids: List[str] = field(default_factory=list)

    # Properties
    entity_ids: Set[str] = field(default_factory=set)
    page_ids: Set[str] = field(default_factory=set)
    internal_corr: int = 0
    internal_contra: int = 0
    mass: float = 0.0
    label: str = ""

    def is_leaf(self) -> bool:
        return self.claim_id is not None


class EURegistry:
    """Registry of all EUs."""
    def __init__(self, snapshot: GraphSnapshot):
        self.snapshot = snapshot
        self.eus: Dict[str, EU] = {}
        self.by_depth: Dict[int, List[str]] = defaultdict(list)
        self._claim_cache: Dict[str, Set[str]] = {}

    def add(self, eu: EU):
        self.eus[eu.id] = eu
        self.by_depth[eu.depth].append(eu.id)
        self._claim_cache.pop(eu.id, None)

    def get(self, eu_id: str) -> Optional[EU]:
        return self.eus.get(eu_id)

    def get_layer(self, depth: int) -> List[EU]:
        return [self.eus[eid] for eid in self.by_depth[depth] if eid in self.eus]

    def all_claim_ids(self, eu_id: str) -> Set[str]:
        if eu_id in self._claim_cache:
            return self._claim_cache[eu_id]

        eu = self.eus.get(eu_id)
        if not eu:
            return set()
        if eu.is_leaf():
            result = {eu.claim_id}
        else:
            result = set()
            for child_id in eu.child_ids:
                result |= self.all_claim_ids(child_id)

        self._claim_cache[eu_id] = result
        return result

    def claim_count(self, eu_id: str) -> int:
        return len(self.all_claim_ids(eu_id))


def compute_energy_between(
    claims1: Set[str],
    claims2: Set[str],
    snapshot: GraphSnapshot
) -> float:
    """Energy from corroboration/contradiction between two claim sets."""
    cross_corr = 0
    cross_contra = 0

    for cid in claims1:
        claim = snapshot.claims.get(cid)
        if claim:
            cross_corr += len([c for c in claim.corroborates_ids if c in claims2])
            cross_contra += len([c for c in claim.contradicts_ids if c in claims2])

    return cross_corr * 2.0 + cross_contra * 1.0


def compute_entity_cohesion(eu1: EU, eu2: EU) -> float:
    """Entity overlap between two EUs."""
    if not eu1.entity_ids or not eu2.entity_ids:
        return 0.0
    shared = len(eu1.entity_ids & eu2.entity_ids)
    return shared * 0.5


def compute_total_energy(
    eu1: EU,
    eu2: EU,
    registry: EURegistry
) -> float:
    """Total energy from combining two EUs."""
    claims1 = registry.all_claim_ids(eu1.id)
    claims2 = registry.all_claim_ids(eu2.id)

    corr_energy = compute_energy_between(claims1, claims2, registry.snapshot)
    cohesion_energy = compute_entity_cohesion(eu1, eu2)

    return corr_energy + cohesion_energy


def create_layer_0(registry: EURegistry):
    """Layer 0: leaf EUs from claims."""
    for cid, claim in registry.snapshot.claims.items():
        eu = EU(
            id=f"L0_{cid}",
            depth=0,
            claim_id=cid,
            text=claim.text,
            entity_ids=set(claim.entity_ids),
            page_ids={claim.page_id} if claim.page_id else set(),
            mass=0.1,
            label=claim.text[:40] + "..."
        )
        registry.add(eu)

    print(f"Layer 0: {len(registry.get_layer(0))} leaf EUs")


def compute_cluster_mass(claims: Set[str], snapshot: GraphSnapshot) -> Tuple[float, int, int]:
    """Compute mass for a set of claims. Returns (mass, internal_corr, internal_contra)."""
    internal_corr = 0
    internal_contra = 0

    for cid in claims:
        claim = snapshot.claims.get(cid)
        if claim:
            internal_corr += len([c for c in claim.corroborates_ids if c in claims])
            internal_contra += len([c for c in claim.contradicts_ids if c in claims])

    internal_corr //= 2
    internal_contra //= 2

    n = len(claims)
    base_mass = n * 0.1
    possible_pairs = n * (n - 1) / 2 if n > 1 else 1
    coherence = internal_corr / possible_pairs if possible_pairs > 0 else 0
    tension_bonus = min(0.5, internal_contra * 0.05)

    mass = base_mass * (1.0 + coherence + tension_bonus)

    return mass, internal_corr, internal_contra


def find_label(claims: Set[str], snapshot: GraphSnapshot) -> str:
    """Find best label for a cluster (most common entity)."""
    entity_counts = defaultdict(int)
    for cid in claims:
        claim = snapshot.claims.get(cid)
        if claim:
            for eid in claim.entity_ids:
                entity_counts[eid] += 1

    if entity_counts:
        top_id = max(entity_counts.items(), key=lambda x: x[1])[0]
        entity = snapshot.entities.get(top_id)
        if entity:
            return entity.canonical_name

    return "cluster"


def merge_eus(eus: List[EU], registry: EURegistry, depth: int, merge_id: int) -> EU:
    """Create a new EU from merging multiple EUs."""
    all_claims = set()
    all_entities = set()
    all_pages = set()
    child_ids = []

    for eu in eus:
        all_claims |= registry.all_claim_ids(eu.id)
        all_entities |= eu.entity_ids
        all_pages |= eu.page_ids
        child_ids.append(eu.id)

    mass, internal_corr, internal_contra = compute_cluster_mass(all_claims, registry.snapshot)
    label = find_label(all_claims, registry.snapshot)

    return EU(
        id=f"L{depth}_{merge_id}",
        depth=depth,
        child_ids=child_ids,
        entity_ids=all_entities,
        page_ids=all_pages,
        internal_corr=internal_corr,
        internal_contra=internal_contra,
        mass=mass,
        label=label
    )


def emerge_layer_v2(
    registry: EURegistry,
    source_depth: int,
    min_energy: float = 1.0
) -> int:
    """
    Emerge next layer with absorption-based growth.

    Algorithm:
    1. Compute all pairwise energies
    2. Build clusters greedily: start with highest energy pair, absorb compatible EUs
    3. Continue until no more high-energy connections
    """
    source_eus = registry.get_layer(source_depth)
    target_depth = source_depth + 1
    n = len(source_eus)

    if n < 2:
        return 0

    # Compute all pairwise energies (only above threshold)
    edges = []  # (energy, i, j)
    for i in range(n):
        for j in range(i + 1, n):
            energy = compute_total_energy(source_eus[i], source_eus[j], registry)
            if energy >= min_energy:
                edges.append((-energy, i, j))  # Negative for min-heap (we want max)

    if not edges:
        # No merges possible, just pass through
        for i, eu in enumerate(source_eus):
            passthrough = EU(
                id=f"L{target_depth}_p{i}",
                depth=target_depth,
                child_ids=[eu.id],
                entity_ids=eu.entity_ids.copy(),
                page_ids=eu.page_ids.copy(),
                internal_corr=eu.internal_corr,
                internal_contra=eu.internal_contra,
                mass=eu.mass,
                label=eu.label
            )
            registry.add(passthrough)
        return 0

    heapq.heapify(edges)

    # Union-Find for cluster building
    parent = list(range(n))
    rank = [0] * n
    cluster_members: Dict[int, Set[int]] = {i: {i} for i in range(n)}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        cluster_members[px] |= cluster_members[py]
        del cluster_members[py]
        return True

    # Process edges in order of energy
    merges_made = 0
    while edges:
        neg_energy, i, j = heapq.heappop(edges)
        energy = -neg_energy

        if energy < min_energy:
            break

        if union(i, j):
            merges_made += 1

    # Build clusters from union-find results
    merge_id = 0
    processed = set()

    for i in range(n):
        root = find(i)
        if root in processed:
            continue
        processed.add(root)

        members = cluster_members.get(root, {root})
        member_eus = [source_eus[m] for m in members]

        if len(member_eus) == 1:
            # Passthrough
            eu = member_eus[0]
            passthrough = EU(
                id=f"L{target_depth}_p{merge_id}",
                depth=target_depth,
                child_ids=[eu.id],
                entity_ids=eu.entity_ids.copy(),
                page_ids=eu.page_ids.copy(),
                internal_corr=eu.internal_corr,
                internal_contra=eu.internal_contra,
                mass=eu.mass,
                label=eu.label
            )
            registry.add(passthrough)
        else:
            # Real merge
            merged = merge_eus(member_eus, registry, target_depth, merge_id)
            registry.add(merged)

        merge_id += 1

    return merges_made


def progressive_emerge_v2(
    snapshot: GraphSnapshot,
    max_depth: int = 6
) -> Tuple[EURegistry, Dict]:
    """Progressive emergence with absorption-based growth."""

    registry = EURegistry(snapshot)

    # Energy thresholds decrease at higher layers (bigger clusters have more edges)
    min_energy_by_depth = {
        0: 1.0,
        1: 0.8,
        2: 0.6,
        3: 0.5,
        4: 0.4,
        5: 0.3,
    }

    stats = {'layers': []}

    create_layer_0(registry)
    stats['layers'].append({'depth': 0, 'eu_count': len(registry.get_layer(0)), 'merges': 0})

    for depth in range(max_depth):
        min_energy = min_energy_by_depth.get(depth, 0.3)
        merges = emerge_layer_v2(registry, depth, min_energy)

        layer_eus = registry.get_layer(depth + 1)
        real_clusters = [eu for eu in layer_eus if len(eu.child_ids) > 1]

        stats['layers'].append({
            'depth': depth + 1,
            'eu_count': len(layer_eus),
            'real_clusters': len(real_clusters),
            'merges': merges
        })

        print(f"Layer {depth + 1}: {merges} merges → {len(real_clusters)} real clusters, {len(layer_eus)} total")

        if merges == 0:
            print(f"  No more merges, stopping.")
            break

    return registry, stats


def analyze_and_compare(registry: EURegistry) -> Tuple[Dict, List[Dict]]:
    """Analyze hierarchy and compare with events."""
    snapshot = registry.snapshot

    # Find all real clusters (not passthroughs)
    all_clusters = []
    max_depth = max(registry.by_depth.keys())

    for depth in range(1, max_depth + 1):
        for eu in registry.get_layer(depth):
            if len(eu.child_ids) > 1:
                all_clusters.append(eu)

    # Analysis by depth
    analysis = {'by_depth': {}}
    for depth in registry.by_depth.keys():
        eus = registry.get_layer(depth)
        real = [eu for eu in eus if len(eu.child_ids) != 1]
        if real:
            analysis['by_depth'][depth] = {
                'count': len(real),
                'top': sorted(
                    [{'label': eu.label, 'mass': eu.mass, 'claims': registry.claim_count(eu.id)} for eu in real],
                    key=lambda x: x['mass'], reverse=True
                )[:5]
            }

    # Compare with existing events
    comparisons = []
    for event in snapshot.events.values():
        event_claims = set(event.claim_ids)

        best_match = None
        best_overlap = 0.0

        for eu in all_clusters:
            eu_claims = registry.all_claim_ids(eu.id)
            intersection = eu_claims & event_claims
            union = eu_claims | event_claims
            overlap = len(intersection) / len(union) if union else 0

            if overlap > best_overlap:
                best_overlap = overlap
                best_match = eu

        comparisons.append({
            'event': event.canonical_name,
            'event_claims': len(event.claim_ids),
            'match_label': best_match.label if best_match else None,
            'match_depth': best_match.depth if best_match else None,
            'match_claims': registry.claim_count(best_match.id) if best_match else 0,
            'overlap': best_overlap,
            'status': 'FOUND' if best_overlap >= 0.5 else 'PARTIAL' if best_overlap >= 0.2 else 'MISSED'
        })

    return analysis, comparisons


def main():
    print("=" * 60)
    print("Progressive Emergence v2 (Absorption-based)")
    print("=" * 60)

    snapshot = load_snapshot()
    print(f"\nLoaded {len(snapshot.claims)} claims\n")

    registry, stats = progressive_emerge_v2(snapshot, max_depth=6)

    analysis, comparisons = analyze_and_compare(registry)

    print("\n" + "=" * 60)
    print("Hierarchy by Depth")
    print("=" * 60)

    for depth, info in analysis['by_depth'].items():
        print(f"\nDepth {depth}: {info['count']} clusters")
        for c in info['top']:
            print(f"  [{c['mass']:.2f}] {c['label']} ({c['claims']} claims)")

    print("\n" + "=" * 60)
    print("Validation vs Existing Events")
    print("=" * 60)

    found = sum(1 for c in comparisons if c['status'] == 'FOUND')
    partial = sum(1 for c in comparisons if c['status'] == 'PARTIAL')
    missed = sum(1 for c in comparisons if c['status'] == 'MISSED')

    print(f"\nResults: {found} FOUND, {partial} PARTIAL, {missed} MISSED")

    for comp in sorted(comparisons, key=lambda x: x['overlap'], reverse=True):
        print(f"\n  [{comp['status']}] {comp['event']} ({comp['event_claims']} claims)")
        if comp['match_label']:
            print(f"      → {comp['match_label']} (d={comp['match_depth']}, {comp['match_claims']} claims, {comp['overlap']:.0%})")

    # Save
    output_path = Path("/app/test_eu/results/progressive_emergence_v2.json")
    with open(output_path, 'w') as f:
        json.dump({'stats': stats, 'analysis': analysis, 'comparisons': comparisons}, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
