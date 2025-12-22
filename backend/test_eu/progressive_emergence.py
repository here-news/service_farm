"""
Progressive Recursive Emergence

Claims ARE events. Clusters ARE events. Same type, different scale.

EU = Claim | Cluster(EU, EU, ...)

Layer-by-layer emergence:
- Layer 0: Claims (leaf EUs)
- Layer 1: Tight pairs/triples (high corroboration)
- Layer 2: Clusters of L1 (shared entities, themes)
- Layer 3: Larger narratives
- ... continues until no more meaningful merges

Each layer is computed fully before moving up.
Lower layers are REUSED (not discarded) by higher layers.

Run inside container:
    docker exec herenews-app python /app/test_eu/progressive_emergence.py
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from load_graph import load_snapshot, GraphSnapshot, ClaimData


@dataclass
class EU:
    """
    EventfulUnit - recursive structure.
    """
    id: str
    depth: int  # 0 = leaf, 1+ = composite

    # If leaf
    claim_id: Optional[str] = None
    text: Optional[str] = None

    # If composite - references to child EU ids (not objects, to allow reuse)
    child_ids: List[str] = field(default_factory=list)

    # Properties
    entity_ids: Set[str] = field(default_factory=set)
    page_ids: Set[str] = field(default_factory=set)
    internal_corr: int = 0
    internal_contra: int = 0

    # Computed
    mass: float = 0.0
    label: str = ""

    def is_leaf(self) -> bool:
        return self.claim_id is not None


class EURegistry:
    """
    Registry of all EUs across all layers.
    Allows lookup and reuse.
    """
    def __init__(self):
        self.eus: Dict[str, EU] = {}
        self.by_depth: Dict[int, List[str]] = defaultdict(list)

    def add(self, eu: EU):
        self.eus[eu.id] = eu
        self.by_depth[eu.depth].append(eu.id)

    def get(self, eu_id: str) -> Optional[EU]:
        return self.eus.get(eu_id)

    def get_layer(self, depth: int) -> List[EU]:
        return [self.eus[eid] for eid in self.by_depth[depth]]

    def all_claim_ids(self, eu_id: str) -> Set[str]:
        """Recursively get all claim IDs under this EU"""
        eu = self.eus.get(eu_id)
        if not eu:
            return set()
        if eu.is_leaf():
            return {eu.claim_id}
        result = set()
        for child_id in eu.child_ids:
            result |= self.all_claim_ids(child_id)
        return result

    def claim_count(self, eu_id: str) -> int:
        return len(self.all_claim_ids(eu_id))


def create_layer_0(snapshot: GraphSnapshot, registry: EURegistry):
    """
    Layer 0: Create leaf EUs from claims.
    """
    for cid, claim in snapshot.claims.items():
        eu = EU(
            id=f"L0_{cid}",
            depth=0,
            claim_id=cid,
            text=claim.text,
            entity_ids=set(claim.entity_ids),
            page_ids={claim.page_id} if claim.page_id else set(),
            mass=0.1,  # Base existence mass
            label=claim.text[:40] + "..."
        )
        registry.add(eu)

    print(f"Layer 0: {len(registry.get_layer(0))} leaf EUs (claims)")


def compute_pair_energy(
    eu1: EU,
    eu2: EU,
    registry: EURegistry,
    snapshot: GraphSnapshot
) -> float:
    """
    Energy released by combining two EUs.
    """
    claims1 = registry.all_claim_ids(eu1.id)
    claims2 = registry.all_claim_ids(eu2.id)
    all_claims = claims1 | claims2

    # Corroboration links between the two
    cross_corr = 0
    cross_contra = 0

    for cid in claims1:
        claim = snapshot.claims.get(cid)
        if claim:
            cross_corr += len([c for c in claim.corroborates_ids if c in claims2])
            cross_contra += len([c for c in claim.contradicts_ids if c in claims2])

    # Entity overlap
    shared_entities = eu1.entity_ids & eu2.entity_ids

    # Energy = corroboration (primary) + contradiction (heat) + entity cohesion
    energy = (
        cross_corr * 2.0 +
        cross_contra * 1.0 +
        len(shared_entities) * 0.5
    )

    return energy


def merge_into_eu(
    eus_to_merge: List[EU],
    registry: EURegistry,
    snapshot: GraphSnapshot,
    depth: int,
    merge_id: int
) -> EU:
    """
    Create a new EU from merging existing EUs.
    """
    # Collect all claims
    all_claims = set()
    for eu in eus_to_merge:
        all_claims |= registry.all_claim_ids(eu.id)

    # Collect entities and pages
    all_entities = set()
    all_pages = set()
    for eu in eus_to_merge:
        all_entities |= eu.entity_ids
        all_pages |= eu.page_ids

    # Count internal links
    internal_corr = 0
    internal_contra = 0
    for cid in all_claims:
        claim = snapshot.claims.get(cid)
        if claim:
            internal_corr += len([c for c in claim.corroborates_ids if c in all_claims])
            internal_contra += len([c for c in claim.contradicts_ids if c in all_claims])

    internal_corr //= 2
    internal_contra //= 2

    # Find label (most common entity)
    entity_counts = defaultdict(int)
    for cid in all_claims:
        claim = snapshot.claims.get(cid)
        if claim:
            for eid in claim.entity_ids:
                entity_counts[eid] += 1

    if entity_counts:
        top_entity_id = max(entity_counts.items(), key=lambda x: x[1])[0]
        entity = snapshot.entities.get(top_entity_id)
        label = entity.canonical_name if entity else f"cluster_L{depth}_{merge_id}"
    else:
        label = f"cluster_L{depth}_{merge_id}"

    # Compute mass
    children_mass = sum(eu.mass for eu in eus_to_merge)
    n = len(all_claims)
    possible_pairs = n * (n - 1) / 2 if n > 1 else 1
    coherence = internal_corr / possible_pairs if possible_pairs > 0 else 0
    tension_bonus = min(0.5, internal_contra * 0.05)

    mass = children_mass * (1.0 + coherence + tension_bonus)

    return EU(
        id=f"L{depth}_{merge_id}",
        depth=depth,
        child_ids=[eu.id for eu in eus_to_merge],
        entity_ids=all_entities,
        page_ids=all_pages,
        internal_corr=internal_corr,
        internal_contra=internal_contra,
        mass=mass,
        label=label
    )


def emerge_layer(
    registry: EURegistry,
    snapshot: GraphSnapshot,
    source_depth: int,
    min_energy: float = 1.0
) -> int:
    """
    Emerge next layer from source layer.
    Find all pairs/groups with sufficient energy and merge them.

    Returns number of new EUs created.
    """
    source_eus = registry.get_layer(source_depth)
    target_depth = source_depth + 1

    # Find all pairs with enough energy
    pairs_with_energy = []
    for i, eu1 in enumerate(source_eus):
        for j, eu2 in enumerate(source_eus[i+1:], i+1):
            energy = compute_pair_energy(eu1, eu2, registry, snapshot)
            if energy >= min_energy:
                pairs_with_energy.append((i, j, energy))

    # Sort by energy
    pairs_with_energy.sort(key=lambda x: x[2], reverse=True)

    # Greedy merge: take highest energy pairs, mark as used
    used = set()
    merge_groups = []

    for i, j, energy in pairs_with_energy:
        if i in used or j in used:
            continue
        merge_groups.append(([i, j], energy))
        used.add(i)
        used.add(j)

    # Create new EUs from merge groups
    new_eus = []
    for merge_id, (indices, energy) in enumerate(merge_groups):
        eus_to_merge = [source_eus[idx] for idx in indices]
        new_eu = merge_into_eu(eus_to_merge, registry, snapshot, target_depth, merge_id)
        new_eus.append(new_eu)

    # Also promote unmerged EUs to next layer (they're still valid at higher level)
    for i, eu in enumerate(source_eus):
        if i not in used:
            # Create a "passthrough" EU at next depth
            passthrough = EU(
                id=f"L{target_depth}_pass_{i}",
                depth=target_depth,
                child_ids=[eu.id],
                entity_ids=eu.entity_ids.copy(),
                page_ids=eu.page_ids.copy(),
                internal_corr=eu.internal_corr,
                internal_contra=eu.internal_contra,
                mass=eu.mass,
                label=eu.label
            )
            new_eus.append(passthrough)

    # Add all to registry
    for eu in new_eus:
        registry.add(eu)

    return len(merge_groups)  # Return actual merges, not passthroughs


def progressive_emerge(
    snapshot: GraphSnapshot,
    max_depth: int = 6,
    min_energy_by_depth: Dict[int, float] = None
) -> Tuple[EURegistry, Dict]:
    """
    Progressively emerge layers until no more merges possible.
    """
    registry = EURegistry()

    # Default energy thresholds (higher layers need more energy to merge)
    if min_energy_by_depth is None:
        min_energy_by_depth = {
            0: 1.0,   # L0→L1: tight corroboration
            1: 2.0,   # L1→L2: need stronger signal
            2: 3.0,   # L2→L3: even stronger
            3: 4.0,
            4: 5.0,
            5: 6.0,
        }

    stats = {
        'layers': []
    }

    # Layer 0
    create_layer_0(snapshot, registry)
    stats['layers'].append({
        'depth': 0,
        'eu_count': len(registry.get_layer(0)),
        'merges': 0
    })

    # Emerge layers progressively
    for depth in range(max_depth):
        min_energy = min_energy_by_depth.get(depth, 5.0)
        merges = emerge_layer(registry, snapshot, depth, min_energy)

        layer_eus = registry.get_layer(depth + 1)
        stats['layers'].append({
            'depth': depth + 1,
            'eu_count': len(layer_eus),
            'merges': merges,
            'min_energy': min_energy
        })

        print(f"Layer {depth + 1}: {merges} merges → {len(layer_eus)} EUs")

        if merges == 0:
            print(f"  No more merges at depth {depth + 1}, stopping.")
            break

    return registry, stats


def analyze_registry(registry: EURegistry, snapshot: GraphSnapshot) -> Dict:
    """Analyze the emerged hierarchy"""

    analysis = {
        'by_depth': {},
        'top_at_each_depth': {}
    }

    for depth in sorted(registry.by_depth.keys()):
        eus = registry.get_layer(depth)
        if not eus:
            continue

        # Filter out passthroughs for stats (they have only 1 child)
        real_clusters = [eu for eu in eus if len(eu.child_ids) != 1]

        analysis['by_depth'][depth] = {
            'total_eus': len(eus),
            'real_clusters': len(real_clusters),
            'avg_mass': sum(eu.mass for eu in eus) / len(eus) if eus else 0,
            'max_mass': max(eu.mass for eu in eus) if eus else 0
        }

        # Top EUs at this depth (by mass, excluding passthroughs)
        sorted_eus = sorted(real_clusters, key=lambda x: x.mass, reverse=True)
        analysis['top_at_each_depth'][depth] = [
            {
                'label': eu.label,
                'mass': eu.mass,
                'claim_count': registry.claim_count(eu.id),
                'corr': eu.internal_corr,
                'contra': eu.internal_contra
            }
            for eu in sorted_eus[:5]
        ]

    return analysis


def compare_with_events(registry: EURegistry, snapshot: GraphSnapshot) -> List[Dict]:
    """Compare emerged clusters with existing events"""

    # Get all non-leaf, non-passthrough EUs
    all_clusters = []
    for depth in registry.by_depth.keys():
        if depth == 0:
            continue
        for eu in registry.get_layer(depth):
            if len(eu.child_ids) > 1:  # Not a passthrough
                all_clusters.append(eu)

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
            'matched_label': best_match.label if best_match else None,
            'matched_depth': best_match.depth if best_match else None,
            'matched_claims': registry.claim_count(best_match.id) if best_match else 0,
            'overlap': best_overlap,
            'status': 'FOUND' if best_overlap >= 0.5 else 'PARTIAL' if best_overlap >= 0.2 else 'MISSED'
        })

    return comparisons


def main():
    print("=" * 60)
    print("Progressive Recursive Emergence")
    print("=" * 60)

    snapshot = load_snapshot()
    print(f"\nLoaded {len(snapshot.claims)} claims, {len(snapshot.events)} existing events\n")

    # Run progressive emergence
    registry, stats = progressive_emerge(
        snapshot,
        max_depth=6,
        min_energy_by_depth={0: 1.0, 1: 1.5, 2: 2.0, 3: 2.5, 4: 3.0, 5: 3.5}
    )

    # Analyze
    analysis = analyze_registry(registry, snapshot)

    print("\n" + "=" * 60)
    print("Hierarchy Summary")
    print("=" * 60)

    for depth, info in analysis['by_depth'].items():
        print(f"\nDepth {depth}:")
        print(f"  Total EUs: {info['total_eus']}, Real clusters: {info['real_clusters']}")
        print(f"  Avg mass: {info['avg_mass']:.2f}, Max mass: {info['max_mass']:.2f}")

        if depth in analysis['top_at_each_depth']:
            print(f"  Top clusters:")
            for c in analysis['top_at_each_depth'][depth]:
                print(f"    [{c['mass']:.2f}] {c['label']} ({c['claim_count']} claims, {c['corr']} corr, {c['contra']} contra)")

    # Compare with existing events
    print("\n" + "=" * 60)
    print("Validation vs Existing Events")
    print("=" * 60)

    comparisons = compare_with_events(registry, snapshot)
    found = sum(1 for c in comparisons if c['status'] == 'FOUND')
    partial = sum(1 for c in comparisons if c['status'] == 'PARTIAL')
    missed = sum(1 for c in comparisons if c['status'] == 'MISSED')

    print(f"\nResults: {found} FOUND, {partial} PARTIAL, {missed} MISSED")

    for comp in sorted(comparisons, key=lambda x: x['overlap'], reverse=True):
        print(f"\n  [{comp['status']}] {comp['event']} ({comp['event_claims']} claims)")
        if comp['matched_label']:
            print(f"      → {comp['matched_label']} (depth={comp['matched_depth']}, {comp['matched_claims']} claims, {comp['overlap']:.0%})")

    # Save results
    output = {
        'stats': stats,
        'analysis': {k: v for k, v in analysis.items() if k != 'top_at_each_depth'},
        'top_clusters_by_depth': analysis['top_at_each_depth'],
        'event_comparisons': comparisons
    }

    output_path = Path("/app/test_eu/results/progressive_emergence.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
