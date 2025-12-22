"""
Recursive Emergence with Energy Budget

Claims ARE events. Clusters ARE events. Same type, different scale.

EU = Claim | Cluster(EU, EU, ...)

Emergence is energy-driven:
- New claims = nutrition
- Corroboration = energy release
- No nutrition → no growth
- Budget determines how far up the hierarchy we compute

Run inside container:
    docker exec herenews-app python /app/test_eu/recursive_emergence.py
"""

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from load_graph import load_snapshot, GraphSnapshot, ClaimData


@dataclass
class EU:
    """
    EventfulUnit - recursive structure.
    Can be a leaf (claim) or composite (cluster of EUs).
    """
    id: str

    # If leaf (claim)
    claim_id: Optional[str] = None
    text: Optional[str] = None

    # If composite (cluster)
    children: List['EU'] = field(default_factory=list)

    # Common properties
    mass: float = 0.0
    energy: float = 0.0  # Available energy for further emergence
    depth: int = 0       # 0 = leaf, 1 = cluster of claims, 2 = cluster of clusters, etc.

    # Emergence signals
    entity_ids: Set[str] = field(default_factory=set)
    internal_corroborations: int = 0
    internal_contradictions: int = 0

    # For display
    label: str = ""

    def is_leaf(self) -> bool:
        return self.claim_id is not None

    def claim_count(self) -> int:
        if self.is_leaf():
            return 1
        return sum(c.claim_count() for c in self.children)

    def all_claim_ids(self) -> Set[str]:
        if self.is_leaf():
            return {self.claim_id}
        result = set()
        for child in self.children:
            result |= child.all_claim_ids()
        return result


def create_leaf_eu(claim: ClaimData) -> EU:
    """Create leaf EU from claim"""
    return EU(
        id=f"eu_{claim.id}",
        claim_id=claim.id,
        text=claim.text,
        depth=0,
        entity_ids=set(claim.entity_ids),
        mass=0.1,  # Base mass for existence
        energy=0.0,  # Leaves don't have energy to grow
        label=claim.text[:50] + "..."
    )


def compute_energy(
    eus: List[EU],
    snapshot: GraphSnapshot
) -> float:
    """
    Compute energy released by combining these EUs.

    Energy sources:
    - Corroboration links between EUs
    - Shared entities (cohesion)
    - Contradiction (tension energy)
    """
    if len(eus) < 2:
        return 0.0

    # Collect all claim IDs
    all_claims = set()
    for eu in eus:
        all_claims |= eu.all_claim_ids()

    # Count internal corroborations
    corr_count = 0
    contra_count = 0
    for cid in all_claims:
        claim = snapshot.claims.get(cid)
        if claim:
            corr_count += len([c for c in claim.corroborates_ids if c in all_claims])
            contra_count += len([c for c in claim.contradicts_ids if c in all_claims])

    corr_count //= 2  # Counted twice
    contra_count //= 2

    # Entity cohesion
    all_entities = set()
    for eu in eus:
        all_entities |= eu.entity_ids

    entity_overlap = 0
    for i, eu1 in enumerate(eus):
        for eu2 in eus[i+1:]:
            shared = eu1.entity_ids & eu2.entity_ids
            entity_overlap += len(shared)

    # Energy formula
    # Corroboration releases energy (agreement)
    # Contradiction releases energy (tension/heat)
    # Entity overlap provides cohesion energy
    energy = (
        corr_count * 1.0 +      # Corroboration is primary energy
        contra_count * 0.5 +    # Contradiction adds heat
        entity_overlap * 0.3    # Shared entities add cohesion
    )

    return energy


def compute_mass(eu: EU) -> float:
    """
    Compute mass recursively.
    Mass = f(children's mass, internal coherence)
    """
    if eu.is_leaf():
        return 0.1  # Base mass

    if not eu.children:
        return 0.0

    # Sum of children's mass
    children_mass = sum(c.mass for c in eu.children)

    # Coherence bonus (corroboration density)
    n = len(eu.children)
    possible_pairs = n * (n - 1) / 2 if n > 1 else 1
    coherence = min(1.0, eu.internal_corroborations / possible_pairs) if possible_pairs > 0 else 0

    # Tension bonus (contradictions add "heat" which is also mass)
    tension = min(0.5, eu.internal_contradictions * 0.1)

    # Mass formula: children's mass amplified by coherence
    mass = children_mass * (1.0 + coherence + tension)

    return mass


def find_mergeable_pairs(
    eus: List[EU],
    snapshot: GraphSnapshot,
    min_energy: float = 1.0
) -> List[Tuple[int, int, float]]:
    """
    Find pairs of EUs that can merge (have enough energy).
    Returns list of (idx1, idx2, energy) sorted by energy descending.
    """
    pairs = []

    for i, eu1 in enumerate(eus):
        for j, eu2 in enumerate(eus[i+1:], i+1):
            energy = compute_energy([eu1, eu2], snapshot)
            if energy >= min_energy:
                pairs.append((i, j, energy))

    # Sort by energy (highest first)
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def merge_eus(
    eu1: EU,
    eu2: EU,
    snapshot: GraphSnapshot,
    depth: int
) -> EU:
    """
    Merge two EUs into a composite EU.
    """
    # Children are the merged EUs (or their children if they're composites at same depth)
    if eu1.depth == eu2.depth == depth - 1:
        children = [eu1, eu2]
    else:
        # Flatten if needed
        children = []
        for eu in [eu1, eu2]:
            if eu.depth == depth - 1:
                children.append(eu)
            else:
                children.extend(eu.children)

    # Collect all claim IDs for corroboration counting
    all_claims = eu1.all_claim_ids() | eu2.all_claim_ids()

    # Count internal links
    corr_count = 0
    contra_count = 0
    for cid in all_claims:
        claim = snapshot.claims.get(cid)
        if claim:
            corr_count += len([c for c in claim.corroborates_ids if c in all_claims])
            contra_count += len([c for c in claim.contradicts_ids if c in all_claims])

    # Merge entities
    merged_entities = eu1.entity_ids | eu2.entity_ids

    # Find most common entity for label
    entity_counts = defaultdict(int)
    for cid in all_claims:
        claim = snapshot.claims.get(cid)
        if claim:
            for eid in claim.entity_ids:
                entity_counts[eid] += 1

    top_entity = max(entity_counts.items(), key=lambda x: x[1])[0] if entity_counts else None
    label = snapshot.entities[top_entity].canonical_name if top_entity and top_entity in snapshot.entities else f"cluster_{depth}"

    merged = EU(
        id=f"eu_d{depth}_{id(eu1)}_{id(eu2)}",
        children=children,
        depth=depth,
        entity_ids=merged_entities,
        internal_corroborations=corr_count // 2,
        internal_contradictions=contra_count // 2,
        energy=compute_energy([eu1, eu2], snapshot),
        label=label
    )

    merged.mass = compute_mass(merged)

    return merged


def emerge_layer(
    eus: List[EU],
    snapshot: GraphSnapshot,
    energy_budget: float,
    min_merge_energy: float = 1.0
) -> Tuple[List[EU], float]:
    """
    Emerge one layer: merge EUs that have enough energy.
    Returns (new_eus, remaining_budget).

    Stops when:
    - No more pairs with enough energy
    - Energy budget exhausted
    """
    current_eus = eus.copy()
    remaining_budget = energy_budget
    target_depth = max(eu.depth for eu in eus) + 1

    merges_made = 0

    while remaining_budget > 0:
        # Find mergeable pairs
        pairs = find_mergeable_pairs(current_eus, snapshot, min_merge_energy)

        if not pairs:
            break

        # Take the highest energy pair
        i, j, energy = pairs[0]

        # Check budget
        merge_cost = 1.0  # Each merge costs 1 unit of budget
        if remaining_budget < merge_cost:
            break

        # Merge
        eu1 = current_eus[i]
        eu2 = current_eus[j]
        merged = merge_eus(eu1, eu2, snapshot, target_depth)

        # Update list (remove merged, add new)
        current_eus = [eu for k, eu in enumerate(current_eus) if k not in (i, j)]
        current_eus.append(merged)

        remaining_budget -= merge_cost
        merges_made += 1

    return current_eus, remaining_budget


def recursive_emerge(
    snapshot: GraphSnapshot,
    energy_budget: float,
    max_depth: int = 5,
    min_merge_energy: float = 1.0
) -> Tuple[List[EU], Dict]:
    """
    Recursively emerge EUs until budget exhausted or max depth reached.

    Returns (final_eus, stats)
    """
    # Layer 0: Create leaf EUs from claims
    eus = [create_leaf_eu(claim) for claim in snapshot.claims.values()]

    stats = {
        'initial_claims': len(eus),
        'layers': [],
        'total_budget': energy_budget,
    }

    remaining_budget = energy_budget
    current_depth = 0

    while current_depth < max_depth and remaining_budget > 0:
        layer_start_count = len(eus)
        layer_start_budget = remaining_budget

        eus, remaining_budget = emerge_layer(
            eus, snapshot, remaining_budget, min_merge_energy
        )

        merges = layer_start_count - len(eus)

        stats['layers'].append({
            'depth': current_depth + 1,
            'merges': merges,
            'eus_after': len(eus),
            'budget_used': layer_start_budget - remaining_budget
        })

        if merges == 0:
            # No more merges possible at this depth
            break

        current_depth += 1

    stats['remaining_budget'] = remaining_budget
    stats['final_depth'] = current_depth
    stats['final_eu_count'] = len(eus)

    return eus, stats


def analyze_hierarchy(eus: List[EU], snapshot: GraphSnapshot) -> Dict:
    """Analyze the emerged hierarchy"""

    # Group by depth
    by_depth = defaultdict(list)

    def collect_all(eu: EU):
        by_depth[eu.depth].append(eu)
        for child in eu.children:
            collect_all(child)

    for eu in eus:
        collect_all(eu)

    analysis = {
        'by_depth': {},
        'top_clusters': []
    }

    for depth, depth_eus in sorted(by_depth.items()):
        analysis['by_depth'][depth] = {
            'count': len(depth_eus),
            'avg_mass': sum(eu.mass for eu in depth_eus) / len(depth_eus) if depth_eus else 0,
            'max_mass': max(eu.mass for eu in depth_eus) if depth_eus else 0
        }

    # Top clusters (highest mass at max depth)
    max_depth = max(eu.depth for eu in eus) if eus else 0
    top_level = [eu for eu in eus if eu.depth == max_depth]
    top_level.sort(key=lambda x: x.mass, reverse=True)

    for eu in top_level[:15]:
        analysis['top_clusters'].append({
            'label': eu.label,
            'mass': eu.mass,
            'depth': eu.depth,
            'claim_count': eu.claim_count(),
            'children': len(eu.children),
            'corroborations': eu.internal_corroborations,
            'contradictions': eu.internal_contradictions
        })

    return analysis


def compare_with_events(eus: List[EU], snapshot: GraphSnapshot) -> List[Dict]:
    """Compare emerged clusters with existing events"""

    # Get all non-leaf EUs
    clusters = [eu for eu in eus if not eu.is_leaf()]

    comparisons = []
    for event in snapshot.events.values():
        event_claims = set(event.claim_ids)

        best_match = None
        best_overlap = 0.0

        for eu in clusters:
            eu_claims = eu.all_claim_ids()
            intersection = eu_claims & event_claims
            union = eu_claims | event_claims
            overlap = len(intersection) / len(union) if union else 0

            if overlap > best_overlap:
                best_overlap = overlap
                best_match = eu

        comparisons.append({
            'event': event.canonical_name,
            'event_claims': len(event.claim_ids),
            'matched_cluster': best_match.label if best_match else None,
            'matched_claims': best_match.claim_count() if best_match else 0,
            'matched_depth': best_match.depth if best_match else 0,
            'overlap': best_overlap,
            'status': 'FOUND' if best_overlap >= 0.5 else 'PARTIAL' if best_overlap >= 0.2 else 'MISSED'
        })

    return comparisons


def main():
    print("=" * 60)
    print("Recursive Emergence with Energy Budget")
    print("=" * 60)

    snapshot = load_snapshot()
    print(f"\nLoaded {len(snapshot.claims)} claims")

    # Test with different energy budgets
    budgets = [50, 100, 200, 500]

    for budget in budgets:
        print(f"\n{'='*60}")
        print(f"Energy Budget: {budget}")
        print("=" * 60)

        eus, stats = recursive_emerge(
            snapshot,
            energy_budget=budget,
            max_depth=5,
            min_merge_energy=1.0
        )

        print(f"\nEmergence Stats:")
        print(f"  Initial claims: {stats['initial_claims']}")
        print(f"  Final EUs: {stats['final_eu_count']}")
        print(f"  Final depth: {stats['final_depth']}")
        print(f"  Budget used: {stats['total_budget'] - stats['remaining_budget']:.0f}")

        print(f"\nLayers:")
        for layer in stats['layers']:
            print(f"  Depth {layer['depth']}: {layer['merges']} merges, {layer['eus_after']} EUs, cost={layer['budget_used']:.0f}")

        # Analyze hierarchy
        analysis = analyze_hierarchy(eus, snapshot)

        print(f"\nHierarchy by depth:")
        for depth, info in analysis['by_depth'].items():
            print(f"  Depth {depth}: {info['count']} EUs, avg_mass={info['avg_mass']:.2f}, max_mass={info['max_mass']:.2f}")

        print(f"\nTop emerged clusters:")
        for cluster in analysis['top_clusters'][:10]:
            print(f"  [{cluster['mass']:.2f}] {cluster['label']} (d={cluster['depth']}, claims={cluster['claim_count']}, corr={cluster['corroborations']})")

        # Compare with existing events
        comparisons = compare_with_events(eus, snapshot)
        found = sum(1 for c in comparisons if c['status'] == 'FOUND')
        partial = sum(1 for c in comparisons if c['status'] == 'PARTIAL')
        missed = sum(1 for c in comparisons if c['status'] == 'MISSED')

        print(f"\nValidation vs existing events: {found} FOUND, {partial} PARTIAL, {missed} MISSED")

    # Final detailed run with budget=200
    print("\n" + "=" * 60)
    print("Detailed Analysis (Budget=200)")
    print("=" * 60)

    eus, stats = recursive_emerge(snapshot, energy_budget=200, max_depth=5, min_merge_energy=1.0)
    comparisons = compare_with_events(eus, snapshot)

    print("\nEvent matching details:")
    for comp in sorted(comparisons, key=lambda x: x['overlap'], reverse=True):
        print(f"\n  [{comp['status']}] {comp['event']} ({comp['event_claims']} claims)")
        if comp['matched_cluster']:
            print(f"      → {comp['matched_cluster']} (d={comp['matched_depth']}, {comp['matched_claims']} claims, {comp['overlap']:.0%} overlap)")

    # Save results
    output = {
        'budget_experiments': {},
        'detailed_200': {
            'stats': stats,
            'comparisons': comparisons
        }
    }

    for budget in budgets:
        eus, stats = recursive_emerge(snapshot, energy_budget=budget, max_depth=5)
        comparisons = compare_with_events(eus, snapshot)
        output['budget_experiments'][budget] = {
            'stats': stats,
            'found': sum(1 for c in comparisons if c['status'] == 'FOUND'),
            'partial': sum(1 for c in comparisons if c['status'] == 'PARTIAL'),
            'missed': sum(1 for c in comparisons if c['status'] == 'MISSED')
        }

    output_path = Path("/app/test_eu/results/recursive_emergence.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
