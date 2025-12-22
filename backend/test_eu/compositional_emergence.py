"""
Compositional Emergence

True recursive emergence where:
1. All existing EUs (any depth) are candidates for merging
2. Smaller EUs can join larger EUs at any point
3. The "layer" concept becomes emergent size, not processing order

EU = Claim | Cluster(EU, EU, ...)

The pool of EUs grows, and any EU can merge with any other EU
if they have sufficient connection energy.

Run inside container:
    docker exec herenews-app python /app/test_eu/compositional_emergence.py
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import heapq

from load_graph import load_snapshot, GraphSnapshot


@dataclass
class EU:
    """EventfulUnit - truly recursive"""
    id: str

    # If leaf
    claim_id: Optional[str] = None
    text: Optional[str] = None

    # If composite - can contain EUs of any depth
    child_ids: List[str] = field(default_factory=list)

    # Computed from all descendants
    entity_ids: Set[str] = field(default_factory=set)
    page_ids: Set[str] = field(default_factory=set)
    corroborations: int = 0
    contradictions: int = 0
    coherence: float = 0.0
    claim_count: int = 0

    # State
    active: bool = True  # Can still merge with others
    label: str = ""

    def is_leaf(self) -> bool:
        return self.claim_id is not None

    def depth(self) -> int:
        """Depth is max child depth + 1"""
        if self.is_leaf():
            return 0
        # Would need registry to compute - simplified for now
        return len(self.child_ids)


class CompositionalRegistry:
    """
    Registry where any EU can merge with any other EU.
    No layer concept - just a pool of EUs.
    """

    def __init__(self, snapshot: GraphSnapshot):
        self.snapshot = snapshot
        self.eus: Dict[str, EU] = {}
        self.active_ids: Set[str] = set()  # EUs that can still merge
        self._claim_cache: Dict[str, Set[str]] = {}
        self.merge_counter = 0

    def add(self, eu: EU):
        self.eus[eu.id] = eu
        if eu.active:
            self.active_ids.add(eu.id)
        self._claim_cache.pop(eu.id, None)

    def deactivate(self, eu_id: str):
        """Mark EU as consumed (merged into another)"""
        if eu_id in self.eus:
            self.eus[eu_id].active = False
        self.active_ids.discard(eu_id)

    def get(self, eu_id: str) -> Optional[EU]:
        return self.eus.get(eu_id)

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

    def compute_energy(self, eu1: EU, eu2: EU) -> float:
        """Energy from potential merge"""
        claims1 = self.all_claim_ids(eu1.id)
        claims2 = self.all_claim_ids(eu2.id)

        cross_corr = 0
        cross_contra = 0

        for cid in claims1:
            claim = self.snapshot.claims.get(cid)
            if claim:
                cross_corr += len([c for c in claim.corroborates_ids if c in claims2])
                cross_contra += len([c for c in claim.contradicts_ids if c in claims2])

        entity_overlap = len(eu1.entity_ids & eu2.entity_ids)

        return cross_corr * 2.0 + cross_contra * 1.0 + entity_overlap * 0.3

    def compute_coherence(self, claim_ids: Set[str]) -> Tuple[int, int, float]:
        """Compute internal coherence metrics"""
        internal_corr = 0
        internal_contra = 0

        for cid in claim_ids:
            claim = self.snapshot.claims.get(cid)
            if claim:
                internal_corr += len([c for c in claim.corroborates_ids if c in claim_ids])
                internal_contra += len([c for c in claim.contradicts_ids if c in claim_ids])

        internal_corr //= 2
        internal_contra //= 2

        total_links = internal_corr + internal_contra
        coherence = internal_corr / total_links if total_links > 0 else 1.0

        return internal_corr, internal_contra, coherence

    def find_label(self, claim_ids: Set[str]) -> str:
        entity_counts = defaultdict(int)
        for cid in claim_ids:
            claim = self.snapshot.claims.get(cid)
            if claim:
                for eid in claim.entity_ids:
                    entity_counts[eid] += 1

        if entity_counts:
            top_id = max(entity_counts.items(), key=lambda x: x[1])[0]
            entity = self.snapshot.entities.get(top_id)
            if entity:
                return entity.canonical_name
        return "cluster"

    def merge(self, eu_ids: List[str]) -> EU:
        """Create new EU from merging existing EUs"""
        self.merge_counter += 1

        all_claims = set()
        all_entities = set()
        all_pages = set()

        for eu_id in eu_ids:
            eu = self.eus[eu_id]
            all_claims |= self.all_claim_ids(eu_id)
            all_entities |= eu.entity_ids
            all_pages |= eu.page_ids
            self.deactivate(eu_id)  # Mark as consumed

        corr, contra, coherence = self.compute_coherence(all_claims)
        label = self.find_label(all_claims)

        new_eu = EU(
            id=f"M_{self.merge_counter}",
            child_ids=eu_ids,
            entity_ids=all_entities,
            page_ids=all_pages,
            corroborations=corr,
            contradictions=contra,
            coherence=coherence,
            claim_count=len(all_claims),
            active=True,
            label=label
        )

        self.add(new_eu)
        return new_eu


def create_leaf_eus(registry: CompositionalRegistry):
    """Create leaf EUs from claims"""
    for cid, claim in registry.snapshot.claims.items():
        eu = EU(
            id=f"C_{cid}",
            claim_id=cid,
            text=claim.text,
            entity_ids=set(claim.entity_ids),
            page_ids={claim.page_id} if claim.page_id else set(),
            coherence=1.0,
            claim_count=1,
            active=True,
            label=claim.text[:40] + "..."
        )
        registry.add(eu)


def compositional_emerge(
    snapshot: GraphSnapshot,
    min_energy: float = 1.0,
    max_rounds: int = 20
) -> CompositionalRegistry:
    """
    Compositional emergence:
    1. Start with all claims as leaf EUs
    2. Find highest energy pair among all active EUs
    3. Merge them into new EU
    4. Repeat until no more high-energy pairs

    Any EU can merge with any other EU - no layer restriction.
    """

    registry = CompositionalRegistry(snapshot)
    create_leaf_eus(registry)

    print(f"Starting with {len(registry.active_ids)} active EUs")

    for round_num in range(max_rounds):
        # Find all pairs with energy above threshold
        active = [registry.eus[eid] for eid in registry.active_ids]
        n = len(active)

        if n < 2:
            print(f"Round {round_num + 1}: Only {n} active EUs, stopping")
            break

        # Find best pair
        best_energy = 0.0
        best_pair = None

        for i in range(n):
            for j in range(i + 1, n):
                energy = registry.compute_energy(active[i], active[j])
                if energy > best_energy:
                    best_energy = energy
                    best_pair = (active[i].id, active[j].id)

        if best_energy < min_energy:
            print(f"Round {round_num + 1}: Best energy {best_energy:.2f} < threshold {min_energy}, stopping")
            break

        # Merge best pair
        new_eu = registry.merge([best_pair[0], best_pair[1]])

        # Report
        active_count = len(registry.active_ids)
        composites = [registry.eus[eid] for eid in registry.active_ids if not registry.eus[eid].is_leaf()]

        if (round_num + 1) % 10 == 0 or round_num < 5:
            print(f"Round {round_num + 1}: Merged {best_pair[0][:20]}... + {best_pair[1][:20]}... "
                  f"→ {new_eu.label} ({new_eu.claim_count} claims, coh={new_eu.coherence:.2f})")
            print(f"  Active: {active_count}, Composites: {len(composites)}")

    return registry


def analyze_results(registry: CompositionalRegistry):
    """Analyze final state"""

    active = [registry.eus[eid] for eid in registry.active_ids]
    composites = [eu for eu in active if not eu.is_leaf()]
    leaves = [eu for eu in active if eu.is_leaf()]

    print(f"\n{'='*60}")
    print("Final State")
    print(f"{'='*60}")
    print(f"\nActive EUs: {len(active)}")
    print(f"  Composite: {len(composites)}")
    print(f"  Leaves (unmerged): {len(leaves)}")

    # Top composites by size
    print(f"\n{'='*60}")
    print("Top Clusters by Size")
    print(f"{'='*60}")

    for eu in sorted(composites, key=lambda x: x.claim_count, reverse=True)[:15]:
        tension = eu.contradictions / (eu.corroborations + eu.contradictions) if (eu.corroborations + eu.contradictions) > 0 else 0
        print(f"\n  {eu.label}: {eu.claim_count} claims")
        print(f"    Coherence: {eu.coherence:.2f}, +{eu.corroborations}/-{eu.contradictions}")
        if eu.contradictions > 0:
            print(f"    Tension: {tension:.0%}")
        print(f"    Children: {len(eu.child_ids)} ({', '.join(eu.child_ids[:3])}...)")

    # Check recursion - do any composites contain other composites?
    print(f"\n{'='*60}")
    print("Recursion Check")
    print(f"{'='*60}")

    recursive_count = 0
    for eu in composites:
        child_composites = [cid for cid in eu.child_ids if not registry.eus[cid].is_leaf()]
        if child_composites:
            recursive_count += 1
            if recursive_count <= 5:
                print(f"\n  {eu.label} contains {len(child_composites)} composite children:")
                for cid in child_composites[:3]:
                    child = registry.eus[cid]
                    print(f"    - {child.label} ({child.claim_count} claims)")

    print(f"\n  Total recursive composites: {recursive_count} / {len(composites)}")


def main():
    print("=" * 60)
    print("Compositional Emergence")
    print("=" * 60)
    print("\nAny EU can merge with any EU. No layer restriction.")
    print("Emergence is truly recursive - clusters contain clusters.\n")

    snapshot = load_snapshot()
    print(f"Loaded {len(snapshot.claims)} claims\n")

    registry = compositional_emerge(
        snapshot,
        min_energy=0.5,
        max_rounds=500
    )

    analyze_results(registry)

    # Save
    output_path = Path("/app/test_eu/results/compositional_emergence.json")

    results = {
        'total_eus': len(registry.eus),
        'active_eus': len(registry.active_ids),
        'merges': registry.merge_counter,
        'top_clusters': [
            {
                'label': eu.label,
                'claims': eu.claim_count,
                'coherence': eu.coherence,
                'corr': eu.corroborations,
                'contra': eu.contradictions,
                'children': len(eu.child_ids)
            }
            for eu in sorted(
                [registry.eus[eid] for eid in registry.active_ids if not registry.eus[eid].is_leaf()],
                key=lambda x: x.claim_count,
                reverse=True
            )[:20]
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
