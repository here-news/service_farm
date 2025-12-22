"""
Entropy-Based Recursive Emergence

The key insight from docs/11.epistemics.analysis.md:

    Hₙ = α·Cd - β·κ - γ·Ds - δ·Sd

Where:
- Cd = Content diversity (increases entropy)
- κ  = Coherence coefficient (decreases entropy)
- Ds = Data verifiability (decreases entropy)

A GOOD merge REDUCES entropy:
- Coherence (κ) increases more than diversity (Cd)
- Claims support each other, forming a tighter unit

A BAD merge INCREASES entropy:
- Diversity increases without coherence gain
- Dilutes the cluster's identity

Metabolism goal: find the lowest entropy state for the event graph.

Run inside container:
    docker exec herenews-app python /app/test_eu/entropy_emergence.py
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
    """EventfulUnit"""
    id: str
    depth: int
    claim_id: Optional[str] = None
    text: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    entity_ids: Set[str] = field(default_factory=set)
    page_ids: Set[str] = field(default_factory=set)
    internal_corr: int = 0
    internal_contra: int = 0
    entropy: float = 0.0
    mass: float = 0.0
    label: str = ""

    def is_leaf(self) -> bool:
        return self.claim_id is not None


class EURegistry:
    """Registry with entropy computation"""

    # Entropy weights (from the formula)
    ALPHA = 1.0   # Diversity coefficient (increases entropy)
    BETA = 2.0    # Coherence coefficient (decreases entropy)
    GAMMA = 0.5   # Verifiability coefficient (decreases entropy)

    def __init__(self, snapshot: GraphSnapshot):
        self.snapshot = snapshot
        self.eus: Dict[str, EU] = {}
        self.by_depth: Dict[int, List[str]] = defaultdict(list)
        self._claim_cache: Dict[str, Set[str]] = {}

        # Precompute entity frequencies for diversity calculation
        self.entity_freq: Dict[str, int] = defaultdict(int)
        for claim in snapshot.claims.values():
            for eid in claim.entity_ids:
                self.entity_freq[eid] += 1

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

    def compute_entropy(self, claim_ids: Set[str]) -> Tuple[float, float, float]:
        """
        Compute entropy for a set of claims.

        Returns: (entropy, coherence, diversity)

        Entropy = α·Cd - β·κ - γ·Ds

        Where:
        - Cd = entity diversity (how scattered are the entities)
        - κ  = coherence (corroboration density)
        - Ds = source diversity (multiple independent sources = more verifiable)
        """
        if len(claim_ids) < 2:
            return 0.0, 1.0, 0.0  # Single claim has zero entropy, perfect coherence

        # Coherence (κ): corroboration density
        internal_corr = 0
        internal_contra = 0
        for cid in claim_ids:
            claim = self.snapshot.claims.get(cid)
            if claim:
                internal_corr += len([c for c in claim.corroborates_ids if c in claim_ids])
                internal_contra += len([c for c in claim.contradicts_ids if c in claim_ids])

        internal_corr //= 2
        internal_contra //= 2

        n = len(claim_ids)
        possible_pairs = n * (n - 1) / 2

        # Coherence = (corroborations - contradictions) / possible_pairs
        # Normalized to [0, 1], with contradictions reducing coherence
        coherence = (internal_corr - internal_contra * 0.5) / possible_pairs if possible_pairs > 0 else 0
        coherence = max(0.0, min(1.0, coherence + 0.1))  # Base coherence of 0.1

        # Diversity (Cd): entity scatter
        # High diversity = claims talk about many different entities
        # Low diversity = claims focused on same entities
        entity_counts = defaultdict(int)
        for cid in claim_ids:
            claim = self.snapshot.claims.get(cid)
            if claim:
                for eid in claim.entity_ids:
                    entity_counts[eid] += 1

        if entity_counts:
            # Compute entity concentration (inverse of diversity)
            total_mentions = sum(entity_counts.values())
            # Herfindahl index: sum of squared shares
            concentration = sum((c / total_mentions) ** 2 for c in entity_counts.values())
            # Diversity = 1 - concentration (higher = more scattered)
            diversity = 1.0 - concentration
        else:
            diversity = 0.0

        # Verifiability (Ds): source diversity
        pages = set()
        for cid in claim_ids:
            claim = self.snapshot.claims.get(cid)
            if claim and claim.page_id:
                pages.add(claim.page_id)

        # More sources = more verifiable = lower entropy
        source_factor = min(1.0, len(pages) / 5)  # Cap at 5 sources

        # Entropy formula: H = α·Cd - β·κ - γ·Ds
        entropy = (
            self.ALPHA * diversity -      # Diversity increases entropy
            self.BETA * coherence -       # Coherence decreases entropy
            self.GAMMA * source_factor    # Verifiability decreases entropy
        )

        return entropy, coherence, diversity

    def merge_reduces_entropy(
        self,
        eu1: EU,
        eu2: EU,
        threshold: float = 0.0
    ) -> Tuple[bool, float, float]:
        """
        Check if merging two EUs would reduce total entropy.

        Returns: (should_merge, pre_entropy, post_entropy)
        """
        claims1 = self.all_claim_ids(eu1.id)
        claims2 = self.all_claim_ids(eu2.id)

        # Pre-merge: weighted average entropy
        h1, _, _ = self.compute_entropy(claims1)
        h2, _, _ = self.compute_entropy(claims2)

        # Weight by claim count
        n1, n2 = len(claims1), len(claims2)
        pre_entropy = (h1 * n1 + h2 * n2) / (n1 + n2)

        # Post-merge entropy
        merged_claims = claims1 | claims2
        post_entropy, _, _ = self.compute_entropy(merged_claims)

        # Merge if entropy decreases (or stays within threshold)
        should_merge = post_entropy <= pre_entropy + threshold

        return should_merge, pre_entropy, post_entropy


def create_layer_0(registry: EURegistry):
    """Create leaf EUs from claims"""
    for cid, claim in registry.snapshot.claims.items():
        eu = EU(
            id=f"L0_{cid}",
            depth=0,
            claim_id=cid,
            text=claim.text,
            entity_ids=set(claim.entity_ids),
            page_ids={claim.page_id} if claim.page_id else set(),
            entropy=0.0,  # Single claim = zero entropy
            mass=0.1,
            label=claim.text[:40] + "..."
        )
        registry.add(eu)

    print(f"Layer 0: {len(registry.get_layer(0))} leaf EUs")


def find_label(claims: Set[str], snapshot: GraphSnapshot) -> str:
    """Find best label (most concentrated entity)"""
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


def create_merged_eu(
    eus: List[EU],
    registry: EURegistry,
    depth: int,
    merge_id: int
) -> EU:
    """Create a merged EU"""
    all_claims = set()
    all_entities = set()
    all_pages = set()
    child_ids = []

    for eu in eus:
        all_claims |= registry.all_claim_ids(eu.id)
        all_entities |= eu.entity_ids
        all_pages |= eu.page_ids
        child_ids.append(eu.id)

    entropy, coherence, diversity = registry.compute_entropy(all_claims)

    # Mass based on claim count and coherence
    base_mass = len(all_claims) * 0.1
    mass = base_mass * (1.0 + coherence)

    # Count internal links
    internal_corr = 0
    internal_contra = 0
    for cid in all_claims:
        claim = registry.snapshot.claims.get(cid)
        if claim:
            internal_corr += len([c for c in claim.corroborates_ids if c in all_claims])
            internal_contra += len([c for c in claim.contradicts_ids if c in all_claims])

    return EU(
        id=f"L{depth}_{merge_id}",
        depth=depth,
        child_ids=child_ids,
        entity_ids=all_entities,
        page_ids=all_pages,
        internal_corr=internal_corr // 2,
        internal_contra=internal_contra // 2,
        entropy=entropy,
        mass=mass,
        label=find_label(all_claims, registry.snapshot)
    )


def emerge_layer_entropy(
    registry: EURegistry,
    source_depth: int,
    entropy_threshold: float = 0.05
) -> int:
    """
    Emerge next layer using entropy-based merge criterion.

    Only merge if:
    1. There's energy (corroboration/entity overlap)
    2. Merge reduces (or doesn't significantly increase) entropy
    """
    source_eus = registry.get_layer(source_depth)
    target_depth = source_depth + 1
    n = len(source_eus)

    if n < 2:
        return 0

    # Find candidate merges that reduce entropy
    candidates = []

    for i in range(n):
        for j in range(i + 1, n):
            eu1, eu2 = source_eus[i], source_eus[j]

            # Check energy (need some connection)
            claims1 = registry.all_claim_ids(eu1.id)
            claims2 = registry.all_claim_ids(eu2.id)

            # Energy from corroboration
            cross_corr = 0
            for cid in claims1:
                claim = registry.snapshot.claims.get(cid)
                if claim:
                    cross_corr += len([c for c in claim.corroborates_ids if c in claims2])

            # Energy from entity overlap
            entity_overlap = len(eu1.entity_ids & eu2.entity_ids)

            energy = cross_corr * 2.0 + entity_overlap * 0.5

            if energy < 0.5:  # Minimum energy to consider
                continue

            # Check entropy criterion
            should_merge, pre_h, post_h = registry.merge_reduces_entropy(
                eu1, eu2, threshold=entropy_threshold
            )

            if should_merge:
                # Score by entropy reduction (more reduction = better)
                entropy_delta = pre_h - post_h
                candidates.append((i, j, entropy_delta, energy))

    if not candidates:
        # No valid merges, pass through
        for i, eu in enumerate(source_eus):
            passthrough = EU(
                id=f"L{target_depth}_p{i}",
                depth=target_depth,
                child_ids=[eu.id],
                entity_ids=eu.entity_ids.copy(),
                page_ids=eu.page_ids.copy(),
                internal_corr=eu.internal_corr,
                internal_contra=eu.internal_contra,
                entropy=eu.entropy,
                mass=eu.mass,
                label=eu.label
            )
            registry.add(passthrough)
        return 0

    # Sort by entropy reduction (best first)
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Greedy merge
    used = set()
    merge_groups = []

    for i, j, entropy_delta, energy in candidates:
        if i in used or j in used:
            continue
        merge_groups.append(([i, j], entropy_delta))
        used.add(i)
        used.add(j)

    # Create merged EUs
    merge_id = 0
    for indices, _ in merge_groups:
        eus_to_merge = [source_eus[idx] for idx in indices]
        merged = create_merged_eu(eus_to_merge, registry, target_depth, merge_id)
        registry.add(merged)
        merge_id += 1

    # Pass through unmerged
    for i, eu in enumerate(source_eus):
        if i not in used:
            passthrough = EU(
                id=f"L{target_depth}_p{merge_id}",
                depth=target_depth,
                child_ids=[eu.id],
                entity_ids=eu.entity_ids.copy(),
                page_ids=eu.page_ids.copy(),
                internal_corr=eu.internal_corr,
                internal_contra=eu.internal_contra,
                entropy=eu.entropy,
                mass=eu.mass,
                label=eu.label
            )
            registry.add(passthrough)
            merge_id += 1

    return len(merge_groups)


def progressive_emerge_entropy(
    snapshot: GraphSnapshot,
    max_depth: int = 8,
    entropy_threshold: float = 0.05
) -> Tuple[EURegistry, Dict]:
    """Progressive emergence with entropy-based criterion"""

    registry = EURegistry(snapshot)
    stats = {'layers': []}

    create_layer_0(registry)
    stats['layers'].append({
        'depth': 0,
        'eu_count': len(registry.get_layer(0)),
        'merges': 0
    })

    for depth in range(max_depth):
        merges = emerge_layer_entropy(registry, depth, entropy_threshold)

        layer_eus = registry.get_layer(depth + 1)
        real_clusters = [eu for eu in layer_eus if len(eu.child_ids) > 1]

        # Compute average entropy at this layer
        avg_entropy = 0
        if real_clusters:
            avg_entropy = sum(eu.entropy for eu in real_clusters) / len(real_clusters)

        stats['layers'].append({
            'depth': depth + 1,
            'eu_count': len(layer_eus),
            'real_clusters': len(real_clusters),
            'merges': merges,
            'avg_entropy': avg_entropy
        })

        print(f"Layer {depth + 1}: {merges} merges → {len(real_clusters)} clusters (avg entropy: {avg_entropy:.3f})")

        if merges == 0:
            print(f"  No entropy-reducing merges possible, stopping.")
            break

    return registry, stats


def analyze_and_compare(registry: EURegistry) -> Tuple[Dict, List[Dict]]:
    """Analyze hierarchy and compare with events"""
    snapshot = registry.snapshot

    all_clusters = []
    for depth in registry.by_depth.keys():
        if depth == 0:
            continue
        for eu in registry.get_layer(depth):
            if len(eu.child_ids) > 1:
                all_clusters.append(eu)

    # Analysis by depth
    analysis = {'by_depth': {}}
    for depth in registry.by_depth.keys():
        eus = registry.get_layer(depth)
        real = [eu for eu in eus if len(eu.child_ids) != 1]
        if real:
            sorted_eus = sorted(real, key=lambda x: x.mass, reverse=True)
            analysis['by_depth'][depth] = {
                'count': len(real),
                'avg_entropy': sum(eu.entropy for eu in real) / len(real),
                'top': [
                    {
                        'label': eu.label,
                        'mass': eu.mass,
                        'entropy': eu.entropy,
                        'claims': registry.claim_count(eu.id),
                        'corr': eu.internal_corr
                    }
                    for eu in sorted_eus[:5]
                ]
            }

    # Compare with events
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
            'match_entropy': best_match.entropy if best_match else None,
            'overlap': best_overlap,
            'status': 'FOUND' if best_overlap >= 0.5 else 'PARTIAL' if best_overlap >= 0.2 else 'MISSED'
        })

    return analysis, comparisons


def main():
    print("=" * 60)
    print("Entropy-Based Recursive Emergence")
    print("=" * 60)
    print("\nMerges only happen if they REDUCE entropy.")
    print("Entropy = α·Diversity - β·Coherence - γ·Verifiability\n")

    snapshot = load_snapshot()
    print(f"Loaded {len(snapshot.claims)} claims\n")

    registry, stats = progressive_emerge_entropy(
        snapshot,
        max_depth=8,
        entropy_threshold=0.05  # Allow small entropy increase
    )

    analysis, comparisons = analyze_and_compare(registry)

    print("\n" + "=" * 60)
    print("Hierarchy by Depth (Entropy-Optimized)")
    print("=" * 60)

    for depth, info in analysis['by_depth'].items():
        print(f"\nDepth {depth}: {info['count']} clusters, avg_entropy={info['avg_entropy']:.3f}")
        for c in info['top']:
            print(f"  [{c['mass']:.1f}m, {c['entropy']:.2f}H] {c['label']} ({c['claims']} claims, {c['corr']} corr)")

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
            print(f"      → {comp['match_label']} (d={comp['match_depth']}, {comp['match_claims']} claims, H={comp['match_entropy']:.2f}, {comp['overlap']:.0%})")

    # Total system entropy
    print("\n" + "=" * 60)
    print("System Entropy Analysis")
    print("=" * 60)

    # Entropy at each layer
    for layer_stat in stats['layers']:
        if layer_stat['merges'] > 0 or layer_stat['depth'] == 0:
            print(f"  Layer {layer_stat['depth']}: {layer_stat.get('avg_entropy', 0):.3f} avg entropy")

    # Save
    output_path = Path("/app/test_eu/results/entropy_emergence.json")
    with open(output_path, 'w') as f:
        json.dump({
            'stats': stats,
            'analysis': {k: v for k, v in analysis.items()},
            'comparisons': comparisons,
            'weights': {
                'alpha': registry.ALPHA,
                'beta': registry.BETA,
                'gamma': registry.GAMMA
            }
        }, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
