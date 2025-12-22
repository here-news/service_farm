"""
Metabolic Emergence

Key insights:
1. Claims start unstable - need metabolism to reach coherence
2. Contradictions are signal, not noise - they indicate active metabolism
3. Clusters grow by absorbing AND metabolizing (resolving/integrating)

Instead of refusing merges that increase entropy, we:
1. Allow merges if there's sufficient energy (connection)
2. Track "unresolved tension" (contradictions not yet metabolized)
3. Cluster is "stable" when tension is low relative to corroboration
4. Cluster is "active" when tension is high (needs metabolism)

Run inside container:
    docker exec herenews-app python /app/test_eu/metabolic_emergence.py
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
    """EventfulUnit with metabolic state"""
    id: str
    depth: int

    # Content
    claim_id: Optional[str] = None
    text: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)

    # Graph properties
    entity_ids: Set[str] = field(default_factory=set)
    page_ids: Set[str] = field(default_factory=set)

    # Metabolic state
    corroborations: int = 0      # Internal support
    contradictions: int = 0      # Internal tension (unresolved)

    # Derived metrics
    coherence: float = 0.0       # corr / (corr + contra)
    stability: float = 0.0       # How settled is this cluster
    mass: float = 0.0
    label: str = ""

    def is_leaf(self) -> bool:
        return self.claim_id is not None

    def tension_ratio(self) -> float:
        """High = lots of unresolved contradictions"""
        total = self.corroborations + self.contradictions
        if total == 0:
            return 0.0
        return self.contradictions / total


class MetabolicRegistry:
    """Registry with metabolic computations"""

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

    def compute_metabolic_state(self, claim_ids: Set[str]) -> Tuple[int, int, float, float]:
        """
        Compute metabolic state for a claim set.

        Returns: (corroborations, contradictions, coherence, stability)
        """
        if len(claim_ids) < 2:
            return 0, 0, 1.0, 1.0  # Single claim is perfectly coherent and stable

        internal_corr = 0
        internal_contra = 0

        for cid in claim_ids:
            claim = self.snapshot.claims.get(cid)
            if claim:
                internal_corr += len([c for c in claim.corroborates_ids if c in claim_ids])
                internal_contra += len([c for c in claim.contradicts_ids if c in claim_ids])

        internal_corr //= 2
        internal_contra //= 2

        # Coherence: corroboration vs contradiction ratio
        total_links = internal_corr + internal_contra
        if total_links > 0:
            coherence = internal_corr / total_links
        else:
            # No links = neutral coherence (not good, not bad)
            coherence = 0.5

        # Stability: based on link density and coherence
        n = len(claim_ids)
        possible_pairs = n * (n - 1) / 2
        link_density = total_links / possible_pairs if possible_pairs > 0 else 0

        # Stable = high link density AND high coherence
        # Unstable = low link density OR low coherence
        stability = link_density * coherence

        return internal_corr, internal_contra, coherence, stability

    def compute_merge_energy(
        self,
        eu1: EU,
        eu2: EU
    ) -> Tuple[float, int, int]:
        """
        Energy from merging two EUs.

        Returns: (energy, cross_corr, cross_contra)

        Energy comes from:
        - Cross-corroborations (strong signal)
        - Cross-contradictions (also signal - things to metabolize)
        - Entity overlap (thematic connection)
        """
        claims1 = self.all_claim_ids(eu1.id)
        claims2 = self.all_claim_ids(eu2.id)

        cross_corr = 0
        cross_contra = 0

        for cid in claims1:
            claim = self.snapshot.claims.get(cid)
            if claim:
                cross_corr += len([c for c in claim.corroborates_ids if c in claims2])
                cross_contra += len([c for c in claim.contradicts_ids if c in claims2])

        # Entity overlap
        entity_overlap = len(eu1.entity_ids & eu2.entity_ids)

        # Energy = connections (both corr and contra are connections)
        # Contradictions are slightly less valuable but still signal
        energy = cross_corr * 2.0 + cross_contra * 1.5 + entity_overlap * 0.3

        return energy, cross_corr, cross_contra


def create_layer_0(registry: MetabolicRegistry):
    """Create leaf EUs"""
    for cid, claim in registry.snapshot.claims.items():
        eu = EU(
            id=f"L0_{cid}",
            depth=0,
            claim_id=cid,
            text=claim.text,
            entity_ids=set(claim.entity_ids),
            page_ids={claim.page_id} if claim.page_id else set(),
            coherence=1.0,
            stability=1.0,
            mass=0.1,
            label=claim.text[:40] + "..."
        )
        registry.add(eu)

    print(f"Layer 0: {len(registry.get_layer(0))} leaf EUs")


def find_label(claims: Set[str], snapshot: GraphSnapshot) -> str:
    """Find dominant entity for label"""
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
    registry: MetabolicRegistry,
    depth: int,
    merge_id: int
) -> EU:
    """Create merged EU with metabolic state"""
    all_claims = set()
    all_entities = set()
    all_pages = set()
    child_ids = []

    for eu in eus:
        all_claims |= registry.all_claim_ids(eu.id)
        all_entities |= eu.entity_ids
        all_pages |= eu.page_ids
        child_ids.append(eu.id)

    corr, contra, coherence, stability = registry.compute_metabolic_state(all_claims)

    # Mass = claim count * stability bonus
    base_mass = len(all_claims) * 0.1
    mass = base_mass * (0.5 + stability)

    label = find_label(all_claims, registry.snapshot)

    return EU(
        id=f"L{depth}_{merge_id}",
        depth=depth,
        child_ids=child_ids,
        entity_ids=all_entities,
        page_ids=all_pages,
        corroborations=corr,
        contradictions=contra,
        coherence=coherence,
        stability=stability,
        mass=mass,
        label=label
    )


def emerge_layer_metabolic(
    registry: MetabolicRegistry,
    source_depth: int,
    min_energy: float = 0.5
) -> Tuple[int, Dict]:
    """
    Emerge next layer with metabolic absorption.

    Unlike entropy-based: we ALLOW merges that create tension,
    because tension is signal that needs metabolism.

    We just require sufficient energy (connection).
    """
    source_eus = registry.get_layer(source_depth)
    target_depth = source_depth + 1
    n = len(source_eus)

    if n < 2:
        return 0, {}

    # Find all pairs with sufficient energy
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            eu1, eu2 = source_eus[i], source_eus[j]
            energy, cross_corr, cross_contra = registry.compute_merge_energy(eu1, eu2)

            if energy >= min_energy:
                candidates.append((i, j, energy, cross_corr, cross_contra))

    if not candidates:
        # Pass through
        for i, eu in enumerate(source_eus):
            passthrough = EU(
                id=f"L{target_depth}_p{i}",
                depth=target_depth,
                child_ids=[eu.id],
                entity_ids=eu.entity_ids.copy(),
                page_ids=eu.page_ids.copy(),
                corroborations=eu.corroborations,
                contradictions=eu.contradictions,
                coherence=eu.coherence,
                stability=eu.stability,
                mass=eu.mass,
                label=eu.label
            )
            registry.add(passthrough)
        return 0, {'candidates': 0}

    # Sort by energy
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Greedy merge
    used = set()
    merges = []
    total_corr = 0
    total_contra = 0

    for i, j, energy, cross_corr, cross_contra in candidates:
        if i in used or j in used:
            continue
        merges.append((i, j))
        used.add(i)
        used.add(j)
        total_corr += cross_corr
        total_contra += cross_contra

    # Create merged EUs
    merge_id = 0
    for i, j in merges:
        eus_to_merge = [source_eus[i], source_eus[j]]
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
                corroborations=eu.corroborations,
                contradictions=eu.contradictions,
                coherence=eu.coherence,
                stability=eu.stability,
                mass=eu.mass,
                label=eu.label
            )
            registry.add(passthrough)
            merge_id += 1

    stats = {
        'candidates': len(candidates),
        'merges': len(merges),
        'total_cross_corr': total_corr,
        'total_cross_contra': total_contra
    }

    return len(merges), stats


def progressive_metabolic_emerge(
    snapshot: GraphSnapshot,
    max_depth: int = 10
) -> Tuple[MetabolicRegistry, Dict]:
    """Progressive emergence with metabolic model"""

    registry = MetabolicRegistry(snapshot)
    stats = {'layers': []}

    # Energy thresholds decrease slightly at higher layers
    # (larger clusters have more potential connection points)
    min_energy_by_depth = {
        0: 1.0,
        1: 0.8,
        2: 0.6,
        3: 0.5,
        4: 0.4,
        5: 0.3,
        6: 0.3,
        7: 0.3,
        8: 0.3,
        9: 0.3,
    }

    create_layer_0(registry)
    stats['layers'].append({
        'depth': 0,
        'eu_count': len(registry.get_layer(0)),
        'merges': 0
    })

    for depth in range(max_depth):
        min_energy = min_energy_by_depth.get(depth, 0.3)
        merges, merge_stats = emerge_layer_metabolic(registry, depth, min_energy)

        layer_eus = registry.get_layer(depth + 1)
        real_clusters = [eu for eu in layer_eus if len(eu.child_ids) > 1]

        # Compute layer statistics
        if real_clusters:
            avg_coherence = sum(eu.coherence for eu in real_clusters) / len(real_clusters)
            avg_stability = sum(eu.stability for eu in real_clusters) / len(real_clusters)
            total_contra = sum(eu.contradictions for eu in real_clusters)
            total_corr = sum(eu.corroborations for eu in real_clusters)
        else:
            avg_coherence = avg_stability = 0
            total_contra = total_corr = 0

        stats['layers'].append({
            'depth': depth + 1,
            'eu_count': len(layer_eus),
            'real_clusters': len(real_clusters),
            'merges': merges,
            'avg_coherence': avg_coherence,
            'avg_stability': avg_stability,
            'total_corr': total_corr,
            'total_contra': total_contra,
            **merge_stats
        })

        tension_str = f", tension={total_contra}" if total_contra > 0 else ""
        print(f"Layer {depth + 1}: {merges} merges → {len(real_clusters)} clusters (coh={avg_coherence:.2f}, stab={avg_stability:.3f}{tension_str})")

        if merges == 0:
            print(f"  No more energy for merges, stopping.")
            break

    return registry, stats


def analyze_clusters(registry: MetabolicRegistry) -> Dict:
    """Analyze emerged clusters"""

    analysis = {
        'by_depth': {},
        'stable_clusters': [],
        'active_clusters': [],  # High tension, needs metabolism
    }

    all_clusters = []
    for depth in registry.by_depth.keys():
        if depth == 0:
            continue
        for eu in registry.get_layer(depth):
            if len(eu.child_ids) > 1:
                all_clusters.append(eu)

    # By depth
    for depth in registry.by_depth.keys():
        eus = registry.get_layer(depth)
        real = [eu for eu in eus if len(eu.child_ids) != 1]
        if real:
            sorted_eus = sorted(real, key=lambda x: x.mass, reverse=True)
            analysis['by_depth'][depth] = {
                'count': len(real),
                'top': [
                    {
                        'label': eu.label,
                        'claims': registry.claim_count(eu.id),
                        'mass': eu.mass,
                        'coherence': eu.coherence,
                        'stability': eu.stability,
                        'corr': eu.corroborations,
                        'contra': eu.contradictions,
                        'tension': eu.tension_ratio()
                    }
                    for eu in sorted_eus[:7]
                ]
            }

    # Categorize by stability
    for eu in all_clusters:
        claims = registry.claim_count(eu.id)
        if claims < 5:
            continue

        info = {
            'label': eu.label,
            'depth': eu.depth,
            'claims': claims,
            'coherence': eu.coherence,
            'stability': eu.stability,
            'corr': eu.corroborations,
            'contra': eu.contradictions
        }

        if eu.stability > 0.1 and eu.coherence > 0.7:
            analysis['stable_clusters'].append(info)
        elif eu.contradictions > 2:
            analysis['active_clusters'].append(info)

    analysis['stable_clusters'].sort(key=lambda x: x['claims'], reverse=True)
    analysis['active_clusters'].sort(key=lambda x: x['contra'], reverse=True)

    return analysis


def main():
    print("=" * 60)
    print("Metabolic Emergence")
    print("=" * 60)
    print("\nClaims start unstable. Contradictions are signal.")
    print("Clusters grow through absorption and metabolism.\n")

    snapshot = load_snapshot()
    print(f"Loaded {len(snapshot.claims)} claims\n")

    registry, stats = progressive_metabolic_emerge(snapshot, max_depth=10)

    analysis = analyze_clusters(registry)

    print("\n" + "=" * 60)
    print("Hierarchy by Depth")
    print("=" * 60)

    for depth, info in analysis['by_depth'].items():
        print(f"\nDepth {depth}: {info['count']} clusters")
        for c in info['top']:
            tension_str = f" T={c['tension']:.0%}" if c['contra'] > 0 else ""
            print(f"  [{c['claims']}] {c['label']} (coh={c['coherence']:.2f}, stab={c['stability']:.3f}, +{c['corr']}/-{c['contra']}{tension_str})")

    print("\n" + "=" * 60)
    print("Stable Clusters (high coherence, low tension)")
    print("=" * 60)
    for c in analysis['stable_clusters'][:10]:
        print(f"  {c['label']}: {c['claims']} claims, coh={c['coherence']:.2f}, +{c['corr']}/-{c['contra']}")

    print("\n" + "=" * 60)
    print("Active Clusters (have contradictions to metabolize)")
    print("=" * 60)
    for c in analysis['active_clusters'][:10]:
        print(f"  {c['label']}: {c['claims']} claims, {c['contra']} contradictions, coh={c['coherence']:.2f}")

    # Compare with known coherent event: Wang Fuk Court Fire
    print("\n" + "=" * 60)
    print("Reference Check: Wang Fuk Court Fire")
    print("=" * 60)

    # Find Wang Fuk Court clusters
    wfc_clusters = []
    for depth in registry.by_depth.keys():
        for eu in registry.get_layer(depth):
            if 'Wang Fuk' in eu.label or 'Wang Fuk Court' in eu.label:
                if len(eu.child_ids) > 1:
                    wfc_clusters.append(eu)

    if wfc_clusters:
        largest = max(wfc_clusters, key=lambda x: registry.claim_count(x.id))
        print(f"\nLargest 'Wang Fuk' cluster:")
        print(f"  Depth: {largest.depth}")
        print(f"  Claims: {registry.claim_count(largest.id)}")
        print(f"  Coherence: {largest.coherence:.2f}")
        print(f"  Stability: {largest.stability:.3f}")
        print(f"  Corroborations: {largest.corroborations}")
        print(f"  Contradictions: {largest.contradictions}")

    # Save
    output_path = Path("/app/test_eu/results/metabolic_emergence.json")
    with open(output_path, 'w') as f:
        json.dump({
            'stats': stats,
            'analysis': analysis
        }, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
