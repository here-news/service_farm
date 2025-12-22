"""
Pure Emergence Experiment

Forget existing events. Start from raw claims.
Let clusters emerge naturally through:
1. Entity co-occurrence
2. Corroboration links (CORROBORATES/CONTRADICTS)
3. Source proximity (same page)

Then compute mass on emergent clusters and see what surfaces.

Run inside container:
    docker exec herenews-app python /app/test_eu/pure_emergence.py
"""

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from load_graph import load_snapshot, GraphSnapshot, ClaimData


@dataclass
class EmergentCluster:
    """A naturally emerged cluster - no reference to existing events"""
    id: str
    claim_ids: Set[str]

    # How it formed
    seed_type: str  # 'entity', 'corroboration', 'page'
    seed_value: str  # entity name, claim id, page id

    # Mass components (computed after formation)
    claim_count: int = 0
    internal_corroborations: int = 0
    internal_contradictions: int = 0
    entity_ids: Set[str] = field(default_factory=set)
    page_ids: Set[str] = field(default_factory=set)

    # Computed mass
    mass: float = 0.0


def build_claim_graph(snapshot: GraphSnapshot) -> Dict[str, Set[str]]:
    """
    Build adjacency list of claims connected by:
    - CORROBORATES
    - CONTRADICTS
    - UPDATES
    - Shared entities (with weight threshold)
    """
    graph = defaultdict(set)

    for cid, claim in snapshot.claims.items():
        # Direct relationships
        for target in claim.corroborates_ids:
            if target in snapshot.claims:
                graph[cid].add(target)
                graph[target].add(cid)

        for target in claim.contradicts_ids:
            if target in snapshot.claims:
                graph[cid].add(target)
                graph[target].add(cid)

        for target in claim.updates_ids:
            if target in snapshot.claims:
                graph[cid].add(target)
                graph[target].add(cid)

    return graph


def build_entity_index(snapshot: GraphSnapshot) -> Dict[str, Set[str]]:
    """Map entity_id -> claim_ids that mention it"""
    index = defaultdict(set)
    for cid, claim in snapshot.claims.items():
        for eid in claim.entity_ids:
            index[eid].add(cid)
    return index


def build_page_index(snapshot: GraphSnapshot) -> Dict[str, Set[str]]:
    """Map page_id -> claim_ids from that page"""
    index = defaultdict(set)
    for cid, claim in snapshot.claims.items():
        if claim.page_id:
            index[claim.page_id].add(cid)
    return index


def find_connected_component(
    start: str,
    graph: Dict[str, Set[str]],
    visited: Set[str]
) -> Set[str]:
    """BFS to find connected component"""
    component = set()
    queue = [start]

    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        component.add(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                queue.append(neighbor)

    return component


def emerge_from_corroboration_graph(
    snapshot: GraphSnapshot,
    min_size: int = 3
) -> List[EmergentCluster]:
    """
    Find clusters via connected components in corroboration graph.
    Claims linked by CORROBORATES/CONTRADICTS/UPDATES form natural clusters.
    """
    graph = build_claim_graph(snapshot)
    visited = set()
    clusters = []

    for cid in snapshot.claims:
        if cid not in visited and cid in graph:
            component = find_connected_component(cid, graph, visited)
            if len(component) >= min_size:
                clusters.append(EmergentCluster(
                    id=f"corr_cluster_{len(clusters)}",
                    claim_ids=component,
                    seed_type='corroboration',
                    seed_value=cid
                ))

    return clusters


def emerge_from_entity_co_occurrence(
    snapshot: GraphSnapshot,
    entity_index: Dict[str, Set[str]],
    min_entity_mentions: int = 15,
    min_cluster_size: int = 5
) -> List[EmergentCluster]:
    """
    Find clusters where claims share significant entities.
    Only use entities with enough mentions to be meaningful.
    """
    clusters = []

    # Find significant entities
    for eid, claim_ids in entity_index.items():
        entity = snapshot.entities.get(eid)
        if not entity:
            continue
        if entity.mention_count < min_entity_mentions:
            continue
        if len(claim_ids) < min_cluster_size:
            continue

        clusters.append(EmergentCluster(
            id=f"entity_cluster_{entity.canonical_name.lower().replace(' ', '_')}",
            claim_ids=claim_ids.copy(),
            seed_type='entity',
            seed_value=entity.canonical_name
        ))

    return clusters


def emerge_from_page_proximity(
    snapshot: GraphSnapshot,
    page_index: Dict[str, Set[str]],
    min_claims_per_page: int = 5
) -> List[EmergentCluster]:
    """
    Claims from same page are naturally related.
    Pages with many claims may represent significant stories.
    """
    clusters = []

    for page_id, claim_ids in page_index.items():
        if len(claim_ids) < min_claims_per_page:
            continue

        page = snapshot.pages.get(page_id)
        page_name = page.title[:50] if page and page.title else page_id

        clusters.append(EmergentCluster(
            id=f"page_cluster_{page_id}",
            claim_ids=claim_ids.copy(),
            seed_type='page',
            seed_value=page_name
        ))

    return clusters


def merge_overlapping_clusters(
    clusters: List[EmergentCluster],
    overlap_threshold: float = 0.5
) -> List[EmergentCluster]:
    """
    Merge clusters with significant claim overlap.
    This consolidates different emergence paths that found the same story.
    """
    if not clusters:
        return []

    # Sort by size (largest first)
    clusters = sorted(clusters, key=lambda c: len(c.claim_ids), reverse=True)

    merged = []
    absorbed = set()

    for i, c1 in enumerate(clusters):
        if i in absorbed:
            continue

        current_claims = c1.claim_ids.copy()
        seed_types = [c1.seed_type]
        seed_values = [c1.seed_value]

        for j, c2 in enumerate(clusters[i+1:], i+1):
            if j in absorbed:
                continue

            intersection = current_claims & c2.claim_ids
            smaller = min(len(current_claims), len(c2.claim_ids))

            if smaller > 0 and len(intersection) / smaller >= overlap_threshold:
                current_claims |= c2.claim_ids
                seed_types.append(c2.seed_type)
                seed_values.append(c2.seed_value)
                absorbed.add(j)

        # Create merged cluster
        merged.append(EmergentCluster(
            id=f"merged_{len(merged)}",
            claim_ids=current_claims,
            seed_type='+'.join(set(seed_types)),
            seed_value=seed_values[0]  # Primary seed
        ))

    return merged


def compute_cluster_mass(
    cluster: EmergentCluster,
    snapshot: GraphSnapshot
) -> float:
    """
    Compute mass for an emergent cluster.
    """
    claims = [snapshot.claims[cid] for cid in cluster.claim_ids if cid in snapshot.claims]
    if not claims:
        return 0.0

    # Claim count (log scale)
    claim_score = min(1.0, math.log1p(len(claims)) / math.log1p(100))

    # Internal corroborations
    internal_corr = 0
    internal_contra = 0
    for cid in cluster.claim_ids:
        claim = snapshot.claims.get(cid)
        if claim:
            internal_corr += len([c for c in claim.corroborates_ids if c in cluster.claim_ids])
            internal_contra += len([c for c in claim.contradicts_ids if c in cluster.claim_ids])

    cluster.internal_corroborations = internal_corr // 2  # Counted twice
    cluster.internal_contradictions = internal_contra // 2

    # Corroboration density
    possible_pairs = len(claims) * (len(claims) - 1) / 2 if len(claims) > 1 else 1
    corr_score = min(1.0, (internal_corr / 2) / possible_pairs * 10)

    # Tension (contradictions are valuable signal)
    tension_score = min(1.0, internal_contra / 10)

    # Entity diversity
    all_entities = set()
    for claim in claims:
        all_entities.update(claim.entity_ids)
    cluster.entity_ids = all_entities
    entity_score = min(1.0, len(all_entities) / 20)

    # Source diversity
    pages = set()
    for claim in claims:
        if claim.page_id:
            pages.add(claim.page_id)
    cluster.page_ids = pages
    source_score = min(1.0, len(pages) / 10)

    cluster.claim_count = len(claims)

    # Weighted sum
    mass = (
        0.25 * claim_score +
        0.20 * corr_score +
        0.20 * tension_score +
        0.15 * entity_score +
        0.20 * source_score
    )

    cluster.mass = mass
    return mass


def compare_with_existing_events(
    clusters: List[EmergentCluster],
    snapshot: GraphSnapshot
) -> List[dict]:
    """
    After pure emergence, compare with existing events.
    This is VALIDATION only - not used in emergence.
    """
    comparisons = []

    for event in snapshot.events.values():
        event_claims = set(event.claim_ids)

        # Find best matching cluster
        best_cluster = None
        best_overlap = 0.0

        for cluster in clusters:
            intersection = cluster.claim_ids & event_claims
            union = cluster.claim_ids | event_claims
            overlap = len(intersection) / len(union) if union else 0

            if overlap > best_overlap:
                best_overlap = overlap
                best_cluster = cluster

        comparisons.append({
            'event_name': event.canonical_name,
            'event_claims': len(event.claim_ids),
            'best_cluster_id': best_cluster.id if best_cluster else None,
            'best_cluster_size': len(best_cluster.claim_ids) if best_cluster else 0,
            'best_cluster_seed': best_cluster.seed_value if best_cluster else None,
            'overlap': best_overlap,
            'status': 'FOUND' if best_overlap >= 0.5 else 'PARTIAL' if best_overlap >= 0.2 else 'MISSED'
        })

    return comparisons


def main():
    print("=" * 60)
    print("Pure Emergence Experiment")
    print("=" * 60)
    print("\nForget existing events. Let clusters emerge from claims.\n")

    # Load data
    snapshot = load_snapshot()
    print(f"Starting with {len(snapshot.claims)} claims (ignoring {len(snapshot.events)} existing events)")

    # Build indices
    entity_index = build_entity_index(snapshot)
    page_index = build_page_index(snapshot)

    # Emerge clusters via different methods
    print("\n--- Emergence Phase ---")

    print("\n1. Corroboration graph clustering...")
    corr_clusters = emerge_from_corroboration_graph(snapshot, min_size=3)
    print(f"   Found {len(corr_clusters)} clusters from corroboration links")

    print("\n2. Entity co-occurrence clustering...")
    entity_clusters = emerge_from_entity_co_occurrence(
        snapshot, entity_index,
        min_entity_mentions=15,
        min_cluster_size=5
    )
    print(f"   Found {len(entity_clusters)} clusters from entity co-occurrence")

    print("\n3. Page proximity clustering...")
    page_clusters = emerge_from_page_proximity(snapshot, page_index, min_claims_per_page=5)
    print(f"   Found {len(page_clusters)} clusters from page proximity")

    # Combine all emergence paths
    all_clusters = corr_clusters + entity_clusters + page_clusters
    print(f"\nTotal before merging: {len(all_clusters)} clusters")

    # Merge overlapping clusters
    print("\n4. Merging overlapping clusters...")
    merged = merge_overlapping_clusters(all_clusters, overlap_threshold=0.4)
    print(f"   After merging: {len(merged)} clusters")

    # Compute mass
    print("\n5. Computing cluster mass...")
    for cluster in merged:
        compute_cluster_mass(cluster, snapshot)

    # Sort by mass
    merged.sort(key=lambda c: c.mass, reverse=True)

    # Results
    print("\n" + "=" * 60)
    print("Emergent Clusters (Top 20 by Mass)")
    print("=" * 60)

    for i, cluster in enumerate(merged[:20], 1):
        # Get representative entity names
        top_entities = []
        entity_counts = defaultdict(int)
        for cid in list(cluster.claim_ids)[:50]:
            claim = snapshot.claims.get(cid)
            if claim:
                for eid in claim.entity_ids:
                    entity = snapshot.entities.get(eid)
                    if entity:
                        entity_counts[entity.canonical_name] += 1

        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        entity_str = ', '.join([e[0] for e in top_entities])

        print(f"\n{i}. Cluster '{cluster.id}' (mass={cluster.mass:.3f})")
        print(f"   Claims: {cluster.claim_count}, Sources: {len(cluster.page_ids)}")
        print(f"   Seed: {cluster.seed_type} -> {cluster.seed_value}")
        print(f"   Top entities: {entity_str}")
        print(f"   Internal: {cluster.internal_corroborations} corr, {cluster.internal_contradictions} contra")

    # Validation against existing events
    print("\n" + "=" * 60)
    print("Validation: Do emergent clusters match existing events?")
    print("=" * 60)

    comparisons = compare_with_existing_events(merged, snapshot)

    found = sum(1 for c in comparisons if c['status'] == 'FOUND')
    partial = sum(1 for c in comparisons if c['status'] == 'PARTIAL')
    missed = sum(1 for c in comparisons if c['status'] == 'MISSED')

    print(f"\nResults: {found} FOUND, {partial} PARTIAL, {missed} MISSED out of {len(comparisons)} events")

    for comp in sorted(comparisons, key=lambda x: x['overlap'], reverse=True):
        print(f"\n  [{comp['status']}] {comp['event_name']}")
        print(f"      Event: {comp['event_claims']} claims")
        if comp['best_cluster_id']:
            print(f"      Best cluster: {comp['best_cluster_size']} claims, overlap={comp['overlap']:.0%}")
            print(f"      Seed: {comp['best_cluster_seed']}")

    # Coverage analysis
    print("\n" + "=" * 60)
    print("Coverage Analysis")
    print("=" * 60)

    all_clustered_claims = set()
    for cluster in merged:
        all_clustered_claims |= cluster.claim_ids

    unclustered = set(snapshot.claims.keys()) - all_clustered_claims

    print(f"\nTotal claims: {len(snapshot.claims)}")
    print(f"Clustered claims: {len(all_clustered_claims)} ({len(all_clustered_claims)/len(snapshot.claims)*100:.1f}%)")
    print(f"Unclustered claims: {len(unclustered)} ({len(unclustered)/len(snapshot.claims)*100:.1f}%)")

    # Save results
    output = {
        'emergence_stats': {
            'corr_clusters': len(corr_clusters),
            'entity_clusters': len(entity_clusters),
            'page_clusters': len(page_clusters),
            'merged_clusters': len(merged)
        },
        'clusters': [
            {
                'id': c.id,
                'claim_count': c.claim_count,
                'mass': c.mass,
                'seed_type': c.seed_type,
                'seed_value': c.seed_value,
                'internal_corr': c.internal_corroborations,
                'internal_contra': c.internal_contradictions,
                'page_count': len(c.page_ids)
            }
            for c in merged
        ],
        'validation': comparisons,
        'coverage': {
            'total_claims': len(snapshot.claims),
            'clustered': len(all_clustered_claims),
            'unclustered': len(unclustered)
        }
    }

    output_path = Path("/app/test_eu/results/pure_emergence.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
