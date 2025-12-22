"""
Experiment 3: Emergence Detection

Can we identify when claim clusters "should" become narrative bundles
without explicit parent/child relationships?

Emergence signals:
- Entity overlap (claims share key entities)
- Corroboration density (claims support each other)
- Temporal clustering (claims within time window)
- Combined mass threshold

Run inside container:
    docker exec herenews-app python /app/test_eu/emergence_detector.py
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

from load_graph import load_snapshot, GraphSnapshot, ClaimData
from mass_calculator import compute_all_masses, MassVector


@dataclass
class EmergentCluster:
    """A detected cluster of claims that could form a narrative bundle"""
    claim_ids: List[str]
    core_entities: List[str]  # Shared entities
    total_mass: float
    mean_mass: float
    internal_links: int  # CORROBORATES/CONTRADICTS within cluster
    coherence_score: float
    # Comparison with existing events
    matched_event_id: Optional[str] = None
    matched_event_name: Optional[str] = None
    overlap_with_event: float = 0.0


def compute_entity_overlap(claims: List[ClaimData]) -> Tuple[Set[str], float]:
    """
    Find entities shared across claims and compute overlap score.
    Returns (shared_entities, overlap_score)
    """
    if len(claims) < 2:
        return set(), 0.0

    # Get entity sets for each claim
    entity_sets = [set(c.entity_ids) for c in claims if c.entity_ids]
    if len(entity_sets) < 2:
        return set(), 0.0

    # Find intersection
    shared = entity_sets[0]
    for es in entity_sets[1:]:
        shared = shared & es

    # Compute Jaccard-like overlap
    union = set()
    for es in entity_sets:
        union = union | es

    overlap_score = len(shared) / len(union) if union else 0.0

    return shared, overlap_score


def compute_internal_links(claim_ids: Set[str], snapshot: GraphSnapshot) -> int:
    """Count CORROBORATES/CONTRADICTS/UPDATES links within cluster"""
    links = 0
    for cid in claim_ids:
        claim = snapshot.claims.get(cid)
        if not claim:
            continue
        # Count outgoing links to other cluster members
        for target_id in claim.corroborates_ids:
            if target_id in claim_ids:
                links += 1
        for target_id in claim.contradicts_ids:
            if target_id in claim_ids:
                links += 1
        for target_id in claim.updates_ids:
            if target_id in claim_ids:
                links += 1
    return links


def find_clusters_by_entity(
    snapshot: GraphSnapshot,
    min_entity_mentions: int = 10,
    min_cluster_size: int = 5
) -> List[Set[str]]:
    """
    Find claim clusters by grouping claims that share high-frequency entities.
    """
    # Find significant entities
    significant_entities = [
        eid for eid, entity in snapshot.entities.items()
        if entity.mention_count >= min_entity_mentions
    ]

    # Build entity -> claims mapping
    entity_to_claims: Dict[str, Set[str]] = defaultdict(set)
    for cid, claim in snapshot.claims.items():
        for eid in claim.entity_ids:
            if eid in significant_entities:
                entity_to_claims[eid].add(cid)

    # Find clusters (claims sharing entity)
    clusters = []
    for eid, claim_set in entity_to_claims.items():
        if len(claim_set) >= min_cluster_size:
            clusters.append(claim_set)

    # Merge overlapping clusters
    merged = merge_overlapping_clusters(clusters, overlap_threshold=0.5)

    return merged


def merge_overlapping_clusters(
    clusters: List[Set[str]],
    overlap_threshold: float = 0.5
) -> List[Set[str]]:
    """Merge clusters with significant overlap"""
    if not clusters:
        return []

    merged = []
    used = set()

    for i, c1 in enumerate(clusters):
        if i in used:
            continue

        current = set(c1)
        for j, c2 in enumerate(clusters[i+1:], i+1):
            if j in used:
                continue
            # Check overlap
            intersection = current & c2
            union = current | c2
            overlap = len(intersection) / len(union) if union else 0

            if overlap >= overlap_threshold:
                current = union
                used.add(j)

        merged.append(current)
        used.add(i)

    return merged


def match_cluster_to_events(
    cluster: Set[str],
    snapshot: GraphSnapshot
) -> Tuple[Optional[str], Optional[str], float]:
    """
    Find which existing event best matches this cluster.
    Returns (event_id, event_name, overlap_score)
    """
    best_match = None
    best_name = None
    best_overlap = 0.0

    for eid, event in snapshot.events.items():
        event_claims = set(event.claim_ids)
        intersection = cluster & event_claims
        union = cluster | event_claims

        overlap = len(intersection) / len(union) if union else 0

        if overlap > best_overlap:
            best_overlap = overlap
            best_match = eid
            best_name = event.canonical_name

    return best_match, best_name, best_overlap


def detect_emergent_clusters(
    snapshot: GraphSnapshot,
    masses: Dict[str, MassVector],
    min_cluster_mass: float = 1.0,
    min_cluster_size: int = 5
) -> List[EmergentCluster]:
    """
    Detect clusters that have accumulated enough mass to be considered
    emergent narrative bundles.
    """
    # Find entity-based clusters
    raw_clusters = find_clusters_by_entity(snapshot, min_entity_mentions=8, min_cluster_size=min_cluster_size)

    emergent = []
    for claim_ids in raw_clusters:
        claims = [snapshot.claims[cid] for cid in claim_ids if cid in snapshot.claims]
        if len(claims) < min_cluster_size:
            continue

        # Compute cluster mass
        cluster_masses = [masses[cid].total() for cid in claim_ids if cid in masses]
        if not cluster_masses:
            continue

        total_mass = sum(cluster_masses)
        mean_mass = total_mass / len(cluster_masses)

        # Skip low-mass clusters
        if total_mass < min_cluster_mass:
            continue

        # Find shared entities
        shared_entities, entity_overlap = compute_entity_overlap(claims)

        # Count internal links
        internal_links = compute_internal_links(claim_ids, snapshot)

        # Compute coherence (combination of entity overlap and internal links)
        link_density = internal_links / (len(claim_ids) * (len(claim_ids) - 1) / 2) if len(claim_ids) > 1 else 0
        coherence = 0.5 * entity_overlap + 0.5 * min(1.0, link_density * 5)

        # Match to existing events
        matched_id, matched_name, overlap = match_cluster_to_events(claim_ids, snapshot)

        # Get entity names for readability
        core_entity_names = [
            snapshot.entities[eid].canonical_name
            for eid in list(shared_entities)[:5]
            if eid in snapshot.entities
        ]

        emergent.append(EmergentCluster(
            claim_ids=list(claim_ids),
            core_entities=core_entity_names,
            total_mass=total_mass,
            mean_mass=mean_mass,
            internal_links=internal_links,
            coherence_score=coherence,
            matched_event_id=matched_id,
            matched_event_name=matched_name,
            overlap_with_event=overlap
        ))

    # Sort by total mass
    emergent.sort(key=lambda x: x.total_mass, reverse=True)

    return emergent


def analyze_emergence_accuracy(
    clusters: List[EmergentCluster],
    snapshot: GraphSnapshot
) -> Dict:
    """
    How well do detected clusters match existing event boundaries?
    """
    results = {
        'total_clusters': len(clusters),
        'high_overlap': 0,      # >70% overlap with existing event
        'medium_overlap': 0,    # 30-70% overlap
        'low_overlap': 0,       # <30% overlap (potential new events)
        'matches': [],
        'potential_new_events': []
    }

    for cluster in clusters:
        if cluster.overlap_with_event >= 0.7:
            results['high_overlap'] += 1
            results['matches'].append({
                'cluster_size': len(cluster.claim_ids),
                'matched_event': cluster.matched_event_name,
                'overlap': cluster.overlap_with_event,
                'core_entities': cluster.core_entities
            })
        elif cluster.overlap_with_event >= 0.3:
            results['medium_overlap'] += 1
        else:
            results['low_overlap'] += 1
            # These might be events that the current system missed
            results['potential_new_events'].append({
                'cluster_size': len(cluster.claim_ids),
                'total_mass': cluster.total_mass,
                'core_entities': cluster.core_entities,
                'sample_claims': [
                    snapshot.claims[cid].text[:80]
                    for cid in cluster.claim_ids[:3]
                    if cid in snapshot.claims
                ]
            })

    # Calculate accuracy
    if clusters:
        results['match_rate'] = results['high_overlap'] / len(clusters)
    else:
        results['match_rate'] = 0

    return results


def main():
    print("=" * 60)
    print("EU Experiment 3: Emergence Detection")
    print("=" * 60)

    # Load data
    snapshot = load_snapshot()
    print(f"\nLoaded: {len(snapshot.claims)} claims, {len(snapshot.events)} events")

    # Compute masses
    print("Computing masses...")
    masses = compute_all_masses(snapshot)

    # Detect emergent clusters
    print("Detecting emergent clusters...")
    clusters = detect_emergent_clusters(
        snapshot, masses,
        min_cluster_mass=2.0,
        min_cluster_size=5
    )

    print(f"\nDetected {len(clusters)} potential narrative bundles")

    # Analyze accuracy
    accuracy = analyze_emergence_accuracy(clusters, snapshot)

    print("\n" + "=" * 60)
    print("Emergence Detection Results")
    print("=" * 60)

    print(f"\nTotal clusters detected: {accuracy['total_clusters']}")
    print(f"High overlap with existing events (>70%): {accuracy['high_overlap']}")
    print(f"Medium overlap (30-70%): {accuracy['medium_overlap']}")
    print(f"Low overlap (<30%): {accuracy['low_overlap']}")
    print(f"Match rate: {accuracy['match_rate']*100:.1f}%")

    print("\n" + "=" * 60)
    print("Matched Clusters (correctly identified existing events)")
    print("=" * 60)
    for match in accuracy['matches'][:10]:
        print(f"\n  -> {match['matched_event']} ({match['overlap']*100:.0f}% overlap)")
        print(f"     Cluster size: {match['cluster_size']}")
        print(f"     Core entities: {', '.join(match['core_entities'][:3])}")

    print("\n" + "=" * 60)
    print("Potential New Events (low overlap with existing)")
    print("=" * 60)
    for pot in accuracy['potential_new_events'][:5]:
        print(f"\n  Cluster of {pot['cluster_size']} claims (mass={pot['total_mass']:.2f})")
        print(f"  Core entities: {', '.join(pot['core_entities'][:3])}")
        print(f"  Sample claims:")
        for claim in pot['sample_claims']:
            print(f"    - {claim}...")

    # Compare with existing events
    print("\n" + "=" * 60)
    print("Comparison: Detected vs Existing Events")
    print("=" * 60)

    print("\nExisting events and their detection status:")
    for eid, event in snapshot.events.items():
        # Find best matching cluster
        best_cluster = None
        best_overlap = 0
        for cluster in clusters:
            if cluster.matched_event_id == eid and cluster.overlap_with_event > best_overlap:
                best_overlap = cluster.overlap_with_event
                best_cluster = cluster

        status = "DETECTED" if best_overlap >= 0.5 else "PARTIAL" if best_overlap >= 0.2 else "MISSED"
        print(f"  [{status}] {event.canonical_name} ({len(event.claim_ids)} claims)")
        if best_cluster:
            print(f"          Best cluster overlap: {best_overlap*100:.0f}%")

    # Save results
    output = {
        'accuracy': accuracy,
        'clusters': [
            {
                'size': len(c.claim_ids),
                'total_mass': c.total_mass,
                'mean_mass': c.mean_mass,
                'coherence': c.coherence_score,
                'core_entities': c.core_entities,
                'matched_event': c.matched_event_name,
                'overlap': c.overlap_with_event
            }
            for c in clusters
        ]
    }
    output_path = Path("/app/test_eu/results/emergence_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
