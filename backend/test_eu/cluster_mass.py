"""
Revised Mass Experiment: Cluster-Level Mass

Mass is NOT a property of individual claims.
Mass is an EMERGENT property of clusters.

A cluster gains mass through:
- Claim count (more claims = more mass)
- Corroboration density (internal support)
- Tension (contradictions create "heat")
- Entity significance (important actors)
- Source diversity (multiple pages)
- Temporal span (sustained attention)

Run inside container:
    docker exec herenews-app python /app/test_eu/cluster_mass.py
"""

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from pathlib import Path
from collections import defaultdict

from load_graph import load_snapshot, GraphSnapshot, ClaimData


@dataclass
class ClusterMass:
    """Mass is a cluster-level property, not claim-level"""
    claim_count: float = 0.0           # Raw count of claims
    corroboration_density: float = 0.0  # Internal CORROBORATES / possible pairs
    tension: float = 0.0                # Internal CONTRADICTS (normalized)
    entity_significance: float = 0.0    # Weighted entity importance
    source_diversity: float = 0.0       # Distinct pages
    temporal_span: float = 0.0          # Days of coverage

    def total(self, weights: Dict[str, float] = None) -> float:
        """Weighted sum - tunable per use case"""
        w = weights or {
            'claim_count': 0.20,
            'corroboration_density': 0.20,
            'tension': 0.15,
            'entity_significance': 0.20,
            'source_diversity': 0.15,
            'temporal_span': 0.10
        }
        return (
            self.claim_count * w['claim_count'] +
            self.corroboration_density * w['corroboration_density'] +
            self.tension * w['tension'] +
            self.entity_significance * w['entity_significance'] +
            self.source_diversity * w['source_diversity'] +
            self.temporal_span * w['temporal_span']
        )

    def __repr__(self):
        return (f"ClusterMass(claims={self.claim_count:.2f}, corr={self.corroboration_density:.2f}, "
                f"tension={self.tension:.2f}, entity={self.entity_significance:.2f}, "
                f"sources={self.source_diversity:.2f}, temporal={self.temporal_span:.2f}, "
                f"total={self.total():.2f})")


@dataclass
class Cluster:
    """A cluster of claims that may become a browsable bundle"""
    id: str
    claim_ids: Set[str]
    core_entities: List[str]  # Entity IDs that define this cluster
    mass: Optional[ClusterMass] = None

    # For comparison with existing events
    matched_event_id: Optional[str] = None
    matched_event_name: Optional[str] = None


def find_entity_clusters(
    snapshot: GraphSnapshot,
    min_entity_mentions: int = 10,
    min_cluster_size: int = 3
) -> List[Cluster]:
    """
    Find clusters by grouping claims around significant entities.
    Each significant entity seeds a potential cluster.
    """
    # Find significant entities
    significant = [
        (eid, entity) for eid, entity in snapshot.entities.items()
        if entity.mention_count >= min_entity_mentions
    ]

    clusters = []
    for eid, entity in significant:
        # Find all claims mentioning this entity
        claim_ids = set()
        for cid, claim in snapshot.claims.items():
            if eid in claim.entity_ids:
                claim_ids.add(cid)

        if len(claim_ids) >= min_cluster_size:
            clusters.append(Cluster(
                id=f"cluster_{entity.canonical_name.lower().replace(' ', '_')}",
                claim_ids=claim_ids,
                core_entities=[eid]
            ))

    return clusters


def compute_cluster_mass(
    cluster: Cluster,
    snapshot: GraphSnapshot,
    max_entity_mentions: int,
    max_pages: int
) -> ClusterMass:
    """
    Compute mass for a cluster of claims.
    All values normalized to roughly 0-1 range.
    """
    mass = ClusterMass()
    claims = [snapshot.claims[cid] for cid in cluster.claim_ids if cid in snapshot.claims]

    if not claims:
        return mass

    # 1. Claim count (log scale, diminishing returns)
    # 10 claims = 0.5, 100 claims = 1.0
    mass.claim_count = min(1.0, math.log1p(len(claims)) / math.log1p(100))

    # 2. Corroboration density
    # Count internal CORROBORATES links / possible pairs
    internal_corr = 0
    for cid in cluster.claim_ids:
        claim = snapshot.claims.get(cid)
        if claim:
            internal_corr += len([c for c in claim.corroborates_ids if c in cluster.claim_ids])

    possible_pairs = len(claims) * (len(claims) - 1) / 2 if len(claims) > 1 else 1
    mass.corroboration_density = min(1.0, (internal_corr / 2) / possible_pairs * 10)  # Scale up

    # 3. Tension (contradictions)
    internal_contra = 0
    for cid in cluster.claim_ids:
        claim = snapshot.claims.get(cid)
        if claim:
            internal_contra += len([c for c in claim.contradicts_ids if c in cluster.claim_ids])

    # Contradictions are rarer, so weight them higher
    mass.tension = min(1.0, internal_contra / 5)  # 5+ contradictions = max tension

    # 4. Entity significance
    # Average importance of entities in this cluster
    all_entity_ids = set()
    for claim in claims:
        all_entity_ids.update(claim.entity_ids)

    if all_entity_ids:
        entity_scores = []
        for eid in all_entity_ids:
            entity = snapshot.entities.get(eid)
            if entity:
                entity_scores.append(entity.mention_count / max_entity_mentions)
        mass.entity_significance = sum(entity_scores) / len(entity_scores) if entity_scores else 0

    # 5. Source diversity
    # How many distinct pages contributed claims
    pages = set()
    for claim in claims:
        if claim.page_id:
            pages.add(claim.page_id)

    mass.source_diversity = min(1.0, len(pages) / max_pages * 5)  # 20% of pages = 1.0

    # 6. Temporal span
    # TODO: Parse event_time and compute span in days
    # For now, estimate from claim count (more claims usually = longer coverage)
    mass.temporal_span = min(1.0, len(claims) / 50)  # 50+ claims = sustained

    return mass


def match_cluster_to_event(
    cluster: Cluster,
    snapshot: GraphSnapshot
) -> tuple[Optional[str], Optional[str], float]:
    """Find best matching existing event"""
    best_id = None
    best_name = None
    best_overlap = 0.0

    for eid, event in snapshot.events.items():
        event_claims = set(event.claim_ids)
        intersection = cluster.claim_ids & event_claims
        union = cluster.claim_ids | event_claims
        overlap = len(intersection) / len(union) if union else 0

        if overlap > best_overlap:
            best_overlap = overlap
            best_id = eid
            best_name = event.canonical_name

    return best_id, best_name, best_overlap


def main():
    print("=" * 60)
    print("EU Experiment: Cluster-Level Mass")
    print("=" * 60)

    # Load data
    snapshot = load_snapshot()
    print(f"\nLoaded: {len(snapshot.claims)} claims, {len(snapshot.entities)} entities, {len(snapshot.events)} events")

    # Normalization constants
    max_entity_mentions = max(e.mention_count for e in snapshot.entities.values())
    max_pages = len(snapshot.pages)

    print(f"Max entity mentions: {max_entity_mentions}")
    print(f"Total pages: {max_pages}")

    # Find clusters
    print("\nFinding entity-based clusters...")
    clusters = find_entity_clusters(snapshot, min_entity_mentions=10, min_cluster_size=5)
    print(f"Found {len(clusters)} clusters")

    # Compute mass for each cluster
    print("Computing cluster mass...")
    for cluster in clusters:
        cluster.mass = compute_cluster_mass(cluster, snapshot, max_entity_mentions, max_pages)
        cluster.matched_event_id, cluster.matched_event_name, overlap = match_cluster_to_event(cluster, snapshot)

    # Sort by mass
    clusters.sort(key=lambda c: c.mass.total(), reverse=True)

    # Results
    print("\n" + "=" * 60)
    print("Clusters Ranked by Mass")
    print("=" * 60)

    for i, cluster in enumerate(clusters[:15], 1):
        entity_names = [
            snapshot.entities[eid].canonical_name
            for eid in cluster.core_entities
            if eid in snapshot.entities
        ]
        print(f"\n{i}. {', '.join(entity_names)} ({len(cluster.claim_ids)} claims)")
        print(f"   Mass: {cluster.mass}")
        if cluster.matched_event_name:
            print(f"   Matches: {cluster.matched_event_name}")

    # Compare with existing events
    print("\n" + "=" * 60)
    print("Existing Events vs Cluster Mass")
    print("=" * 60)

    for eid, event in snapshot.events.items():
        # Create a "cluster" from this event's claims
        event_cluster = Cluster(
            id=eid,
            claim_ids=set(event.claim_ids),
            core_entities=event.entity_ids
        )
        event_cluster.mass = compute_cluster_mass(event_cluster, snapshot, max_entity_mentions, max_pages)

        print(f"\n{event.canonical_name}")
        print(f"  Claims: {len(event.claim_ids)}")
        print(f"  Mass: {event_cluster.mass.total():.3f}")
        print(f"    - claim_count: {event_cluster.mass.claim_count:.2f}")
        print(f"    - corr_density: {event_cluster.mass.corroboration_density:.2f}")
        print(f"    - tension: {event_cluster.mass.tension:.2f}")
        print(f"    - entity_sig: {event_cluster.mass.entity_significance:.2f}")
        print(f"    - source_div: {event_cluster.mass.source_diversity:.2f}")

    # Browsable threshold analysis
    print("\n" + "=" * 60)
    print("Browsable Threshold Analysis (Cluster Level)")
    print("=" * 60)

    all_masses = [c.mass.total() for c in clusters]
    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
        browsable = [c for c in clusters if c.mass.total() >= threshold]
        print(f"\nThreshold >= {threshold}:")
        print(f"  {len(browsable)} clusters would be browsable")
        for c in browsable[:5]:
            entity_names = [snapshot.entities[eid].canonical_name for eid in c.core_entities if eid in snapshot.entities]
            print(f"    - {', '.join(entity_names)} (mass={c.mass.total():.2f}, claims={len(c.claim_ids)})")

    # Save results
    output = {
        'clusters': [
            {
                'id': c.id,
                'claim_count': len(c.claim_ids),
                'core_entities': [snapshot.entities[eid].canonical_name for eid in c.core_entities if eid in snapshot.entities],
                'mass_total': c.mass.total(),
                'mass_components': {
                    'claim_count': c.mass.claim_count,
                    'corroboration_density': c.mass.corroboration_density,
                    'tension': c.mass.tension,
                    'entity_significance': c.mass.entity_significance,
                    'source_diversity': c.mass.source_diversity,
                    'temporal_span': c.mass.temporal_span
                },
                'matched_event': c.matched_event_name
            }
            for c in clusters
        ]
    }

    output_path = Path("/app/test_eu/results/cluster_mass.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
