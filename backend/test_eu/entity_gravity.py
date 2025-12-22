"""
Entity Gravity Model

Claims are attracted to EUs by shared entities.
Pages dissolve as claims find their entity-gravity homes.

Key insight: Most claims don't have explicit corr/contra links,
but they DO mention entities. Entities create gravitational pull.

EU formation:
1. Page arrives with claims
2. Each claim has entities
3. Claims with overlapping entities attract each other
4. Strong entity overlap across pages → topic-EU forms

Run inside container:
    docker exec herenews-app python /app/test_eu/entity_gravity.py
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from load_graph import load_snapshot, GraphSnapshot


@dataclass
class TopicEU:
    """Topic-based EU formed by entity gravity"""
    id: str
    anchor_entity_id: str  # The entity this EU formed around
    anchor_entity_name: str

    claim_ids: Set[str] = field(default_factory=set)
    page_ids: Set[str] = field(default_factory=set)
    all_entity_ids: Set[str] = field(default_factory=set)

    # Link stats
    internal_corr: int = 0
    internal_contra: int = 0

    def claim_count(self) -> int:
        return len(self.claim_ids)

    def page_count(self) -> int:
        return len(self.page_ids)

    def coherence(self) -> float:
        total = self.internal_corr + self.internal_contra
        if total == 0:
            return 1.0
        return self.internal_corr / total


def build_entity_gravity_eus(snapshot: GraphSnapshot, min_claims: int = 3, min_pages: int = 2):
    """
    Build topic-EUs around high-gravity entities.

    An entity has high gravity if:
    - Many claims mention it
    - Those claims come from multiple pages

    Claims are pulled into the EU of their highest-gravity entity.
    """

    # Step 1: Count entity mentions across pages
    entity_claims: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> claim_ids
    entity_pages: Dict[str, Set[str]] = defaultdict(set)   # entity_id -> page_ids

    for cid, claim in snapshot.claims.items():
        for eid in claim.entity_ids:
            entity_claims[eid].add(cid)
            if claim.page_id:
                entity_pages[eid].add(claim.page_id)

    # Step 2: Rank entities by gravity (claims × pages)
    entity_gravity = {}
    for eid in entity_claims:
        claims = len(entity_claims[eid])
        pages = len(entity_pages[eid])
        if claims >= min_claims and pages >= min_pages:
            gravity = claims * pages  # Simple gravity formula
            entity_gravity[eid] = gravity

    print(f"Entities with gravity (>={min_claims} claims, >={min_pages} pages): {len(entity_gravity)}")

    # Step 3: Create topic-EUs around high-gravity entities
    # Sort by gravity, create EUs greedily
    sorted_entities = sorted(entity_gravity.items(), key=lambda x: x[1], reverse=True)

    topic_eus: Dict[str, TopicEU] = {}
    claim_assignments: Dict[str, str] = {}  # claim_id -> topic_eu_id

    for eid, gravity in sorted_entities:
        entity = snapshot.entities.get(eid)
        entity_name = entity.canonical_name if entity else eid

        # Get claims for this entity that aren't already assigned
        available_claims = entity_claims[eid] - set(claim_assignments.keys())

        if len(available_claims) < min_claims:
            continue  # Not enough unassigned claims

        # Check page diversity of available claims
        available_pages = set()
        for cid in available_claims:
            claim = snapshot.claims.get(cid)
            if claim and claim.page_id:
                available_pages.add(claim.page_id)

        if len(available_pages) < min_pages:
            continue  # Not enough page diversity

        # Create topic-EU
        topic_eu = TopicEU(
            id=f"topic_{eid}",
            anchor_entity_id=eid,
            anchor_entity_name=entity_name,
            claim_ids=available_claims,
            page_ids=available_pages
        )

        # Compute all entities and internal links
        for cid in available_claims:
            claim = snapshot.claims.get(cid)
            if claim:
                topic_eu.all_entity_ids |= set(claim.entity_ids)

                # Count internal links
                for corr_id in claim.corroborates_ids:
                    if corr_id in available_claims:
                        topic_eu.internal_corr += 1

                for contra_id in claim.contradicts_ids:
                    if contra_id in available_claims:
                        topic_eu.internal_contra += 1

        topic_eu.internal_corr //= 2  # Counted twice
        topic_eu.internal_contra //= 2

        topic_eus[eid] = topic_eu

        # Mark claims as assigned
        for cid in available_claims:
            claim_assignments[cid] = topic_eu.id

    return topic_eus, claim_assignments, entity_gravity


def analyze_results(
    topic_eus: Dict[str, TopicEU],
    claim_assignments: Dict[str, str],
    snapshot: GraphSnapshot
):
    """Analyze entity gravity results"""

    print(f"\n{'='*60}")
    print("Entity Gravity Results")
    print(f"{'='*60}")

    total_claims = len(snapshot.claims)
    assigned_claims = len(claim_assignments)
    unassigned_claims = total_claims - assigned_claims

    print(f"\nTotal claims: {total_claims}")
    print(f"Assigned to topic-EUs: {assigned_claims} ({assigned_claims/total_claims*100:.1f}%)")
    print(f"Unassigned (orphans): {unassigned_claims} ({unassigned_claims/total_claims*100:.1f}%)")
    print(f"Topic-EUs formed: {len(topic_eus)}")

    # Top topic-EUs
    print(f"\n{'='*60}")
    print("Top Topic-EUs by Entity Gravity")
    print(f"{'='*60}")

    sorted_eus = sorted(topic_eus.values(), key=lambda x: x.claim_count(), reverse=True)

    for eu in sorted_eus[:20]:
        coh = eu.coherence()
        tension = eu.internal_contra
        print(f"\n  {eu.anchor_entity_name}:")
        print(f"    Claims: {eu.claim_count()}, Pages: {eu.page_count()}")
        print(f"    Links: +{eu.internal_corr}/-{eu.internal_contra}, Coherence: {coh:.0%}")

    # Compare with existing events
    print(f"\n{'='*60}")
    print("Comparison with Existing Events")
    print(f"{'='*60}")

    for event in snapshot.events.values():
        event_claims = set(event.claim_ids)

        # Find best matching topic-EU
        best_match = None
        best_overlap = 0.0

        for eu in topic_eus.values():
            intersection = eu.claim_ids & event_claims
            union = eu.claim_ids | event_claims
            overlap = len(intersection) / len(union) if union else 0

            if overlap > best_overlap:
                best_overlap = overlap
                best_match = eu

        status = 'FOUND' if best_overlap >= 0.5 else 'PARTIAL' if best_overlap >= 0.2 else 'MISSED'
        print(f"\n  [{status}] {event.canonical_name} ({len(event.claim_ids)} claims)")
        if best_match:
            print(f"      → {best_match.anchor_entity_name} ({best_match.claim_count()} claims, {best_overlap:.0%})")


def main():
    print("=" * 60)
    print("Entity Gravity Model")
    print("=" * 60)
    print("\nClaims are attracted to EUs by shared entities.")
    print("High-gravity entities (many claims, many pages) anchor topic-EUs.\n")

    snapshot = load_snapshot()
    print(f"Loaded {len(snapshot.claims)} claims, {len(snapshot.entities)} entities\n")

    topic_eus, claim_assignments, entity_gravity = build_entity_gravity_eus(
        snapshot,
        min_claims=3,
        min_pages=2
    )

    analyze_results(topic_eus, claim_assignments, snapshot)

    # Save
    output_path = Path("/app/test_eu/results/entity_gravity.json")

    output = {
        'topic_eus': [
            {
                'entity': eu.anchor_entity_name,
                'claims': eu.claim_count(),
                'pages': eu.page_count(),
                'corr': eu.internal_corr,
                'contra': eu.internal_contra,
                'coherence': eu.coherence()
            }
            for eu in sorted(topic_eus.values(), key=lambda x: x.claim_count(), reverse=True)
        ],
        'assigned_claims': len(claim_assignments),
        'total_claims': len(snapshot.claims)
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
