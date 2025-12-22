"""
Entity Constellation Model

EUs form around entity constellations (co-occurring entity sets),
not single entities.

"Hong Kong" alone is too broad.
"Hong Kong + Wang Fuk Court + Fire Services" is specific.

Key insight: The intersection of entity mentions defines the topic.

Run inside container:
    docker exec herenews-app python /app/test_eu/entity_constellation.py
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, FrozenSet
from pathlib import Path
from collections import defaultdict

from load_graph import load_snapshot, GraphSnapshot


@dataclass
class ConstellationEU:
    """EU formed around an entity constellation"""
    id: str
    core_entities: FrozenSet[str]  # The defining entity set
    core_entity_names: List[str]

    claim_ids: Set[str] = field(default_factory=set)
    page_ids: Set[str] = field(default_factory=set)

    internal_corr: int = 0
    internal_contra: int = 0

    def label(self) -> str:
        return " + ".join(self.core_entity_names[:3])

    def coherence(self) -> float:
        total = self.internal_corr + self.internal_contra
        return self.internal_corr / total if total > 0 else 1.0


def find_entity_constellations(snapshot: GraphSnapshot, min_claims: int = 3):
    """
    Find entity constellations - sets of entities that frequently co-occur.

    A constellation is a set of 2+ entities that appear together in multiple claims.
    """

    # For each claim, get its entity set
    claim_entity_sets: Dict[str, FrozenSet[str]] = {}
    for cid, claim in snapshot.claims.items():
        if len(claim.entity_ids) >= 2:
            claim_entity_sets[cid] = frozenset(claim.entity_ids)

    # Count entity pairs
    pair_claims: Dict[FrozenSet[str], Set[str]] = defaultdict(set)

    for cid, entity_set in claim_entity_sets.items():
        entities = list(entity_set)
        # Generate all pairs
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                pair = frozenset([entities[i], entities[j]])
                pair_claims[pair].add(cid)

    # Filter pairs that appear in multiple claims
    strong_pairs = {
        pair: claims
        for pair, claims in pair_claims.items()
        if len(claims) >= min_claims
    }

    print(f"Entity pairs with >={min_claims} claims: {len(strong_pairs)}")

    return strong_pairs


def build_constellation_eus(
    snapshot: GraphSnapshot,
    min_claims: int = 3,
    min_pages: int = 2
) -> Tuple[Dict[str, ConstellationEU], Dict[str, str]]:
    """Build EUs from entity constellations"""

    # Find strong entity pairs
    pair_claims = find_entity_constellations(snapshot, min_claims)

    # Score pairs by: claim_count * page_diversity
    pair_scores = {}
    for pair, claim_ids in pair_claims.items():
        pages = set()
        for cid in claim_ids:
            claim = snapshot.claims.get(cid)
            if claim and claim.page_id:
                pages.add(claim.page_id)

        if len(pages) >= min_pages:
            pair_scores[pair] = len(claim_ids) * len(pages)

    # Sort by score
    sorted_pairs = sorted(pair_scores.items(), key=lambda x: x[1], reverse=True)

    # Build EUs greedily - each claim can only belong to one EU
    constellation_eus: Dict[str, ConstellationEU] = {}
    claim_assignments: Dict[str, str] = {}
    eu_counter = 0

    for pair, score in sorted_pairs:
        # Get unassigned claims for this pair
        available_claims = pair_claims[pair] - set(claim_assignments.keys())

        if len(available_claims) < min_claims:
            continue

        # Check page diversity of available claims
        pages = set()
        for cid in available_claims:
            claim = snapshot.claims.get(cid)
            if claim and claim.page_id:
                pages.add(claim.page_id)

        if len(pages) < min_pages:
            continue

        # Get entity names
        entity_names = []
        for eid in pair:
            e = snapshot.entities.get(eid)
            entity_names.append(e.canonical_name if e else eid)

        eu_counter += 1
        eu = ConstellationEU(
            id=f"const_{eu_counter}",
            core_entities=pair,
            core_entity_names=entity_names,
            claim_ids=available_claims,
            page_ids=pages
        )

        # Compute internal links
        for cid in available_claims:
            claim = snapshot.claims.get(cid)
            if claim:
                for corr_id in claim.corroborates_ids:
                    if corr_id in available_claims:
                        eu.internal_corr += 1
                for contra_id in claim.contradicts_ids:
                    if contra_id in available_claims:
                        eu.internal_contra += 1

        eu.internal_corr //= 2
        eu.internal_contra //= 2

        constellation_eus[eu.id] = eu

        for cid in available_claims:
            claim_assignments[cid] = eu.id

    return constellation_eus, claim_assignments


def analyze_and_compare(
    constellation_eus: Dict[str, ConstellationEU],
    claim_assignments: Dict[str, str],
    snapshot: GraphSnapshot
):
    """Analyze constellation results"""

    print(f"\n{'='*60}")
    print("Constellation Results")
    print(f"{'='*60}")

    assigned = len(claim_assignments)
    total = len(snapshot.claims)

    print(f"\nClaims assigned: {assigned}/{total} ({assigned/total*100:.1f}%)")
    print(f"Constellation-EUs: {len(constellation_eus)}")

    # Top EUs
    print(f"\n{'='*60}")
    print("Top Constellation-EUs")
    print(f"{'='*60}")

    sorted_eus = sorted(constellation_eus.values(), key=lambda x: len(x.claim_ids), reverse=True)

    for eu in sorted_eus[:20]:
        print(f"\n  {eu.label()}:")
        print(f"    Claims: {len(eu.claim_ids)}, Pages: {len(eu.page_ids)}")
        print(f"    Links: +{eu.internal_corr}/-{eu.internal_contra}, Coherence: {eu.coherence():.0%}")

    # Compare with events
    print(f"\n{'='*60}")
    print("Event Matching")
    print(f"{'='*60}")

    results = {'found': 0, 'partial': 0, 'missed': 0}

    for event in snapshot.events.values():
        event_claims = set(event.claim_ids)

        best_match = None
        best_overlap = 0.0

        for eu in constellation_eus.values():
            intersection = eu.claim_ids & event_claims
            union = eu.claim_ids | event_claims
            overlap = len(intersection) / len(union) if union else 0

            if overlap > best_overlap:
                best_overlap = overlap
                best_match = eu

        if best_overlap >= 0.5:
            status = 'FOUND'
            results['found'] += 1
        elif best_overlap >= 0.2:
            status = 'PARTIAL'
            results['partial'] += 1
        else:
            status = 'MISSED'
            results['missed'] += 1

        print(f"\n  [{status}] {event.canonical_name} ({len(event.claim_ids)} claims)")
        if best_match:
            print(f"      → {best_match.label()} ({len(best_match.claim_ids)} claims, {best_overlap:.0%})")

    print(f"\n\nSummary: {results['found']} FOUND, {results['partial']} PARTIAL, {results['missed']} MISSED")

    return results


def main():
    print("=" * 60)
    print("Entity Constellation Model")
    print("=" * 60)
    print("\nEUs form around entity pairs/sets, not single entities.")
    print("'Hong Kong + Wang Fuk Court' is more specific than 'Hong Kong'.\n")

    snapshot = load_snapshot()
    print(f"Loaded {len(snapshot.claims)} claims\n")

    constellation_eus, claim_assignments = build_constellation_eus(
        snapshot,
        min_claims=3,
        min_pages=2
    )

    results = analyze_and_compare(constellation_eus, claim_assignments, snapshot)

    # Save
    output_path = Path("/app/test_eu/results/entity_constellation.json")

    output = {
        'constellation_eus': [
            {
                'entities': eu.core_entity_names,
                'claims': len(eu.claim_ids),
                'pages': len(eu.page_ids),
                'corr': eu.internal_corr,
                'contra': eu.internal_contra,
                'coherence': eu.coherence()
            }
            for eu in sorted(constellation_eus.values(), key=lambda x: len(x.claim_ids), reverse=True)
        ],
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
