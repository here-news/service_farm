"""
Anchor + Context Model

EUs form around specific anchor entities, then absorb claims
that share context entities AND have some connection signal.

Anchor entity: Specific (Wang Fuk Court, Jimmy Lai, Charlie Kirk)
Context entities: Broad (Hong Kong, United States, Fire)

A claim joins an EU if:
1. It mentions the anchor entity, OR
2. It shares 2+ entities with existing claims AND has link/semantic connection

Run inside container:
    docker exec herenews-app python /app/test_eu/anchor_context.py
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from load_graph import load_snapshot, GraphSnapshot


@dataclass
class AnchoredEU:
    """EU anchored by a specific entity"""
    id: str
    anchor_entity_id: str
    anchor_entity_name: str

    claim_ids: Set[str] = field(default_factory=set)
    page_ids: Set[str] = field(default_factory=set)
    context_entities: Set[str] = field(default_factory=set)  # All entities in claims

    internal_corr: int = 0
    internal_contra: int = 0

    def coherence(self) -> float:
        total = self.internal_corr + self.internal_contra
        return self.internal_corr / total if total > 0 else 1.0


def identify_anchor_entities(snapshot: GraphSnapshot, min_claims: int = 5, min_pages: int = 3):
    """
    Find anchor entities - specific enough to define a topic.

    Anchor = high claims-per-page ratio (concentrated topic)
    vs Context = low claims-per-page ratio (appears everywhere)
    """

    entity_claims: Dict[str, Set[str]] = defaultdict(set)
    entity_pages: Dict[str, Set[str]] = defaultdict(set)

    for cid, claim in snapshot.claims.items():
        for eid in claim.entity_ids:
            entity_claims[eid].add(cid)
            if claim.page_id:
                entity_pages[eid].add(claim.page_id)

    anchor_scores = {}
    for eid in entity_claims:
        claims = len(entity_claims[eid])
        pages = len(entity_pages[eid])

        if claims >= min_claims and pages >= min_pages:
            # Concentration = claims / pages
            # High concentration = anchor (specific topic)
            # Low concentration = context (appears everywhere)
            concentration = claims / pages
            anchor_scores[eid] = {
                'claims': claims,
                'pages': pages,
                'concentration': concentration,
                'score': claims * concentration  # Favor both size and specificity
            }

    return anchor_scores, entity_claims, entity_pages


def build_anchored_eus(snapshot: GraphSnapshot):
    """Build EUs around anchor entities with context expansion"""

    anchor_scores, entity_claims, entity_pages = identify_anchor_entities(snapshot)

    print(f"Potential anchors: {len(anchor_scores)}")

    # Sort by score (size × concentration)
    sorted_anchors = sorted(
        anchor_scores.items(),
        key=lambda x: x[1]['score'],
        reverse=True
    )

    # Show top anchors
    print("\nTop anchor candidates:")
    for eid, info in sorted_anchors[:10]:
        e = snapshot.entities.get(eid)
        name = e.canonical_name if e else eid
        print(f"  {name}: {info['claims']} claims, {info['pages']} pages, conc={info['concentration']:.1f}")

    # Build EUs greedily
    anchored_eus: Dict[str, AnchoredEU] = {}
    claim_assignments: Dict[str, str] = {}
    eu_counter = 0

    for anchor_eid, info in sorted_anchors:
        # Get unassigned claims that mention this anchor
        anchor_claim_ids = entity_claims[anchor_eid] - set(claim_assignments.keys())

        if len(anchor_claim_ids) < 3:
            continue

        # Check page diversity
        pages = set()
        for cid in anchor_claim_ids:
            claim = snapshot.claims.get(cid)
            if claim and claim.page_id:
                pages.add(claim.page_id)

        if len(pages) < 2:
            continue

        entity = snapshot.entities.get(anchor_eid)
        anchor_name = entity.canonical_name if entity else anchor_eid

        eu_counter += 1
        eu = AnchoredEU(
            id=f"anchor_{eu_counter}",
            anchor_entity_id=anchor_eid,
            anchor_entity_name=anchor_name,
            claim_ids=anchor_claim_ids.copy(),
            page_ids=pages.copy()
        )

        # Collect context entities from anchor claims
        for cid in anchor_claim_ids:
            claim = snapshot.claims.get(cid)
            if claim:
                eu.context_entities |= set(claim.entity_ids)

        # Now try to absorb additional claims that:
        # 1. Don't mention anchor but share 2+ context entities
        # 2. AND have a link (corr/contra) to an existing claim in EU

        for cid, claim in snapshot.claims.items():
            if cid in claim_assignments or cid in eu.claim_ids:
                continue

            # Check entity overlap (at least 2 shared context entities, excluding anchor)
            claim_entities = set(claim.entity_ids) - {anchor_eid}
            shared = claim_entities & eu.context_entities
            if len(shared) < 2:
                continue

            # Check for link to existing EU claim
            has_link = False
            for link_id in claim.corroborates_ids + claim.contradicts_ids:
                if link_id in eu.claim_ids:
                    has_link = True
                    break

            if has_link:
                eu.claim_ids.add(cid)
                if claim.page_id:
                    eu.page_ids.add(claim.page_id)
                eu.context_entities |= set(claim.entity_ids)

        # Compute internal links
        for cid in eu.claim_ids:
            claim = snapshot.claims.get(cid)
            if claim:
                for corr_id in claim.corroborates_ids:
                    if corr_id in eu.claim_ids:
                        eu.internal_corr += 1
                for contra_id in claim.contradicts_ids:
                    if contra_id in eu.claim_ids:
                        eu.internal_contra += 1

        eu.internal_corr //= 2
        eu.internal_contra //= 2

        anchored_eus[eu.id] = eu

        for cid in eu.claim_ids:
            claim_assignments[cid] = eu.id

    return anchored_eus, claim_assignments


def analyze_results(
    anchored_eus: Dict[str, AnchoredEU],
    claim_assignments: Dict[str, str],
    snapshot: GraphSnapshot
):
    """Analyze anchored EU results"""

    print(f"\n{'='*60}")
    print("Anchored EU Results")
    print(f"{'='*60}")

    assigned = len(claim_assignments)
    total = len(snapshot.claims)

    print(f"\nClaims assigned: {assigned}/{total} ({assigned/total*100:.1f}%)")
    print(f"Anchored-EUs: {len(anchored_eus)}")

    # Top EUs
    print(f"\n{'='*60}")
    print("Top Anchored-EUs")
    print(f"{'='*60}")

    sorted_eus = sorted(anchored_eus.values(), key=lambda x: len(x.claim_ids), reverse=True)

    for eu in sorted_eus[:15]:
        print(f"\n  {eu.anchor_entity_name}:")
        print(f"    Claims: {len(eu.claim_ids)}, Pages: {len(eu.page_ids)}")
        print(f"    Context entities: {len(eu.context_entities)}")
        print(f"    Links: +{eu.internal_corr}/-{eu.internal_contra}, Coherence: {eu.coherence():.0%}")

    # Event matching
    print(f"\n{'='*60}")
    print("Event Matching")
    print(f"{'='*60}")

    results = {'found': 0, 'partial': 0, 'missed': 0}

    for event in snapshot.events.values():
        event_claims = set(event.claim_ids)

        best_match = None
        best_overlap = 0.0

        for eu in anchored_eus.values():
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
            print(f"      → {best_match.anchor_entity_name} ({len(best_match.claim_ids)} claims, {best_overlap:.0%})")

    print(f"\n\nSummary: {results['found']} FOUND, {results['partial']} PARTIAL, {results['missed']} MISSED")

    return results


def main():
    print("=" * 60)
    print("Anchor + Context Model")
    print("=" * 60)
    print("\nEUs form around specific anchor entities.")
    print("Claims with shared context + links get absorbed.\n")

    snapshot = load_snapshot()
    print(f"Loaded {len(snapshot.claims)} claims\n")

    anchored_eus, claim_assignments = build_anchored_eus(snapshot)

    results = analyze_results(anchored_eus, claim_assignments, snapshot)

    # Save
    output_path = Path("/app/test_eu/results/anchor_context.json")

    output = {
        'anchored_eus': [
            {
                'anchor': eu.anchor_entity_name,
                'claims': len(eu.claim_ids),
                'pages': len(eu.page_ids),
                'context_entities': len(eu.context_entities),
                'corr': eu.internal_corr,
                'contra': eu.internal_contra
            }
            for eu in sorted(anchored_eus.values(), key=lambda x: len(x.claim_ids), reverse=True)
        ],
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
