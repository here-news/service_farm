"""
Page Dissolution Model

Pages arrive as pre-bundled EUs (source grouping).
Claims then find semantic homes (topic grouping).
Page-EUs dissolve as claims get absorbed into topic-EUs.

Key idea:
- Page = initial weak EU (external grouping by source)
- Topic EU = emerges from cross-page connections (semantic grouping)
- A claim starts in its page-EU, then migrates to topic-EU

Run inside container:
    docker exec herenews-app python /app/test_eu/page_dissolution.py
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from collections import defaultdict

from load_graph import load_snapshot, GraphSnapshot


@dataclass
class EU:
    """EventfulUnit"""
    id: str
    eu_type: str  # 'page' or 'topic'

    # Claims currently in this EU
    claim_ids: Set[str] = field(default_factory=set)

    # For page-EUs
    page_id: Optional[str] = None

    # Properties (computed from claims)
    entity_ids: Set[str] = field(default_factory=set)
    internal_corr: int = 0
    internal_contra: int = 0
    cross_page_count: int = 0  # How many pages contribute

    label: str = ""

    def strength(self) -> float:
        """How strong is this EU's internal binding?"""
        # Topic EUs with cross-page corroboration are strong
        # Page EUs with only same-source claims are weak
        if self.eu_type == 'page':
            return 0.1  # Weak - just source grouping
        else:
            # Strength from corroboration + source diversity
            return self.internal_corr * 1.0 + self.cross_page_count * 0.5


def simulate_page_arrival(snapshot: GraphSnapshot):
    """
    Simulate pages arriving one by one.
    Track how claims migrate from page-EUs to topic-EUs.
    """

    # Group claims by page
    claims_by_page: Dict[str, List[str]] = defaultdict(list)
    for cid, claim in snapshot.claims.items():
        if claim.page_id:
            claims_by_page[claim.page_id].append(cid)

    print(f"Pages: {len(claims_by_page)}")
    print(f"Claims: {len(snapshot.claims)}")

    # State
    page_eus: Dict[str, EU] = {}  # page_id -> EU
    topic_eus: Dict[str, EU] = {}  # topic_id -> EU
    claim_to_eu: Dict[str, str] = {}  # claim_id -> current EU id

    topic_counter = 0

    # Process pages in order (simulate arrival)
    for page_id, claim_ids in claims_by_page.items():
        page = snapshot.pages.get(page_id)

        # Create page-EU
        page_eu = EU(
            id=f"page_{page_id}",
            eu_type='page',
            claim_ids=set(claim_ids),
            page_id=page_id,
            label=page.url[:40] if page else page_id
        )

        # Compute entities
        for cid in claim_ids:
            claim = snapshot.claims.get(cid)
            if claim:
                page_eu.entity_ids |= set(claim.entity_ids)

        page_eus[page_id] = page_eu

        # Initially, all claims belong to their page-EU
        for cid in claim_ids:
            claim_to_eu[cid] = page_eu.id

        # Now check: do any claims connect to existing topic-EUs?
        for cid in claim_ids:
            claim = snapshot.claims.get(cid)
            if not claim:
                continue

            # Check corroboration links to other claims
            for corr_id in claim.corroborates_ids:
                if corr_id in claim_to_eu:
                    other_eu_id = claim_to_eu[corr_id]

                    # If other claim is in a topic-EU, join it
                    if other_eu_id.startswith('topic_'):
                        topic_eu = topic_eus[other_eu_id.replace('topic_', '')]
                        topic_eu.claim_ids.add(cid)
                        topic_eu.entity_ids |= set(claim.entity_ids)
                        topic_eu.internal_corr += 1

                        # Track cross-page
                        other_claim = snapshot.claims.get(corr_id)
                        if other_claim and other_claim.page_id != page_id:
                            topic_eu.cross_page_count += 1

                        claim_to_eu[cid] = other_eu_id
                        page_eu.claim_ids.discard(cid)

                    # If other claim is in page-EU, create new topic-EU
                    elif other_eu_id.startswith('page_'):
                        other_page_id = other_eu_id.replace('page_', '')

                        # Only create topic if cross-page connection
                        if other_page_id != page_id:
                            topic_counter += 1

                            # Find dominant entity for label
                            entities = set(claim.entity_ids)
                            other_claim = snapshot.claims.get(corr_id)
                            if other_claim:
                                entities |= set(other_claim.entity_ids)

                            label = "topic"
                            if entities:
                                eid = list(entities)[0]
                                e = snapshot.entities.get(eid)
                                if e:
                                    label = e.canonical_name

                            topic_eu = EU(
                                id=f"topic_{topic_counter}",
                                eu_type='topic',
                                claim_ids={cid, corr_id},
                                entity_ids=entities,
                                internal_corr=1,
                                cross_page_count=1,
                                label=label
                            )
                            topic_eus[str(topic_counter)] = topic_eu

                            # Migrate both claims
                            claim_to_eu[cid] = topic_eu.id
                            claim_to_eu[corr_id] = topic_eu.id
                            page_eu.claim_ids.discard(cid)
                            page_eus[other_page_id].claim_ids.discard(corr_id)

            # Also check contradiction links (they also indicate connection)
            for contra_id in claim.contradicts_ids:
                if contra_id in claim_to_eu:
                    other_eu_id = claim_to_eu[contra_id]

                    if other_eu_id.startswith('topic_'):
                        topic_eu = topic_eus[other_eu_id.replace('topic_', '')]
                        topic_eu.claim_ids.add(cid)
                        topic_eu.entity_ids |= set(claim.entity_ids)
                        topic_eu.internal_contra += 1

                        claim_to_eu[cid] = other_eu_id
                        page_eu.claim_ids.discard(cid)

    return page_eus, topic_eus, claim_to_eu


def analyze_results(
    page_eus: Dict[str, EU],
    topic_eus: Dict[str, EU],
    claim_to_eu: Dict[str, str],
    snapshot: GraphSnapshot
):
    """Analyze the dissolution results"""

    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")

    # Count claims in each type
    claims_in_pages = sum(len(eu.claim_ids) for eu in page_eus.values())
    claims_in_topics = sum(len(eu.claim_ids) for eu in topic_eus.values())

    print(f"\nClaims still in page-EUs: {claims_in_pages}")
    print(f"Claims migrated to topic-EUs: {claims_in_topics}")
    print(f"Topic-EUs formed: {len(topic_eus)}")

    # Page dissolution rate
    dissolved_pages = sum(1 for eu in page_eus.values() if len(eu.claim_ids) == 0)
    partial_pages = sum(1 for eu in page_eus.values() if 0 < len(eu.claim_ids) < 5)

    print(f"\nPage dissolution:")
    print(f"  Fully dissolved (0 claims): {dissolved_pages}")
    print(f"  Partially dissolved (1-4 claims): {partial_pages}")
    print(f"  Intact (5+ claims): {len(page_eus) - dissolved_pages - partial_pages}")

    # Top topic-EUs
    print(f"\n{'='*60}")
    print("Top Topic-EUs (by size)")
    print(f"{'='*60}")

    sorted_topics = sorted(topic_eus.values(), key=lambda x: len(x.claim_ids), reverse=True)

    for eu in sorted_topics[:15]:
        pages = set()
        for cid in eu.claim_ids:
            claim = snapshot.claims.get(cid)
            if claim and claim.page_id:
                pages.add(claim.page_id)

        print(f"\n  {eu.label}: {len(eu.claim_ids)} claims from {len(pages)} pages")
        print(f"    Corr: {eu.internal_corr}, Contra: {eu.internal_contra}")

    # Orphan analysis - claims that stayed in page-EUs
    print(f"\n{'='*60}")
    print("Orphan Analysis (claims still in page-EUs)")
    print(f"{'='*60}")

    orphan_reasons = defaultdict(int)
    for eu in page_eus.values():
        for cid in eu.claim_ids:
            claim = snapshot.claims.get(cid)
            if claim:
                if not claim.corroborates_ids and not claim.contradicts_ids:
                    orphan_reasons['no_links'] += 1
                elif all(c not in claim_to_eu for c in claim.corroborates_ids):
                    orphan_reasons['links_to_unknown'] += 1
                else:
                    orphan_reasons['same_page_only'] += 1

    print(f"\n  No corr/contra links: {orphan_reasons['no_links']}")
    print(f"  Links only to same page: {orphan_reasons['same_page_only']}")
    print(f"  Links to claims not processed: {orphan_reasons['links_to_unknown']}")

    return {
        'claims_in_pages': claims_in_pages,
        'claims_in_topics': claims_in_topics,
        'topic_count': len(topic_eus),
        'dissolved_pages': dissolved_pages
    }


def main():
    print("=" * 60)
    print("Page Dissolution Model")
    print("=" * 60)
    print("\nPages arrive as pre-bundled EUs.")
    print("Claims migrate to topic-EUs based on cross-page connections.\n")

    snapshot = load_snapshot()

    page_eus, topic_eus, claim_to_eu = simulate_page_arrival(snapshot)

    results = analyze_results(page_eus, topic_eus, claim_to_eu, snapshot)

    # Save
    output_path = Path("/app/test_eu/results/page_dissolution.json")

    output = {
        'summary': results,
        'topic_eus': [
            {
                'label': eu.label,
                'claims': len(eu.claim_ids),
                'corr': eu.internal_corr,
                'contra': eu.internal_contra
            }
            for eu in sorted(topic_eus.values(), key=lambda x: len(x.claim_ids), reverse=True)[:30]
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
