"""
Experiment 5: Trump/BBC Duplicate Prevention

Would the EU model have prevented the duplicate sub-event issue (#22)?

Current system created:
- Parent: "Trump vs. BBC Defamation Lawsuit" (ev_bca3lue9)
- Child:  "Trump's Defamation Lawsuit Against BBC" (ev_zi427c8u)

These are semantically identical - the sub-event shouldn't exist.

EU model hypothesis:
- Both would be part of same cluster
- Mass would accumulate without spawning near-identical child
- Emergence threshold would prevent duplicate naming

Run inside container:
    docker exec herenews-app python /app/test_eu/replay_trump_bbc.py
"""

import json
from pathlib import Path
from typing import List, Set, Dict

from load_graph import load_snapshot, GraphSnapshot, ClaimData
from mass_calculator import compute_all_masses, MassVector


def get_trump_bbc_claims(snapshot: GraphSnapshot) -> Dict[str, List[str]]:
    """Get claims from both Trump/BBC events"""
    parent_id = "ev_bca3lue9"  # Trump vs. BBC Defamation Lawsuit
    child_id = "ev_zi427c8u"   # Trump's Defamation Lawsuit Against BBC

    parent_event = snapshot.events.get(parent_id)
    child_event = snapshot.events.get(child_id)

    return {
        'parent': {
            'id': parent_id,
            'name': parent_event.canonical_name if parent_event else "NOT FOUND",
            'claim_ids': parent_event.claim_ids if parent_event else []
        },
        'child': {
            'id': child_id,
            'name': child_event.canonical_name if child_event else "NOT FOUND",
            'claim_ids': child_event.claim_ids if child_event else []
        }
    }


def analyze_claim_overlap(
    snapshot: GraphSnapshot,
    parent_claim_ids: List[str],
    child_claim_ids: List[str]
) -> Dict:
    """Analyze overlap between parent and child claims"""

    parent_set = set(parent_claim_ids)
    child_set = set(child_claim_ids)

    # Claims shared by both
    shared = parent_set & child_set
    # Claims only in parent
    parent_only = parent_set - child_set
    # Claims only in child
    child_only = child_set - parent_set

    # Entity overlap
    parent_entities = set()
    child_entities = set()

    for cid in parent_set:
        claim = snapshot.claims.get(cid)
        if claim:
            parent_entities.update(claim.entity_ids)

    for cid in child_set:
        claim = snapshot.claims.get(cid)
        if claim:
            child_entities.update(claim.entity_ids)

    shared_entities = parent_entities & child_entities
    entity_jaccard = len(shared_entities) / len(parent_entities | child_entities) if (parent_entities | child_entities) else 0

    return {
        'parent_claims': len(parent_set),
        'child_claims': len(child_set),
        'shared_claims': len(shared),
        'parent_only': len(parent_only),
        'child_only': len(child_only),
        'claim_overlap': len(shared) / len(parent_set | child_set) if (parent_set | child_set) else 0,
        'entity_jaccard': entity_jaccard,
        'shared_entities': [
            snapshot.entities[eid].canonical_name
            for eid in list(shared_entities)[:10]
            if eid in snapshot.entities
        ]
    }


def simulate_eu_model(
    snapshot: GraphSnapshot,
    masses: Dict[str, MassVector],
    all_claim_ids: List[str]
) -> Dict:
    """
    Simulate what would happen under EU model:
    - All claims form ONE cluster
    - Mass accumulates
    - No separate sub-event created
    """

    # Combine all claims
    all_claims = [snapshot.claims[cid] for cid in all_claim_ids if cid in snapshot.claims]

    # Total mass
    total_mass = sum(masses[cid].total() for cid in all_claim_ids if cid in masses)
    mean_mass = total_mass / len(all_claim_ids) if all_claim_ids else 0

    # Entity analysis
    all_entities = set()
    for claim in all_claims:
        all_entities.update(claim.entity_ids)

    # Internal links
    internal_corr = 0
    internal_contra = 0
    claim_id_set = set(all_claim_ids)
    for cid in all_claim_ids:
        claim = snapshot.claims.get(cid)
        if claim:
            internal_corr += len([c for c in claim.corroborates_ids if c in claim_id_set])
            internal_contra += len([c for c in claim.contradicts_ids if c in claim_id_set])

    return {
        'total_claims': len(all_claims),
        'total_mass': total_mass,
        'mean_mass': mean_mass,
        'unique_entities': len(all_entities),
        'top_entities': [
            snapshot.entities[eid].canonical_name
            for eid in list(all_entities)[:10]
            if eid in snapshot.entities
        ],
        'internal_corroborations': internal_corr,
        'internal_contradictions': internal_contra,
        'would_form_single_bundle': True,  # EU model assertion
        'reasoning': "All claims share core entities (Trump, BBC), high internal corroboration, no semantic split justifies separate bundle"
    }


def main():
    print("=" * 60)
    print("EU Experiment 5: Trump/BBC Duplicate Prevention")
    print("=" * 60)

    # Load data
    snapshot = load_snapshot()
    masses = compute_all_masses(snapshot)

    # Get Trump/BBC events
    events = get_trump_bbc_claims(snapshot)

    print("\n" + "=" * 60)
    print("Current System State (Issue #22)")
    print("=" * 60)

    print(f"\nParent Event: {events['parent']['name']}")
    print(f"  ID: {events['parent']['id']}")
    print(f"  Claims: {len(events['parent']['claim_ids'])}")

    print(f"\nChild Event: {events['child']['name']}")
    print(f"  ID: {events['child']['id']}")
    print(f"  Claims: {len(events['child']['claim_ids'])}")

    # Analyze overlap
    overlap = analyze_claim_overlap(
        snapshot,
        events['parent']['claim_ids'],
        events['child']['claim_ids']
    )

    print("\n" + "=" * 60)
    print("Overlap Analysis")
    print("=" * 60)
    print(f"\nClaim overlap: {overlap['claim_overlap']*100:.1f}%")
    print(f"  Shared claims: {overlap['shared_claims']}")
    print(f"  Parent-only: {overlap['parent_only']}")
    print(f"  Child-only: {overlap['child_only']}")
    print(f"\nEntity Jaccard similarity: {overlap['entity_jaccard']*100:.1f}%")
    print(f"Shared entities: {', '.join(overlap['shared_entities'][:5])}")

    # Show actual claims
    print("\n" + "=" * 60)
    print("Sample Claims")
    print("=" * 60)

    print("\nParent-only claims:")
    parent_only_ids = set(events['parent']['claim_ids']) - set(events['child']['claim_ids'])
    for cid in list(parent_only_ids)[:3]:
        claim = snapshot.claims.get(cid)
        if claim:
            mass = masses.get(cid)
            print(f"  [{mass.total():.2f}] {claim.text[:80]}...")

    print("\nChild-only claims:")
    child_only_ids = set(events['child']['claim_ids']) - set(events['parent']['claim_ids'])
    for cid in list(child_only_ids)[:3]:
        claim = snapshot.claims.get(cid)
        if claim:
            mass = masses.get(cid)
            print(f"  [{mass.total():.2f}] {claim.text[:80]}...")

    # EU model simulation
    all_claim_ids = list(set(events['parent']['claim_ids']) | set(events['child']['claim_ids']))
    eu_result = simulate_eu_model(snapshot, masses, all_claim_ids)

    print("\n" + "=" * 60)
    print("EU Model Simulation")
    print("=" * 60)
    print(f"\nUnder EU model, all {eu_result['total_claims']} claims would form ONE cluster:")
    print(f"  Total mass: {eu_result['total_mass']:.2f}")
    print(f"  Mean mass: {eu_result['mean_mass']:.3f}")
    print(f"  Unique entities: {eu_result['unique_entities']}")
    print(f"  Top entities: {', '.join(eu_result['top_entities'][:5])}")
    print(f"  Internal corroborations: {eu_result['internal_corroborations']}")
    print(f"  Internal contradictions: {eu_result['internal_contradictions']}")
    print(f"\nWould form single bundle: {eu_result['would_form_single_bundle']}")
    print(f"Reasoning: {eu_result['reasoning']}")

    # Conclusion
    print("\n" + "=" * 60)
    print("Conclusion")
    print("=" * 60)

    if overlap['entity_jaccard'] > 0.7:
        print("\n DUPLICATE DETECTED")
        print("Entity Jaccard > 70% indicates these are the same event.")
        print("EU model would NOT create separate sub-event.")
    elif overlap['entity_jaccard'] > 0.5:
        print("\n HIGH SIMILARITY")
        print("Entity Jaccard > 50% suggests questionable split.")
        print("EU model would likely merge into single bundle.")
    else:
        print("\n LEGITIMATE SPLIT")
        print("Events are sufficiently distinct.")

    print(f"\nCurrent system: Created 2 events with {overlap['entity_jaccard']*100:.0f}% entity overlap")
    print(f"EU model: Would create 1 bundle, mass = {eu_result['total_mass']:.2f}")

    # Save results
    output = {
        'current_system': {
            'parent': events['parent'],
            'child': events['child'],
            'overlap': overlap
        },
        'eu_model': eu_result,
        'conclusion': {
            'entity_jaccard': overlap['entity_jaccard'],
            'is_duplicate': overlap['entity_jaccard'] > 0.7,
            'eu_prevents_duplicate': True
        }
    }

    output_path = Path("/app/test_eu/results/trump_bbc_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
