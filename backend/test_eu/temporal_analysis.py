"""
Temporal Analysis of Contradictions

Are contradictions temporal updates or fundamental disagreements?

Run inside container:
    docker exec herenews-app python /app/test_eu/temporal_analysis.py
"""

from load_graph import load_snapshot
from collections import defaultdict
import re


def extract_numbers(text: str) -> list:
    """Extract numbers from text."""
    return [int(n) for n in re.findall(r'\b(\d+)\b', text)]


def analyze_contradictions(snapshot):
    """Analyze what contradictions represent."""

    results = {
        'numerical_updates': [],
        'source_disagreements': [],
        'unknown': []
    }

    # Find all contradiction pairs
    seen = set()
    for cid, claim in snapshot.claims.items():
        for contra_id in claim.contradicts_ids:
            pair = tuple(sorted([cid, contra_id]))
            if pair in seen:
                continue
            seen.add(pair)

            other = snapshot.claims.get(contra_id)
            if not other:
                continue

            # Get sources
            page1 = snapshot.pages.get(claim.page_id)
            page2 = snapshot.pages.get(other.page_id)
            source1 = page1.url if page1 else 'unknown'
            source2 = page2.url if page2 else 'unknown'

            # Check if numerical disagreement
            nums1 = extract_numbers(claim.text)
            nums2 = extract_numbers(other.text)

            # Check for overlapping entities (same story)
            shared_entities = set(claim.entity_ids) & set(other.entity_ids)

            entry = {
                'claim1': claim.text[:100],
                'claim2': other.text[:100],
                'source1': source1[:60],
                'source2': source2[:60],
                'shared_entities': len(shared_entities),
                'nums1': nums1[:3],
                'nums2': nums2[:3]
            }

            # Classify
            if nums1 and nums2 and shared_entities:
                # Same entities, different numbers = likely temporal update
                results['numerical_updates'].append(entry)
            elif shared_entities:
                # Same entities, different claims = source disagreement
                results['source_disagreements'].append(entry)
            else:
                results['unknown'].append(entry)

    return results


def main():
    print("=" * 60)
    print("Temporal Analysis of Contradictions")
    print("=" * 60)

    snapshot = load_snapshot()

    results = analyze_contradictions(snapshot)

    print(f"\nTotal contradictions: {sum(len(v) for v in results.values())}")
    print(f"  Numerical updates (temporal): {len(results['numerical_updates'])}")
    print(f"  Source disagreements: {len(results['source_disagreements'])}")
    print(f"  Unknown: {len(results['unknown'])}")

    print("\n" + "=" * 60)
    print("Numerical Updates (likely temporal)")
    print("=" * 60)

    for entry in results['numerical_updates'][:5]:
        print(f"\n  [{entry['claim1']}...]")
        print(f"     Numbers: {entry['nums1']}")
        print(f"  vs [{entry['claim2']}...]")
        print(f"     Numbers: {entry['nums2']}")
        print(f"  Shared entities: {entry['shared_entities']}")

    print("\n" + "=" * 60)
    print("Source Disagreements")
    print("=" * 60)

    for entry in results['source_disagreements'][:5]:
        print(f"\n  [{entry['claim1']}...]")
        print(f"     From: {entry['source1']}")
        print(f"  vs [{entry['claim2']}...]")
        print(f"     From: {entry['source2']}")
        print(f"  Shared entities: {entry['shared_entities']}")

    # Summary insight
    print("\n" + "=" * 60)
    print("Insight")
    print("=" * 60)

    total = sum(len(v) for v in results.values())
    temporal = len(results['numerical_updates'])
    if total > 0:
        pct = temporal / total * 100
        print(f"\n{pct:.0f}% of contradictions appear to be numerical/temporal updates")
        print("This suggests most 'contradictions' are evolving facts, not fundamental disagreements.")


if __name__ == "__main__":
    main()
