#!/usr/bin/env python3
"""
Evaluate EU Clustering vs Legacy Events

Comparison metrics:
1. Event count and claim distribution
2. Semantic coherence within clusters
3. Topic separation (are distinct stories in different clusters?)
4. Sibling grouping quality
"""

import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/test_eu')

import json
import httpx
from collections import defaultdict
from load_graph import load_snapshot

def get_new_eu_data():
    """Get current EU clustering from demo server"""
    response = httpx.get("http://localhost:8765/api/eus", timeout=30)
    return response.json()

def get_new_eu_status():
    """Get demo status"""
    response = httpx.get("http://localhost:8765/api/status", timeout=30)
    return response.json()

def analyze_topic_keywords(texts):
    """Extract key topics from texts"""
    keywords = defaultdict(int)
    topic_markers = [
        "hong kong", "fire", "tai po", "killed", "missing",
        "charlie kirk", "assassination", "utah", "amanda seyfried",
        "jimmy lai", "trial", "sentencing",
        "do kwon", "crypto", "terraform",
        "brown university", "shooting", "trump",
        "venezuela", "oil", "tanker", "sanctions",
        "ai", "musk", "zuckerberg", "nvidia", "jensen huang",
        "king charles", "australia", "royal",
        "lee jae myung", "condolences"
    ]

    text_lower = " ".join(texts).lower()
    for marker in topic_markers:
        if marker in text_lower:
            keywords[marker] += 1

    return dict(keywords)

def main():
    print("=" * 80)
    print("EU CLUSTERING EVALUATION")
    print("=" * 80)

    # Load snapshot for ground truth
    snapshot = load_snapshot()
    total_claims = len(snapshot.claims)
    print(f"\nTotal claims in snapshot: {total_claims}")

    # Get new EU data
    try:
        eu_data = get_new_eu_data()
        status = get_new_eu_status()
    except Exception as e:
        print(f"Error connecting to demo server: {e}")
        print("Make sure demo is running at http://localhost:8765")
        return

    claims_processed = status['stats']['claims_processed']

    print(f"Claims processed by EU system: {claims_processed}")
    print(f"Progress: {status['progress']['percent']:.1f}%")

    # Analyze EU structure
    print("\n" + "=" * 80)
    print("EU HIERARCHY STRUCTURE")
    print("=" * 80)

    eus = eu_data.get('eus', {})
    l1_eus = eus.get('level_1', [])
    l2_eus = eus.get('level_2', [])
    l3_eus = eus.get('level_3', [])

    print(f"\nLevel 1 (Sub-events): {len(l1_eus)}")
    print(f"Level 2 (Events):     {len(l2_eus)}")
    print(f"Level 3 (Parents):    {len(l3_eus)}")

    # Claim distribution
    print("\n" + "-" * 40)
    print("CLAIM DISTRIBUTION PER L2 EVENT")
    print("-" * 40)

    l2_sorted = sorted(l2_eus, key=lambda x: x['claim_count'], reverse=True)
    for eu in l2_sorted[:10]:
        bar = "█" * min(eu['claim_count'], 40)
        print(f"[{eu['claim_count']:3d}] {bar}")
        print(f"      {eu['headline'][:60]}")

    # Topic analysis per L2
    print("\n" + "-" * 40)
    print("TOPIC COHERENCE PER L2 EVENT")
    print("-" * 40)

    for eu in l2_sorted[:8]:
        eu_id = eu['id']
        # Get full EU with claims
        try:
            detail = httpx.get(f"http://localhost:8765/api/eus/{eu_id}", timeout=30).json()
            claims = detail.get('all_claims', eu.get('sample_claims', []))
            topics = analyze_topic_keywords(claims)

            if topics:
                top_topics = sorted(topics.items(), key=lambda x: -x[1])[:3]
                topic_str = ", ".join([f"{t}({c})" for t, c in top_topics])
            else:
                topic_str = "(no keywords found)"

            print(f"\n{eu['headline'][:50]}...")
            print(f"  Topics: {topic_str}")
            print(f"  Claims: {len(claims)}, Sources: {eu['source_count']}")
        except:
            pass

    # L3 Parent analysis (sibling grouping)
    print("\n" + "=" * 80)
    print("SIBLING GROUPING (L3 PARENTS)")
    print("=" * 80)

    for eu in l3_eus:
        print(f"\n[L3] {eu['headline']}")
        print(f"     Claims: {eu['claim_count']}, Sources: {eu['source_count']}")

        # Get children
        try:
            detail = httpx.get(f"http://localhost:8765/api/eus/{eu['id']}", timeout=30).json()
            children = detail.get('children_tree', [])
            print(f"     Children ({len(children)}):")
            for child in children:
                print(f"       └─ [L{child['level']}] {child['headline'][:45]}... ({child['claim_count']} claims)")
        except:
            pass

    # Metrics summary
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)

    # Fragmentation: claims per L2 event
    if l2_eus:
        claims_per_l2 = [eu['claim_count'] for eu in l2_eus]
        avg_claims = sum(claims_per_l2) / len(claims_per_l2)
        max_claims = max(claims_per_l2)
        min_claims = min(claims_per_l2)

        print(f"\nClaims per L2 Event:")
        print(f"  Average: {avg_claims:.1f}")
        print(f"  Max:     {max_claims}")
        print(f"  Min:     {min_claims}")

    # Grouping ratio
    if l2_eus and l3_eus:
        grouped_events = sum(1 for eu in l2_eus if any(
            l3['id'] in str(eu) for l3 in l3_eus
        ))
        # Actually count events with parents
        grouped = len([eu for eu in l2_eus if eu.get('parent_id')])
        print(f"\nSibling Grouping:")
        print(f"  L2 events with L3 parent: {grouped}/{len(l2_eus)}")
        print(f"  L3 parent events: {len(l3_eus)}")

    # Hierarchy depth
    total_l2_claims = sum(eu['claim_count'] for eu in l2_eus)
    total_l3_claims = sum(eu['claim_count'] for eu in l3_eus)

    print(f"\nClaim Coverage:")
    print(f"  Total in L2 events: {total_l2_claims}")
    print(f"  Total in L3 parents: {total_l3_claims}")
    print(f"  Processing rate: {claims_processed}/{total_claims} ({100*claims_processed/total_claims:.1f}%)")

    # Comparison with what we'd expect
    print("\n" + "-" * 40)
    print("EXPECTED vs ACTUAL")
    print("-" * 40)
    print("""
Expected major stories in dataset:
  1. Hong Kong Tai Po Fire (deaths, missing, condolences)
  2. Charlie Kirk assassination at Utah
  3. Jimmy Lai trial/detention
  4. Do Kwon crypto case
  5. Brown University shooting
  6. Venezuela oil tanker sanctions
  7. AI leaders (Musk, Zuckerberg, Altman, Huang)
  8. King Charles Australia visit
""")

    print("Stories detected in L2+ events:")
    all_headlines = [eu['headline'] for eu in l2_eus + l3_eus]
    detected = []
    story_markers = {
        "fire": "Hong Kong Fire",
        "kirk": "Charlie Kirk",
        "lai": "Jimmy Lai",
        "kwon": "Do Kwon",
        "brown": "Brown Shooting",
        "venezuela": "Venezuela Oil",
        "ai": "AI Leaders",
        "charles": "King Charles"
    }

    for marker, story in story_markers.items():
        for h in all_headlines:
            if marker in h.lower():
                detected.append(f"  ✓ {story}")
                break
        else:
            detected.append(f"  ✗ {story} (not found)")

    print("\n".join(detected))

if __name__ == "__main__":
    main()
