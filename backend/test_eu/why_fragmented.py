"""
WHY IS HK FIRE FRAGMENTED?

Investigate why claims about the same event don't connect.
"""
import sys
sys.path.insert(0, '/app/backend')
sys.path.insert(0, '/app/test_eu')

from collections import defaultdict
from load_graph import load_snapshot
import numpy as np


def main():
    snapshot = load_snapshot()
    claims = snapshot.claims
    entities = snapshot.entities

    # Get HK fire claims
    fire_claims = {}
    for cid, claim in claims.items():
        text = claim.text.lower()
        has_fire = 'fire' in text or 'blaze' in text
        has_hk = 'hong kong' in text or 'tai po' in text or 'wang fuk' in text
        if has_fire and has_hk:
            fire_claims[cid] = claim

    print(f"Total HK fire claims: {len(fire_claims)}")

    # Group by their entities
    entity_to_claims = defaultdict(list)
    claim_entities = {}

    for cid, claim in fire_claims.items():
        claim_entities[cid] = set(claim.entity_ids)
        for eid in claim.entity_ids:
            entity_to_claims[eid].append(cid)

    # Find which entities are shared most
    print("\n--- Entities shared across fire claims ---")
    entity_coverage = []
    for eid, claim_list in entity_to_claims.items():
        if len(claim_list) >= 2:
            ent = entities.get(eid)
            name = ent.canonical_name if ent else eid
            entity_coverage.append((len(claim_list), name, eid))

    entity_coverage.sort(reverse=True)
    for count, name, eid in entity_coverage[:15]:
        print(f"  {count:3d} claims share: {name}")

    # Find claims with NO shared entities
    print("\n--- Claims with no overlap ---")

    # Pick two claims at random from different "fragments"
    claim_list = list(fire_claims.keys())

    # Find pairs with zero entity overlap
    no_overlap = []
    for i, cid1 in enumerate(claim_list[:30]):
        for cid2 in claim_list[i+1:30]:
            overlap = claim_entities[cid1] & claim_entities[cid2]
            if not overlap:
                no_overlap.append((cid1, cid2))

    print(f"Pairs with zero entity overlap: {len(no_overlap)} (out of first 30 claims)")

    # Show examples
    for cid1, cid2 in no_overlap[:5]:
        c1, c2 = fire_claims[cid1], fire_claims[cid2]
        print(f"\n  Claim 1: {c1.text[:60]}...")
        print(f"    Entities: {[entities.get(e).canonical_name if entities.get(e) else e for e in c1.entity_ids]}")
        print(f"  Claim 2: {c2.text[:60]}...")
        print(f"    Entities: {[entities.get(e).canonical_name if entities.get(e) else e for e in c2.entity_ids]}")

    # Load embeddings and check semantic similarity
    print("\n--- Semantic similarity of non-overlapping pairs ---")
    import json
    from pathlib import Path
    cache_path = "/app/test_eu/results/embeddings_cache.json"
    embeddings = {}
    if Path(cache_path).exists():
        with open(cache_path) as f:
            embeddings = json.load(f)

    for cid1, cid2 in no_overlap[:5]:
        if cid1 in embeddings and cid2 in embeddings:
            e1 = np.array(embeddings[cid1])
            e2 = np.array(embeddings[cid2])
            sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
            print(f"  sim({cid1[:8]}..., {cid2[:8]}...) = {sim:.3f}")

    print("\n" + "=" * 60)
    print("THE PROBLEM")
    print("=" * 60)
    print("""
    Claims about the SAME EVENT don't share entities.

    Example:
      "Fire at Wang Fuk Court killed 36" → entities: [Wang Fuk Court, ...]
      "Xi Jinping urged HK to handle fire" → entities: [Xi Jinping, Hong Kong, ...]

    These have ZERO entity overlap, so entity routing never compares them.
    Even if semantically similar, they never get connected.

    SOLUTIONS:
    1. Multi-hop: If A shares with B, and B shares with C, compare A↔C
    2. Location entity expansion: "Wang Fuk Court" ⊂ "Tai Po" ⊂ "Hong Kong"
    3. Temporal clustering: Claims from same time window get compared
    4. Lower semantic threshold for candidate expansion
    5. "Event seed" entities: Specific entities (Wang Fuk Court) pull in related claims
    """)


if __name__ == "__main__":
    main()
