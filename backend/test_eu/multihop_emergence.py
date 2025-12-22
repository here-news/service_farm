"""
MULTI-HOP EVENT EMERGENCE

The problem: Entity routing misses connections.
  - "Hong Kong" and "Tai Po District" are separate entities
  - Claims don't share entities → never compared

Solution: Multi-hop / transitive closure
  - After initial graph, check if disconnected components should merge
  - If component A has edge to X, and component B has edge to X → merge
"""
import sys
sys.path.insert(0, '/app/backend')
sys.path.insert(0, '/app/test_eu')

from collections import defaultdict
from load_graph import load_snapshot
import numpy as np
import json
from pathlib import Path


def load_embeddings():
    cache_path = "/app/test_eu/results/embeddings_cache.json"
    if Path(cache_path).exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def get_component_embedding(claim_ids, claims, embeddings):
    """Average embedding of component."""
    vecs = []
    for cid in claim_ids[:20]:  # Sample
        if cid in embeddings:
            vecs.append(np.array(embeddings[cid]))
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def should_merge_components(comp1, comp2, claims, embeddings, threshold=0.6):
    """Should these two components be merged?"""
    # Get average embeddings
    e1 = get_component_embedding(comp1, claims, embeddings)
    e2 = get_component_embedding(comp2, claims, embeddings)

    if e1 is None or e2 is None:
        return False, 0.0

    sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))
    return sim > threshold, sim


def find_components(claim_ids, edges):
    """Find connected components."""
    parent = {c: c for c in claim_ids}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    for c1, c2, _ in edges:
        if c1 in parent and c2 in parent:
            union(c1, c2)

    components = defaultdict(list)
    for cid in claim_ids:
        components[find(cid)].append(cid)

    return list(components.values())


def main():
    print("=" * 70)
    print("MULTI-HOP EVENT EMERGENCE")
    print("=" * 70)

    snapshot = load_snapshot()
    claims = snapshot.claims
    entities = snapshot.entities
    embeddings = load_embeddings()

    # Get HK fire claims
    fire_claims = {}
    for cid, claim in claims.items():
        text = claim.text.lower()
        has_fire = 'fire' in text or 'blaze' in text
        has_hk = 'hong kong' in text or 'tai po' in text or 'wang fuk' in text
        if has_fire and has_hk:
            fire_claims[cid] = claim

    print(f"\nHK Fire claims: {len(fire_claims)}")

    # Get existing edges between fire claims
    edges = []
    seen = set()
    for claim in fire_claims.values():
        for cid in (claim.corroborates_ids or []):
            if cid in fire_claims:
                pair = tuple(sorted([claim.id, cid]))
                if pair not in seen:
                    edges.append((claim.id, cid, 'CORROBORATES'))
                    seen.add(pair)

    print(f"Existing edges: {len(edges)}")

    # Initial components (fragmented)
    components = find_components(list(fire_claims.keys()), edges)
    components = sorted(components, key=len, reverse=True)

    print(f"Initial components: {len(components)}")
    for i, comp in enumerate(components[:5]):
        print(f"  Component {i+1}: {len(comp)} claims")

    # MULTI-HOP MERGE: Check if components should merge
    print("\n--- Multi-hop merge phase ---")

    merge_count = 0
    merged = set()

    for i, comp1 in enumerate(components):
        if i in merged:
            continue

        for j, comp2 in enumerate(components):
            if j <= i or j in merged:
                continue

            should_merge, sim = should_merge_components(
                comp1, comp2, fire_claims, embeddings, threshold=0.55
            )

            if should_merge:
                print(f"  Merge {i+1} ({len(comp1)}) + {j+1} ({len(comp2)}), sim={sim:.3f}")
                comp1.extend(comp2)
                merged.add(j)
                merge_count += 1

    print(f"Merged {merge_count} component pairs")

    # Final components
    final_components = [c for i, c in enumerate(components) if i not in merged]
    final_components = sorted(final_components, key=len, reverse=True)

    print(f"\nFinal components: {len(final_components)}")
    for i, comp in enumerate(final_components[:5]):
        print(f"  Component {i+1}: {len(comp)} claims")

    # Did we consolidate HK fire?
    largest = final_components[0]
    print(f"\n>>> Largest component has {len(largest)} of {len(fire_claims)} fire claims ({100*len(largest)/len(fire_claims):.0f}%)")

    if len(largest) >= len(fire_claims) * 0.8:
        print("✓ SUCCESS: HK Fire mostly consolidated!")
    else:
        print("⚠ Still fragmented")

    print("\n" + "=" * 70)
    print("THE MULTI-HOP IDEA")
    print("=" * 70)
    print("""
    Entity routing creates INITIAL edges.
    But claims about same event may not share entities.

    Multi-hop: After building initial graph, check if components should merge.

    If avg(embedding(component_A)) ≈ avg(embedding(component_B))
       → Merge A and B

    This is a coarser signal but catches what entity routing misses.

    The hierarchy:
    1. FINE-GRAINED: Entity overlap + semantic similarity → edges
    2. COARSE-GRAINED: Component-level semantic similarity → merges

    Like a funnel: coarse first, then fine.
    """)


if __name__ == "__main__":
    main()
