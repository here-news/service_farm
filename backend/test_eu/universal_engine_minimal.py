"""
Minimal Universal Epistemic Engine Test

Core idea: Edges first, events emerge from connected components.
"""
import sys
sys.path.insert(0, '/app/backend')
sys.path.insert(0, '/app/test_eu')

from collections import defaultdict
from load_graph import load_snapshot

def relate(c1, c2, embeddings=None, entity_freq=None) -> str:
    """The atomic epistemic operation - combines numeric, semantic, entity signals"""
    import re
    import numpy as np

    # Extract death counts
    def get_deaths(text):
        patterns = [r'(\d+)\s*(?:people\s+)?(?:killed|dead|died|deaths)',
                    r'(?:killed|death toll)[:\s]+(\d+)']
        for p in patterns:
            m = re.search(p, text.lower())
            if m: return int(m.group(1))
        return None

    d1, d2 = get_deaths(c1.text), get_deaths(c2.text)

    # 1. Numeric matching (strongest signal)
    if d1 and d2:
        if d1 == d2:
            return 'CORROBORATES'
        else:
            return 'CONTRADICTS'

    # Get semantic similarity
    sim = 0.0
    if embeddings and c1.id in embeddings and c2.id in embeddings:
        e1, e2 = np.array(embeddings[c1.id]), np.array(embeddings[c2.id])
        sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))

    # Get entity overlap with specificity weighting
    shared_entities = set(c1.entity_ids) & set(c2.entity_ids)

    # Compute specificity-weighted overlap
    # Specific entities (low freq) = high weight, common entities = low weight
    if entity_freq:
        max_freq = max(entity_freq.values()) if entity_freq else 100
        specificity_score = sum(
            1.0 - (entity_freq.get(e, max_freq) / max_freq)
            for e in shared_entities
        )
    else:
        specificity_score = len(shared_entities)

    # 2. High semantic similarity = same statement
    if sim > 0.85:
        return 'CORROBORATES'

    # 3. Specific entity overlap + moderate similarity = same event
    # Higher specificity score → lower similarity threshold needed
    if specificity_score > 1.5 and sim > 0.3:  # Very specific overlap
        return 'CORROBORATES'
    if specificity_score > 0.8 and sim > 0.5:  # Moderate specific overlap
        return 'CORROBORATES'
    if specificity_score > 0.5 and sim > 0.65:  # Some specific overlap
        return 'CORROBORATES'

    return 'INDEPENDENT'


def find_candidates(claim, entity_index, limit=20):
    """Find candidate claims via entity routing"""
    candidates = set()
    for eid in claim.entity_ids:
        candidates.update(entity_index.get(eid, []))
    candidates.discard(claim.id)
    return list(candidates)[:limit]


def build_graph(claims, entity_index, embeddings=None):
    """Build epistemic graph: EXISTING edges + inferred edges"""
    edges = []
    seen_pairs = set()

    # 1. FIRST: Add all existing epistemic edges from the data
    existing_count = 0
    for claim in claims.values():
        for cid in (claim.corroborates_ids or []):
            if cid in claims:
                pair = tuple(sorted([claim.id, cid]))
                if pair not in seen_pairs:
                    edges.append((claim.id, cid, 'CORROBORATES'))
                    seen_pairs.add(pair)
                    existing_count += 1
        for cid in (claim.contradicts_ids or []):
            if cid in claims:
                pair = tuple(sorted([claim.id, cid]))
                if pair not in seen_pairs:
                    edges.append((claim.id, cid, 'CONTRADICTS'))
                    seen_pairs.add(pair)
                    existing_count += 1
        for cid in (claim.updates_ids or []):
            if cid in claims:
                pair = tuple(sorted([claim.id, cid]))
                if pair not in seen_pairs:
                    edges.append((claim.id, cid, 'UPDATES'))
                    seen_pairs.add(pair)
                    existing_count += 1

    print(f"  Existing edges from data: {existing_count}")

    # 2. THEN: Infer additional edges for unconnected claims
    entity_freq = {eid: len(cids) for eid, cids in entity_index.items()}
    inferred_count = 0

    for claim in claims.values():
        candidates = find_candidates(claim, entity_index)

        for cid in candidates:
            pair = tuple(sorted([claim.id, cid]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            other = claims[cid]
            rel = relate(claim, other, embeddings, entity_freq)

            if rel != 'INDEPENDENT':
                edges.append((claim.id, cid, rel))
                inferred_count += 1

    print(f"  Inferred new edges: {inferred_count}")
    return edges


def find_components(claims, edges):
    """Find connected components = emergent events"""
    parent = {c: c for c in claims}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    for c1, c2, _ in edges:
        union(c1, c2)

    components = defaultdict(list)
    for cid in claims:
        components[find(cid)].append(cid)

    return sorted(components.values(), key=len, reverse=True)


def get_embeddings(claims, cache_path="/app/test_eu/results/embeddings_cache.json"):
    """Get embeddings - load from cache or compute and save"""
    import os
    import json
    from pathlib import Path

    # Try loading from cache
    if Path(cache_path).exists():
        with open(cache_path) as f:
            cached = json.load(f)
        # Check if cache has all claims
        if set(cached.keys()) >= set(claims.keys()):
            print(f"  Loaded {len(cached)} embeddings from cache")
            return cached

    # Compute embeddings
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    embeddings = {}

    texts = [(c.id, c.text[:500]) for c in claims.values()]
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"  Computing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[t[1] for t in batch]
        )
        for j, item in enumerate(batch):
            embeddings[item[0]] = response.data[j].embedding

    # Save to cache
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(embeddings, f)
    print(f"  Saved {len(embeddings)} embeddings to cache")

    return embeddings


def main():
    print("=" * 60)
    print("UNIVERSAL EPISTEMIC ENGINE - MINIMAL TEST")
    print("=" * 60)

    snapshot = load_snapshot()
    claims = snapshot.claims

    # Build entity index
    entity_index = defaultdict(list)
    for c in claims.values():
        for eid in c.entity_ids:
            entity_index[eid].append(c.id)

    print(f"\nClaims: {len(claims)}")
    print(f"Entities: {len(entity_index)}")

    # Get embeddings for semantic similarity
    print("\nComputing embeddings...")
    embeddings = get_embeddings(claims)
    print(f"Embeddings: {len(embeddings)}")

    # Build graph
    print("\nBuilding epistemic graph...")
    edges = build_graph(claims, entity_index, embeddings)

    corr = sum(1 for _, _, r in edges if r == 'CORROBORATES')
    cont = sum(1 for _, _, r in edges if r == 'CONTRADICTS')
    print(f"Edges: {len(edges)} (CORROBORATES: {corr}, CONTRADICTS: {cont})")

    # Find emergent events
    print("\nFinding emergent events (connected components)...")
    components = find_components(claims, edges)

    non_trivial = [c for c in components if len(c) >= 2]
    print(f"Total components: {len(components)}")
    print(f"Non-trivial (2+ claims): {len(non_trivial)}")

    # Show top events
    print("\n" + "=" * 60)
    print("EMERGENT EVENTS (Top 8)")
    print("=" * 60)

    for i, comp in enumerate(components[:8]):
        if len(comp) < 2:
            break
        print(f"\n--- Event {i+1}: {len(comp)} claims ---")
        for cid in comp[:4]:
            print(f"  • {claims[cid].text[:70]}...")
        if len(comp) > 4:
            print(f"  ... and {len(comp) - 4} more")


if __name__ == "__main__":
    main()
