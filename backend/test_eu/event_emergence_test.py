"""
EVENT EMERGENCE TEST

The fundamental question: How does an event EMERGE from scratch?

We have N claims. We don't know what events exist.
Can the engine discover "Hong Kong Fire" without us telling it?

Process:
1. Load ALL claims (no filtering)
2. Build edges via entity routing + relate()
3. Find connected components
4. See what events emerge
"""
import sys
sys.path.insert(0, '/app/backend')
sys.path.insert(0, '/app/test_eu')

from collections import defaultdict
from load_graph import load_snapshot
import numpy as np


def get_embeddings_cached():
    """Load cached embeddings."""
    import json
    from pathlib import Path
    cache_path = "/app/test_eu/results/embeddings_cache.json"
    if Path(cache_path).exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def relate(c1, c2, embeddings, entity_freq):
    """
    The atomic epistemic operation.

    Returns: (relation, confidence)
    """
    # Get semantic similarity
    sim = 0.0
    if c1.id in embeddings and c2.id in embeddings:
        e1 = np.array(embeddings[c1.id])
        e2 = np.array(embeddings[c2.id])
        sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))

    # Get entity overlap with specificity weighting
    shared = set(c1.entity_ids) & set(c2.entity_ids)

    if not shared:
        return 'INDEPENDENT', 0.0

    # Specificity score (rare entities = high weight)
    max_freq = max(entity_freq.values()) if entity_freq else 1
    specificity = sum(1.0 - (entity_freq.get(e, max_freq) / max_freq)
                      for e in shared)

    # Decision logic
    if sim > 0.85:
        return 'CORROBORATES', sim

    if specificity > 1.5 and sim > 0.3:
        return 'CORROBORATES', sim
    if specificity > 0.8 and sim > 0.5:
        return 'CORROBORATES', sim
    if specificity > 0.5 and sim > 0.65:
        return 'CORROBORATES', sim

    return 'INDEPENDENT', 0.0


def find_candidates(claim, entity_index, limit=30):
    """Find candidate claims via entity routing."""
    candidates = set()
    for eid in claim.entity_ids:
        candidates.update(entity_index.get(eid, []))
    candidates.discard(claim.id)
    return list(candidates)[:limit]


def build_graph(claims, entity_index, embeddings):
    """Build epistemic graph from scratch."""
    entity_freq = {eid: len(cids) for eid, cids in entity_index.items()}

    edges = []
    seen = set()

    # Also add existing edges from data
    existing = 0
    for claim in claims.values():
        for cid in (claim.corroborates_ids or []):
            if cid in claims:
                pair = tuple(sorted([claim.id, cid]))
                if pair not in seen:
                    edges.append((claim.id, cid, 'CORROBORATES'))
                    seen.add(pair)
                    existing += 1
        for cid in (claim.contradicts_ids or []):
            if cid in claims:
                pair = tuple(sorted([claim.id, cid]))
                if pair not in seen:
                    edges.append((claim.id, cid, 'CONTRADICTS'))
                    seen.add(pair)
                    existing += 1

    print(f"  Existing edges from data: {existing}")

    # Infer new edges via entity routing
    inferred = 0
    for i, claim in enumerate(claims.values()):
        if i % 200 == 0:
            print(f"  Processing claim {i}/{len(claims)}...")

        candidates = find_candidates(claim, entity_index)

        for cid in candidates:
            pair = tuple(sorted([claim.id, cid]))
            if pair in seen:
                continue
            seen.add(pair)

            other = claims[cid]
            rel, conf = relate(claim, other, embeddings, entity_freq)

            if rel != 'INDEPENDENT':
                edges.append((claim.id, cid, rel))
                inferred += 1

    print(f"  Inferred edges: {inferred}")
    return edges


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

    return sorted(components.values(), key=len, reverse=True)


def describe_event(claims, claim_ids, pages):
    """What is this event about?"""
    # Get sample texts
    texts = [claims[cid].text.lower() for cid in claim_ids[:20]]
    combined = ' '.join(texts)

    # Simple keyword extraction
    keywords = defaultdict(int)
    for word in combined.split():
        if len(word) > 4:
            keywords[word] += 1

    top_words = sorted(keywords.items(), key=lambda x: -x[1])[:10]

    # Get sources
    sources = set()
    for cid in claim_ids[:10]:
        page = pages.get(claims[cid].page_id)
        if page and page.url:
            try:
                from urllib.parse import urlparse
                domain = urlparse(page.url).netloc.replace('www.', '')
                sources.add(domain)
            except:
                pass

    return {
        'keywords': [w for w, _ in top_words],
        'sources': list(sources)[:5],
        'sample': claims[claim_ids[0]].text[:100]
    }


def main():
    print("=" * 70)
    print("EVENT EMERGENCE TEST")
    print("Can events emerge from raw claims without pre-definition?")
    print("=" * 70)

    # Load everything
    print("\nLoading data...")
    snapshot = load_snapshot()
    claims = snapshot.claims
    pages = snapshot.pages

    print(f"Total claims: {len(claims)}")

    # Build entity index
    entity_index = defaultdict(list)
    for c in claims.values():
        for eid in c.entity_ids:
            entity_index[eid].append(c.id)

    print(f"Total entities: {len(entity_index)}")

    # Load embeddings
    print("\nLoading embeddings...")
    embeddings = get_embeddings_cached()
    print(f"Embeddings available: {len(embeddings)}")

    # Build graph from scratch
    print("\nBuilding epistemic graph from scratch...")
    edges = build_graph(claims, entity_index, embeddings)
    print(f"Total edges: {len(edges)}")

    # Find emergent events
    print("\nFinding emergent events (connected components)...")
    components = find_components(list(claims.keys()), edges)

    non_trivial = [c for c in components if len(c) >= 3]
    print(f"Total components: {len(components)}")
    print(f"Non-trivial (3+ claims): {len(non_trivial)}")

    # Show top emergent events
    print("\n" + "=" * 70)
    print("EMERGENT EVENTS (Top 10)")
    print("=" * 70)

    for i, comp in enumerate(components[:10]):
        if len(comp) < 2:
            break

        desc = describe_event(claims, comp, pages)

        print(f"\n--- Event {i+1}: {len(comp)} claims ---")
        print(f"  Keywords: {', '.join(desc['keywords'][:6])}")
        print(f"  Sources: {', '.join(desc['sources'])}")
        print(f"  Sample: {desc['sample']}...")

        # Check if this is HK fire
        texts = ' '.join(claims[cid].text.lower() for cid in comp[:20])
        if 'hong kong' in texts and 'fire' in texts:
            print(f"  >>> THIS IS HONG KONG FIRE <<<")

    # Did HK fire emerge?
    print("\n" + "=" * 70)
    print("DID HONG KONG FIRE EMERGE?")
    print("=" * 70)

    # Count HK fire claims in each component
    hk_fire_distribution = []
    for i, comp in enumerate(components[:50]):
        texts = ' '.join(claims[cid].text.lower() for cid in comp)
        has_fire = 'fire' in texts or 'blaze' in texts
        has_hk = 'hong kong' in texts or 'tai po' in texts

        if has_fire and has_hk:
            # Count actual fire claims
            fire_count = sum(1 for cid in comp
                           if 'fire' in claims[cid].text.lower() or 'blaze' in claims[cid].text.lower())
            if fire_count > 0:
                hk_fire_distribution.append((i+1, len(comp), fire_count))

    if hk_fire_distribution:
        print("\nHK Fire claims found in components:")
        for comp_num, total, fire_count in hk_fire_distribution:
            print(f"  Event {comp_num}: {fire_count} fire claims out of {total} total")

        if len(hk_fire_distribution) == 1:
            print("\n✓ SUCCESS: HK Fire emerged as ONE coherent event!")
        else:
            print(f"\n⚠ FRAGMENTED: HK Fire split across {len(hk_fire_distribution)} components")
    else:
        print("\n✗ FAILED: HK Fire did not emerge as a recognizable event")


if __name__ == "__main__":
    main()
