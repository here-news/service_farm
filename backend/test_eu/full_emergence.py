"""
FULL EVENT EMERGENCE - No Pre-filtering

Can events emerge from ALL claims using multi-hop?

1. Build initial graph (entity routing + existing edges)
2. Find components
3. Multi-hop merge similar components
4. See what events emerge
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
    for cid in claim_ids[:20]:
        if cid in embeddings:
            vecs.append(np.array(embeddings[cid]))
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


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


def describe_component(comp, claims, pages):
    """Quick description of what a component is about."""
    texts = [claims[cid].text.lower() for cid in comp[:10]]
    combined = ' '.join(texts)

    # Check for known events
    if ('fire' in combined or 'blaze' in combined) and ('hong kong' in combined or 'tai po' in combined):
        return "HK_FIRE"
    if 'brown university' in combined and 'shooting' in combined:
        return "BROWN_SHOOTING"
    if 'do kwon' in combined or 'terraform' in combined:
        return "DO_KWON"
    if 'king charles' in combined and 'cancer' in combined:
        return "KING_CHARLES"
    if 'jimmy lai' in combined:
        return "JIMMY_LAI"
    if 'tanker' in combined and 'venezuela' in combined:
        return "VENEZUELA_TANKER"
    if 'reiner' in combined and ('deaths' in combined or 'killed' in combined):
        return "REINER_CASE"

    # Generic
    return combined[:50] + "..."


def main():
    print("=" * 70)
    print("FULL EVENT EMERGENCE (No Pre-filtering)")
    print("=" * 70)

    snapshot = load_snapshot()
    claims = snapshot.claims
    pages = snapshot.pages
    embeddings = load_embeddings()

    print(f"\nTotal claims: {len(claims)}")

    # Build entity index for routing
    entity_index = defaultdict(list)
    for c in claims.values():
        for eid in c.entity_ids:
            entity_index[eid].append(c.id)

    entity_freq = {eid: len(cids) for eid, cids in entity_index.items()}

    # Build initial graph from existing edges
    edges = []
    seen = set()
    for claim in claims.values():
        for cid in (claim.corroborates_ids or []):
            if cid in claims:
                pair = tuple(sorted([claim.id, cid]))
                if pair not in seen:
                    edges.append((claim.id, cid, 'CORROBORATES'))
                    seen.add(pair)
        for cid in (claim.contradicts_ids or []):
            if cid in claims:
                pair = tuple(sorted([claim.id, cid]))
                if pair not in seen:
                    edges.append((claim.id, cid, 'CONTRADICTS'))
                    seen.add(pair)

    print(f"Existing edges: {len(edges)}")

    # PHASE 1: Infer edges via entity routing + semantic similarity
    print("\n--- Phase 1: Entity-routed edge inference ---")
    inferred = 0

    for i, claim in enumerate(claims.values()):
        if i % 300 == 0:
            print(f"  Processing {i}/{len(claims)}...")

        # Find candidates via entity routing
        candidates = set()
        for eid in claim.entity_ids:
            candidates.update(entity_index.get(eid, []))
        candidates.discard(claim.id)

        for cid in list(candidates)[:30]:
            pair = tuple(sorted([claim.id, cid]))
            if pair in seen:
                continue
            seen.add(pair)

            other = claims[cid]

            # Check if they should connect
            if claim.id not in embeddings or cid not in embeddings:
                continue

            e1 = np.array(embeddings[claim.id])
            e2 = np.array(embeddings[cid])
            sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))

            # Shared entities with specificity
            shared = set(claim.entity_ids) & set(other.entity_ids)
            max_freq = max(entity_freq.values())
            specificity = sum(1.0 - (entity_freq.get(e, max_freq) / max_freq)
                              for e in shared)

            # Connect if high semantic similarity OR specific entity overlap
            if sim > 0.75 or (specificity > 0.8 and sim > 0.4):
                edges.append((claim.id, cid, 'CORROBORATES'))
                inferred += 1

    print(f"Inferred edges: {inferred}")
    print(f"Total edges: {len(edges)}")

    # Initial components
    components = find_components(list(claims.keys()), edges)
    multi_claim = [c for c in components if len(c) >= 2]

    print(f"\nInitial components: {len(components)}")
    print(f"Multi-claim (2+): {len(multi_claim)}")

    # MULTI-HOP MERGE
    print("\n--- Multi-hop merge phase ---")

    merge_threshold = 0.65  # Higher threshold for all-claims (avoid over-merge)
    merge_count = 0
    merged = set()

    for i, comp1 in enumerate(components[:100]):  # Top 100 only
        if i in merged or len(comp1) < 2:
            continue

        e1 = get_component_embedding(comp1, claims, embeddings)
        if e1 is None:
            continue

        for j, comp2 in enumerate(components[:100]):
            if j <= i or j in merged or len(comp2) < 2:
                continue

            e2 = get_component_embedding(comp2, claims, embeddings)
            if e2 is None:
                continue

            sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))

            if sim > merge_threshold:
                desc1 = describe_component(comp1, claims, pages)
                desc2 = describe_component(comp2, claims, pages)
                print(f"  Merge: {desc1[:20]} + {desc2[:20]} (sim={sim:.3f})")
                comp1.extend(comp2)
                merged.add(j)
                merge_count += 1

                # Update embedding
                e1 = get_component_embedding(comp1, claims, embeddings)

    print(f"\nMerged {merge_count} component pairs")

    # Final components
    final_components = [c for i, c in enumerate(components) if i not in merged]
    final_components = sorted(final_components, key=len, reverse=True)

    print(f"\nFinal components (after merge): {len(final_components)}")

    # Show top emergent events
    print("\n" + "=" * 70)
    print("EMERGENT EVENTS")
    print("=" * 70)

    for i, comp in enumerate(final_components[:12]):
        if len(comp) < 3:
            break

        desc = describe_component(comp, claims, pages)
        print(f"\n  Event {i+1}: {len(comp)} claims - {desc}")

        # Sample claims
        for cid in comp[:2]:
            print(f"    • {claims[cid].text[:60]}...")

    # Check HK fire specifically
    print("\n" + "=" * 70)
    print("HK FIRE CHECK")
    print("=" * 70)

    hk_fire_claims = 0
    hk_fire_components = []

    for i, comp in enumerate(final_components):
        texts = ' '.join(claims[cid].text.lower() for cid in comp)
        has_fire = 'fire' in texts or 'blaze' in texts
        has_hk = 'hong kong' in texts or 'tai po' in texts or 'wang fuk' in texts

        if has_fire and has_hk:
            fire_count = sum(1 for cid in comp
                           if 'fire' in claims[cid].text.lower() or 'blaze' in claims[cid].text.lower())
            if fire_count > 0:
                hk_fire_claims += fire_count
                hk_fire_components.append((i+1, len(comp), fire_count))

    print(f"\nTotal HK fire claims found: {hk_fire_claims}")
    print(f"Distributed across {len(hk_fire_components)} components:")
    for comp_num, total, fire in hk_fire_components[:5]:
        print(f"  Event {comp_num}: {fire} fire claims / {total} total")

    if len(hk_fire_components) == 1:
        pct = hk_fire_components[0][2] / hk_fire_claims * 100
        print(f"\n✓ SUCCESS: HK Fire in ONE component! ({pct:.0f}%)")
    elif hk_fire_components and hk_fire_components[0][2] >= hk_fire_claims * 0.7:
        pct = hk_fire_components[0][2] / hk_fire_claims * 100
        print(f"\n✓ MOSTLY CONSOLIDATED: Largest has {pct:.0f}% of fire claims")
    else:
        print(f"\n⚠ FRAGMENTED across {len(hk_fire_components)} components")


if __name__ == "__main__":
    main()
