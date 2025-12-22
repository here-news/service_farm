"""
EVENT GROWTH TRACE

Start with 2 claims. Watch the event grow claim by claim.
Same engine, just traced step by step.
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


def relate(c1, c2, embeddings):
    """Same relate() function - returns (should_connect, reason, score)"""
    if c1.id not in embeddings or c2.id not in embeddings:
        return False, "no_embedding", 0.0

    e1 = np.array(embeddings[c1.id])
    e2 = np.array(embeddings[c2.id])
    sim = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))

    shared_entities = set(c1.entity_ids) & set(c2.entity_ids)

    if sim > 0.75:
        return True, "semantic", sim

    if len(shared_entities) >= 1 and sim > 0.5:
        return True, "entity+semantic", sim

    if len(shared_entities) >= 2:
        return True, "entity_overlap", sim

    return False, "independent", sim


def main():
    print("=" * 70)
    print("EVENT GROWTH TRACE")
    print("=" * 70)

    snapshot = load_snapshot()
    claims = snapshot.claims
    entities = snapshot.entities
    embeddings = load_embeddings()

    # Get HK fire claims and shuffle them (simulate arrival order)
    fire_claims = []
    for cid, claim in claims.items():
        text = claim.text.lower()
        if ('fire' in text or 'blaze' in text) and ('hong kong' in text or 'tai po' in text or 'wang fuk' in text):
            fire_claims.append(claim)

    print(f"\nTotal fire claims: {len(fire_claims)}")

    # Start with first 2 claims as seed
    event_claims = [fire_claims[0], fire_claims[1]]
    remaining = fire_claims[2:]

    print("\n--- SEED EVENT ---")
    for c in event_claims:
        ent_names = [entities.get(e).canonical_name if entities.get(e) else e for e in c.entity_ids]
        print(f"  • {c.text[:50]}...")
        print(f"    entities: {ent_names}")

    # Event's entity surface
    def get_event_entities(event_claims):
        surface = set()
        for c in event_claims:
            surface.update(c.entity_ids)
        return surface

    print(f"\nEvent entity surface: {len(get_event_entities(event_claims))} entities")

    # Try to add claims one by one
    print("\n--- GROWTH TRACE ---")

    for i, new_claim in enumerate(remaining[:20]):
        event_surface = get_event_entities(event_claims)

        # Check if new claim shares entities with event
        shared = set(new_claim.entity_ids) & event_surface
        shared_names = [entities.get(e).canonical_name if entities.get(e) else e for e in shared]

        # Check semantic similarity with event (average)
        event_embedding = np.mean([
            np.array(embeddings[c.id]) for c in event_claims if c.id in embeddings
        ], axis=0)

        if new_claim.id in embeddings:
            nc_emb = np.array(embeddings[new_claim.id])
            sim = float(np.dot(event_embedding, nc_emb) /
                       (np.linalg.norm(event_embedding) * np.linalg.norm(nc_emb) + 1e-9))
        else:
            sim = 0.0

        # Decision: should this claim join?
        joins = False
        reason = ""

        if sim > 0.6:
            joins = True
            reason = f"semantic sim={sim:.2f}"
        elif len(shared) >= 1 and sim > 0.4:
            joins = True
            reason = f"entity+semantic (shared: {shared_names}, sim={sim:.2f})"
        elif len(shared) >= 2:
            joins = True
            reason = f"entity overlap (shared: {shared_names})"

        if joins:
            event_claims.append(new_claim)
            status = "✓ JOINED"
        else:
            status = "✗ rejected"

        print(f"\n  Claim {i+3}: {new_claim.text[:40]}...")
        print(f"    shared entities: {shared_names if shared else 'none'}")
        print(f"    semantic sim: {sim:.2f}")
        print(f"    {status}: {reason if joins else 'no match'}")
        print(f"    Event size: {len(event_claims)}, surface: {len(get_event_entities(event_claims))} entities")

    print("\n" + "=" * 70)
    print("FINAL EVENT")
    print("=" * 70)
    print(f"Event grew from 2 → {len(event_claims)} claims")
    print(f"Entity surface: {len(get_event_entities(event_claims))} entities")

    # Show all entities in the event
    surface = get_event_entities(event_claims)
    surface_names = [entities.get(e).canonical_name if entities.get(e) else e for e in surface]
    print(f"Entities: {surface_names}")


if __name__ == "__main__":
    main()
