"""
FULL CHAIN DEMO

Trace the entire process:
1. Claims arrive → seed/join events
2. Events grow (accumulate entity surface)
3. Events merge (same rules)
4. Distill final events (Jaynesian quantities)

Show both PROCESS and RESULT.
"""
import sys
sys.path.insert(0, '/app/backend')
sys.path.insert(0, '/app/test_eu')

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
import numpy as np
import json
from pathlib import Path
from math import log, log2


SOURCE_CREDIBILITY = {
    'bbc.com': 0.90, 'reuters.com': 0.88, 'apnews.com': 0.88,
    'theguardian.com': 0.85, 'dw.com': 0.85, 'cnn.com': 0.80,
    'scmp.com': 0.82, 'newsweek.com': 0.70, 'nypost.com': 0.60,
    'default': 0.5
}


@dataclass
class Event:
    id: str
    claim_ids: list = field(default_factory=list)
    entity_surface: set = field(default_factory=set)
    embedding_centroid: np.ndarray = None
    mass: float = 0.0
    sources: set = field(default_factory=set)
    # For tracking
    growth_history: list = field(default_factory=list)
    merge_history: list = field(default_factory=list)


class Engine:
    def __init__(self, entities, embeddings, entity_freq):
        self.events = []
        self.entities = entities
        self.embeddings = embeddings
        self.entity_freq = entity_freq
        self.max_freq = max(entity_freq.values()) if entity_freq else 1

    def get_credibility(self, url):
        if not url:
            return 0.5
        for domain, cred in SOURCE_CREDIBILITY.items():
            if domain in url.lower():
                return cred
        return 0.5

    def compute_affinity(self, claim, event):
        scores = {}

        # Semantic
        if claim.id in self.embeddings and event.embedding_centroid is not None:
            emb = np.array(self.embeddings[claim.id])
            sim = float(np.dot(emb, event.embedding_centroid) /
                       (np.linalg.norm(emb) * np.linalg.norm(event.embedding_centroid) + 1e-9))
            scores['semantic'] = sim
        else:
            scores['semantic'] = 0.0

        # Entity overlap
        shared = set(claim.entity_ids) & event.entity_surface
        scores['entity'] = len(shared) / max(len(claim.entity_ids), 1)

        # Specificity
        if shared:
            scores['specificity'] = sum(
                1.0 - (self.entity_freq.get(e, self.max_freq) / self.max_freq)
                for e in shared
            ) / len(shared)
        else:
            scores['specificity'] = 0.0

        combined = 0.6 * scores['semantic'] + 0.25 * scores['entity'] + 0.15 * scores['specificity']
        return combined, scores

    def compute_event_affinity(self, a, b):
        if a.embedding_centroid is None or b.embedding_centroid is None:
            return 0.0, {}

        sim = float(np.dot(a.embedding_centroid, b.embedding_centroid) /
                   (np.linalg.norm(a.embedding_centroid) * np.linalg.norm(b.embedding_centroid) + 1e-9))

        shared = a.entity_surface & b.entity_surface
        union = a.entity_surface | b.entity_surface
        jaccard = len(shared) / len(union) if union else 0.0

        if shared:
            spec = sum(1.0 - (self.entity_freq.get(e, self.max_freq) / self.max_freq)
                      for e in shared) / len(shared)
        else:
            spec = 0.0

        combined = 0.6 * sim + 0.25 * jaccard + 0.15 * spec
        return combined, {'semantic': sim, 'jaccard': jaccard, 'specificity': spec}

    def process_claim(self, claim, url, threshold=0.45):
        cred = self.get_credibility(url)
        emb = self.embeddings.get(claim.id)

        best_event, best_score = None, 0.0
        for event in self.events:
            score, _ = self.compute_affinity(claim, event)
            if score > best_score:
                best_score = score
                best_event = event

        if best_score >= threshold and best_event:
            # Join
            best_event.claim_ids.append(claim.id)
            best_event.entity_surface.update(claim.entity_ids)
            best_event.mass += cred
            if url:
                domain = url.split('/')[2].replace('www.', '')
                best_event.sources.add(domain)

            n = len(best_event.claim_ids)
            if emb is not None:
                if best_event.embedding_centroid is None:
                    best_event.embedding_centroid = np.array(emb)
                else:
                    best_event.embedding_centroid = (
                        best_event.embedding_centroid * (n-1) + np.array(emb)
                    ) / n

            best_event.growth_history.append((claim.id, best_score))
            return "joined", best_event, best_score
        else:
            # Seed
            new_event = Event(id=f"E{len(self.events)+1}")
            new_event.claim_ids.append(claim.id)
            new_event.entity_surface.update(claim.entity_ids)
            new_event.mass = cred
            if url:
                domain = url.split('/')[2].replace('www.', '')
                new_event.sources.add(domain)
            if emb is not None:
                new_event.embedding_centroid = np.array(emb)
            self.events.append(new_event)
            return "seeded", new_event, 0.0

    def merge_events(self, threshold=0.50):
        merged = {}
        merges = []

        for i, a in enumerate(self.events):
            if a.id in merged or len(a.claim_ids) < 2:
                continue
            for j, b in enumerate(self.events):
                if j <= i or b.id in merged or len(b.claim_ids) < 2:
                    continue

                score, breakdown = self.compute_event_affinity(a, b)
                if score >= threshold:
                    a.claim_ids.extend(b.claim_ids)
                    a.entity_surface.update(b.entity_surface)
                    a.mass += b.mass
                    a.sources.update(b.sources)
                    a.merge_history.append((b.id, score))

                    n_a = len(a.claim_ids) - len(b.claim_ids)
                    n_b = len(b.claim_ids)
                    if a.embedding_centroid is not None and b.embedding_centroid is not None:
                        a.embedding_centroid = (
                            a.embedding_centroid * n_a + b.embedding_centroid * n_b
                        ) / (n_a + n_b)

                    merged[b.id] = a.id
                    merges.append((a.id, b.id, score))

        self.events = [e for e in self.events if e.id not in merged]
        return merges


def compute_event_coherence(event, claims, edges_data):
    """Compute internal coherence of event."""
    event_claims = set(event.claim_ids)
    corr, cont = 0, 0

    for cid in event.claim_ids:
        claim = claims.get(cid)
        if not claim:
            continue
        for other_id in (claim.corroborates_ids or []):
            if other_id in event_claims:
                corr += 1
        for other_id in (claim.contradicts_ids or []):
            if other_id in event_claims:
                cont += 1

    if corr + cont == 0:
        return 1.0
    return corr / (corr + cont)


def distill_event(event, claims, entities, pages):
    """Create Jaynesian distillation of event."""
    # Get claim texts
    claim_texts = []
    for cid in event.claim_ids:
        c = claims.get(cid)
        if c:
            page = pages.get(c.page_id)
            url = page.url if page else ""
            cred = 0.5
            for domain, cr in SOURCE_CREDIBILITY.items():
                if domain in url.lower():
                    cred = cr
                    break
            claim_texts.append((c.text, cred, cid))

    # Sort by credibility
    claim_texts.sort(key=lambda x: -x[1])

    # Entity names
    ent_names = [
        entities.get(e).canonical_name if entities.get(e) else e
        for e in list(event.entity_surface)[:8]
    ]

    return {
        'id': event.id,
        'claims': len(event.claim_ids),
        'mass': event.mass,
        'sources': len(event.sources),
        'entities': ent_names,
        'top_claims': [(t[:80], c) for t, c, _ in claim_texts[:5]],
        'coherence': 1.0,  # Would need edges data
    }


def main():
    from load_graph import load_snapshot

    print("=" * 70)
    print("FULL CHAIN DEMO: Claims → Events → Merge → Distill")
    print("=" * 70)

    snapshot = load_snapshot()
    claims = snapshot.claims
    entities = snapshot.entities
    pages = snapshot.pages

    with open("/app/test_eu/results/embeddings_cache.json") as f:
        embeddings = json.load(f)

    entity_freq = defaultdict(int)
    for c in claims.values():
        for eid in c.entity_ids:
            entity_freq[eid] += 1

    engine = Engine(entities, embeddings, dict(entity_freq))

    # =========================================================
    # PHASE 1: CLAIM PROCESSING (show growth)
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 1: CLAIM PROCESSING")
    print("=" * 70)

    claim_list = list(claims.values())
    print(f"\nProcessing {len(claim_list)} claims...\n")

    # Track a few events in detail
    tracked_events = set()

    for i, claim in enumerate(claim_list):
        page = pages.get(claim.page_id)
        url = page.url if page else None

        action, event, score = engine.process_claim(claim, url)

        # Show first 30 claims
        if i < 30:
            if action == "seeded":
                print(f"  [{i:3d}] SEED {event.id:4} | {claim.text[:45]}...")
                if len(tracked_events) < 3:
                    tracked_events.add(event.id)
            else:
                marker = "→" if event.id in tracked_events else " "
                print(f"  [{i:3d}] JOIN {event.id:4} (score={score:.2f}) {marker} | {claim.text[:40]}...")

    # Show growth summary for tracked events
    print("\n--- Growth Summary (tracked events) ---")
    for event in engine.events:
        if event.id in tracked_events and len(event.claim_ids) >= 3:
            print(f"\n{event.id}: {len(event.claim_ids)} claims, mass={event.mass:.1f}")
            print(f"  Entity surface: {len(event.entity_surface)} entities")
            ent_names = [entities.get(e).canonical_name if entities.get(e) else e
                        for e in list(event.entity_surface)[:5]]
            print(f"  Entities: {', '.join(ent_names)}")

    # =========================================================
    # PHASE 2: EVENT MERGE (same rules, higher level)
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 2: EVENT MERGE (same rules at event level)")
    print("=" * 70)

    print(f"\nBefore: {len(engine.events)} events")

    merges = engine.merge_events(threshold=0.50)

    print(f"\nMerges performed: {len(merges)}")
    for a_id, b_id, score in merges[:10]:
        print(f"  {b_id} → {a_id} (score={score:.2f})")

    print(f"\nAfter: {len(engine.events)} events")

    # =========================================================
    # PHASE 3: DISTILLED EVENTS (Jaynesian output)
    # =========================================================
    print("\n" + "=" * 70)
    print("PHASE 3: DISTILLED EVENTS (Jaynesian quantities)")
    print("=" * 70)

    engine.events.sort(key=lambda e: e.mass, reverse=True)

    for event in engine.events[:8]:
        if len(event.claim_ids) < 5:
            continue

        dist = distill_event(event, claims, entities, pages)

        print(f"\n{'─' * 60}")
        print(f"EVENT: {dist['id']}")
        print(f"{'─' * 60}")
        print(f"  Claims:   {dist['claims']}")
        print(f"  Mass:     {dist['mass']:.1f} (credibility-weighted evidence)")
        print(f"  Sources:  {dist['sources']} distinct")
        print(f"  Entities: {', '.join(dist['entities'][:5])}")

        print(f"\n  📍 WHAT WE KNOW (top claims by credibility):")
        for text, cred in dist['top_claims'][:3]:
            print(f"     [{cred:.2f}] {text}...")

        # Check for merge history
        if event.merge_history:
            print(f"\n  🔀 MERGE HISTORY:")
            for merged_id, score in event.merge_history:
                print(f"     Absorbed {merged_id} (score={score:.2f})")

    # =========================================================
    # FINAL: HK FIRE TRACE
    # =========================================================
    print("\n" + "=" * 70)
    print("CASE STUDY: HONG KONG FIRE")
    print("=" * 70)

    # Find fire events
    fire_events = []
    for event in engine.events:
        fire_count = sum(1 for cid in event.claim_ids
                        if 'fire' in claims[cid].text.lower() or 'blaze' in claims[cid].text.lower())
        if fire_count > 0:
            fire_events.append((event, fire_count))

    fire_events.sort(key=lambda x: -x[1])
    total_fire = sum(x[1] for x in fire_events)

    print(f"\nTotal fire claims: {total_fire}")
    print(f"Distributed across {len(fire_events)} events\n")

    if fire_events:
        main_event, main_count = fire_events[0]
        print(f"MAIN FIRE EVENT: {main_event.id}")
        print(f"  Contains {main_count}/{total_fire} = {100*main_count/total_fire:.0f}% of fire claims")
        print(f"  Total claims in event: {len(main_event.claim_ids)}")
        print(f"  Mass: {main_event.mass:.1f}")
        print(f"  Sources: {len(main_event.sources)}")

        if main_event.merge_history:
            print(f"\n  This event absorbed {len(main_event.merge_history)} other events:")
            for merged_id, score in main_event.merge_history:
                print(f"    - {merged_id} (score={score:.2f})")

        # Show entity surface growth
        print(f"\n  Entity surface: {len(main_event.entity_surface)} entities")
        ent_names = [entities.get(e).canonical_name if entities.get(e) else e
                    for e in list(main_event.entity_surface)[:10]]
        print(f"  Key entities: {', '.join(ent_names)}")

    print("\n" + "=" * 70)
    print("SUMMARY: THE FULL CHAIN")
    print("=" * 70)
    print(f"""
    1. CLAIMS ARRIVE (1215 total)
           ↓
    2. SEED or JOIN events
       - compute_affinity(claim, event)
       - semantic + entity + specificity
           ↓
    3. EVENTS GROW
       - entity surface expands
       - mass accumulates
       - becomes "stickier"
           ↓
    4. EVENTS MERGE (same rules!)
       - compute_event_affinity(event_a, event_b)
       - {len(merges)} merges performed
           ↓
    5. DISTILLED OUTPUT
       - {len([e for e in engine.events if len(e.claim_ids) >= 5])} significant events
       - Each with: mass, entities, what we know

    Same engine, same rules, all scales.
    """)


if __name__ == "__main__":
    main()
