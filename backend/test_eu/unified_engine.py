"""
UNIFIED EPISTEMIC ENGINE

All signals combined:
- Jaynesian: source credibility, entropy, mass
- Semantic: embedding similarity
- Entity: overlap + specificity
- Growth: events accumulate surface, become "stickier"

Process:
1. Claims arrive one by one
2. For each claim, compute P(belongs to event_i) for all events
3. Join best match OR seed new event
4. Events grow, compete for claims
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
from math import log, log2, exp


# Source credibility priors
SOURCE_CREDIBILITY = {
    'bbc.com': 0.90, 'reuters.com': 0.88, 'apnews.com': 0.88,
    'theguardian.com': 0.85, 'dw.com': 0.85, 'cnn.com': 0.80,
    'scmp.com': 0.82, 'aljazeera.com': 0.80, 'newsweek.com': 0.70,
    'nypost.com': 0.60, 'foxnews.com': 0.65, 'default': 0.5
}


@dataclass
class Event:
    id: str
    claim_ids: list = field(default_factory=list)
    entity_surface: set = field(default_factory=set)
    embedding_centroid: np.ndarray = None
    mass: float = 0.0
    entropy: float = 0.0

    def add_claim(self, claim, embedding, credibility):
        self.claim_ids.append(claim.id)
        self.entity_surface.update(claim.entity_ids)
        self.mass += credibility

        # Update centroid
        if embedding is not None:
            if self.embedding_centroid is None:
                self.embedding_centroid = np.array(embedding)
            else:
                n = len(self.claim_ids)
                self.embedding_centroid = (
                    self.embedding_centroid * (n-1) + np.array(embedding)
                ) / n


class UnifiedEngine:
    def __init__(self, entities, embeddings, entity_freq):
        self.events = []
        self.entities = entities
        self.embeddings = embeddings
        self.entity_freq = entity_freq
        self.max_freq = max(entity_freq.values()) if entity_freq else 1

    def get_source_credibility(self, url):
        if not url:
            return SOURCE_CREDIBILITY['default']
        for domain, cred in SOURCE_CREDIBILITY.items():
            if domain in url.lower():
                return cred
        return SOURCE_CREDIBILITY['default']

    def compute_affinity(self, claim, event, url=None):
        """
        Compute P(claim belongs to event) using all signals.

        Returns: (score, breakdown)
        """
        scores = {}

        # 1. SEMANTIC SIMILARITY
        if claim.id in self.embeddings and event.embedding_centroid is not None:
            emb = np.array(self.embeddings[claim.id])
            sim = float(np.dot(emb, event.embedding_centroid) /
                       (np.linalg.norm(emb) * np.linalg.norm(event.embedding_centroid) + 1e-9))
            scores['semantic'] = sim
        else:
            scores['semantic'] = 0.0

        # 2. ENTITY OVERLAP
        shared = set(claim.entity_ids) & event.entity_surface
        scores['entity_count'] = len(shared) / max(len(claim.entity_ids), 1)

        # 3. ENTITY SPECIFICITY (IDF-like)
        if shared:
            specificity = sum(
                1.0 - (self.entity_freq.get(e, self.max_freq) / self.max_freq)
                for e in shared
            ) / len(shared)
            scores['entity_specificity'] = specificity
        else:
            scores['entity_specificity'] = 0.0

        # 4. EVENT MASS (larger events are more "attractive")
        # Log scale to avoid runaway growth
        scores['mass_bonus'] = min(0.1, 0.02 * log(1 + event.mass))

        # 5. COMBINED SCORE (weighted)
        # Semantic is primary, entity provides boost
        combined = (
            0.6 * scores['semantic'] +
            0.2 * scores['entity_count'] +
            0.15 * scores['entity_specificity'] +
            0.05 * scores['mass_bonus']
        )

        return combined, scores

    def process_claim(self, claim, url=None, threshold=0.45):
        """
        Process a new claim: join existing event or seed new one.

        Returns: (action, event_id, score, breakdown)
        """
        credibility = self.get_source_credibility(url)
        embedding = self.embeddings.get(claim.id)

        best_event = None
        best_score = 0.0
        best_breakdown = {}

        # Check affinity with each existing event
        for event in self.events:
            score, breakdown = self.compute_affinity(claim, event, url)
            if score > best_score:
                best_score = score
                best_event = event
                best_breakdown = breakdown

        # Decision
        if best_score >= threshold and best_event:
            best_event.add_claim(claim, embedding, credibility)
            return "joined", best_event.id, best_score, best_breakdown
        else:
            # Seed new event
            new_event = Event(id=f"event_{len(self.events)+1}")
            new_event.add_claim(claim, embedding, credibility)
            self.events.append(new_event)
            return "seeded", new_event.id, 0.0, {}

    def compute_event_affinity(self, event_a, event_b):
        """
        Same rules applied to events.
        P(event_a and event_b should merge)
        """
        scores = {}

        # 1. SEMANTIC: centroid similarity
        if event_a.embedding_centroid is not None and event_b.embedding_centroid is not None:
            sim = float(np.dot(event_a.embedding_centroid, event_b.embedding_centroid) /
                       (np.linalg.norm(event_a.embedding_centroid) *
                        np.linalg.norm(event_b.embedding_centroid) + 1e-9))
            scores['semantic'] = sim
        else:
            scores['semantic'] = 0.0

        # 2. ENTITY OVERLAP
        shared = event_a.entity_surface & event_b.entity_surface
        union = event_a.entity_surface | event_b.entity_surface
        scores['entity_jaccard'] = len(shared) / len(union) if union else 0.0

        # 3. ENTITY SPECIFICITY of shared entities
        if shared:
            specificity = sum(
                1.0 - (self.entity_freq.get(e, self.max_freq) / self.max_freq)
                for e in shared
            ) / len(shared)
            scores['entity_specificity'] = specificity
        else:
            scores['entity_specificity'] = 0.0

        # Combined (same weights as claim affinity)
        combined = (
            0.6 * scores['semantic'] +
            0.25 * scores['entity_jaccard'] +
            0.15 * scores['entity_specificity']
        )

        return combined, scores

    def merge_events(self, threshold=0.55):
        """
        Periodically merge similar events.
        Same rules as claim→event, now event→event.
        """
        merged_into = {}  # event_id -> merged_into_event_id

        for i, event_a in enumerate(self.events):
            if event_a.id in merged_into:
                continue
            if len(event_a.claim_ids) < 2:
                continue

            for j, event_b in enumerate(self.events):
                if j <= i:
                    continue
                if event_b.id in merged_into:
                    continue
                if len(event_b.claim_ids) < 2:
                    continue

                score, breakdown = self.compute_event_affinity(event_a, event_b)

                if score >= threshold:
                    # Merge B into A
                    event_a.claim_ids.extend(event_b.claim_ids)
                    event_a.entity_surface.update(event_b.entity_surface)
                    event_a.mass += event_b.mass

                    # Update centroid
                    n_a = len(event_a.claim_ids) - len(event_b.claim_ids)
                    n_b = len(event_b.claim_ids)
                    if event_a.embedding_centroid is not None and event_b.embedding_centroid is not None:
                        event_a.embedding_centroid = (
                            event_a.embedding_centroid * n_a + event_b.embedding_centroid * n_b
                        ) / (n_a + n_b)

                    merged_into[event_b.id] = event_a.id
                    yield (event_a.id, event_b.id, score, breakdown)

        # Remove merged events
        self.events = [e for e in self.events if e.id not in merged_into]

    def get_event_summary(self, event, claims):
        """Get summary of an event."""
        texts = [claims[cid].text[:50] for cid in event.claim_ids[:3]]
        entities = [
            self.entities.get(e).canonical_name if self.entities.get(e) else e
            for e in list(event.entity_surface)[:5]
        ]
        return {
            'claims': len(event.claim_ids),
            'mass': event.mass,
            'entities': entities,
            'sample': texts[0] if texts else ""
        }


def main():
    from load_graph import load_snapshot

    print("=" * 70)
    print("UNIFIED EPISTEMIC ENGINE")
    print("All signals: Jaynes + semantic + entity + growth")
    print("=" * 70)

    # Load data
    snapshot = load_snapshot()
    claims = snapshot.claims
    entities = snapshot.entities
    pages = snapshot.pages

    # Load embeddings
    cache_path = "/app/test_eu/results/embeddings_cache.json"
    with open(cache_path) as f:
        embeddings = json.load(f)

    # Build entity frequency
    entity_freq = defaultdict(int)
    for c in claims.values():
        for eid in c.entity_ids:
            entity_freq[eid] += 1

    # Initialize engine
    engine = UnifiedEngine(entities, embeddings, dict(entity_freq))

    # Simulate streaming claims (use order from data)
    claim_list = list(claims.values())
    print(f"\nProcessing {len(claim_list)} claims...")

    # Process claims
    for i, claim in enumerate(claim_list):
        page = pages.get(claim.page_id)
        url = page.url if page else None

        action, event_id, score, breakdown = engine.process_claim(claim, url)

        # Log every 100th or interesting events
        if i < 20 or (i % 200 == 0):
            if action == "joined":
                print(f"  [{i:4d}] JOINED {event_id} (score={score:.2f}) | {claim.text[:40]}...")
            else:
                print(f"  [{i:4d}] SEEDED {event_id} | {claim.text[:40]}...")

    # EVENT MERGE PASS (same rules, event→event)
    print("\n" + "=" * 70)
    print("EVENT MERGE PASS (same rules applied to events)")
    print("=" * 70)

    print(f"\nBefore merge: {len(engine.events)} events")

    merges = list(engine.merge_events(threshold=0.50))
    for a_id, b_id, score, breakdown in merges[:15]:
        print(f"  MERGED {b_id} → {a_id} (score={score:.2f})")

    print(f"\nAfter merge: {len(engine.events)} events")
    print(f"Total merges: {len(merges)}")

    # Results
    print("\n" + "=" * 70)
    print("EMERGENT EVENTS (after merge)")
    print("=" * 70)

    # Sort by mass
    engine.events.sort(key=lambda e: e.mass, reverse=True)

    for i, event in enumerate(engine.events[:15]):
        if len(event.claim_ids) < 2:
            continue

        summary = engine.get_event_summary(event, claims)
        print(f"\n{event.id}: {summary['claims']} claims, mass={summary['mass']:.1f}")
        print(f"  Entities: {', '.join(summary['entities'])}")
        print(f"  Sample: {summary['sample']}...")

    # HK Fire check
    print("\n" + "=" * 70)
    print("HK FIRE CHECK")
    print("=" * 70)

    fire_distribution = []
    for event in engine.events:
        fire_count = 0
        for cid in event.claim_ids:
            text = claims[cid].text.lower()
            if ('fire' in text or 'blaze' in text) and ('hong kong' in text or 'tai po' in text):
                fire_count += 1
        if fire_count > 0:
            fire_distribution.append((event.id, len(event.claim_ids), fire_count))

    fire_distribution.sort(key=lambda x: -x[2])
    total_fire = sum(x[2] for x in fire_distribution)

    print(f"\nTotal fire claims: {total_fire}")
    print(f"Distributed across {len(fire_distribution)} events:")
    for eid, total, fire in fire_distribution[:5]:
        pct = 100 * fire / total_fire if total_fire else 0
        print(f"  {eid}: {fire}/{total} claims ({pct:.0f}% of fire)")

    if fire_distribution and fire_distribution[0][2] >= total_fire * 0.7:
        print(f"\n✓ SUCCESS: {fire_distribution[0][2]}/{total_fire} = {100*fire_distribution[0][2]/total_fire:.0f}% consolidated")
    else:
        print(f"\n⚠ Fragmented")


if __name__ == "__main__":
    main()
