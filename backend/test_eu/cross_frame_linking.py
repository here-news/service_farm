"""
Cross-Frame Entity Linking

Identify entities that appear across multiple events/frames.
These create graph structures (not just trees) at higher levels.

Example: Jimmy Lai might appear in:
- "Hong Kong Crackdown" (primary)
- "US-China Relations" (Trump raised case)
- "Press Freedom Global" (symbol)

Run inside container:
    docker exec herenews-app python /app/test_eu/cross_frame_linking.py
"""

import os
import json
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
import httpx
import psycopg2
from collections import defaultdict
import time

from load_graph import load_snapshot, GraphSnapshot


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"

PG_HOST = os.environ.get("POSTGRES_HOST", "herenews-postgres")
PG_DB = os.environ.get("POSTGRES_DB", "herenews")
PG_USER = os.environ.get("POSTGRES_USER", "herenews_user")
PG_PASS = os.environ.get("POSTGRES_PASSWORD", "herenews_pass")

SIM_THRESHOLD = 0.70
LLM_THRESHOLD = 0.55
EVENT_MERGE_THRESHOLD = 0.60


def get_pg_connection():
    return psycopg2.connect(host=PG_HOST, database=PG_DB, user=PG_USER, password=PG_PASS)


def load_cached_embeddings() -> Dict[str, List[float]]:
    conn = get_pg_connection()
    cur = conn.cursor()
    cur.execute("SELECT claim_id, embedding FROM core.claim_embeddings")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    embeddings = {}
    for claim_id, emb in rows:
        if isinstance(emb, str):
            emb = [float(x) for x in emb.strip('[]').split(',')]
        embeddings[claim_id] = list(emb)
    return embeddings


def llm_same_event(text1: str, text2: str) -> bool:
    prompt = f"""Are these claims about the same news story/event? Answer YES or NO.

Claim 1: {text1[:250]}
Claim 2: {text2[:250]}

Same story?"""

    for attempt in range(3):
        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 5,
                    "temperature": 0
                },
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip().upper().startswith("YES")
        except:
            if attempt < 2:
                time.sleep(2)
            else:
                return False


def llm_same_story(summary1: str, summary2: str) -> bool:
    prompt = f"""Are these two descriptions part of the SAME broader news event/story?
Answer YES or NO.

Description 1: {summary1[:300]}
Description 2: {summary2[:300]}

Same broader story?"""

    for attempt in range(3):
        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 5,
                    "temperature": 0
                },
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip().upper().startswith("YES")
        except:
            if attempt < 2:
                time.sleep(2)
            else:
                return False


def llm_extract_entities(text: str) -> List[str]:
    """Extract key named entities from text"""
    prompt = f"""Extract the key named entities (people, organizations, places) from this text.
Return as comma-separated list. Only include prominent entities.

Text: {text[:500]}

Entities:"""

    for attempt in range(3):
        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0
                },
                timeout=60.0
            )
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"].strip()
            entities = [e.strip() for e in result.split(',') if e.strip()]
            return entities
        except:
            if attempt < 2:
                time.sleep(2)
            else:
                return []


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@dataclass
class EU:
    id: str
    level: int = 0
    claim_ids: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    page_ids: Set[str] = field(default_factory=set)
    embedding: Optional[List[float]] = None
    internal_corr: int = 0
    internal_contra: int = 0
    children: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    entities: Set[str] = field(default_factory=set)  # Named entities

    def size(self) -> int:
        return len(self.claim_ids)

    def coherence(self) -> float:
        total = self.internal_corr + self.internal_contra
        return self.internal_corr / total if total > 0 else 1.0

    def label(self) -> str:
        return self.texts[0][:50] + "..." if self.texts else "empty"


class CrossFrameSystem:
    def __init__(self, snapshot: GraphSnapshot):
        self.snapshot = snapshot
        self.eus: Dict[str, EU] = {}
        self.claim_to_eu: Dict[str, str] = {}
        self.sub_counter = 0
        self.event_counter = 0
        self.llm_calls = 0
        self.total_claims = 0
        self.merges = 0

        # Entity tracking
        self.entity_to_events: Dict[str, Set[str]] = defaultdict(set)
        self.event_entities: Dict[str, Set[str]] = defaultdict(set)

    def process_claim(self, claim_id: str, text: str, page_id: str, embedding: List[float]) -> str:
        self.total_claims += 1

        level0_eus = [eu for eu in self.eus.values() if eu.level == 0 and eu.parent_id is None]

        best_eu = None
        best_sim = 0.0

        for eu in level0_eus:
            if eu.embedding:
                sim = cosine_sim(embedding, eu.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_eu = eu

        should_merge = False

        if best_sim >= SIM_THRESHOLD:
            should_merge = True
        elif best_sim >= LLM_THRESHOLD and best_eu:
            self.llm_calls += 1
            should_merge = llm_same_event(text, best_eu.texts[0])

        if should_merge and best_eu:
            self.merges += 1
            best_eu.claim_ids.append(claim_id)
            best_eu.texts.append(text)
            best_eu.page_ids.add(page_id)

            old_emb = np.array(best_eu.embedding)
            new_emb = np.array(embedding)
            n = len(best_eu.claim_ids)
            best_eu.embedding = ((old_emb * (n - 1) + new_emb) / n).tolist()

            self.claim_to_eu[claim_id] = best_eu.id
            return best_eu.id
        else:
            self.sub_counter += 1
            new_eu = EU(
                id=f"sub_{self.sub_counter}",
                level=0,
                claim_ids=[claim_id],
                texts=[text],
                page_ids={page_id},
                embedding=embedding
            )
            self.eus[new_eu.id] = new_eu
            self.claim_to_eu[claim_id] = new_eu.id
            return new_eu.id

    def merge_into_events(self, min_size: int = 3):
        print("\n--- Merging sub-events into events ---")

        candidates = [
            eu for eu in self.eus.values()
            if eu.level == 0 and eu.size() >= min_size and eu.parent_id is None
        ]

        print(f"Candidates: {len(candidates)}")

        if len(candidates) < 2:
            return

        candidates.sort(key=lambda x: x.size(), reverse=True)

        used = set()
        events_created = 0

        for eu in candidates:
            if eu.id in used:
                continue

            group = [eu]
            used.add(eu.id)

            for other in candidates:
                if other.id in used:
                    continue

                sim = cosine_sim(eu.embedding, other.embedding)

                if sim >= EVENT_MERGE_THRESHOLD:
                    self.llm_calls += 1
                    if llm_same_story(eu.texts[0], other.texts[0]):
                        group.append(other)
                        used.add(other.id)

            if len(group) > 1:
                self.event_counter += 1
                event = EU(
                    id=f"event_{self.event_counter}",
                    level=1,
                    claim_ids=[],
                    texts=[],
                    page_ids=set(),
                    children=[sub.id for sub in group]
                )

                all_embs = []
                for sub in group:
                    event.claim_ids.extend(sub.claim_ids)
                    event.texts.extend(sub.texts[:2])
                    event.page_ids |= sub.page_ids
                    if sub.embedding:
                        all_embs.append(sub.embedding)
                    sub.parent_id = event.id

                event.embedding = np.mean(all_embs, axis=0).tolist()
                self.eus[event.id] = event
                events_created += 1

        print(f"Events created: {events_created}")

    def extract_event_entities(self):
        """Extract named entities from all events (level 1 and large level 0)"""
        print("\n--- Extracting entities from events ---")

        # Get events and large sub-events
        candidates = [
            eu for eu in self.eus.values()
            if (eu.level == 1) or (eu.level == 0 and eu.size() >= 10 and eu.parent_id is None)
        ]

        print(f"Extracting entities from {len(candidates)} events...")

        for eu in candidates:
            # Get representative text
            sample_text = " ".join(eu.texts[:5])
            entities = llm_extract_entities(sample_text)
            self.llm_calls += 1

            eu.entities = set(entities)
            self.event_entities[eu.id] = set(entities)

            # Track which events each entity appears in
            for entity in entities:
                normalized = entity.lower().strip()
                self.entity_to_events[normalized].add(eu.id)

            if entities:
                print(f"  {eu.id}: {', '.join(entities[:5])}")

    def find_cross_event_entities(self) -> List[Dict]:
        """Find entities that appear in multiple events"""
        cross_entities = []

        for entity, event_ids in self.entity_to_events.items():
            if len(event_ids) > 1:
                events_info = []
                for eid in event_ids:
                    eu = self.eus.get(eid)
                    if eu:
                        events_info.append({
                            'id': eid,
                            'size': eu.size(),
                            'label': eu.label()
                        })

                cross_entities.append({
                    'entity': entity,
                    'appears_in': len(event_ids),
                    'events': events_info
                })

        # Sort by number of appearances
        cross_entities.sort(key=lambda x: x['appears_in'], reverse=True)
        return cross_entities

    def compute_entity_overlap(self) -> Dict[Tuple[str, str], float]:
        """Compute Jaccard similarity of entity sets between events"""
        events = [
            eu for eu in self.eus.values()
            if (eu.level == 1) or (eu.level == 0 and eu.size() >= 10 and eu.parent_id is None)
        ]

        overlaps = {}
        for i, e1 in enumerate(events):
            for e2 in events[i+1:]:
                if e1.entities and e2.entities:
                    intersection = len(e1.entities & e2.entities)
                    union = len(e1.entities | e2.entities)
                    jaccard = intersection / union if union > 0 else 0

                    if jaccard > 0:
                        overlaps[(e1.id, e2.id)] = jaccard

        return overlaps


def run_analysis(snapshot: GraphSnapshot):
    print("Loading cached embeddings...")
    cached = load_cached_embeddings()
    print(f"  Found {len(cached)} cached embeddings")

    all_claims = list(snapshot.claims.keys())
    random.seed(42)
    random.shuffle(all_claims)

    system = CrossFrameSystem(snapshot)

    # Phase 1: Claims → Sub-events
    print(f"\n{'='*70}")
    print(f"Phase 1: Streaming claims")
    print(f"{'='*70}\n")

    for i, cid in enumerate(all_claims):
        claim = snapshot.claims[cid]
        system.process_claim(cid, claim.text, claim.page_id or "?", cached[cid])

    # Phase 2: Sub-events → Events
    print(f"\n{'='*70}")
    print("Phase 2: Merging into events")
    print(f"{'='*70}")

    system.merge_into_events(min_size=3)

    # Phase 3: Extract entities
    print(f"\n{'='*70}")
    print("Phase 3: Extracting entities")
    print(f"{'='*70}")

    system.extract_event_entities()

    # Phase 4: Find cross-event entities
    print(f"\n{'='*70}")
    print("Phase 4: Cross-event entity analysis")
    print(f"{'='*70}\n")

    cross_entities = system.find_cross_event_entities()
    entity_overlaps = system.compute_entity_overlap()

    # Report cross-event entities
    print("ENTITIES APPEARING IN MULTIPLE EVENTS:")
    print("-" * 50)

    for item in cross_entities[:20]:
        print(f"\n  '{item['entity']}' appears in {item['appears_in']} events:")
        for ev in item['events']:
            print(f"    └─ {ev['id']}: {ev['label'][:40]} ({ev['size']} claims)")

    # Report event pairs with entity overlap
    print(f"\n{'='*70}")
    print("EVENT PAIRS WITH ENTITY OVERLAP")
    print(f"{'='*70}\n")

    sorted_overlaps = sorted(entity_overlaps.items(), key=lambda x: x[1], reverse=True)

    for (e1_id, e2_id), jaccard in sorted_overlaps[:15]:
        e1 = system.eus.get(e1_id)
        e2 = system.eus.get(e2_id)
        if e1 and e2:
            shared = e1.entities & e2.entities
            print(f"Jaccard={jaccard:.2f}: {e1_id} ↔ {e2_id}")
            print(f"  Event 1: {e1.label()[:40]} ({e1.size()} claims)")
            print(f"  Event 2: {e2.label()[:40]} ({e2.size()} claims)")
            print(f"  Shared entities: {', '.join(shared)}")
            print()

    # Graph structure implications
    print(f"{'='*70}")
    print("GRAPH STRUCTURE IMPLICATIONS")
    print(f"{'='*70}\n")

    multi_event_entities = [e for e in cross_entities if e['appears_in'] >= 2]
    print(f"Entities appearing in 2+ events: {len(multi_event_entities)}")
    print(f"Event pairs with entity overlap: {len(entity_overlaps)}")
    print(f"Average overlap (where exists): {np.mean(list(entity_overlaps.values())):.3f}")

    print("\nThis suggests potential graph links at higher levels:")
    for item in multi_event_entities[:10]:
        print(f"  - '{item['entity']}' could link {item['appears_in']} events")

    return system, cross_entities, entity_overlaps


def main():
    print("=" * 70)
    print("Cross-Frame Entity Linking Analysis")
    print("=" * 70 + "\n")

    snapshot = load_snapshot()
    print(f"Loaded {len(snapshot.claims)} claims\n")

    system, cross_entities, overlaps = run_analysis(snapshot)

    # Save
    output = {
        'cross_entities': cross_entities[:30],
        'entity_overlaps': [
            {
                'event1': e1,
                'event2': e2,
                'jaccard': j
            }
            for (e1, e2), j in sorted(overlaps.items(), key=lambda x: x[1], reverse=True)[:20]
        ],
        'summary': {
            'total_cross_entities': len([e for e in cross_entities if e['appears_in'] >= 2]),
            'total_overlapping_pairs': len(overlaps),
            'llm_calls': system.llm_calls
        }
    }

    output_path = Path("/app/test_eu/results/cross_frame_linking.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
