"""
Full Streaming Emergence v2 - With Hierarchical Merging

After claims cluster into sub-events,
merge related sub-events into parent events.

Run inside container:
    docker exec herenews-app python /app/test_eu/streaming_full_v2.py
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
from psycopg2.extras import execute_values
import time

from load_graph import load_snapshot, GraphSnapshot


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

PG_HOST = os.environ.get("POSTGRES_HOST", "herenews-postgres")
PG_DB = os.environ.get("POSTGRES_DB", "herenews")
PG_USER = os.environ.get("POSTGRES_USER", "herenews_user")
PG_PASS = os.environ.get("POSTGRES_PASSWORD", "herenews_pass")

SIM_THRESHOLD = 0.70
LLM_THRESHOLD = 0.55
EVENT_MERGE_THRESHOLD = 0.60  # Lower threshold for merging sub-events


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


def llm_same_story(summary1: str, summary2: str) -> bool:
    """Ask LLM if two sub-events are part of the same broader story"""
    prompt = f"""Are these two descriptions part of the SAME broader news event/story?
They might be different aspects (e.g., casualties vs investigation) of the same incident.
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


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@dataclass
class EU:
    id: str
    level: int = 0  # 0 = sub-event, 1 = event
    claim_ids: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    page_ids: Set[str] = field(default_factory=set)
    embedding: Optional[List[float]] = None
    internal_corr: int = 0
    internal_contra: int = 0
    children: List[str] = field(default_factory=list)  # Child EU ids for level 1+
    parent_id: Optional[str] = None

    def size(self) -> int:
        return len(self.claim_ids)

    def coherence(self) -> float:
        total = self.internal_corr + self.internal_contra
        return self.internal_corr / total if total > 0 else 1.0

    def tension(self) -> float:
        total = self.internal_corr + self.internal_contra
        return self.internal_contra / total if total > 0 else 0.0

    def mass(self) -> float:
        return self.size() * 0.1 * (0.5 + self.coherence()) * (1 + 0.1 * len(self.page_ids))

    def label(self) -> str:
        return self.texts[0][:50] + "..." if self.texts else "empty"


class HierarchicalSystem:
    def __init__(self, snapshot: GraphSnapshot):
        self.snapshot = snapshot
        self.eus: Dict[str, EU] = {}
        self.claim_to_eu: Dict[str, str] = {}
        self.eu_counter = 0
        self.event_counter = 0
        self.llm_calls = 0
        self.total_claims = 0
        self.merges = 0

    def process_claim(self, claim_id: str, text: str, page_id: str, embedding: List[float]) -> str:
        """Process claim into sub-event (level 0)"""
        self.total_claims += 1

        # Only consider level 0 EUs for claim clustering
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

            claim = self.snapshot.claims.get(claim_id)
            if claim:
                for corr_id in claim.corroborates_ids:
                    if corr_id in best_eu.claim_ids:
                        best_eu.internal_corr += 1
                for contra_id in claim.contradicts_ids:
                    if contra_id in best_eu.claim_ids:
                        best_eu.internal_contra += 1

            self.claim_to_eu[claim_id] = best_eu.id
            return best_eu.id
        else:
            self.eu_counter += 1
            new_eu = EU(
                id=f"sub_{self.eu_counter}",
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
        """
        Second pass: merge sub-events (level 0) into events (level 1).
        Only consider sub-events with >= min_size claims.
        """
        print("\n--- Merging sub-events into events ---")

        # Get significant sub-events that aren't already parented
        candidates = [
            eu for eu in self.eus.values()
            if eu.level == 0 and eu.size() >= min_size and eu.parent_id is None
        ]

        print(f"Candidates for event merging: {len(candidates)}")

        if len(candidates) < 2:
            return

        # Sort by size
        candidates.sort(key=lambda x: x.size(), reverse=True)

        used = set()
        events_created = 0
        llm_merge_calls = 0

        for eu in candidates:
            if eu.id in used:
                continue

            # Find similar sub-events to merge
            group = [eu]
            used.add(eu.id)

            for other in candidates:
                if other.id in used:
                    continue

                sim = cosine_sim(eu.embedding, other.embedding)

                if sim >= EVENT_MERGE_THRESHOLD:
                    # Use LLM to confirm they're part of same broader story
                    llm_merge_calls += 1
                    if llm_same_story(eu.texts[0], other.texts[0]):
                        group.append(other)
                        used.add(other.id)

            if len(group) > 1:
                # Create parent event
                self.event_counter += 1
                event = EU(
                    id=f"event_{self.event_counter}",
                    level=1,
                    claim_ids=[],
                    texts=[],
                    page_ids=set(),
                    children=[sub.id for sub in group]
                )

                # Aggregate from children
                all_embs = []
                for sub in group:
                    event.claim_ids.extend(sub.claim_ids)
                    event.texts.extend(sub.texts[:2])  # Sample texts
                    event.page_ids |= sub.page_ids
                    event.internal_corr += sub.internal_corr
                    event.internal_contra += sub.internal_contra
                    if sub.embedding:
                        all_embs.append(sub.embedding)
                    sub.parent_id = event.id

                event.embedding = np.mean(all_embs, axis=0).tolist()
                self.eus[event.id] = event
                events_created += 1

                print(f"  Created {event.id}: {len(group)} sub-events, {event.size()} claims")
                for sub in group:
                    print(f"    └─ {sub.id}: {sub.label()[:40]}")

        print(f"\nEvents created: {events_created}")
        print(f"LLM calls for event merging: {llm_merge_calls}")

    def get_top_level_eus(self, n: int = 20) -> List[EU]:
        """Get top-level EUs (events if they exist, otherwise large sub-events)"""
        # Prefer level 1 events
        events = [eu for eu in self.eus.values() if eu.level == 1]

        # Add unparented sub-events
        orphan_subs = [eu for eu in self.eus.values() if eu.level == 0 and eu.parent_id is None]

        all_top = events + orphan_subs
        return sorted(all_top, key=lambda x: x.size(), reverse=True)[:n]

    def stats(self) -> Dict:
        level0 = [eu for eu in self.eus.values() if eu.level == 0]
        level1 = [eu for eu in self.eus.values() if eu.level == 1]
        multi = [eu for eu in level0 if eu.size() > 1]
        large = [eu for eu in level0 if eu.size() >= 5]

        return {
            'claims': self.total_claims,
            'sub_events': len(level0),
            'events': len(level1),
            'multi_claim': len(multi),
            'large': len(large),
            'largest': max(eu.size() for eu in self.eus.values()) if self.eus else 0,
            'merges': self.merges,
            'llm_calls': self.llm_calls
        }


def run_simulation(snapshot: GraphSnapshot):
    # Load embeddings
    print("Loading cached embeddings...")
    cached = load_cached_embeddings()
    print(f"  Found {len(cached)} cached embeddings")

    all_claims = list(snapshot.claims.keys())
    random.seed(42)
    random.shuffle(all_claims)

    system = HierarchicalSystem(snapshot)

    print(f"\n{'='*70}")
    print(f"Phase 1: Streaming {len(all_claims)} claims into sub-events")
    print(f"{'='*70}\n")

    for i, cid in enumerate(all_claims):
        claim = snapshot.claims[cid]
        system.process_claim(cid, claim.text, claim.page_id or "?", cached[cid])

        if (i + 1) % 200 == 0:
            s = system.stats()
            print(f"[{i+1:4d}] SubEvents:{s['sub_events']:3d} Multi:{s['multi_claim']:2d} Large:{s['large']:2d}")

    print(f"\n{'='*70}")
    print("Phase 2: Merging sub-events into events")
    print(f"{'='*70}")

    system.merge_into_events(min_size=3)

    # Final report
    print(f"\n{'='*70}")
    print("FINAL HIERARCHY")
    print(f"{'='*70}\n")

    s = system.stats()
    print(f"Total claims: {s['claims']}")
    print(f"Sub-events (level 0): {s['sub_events']}")
    print(f"Events (level 1): {s['events']}")
    print(f"Largest: {s['largest']} claims")

    print(f"\n{'='*70}")
    print("TOP EVENTS & SUB-EVENTS")
    print(f"{'='*70}\n")

    for eu in system.get_top_level_eus(15):
        state = "ACTIVE ⚡" if eu.tension() > 0.1 else "STABLE"
        level_str = "[EVENT]" if eu.level == 1 else "[sub]"

        print(f"{level_str} [{eu.size():2d} claims, {len(eu.page_ids):2d} pages] mass={eu.mass():.1f} coh={eu.coherence():.0%} {state}")
        print(f"    {eu.label()}")

        if eu.level == 1:
            # Show children
            for child_id in eu.children[:5]:
                child = system.eus.get(child_id)
                if child:
                    print(f"      └─ [{child.size()}] {child.label()[:35]}")
        print()

    return system


def main():
    print("=" * 70)
    print("Full Streaming Emergence v2 (Hierarchical)")
    print("=" * 70 + "\n")

    snapshot = load_snapshot()
    print(f"Loaded {len(snapshot.claims)} claims\n")

    system = run_simulation(snapshot)

    # Save
    output = {
        'stats': system.stats(),
        'hierarchy': [
            {
                'id': eu.id,
                'level': eu.level,
                'size': eu.size(),
                'pages': len(eu.page_ids),
                'mass': eu.mass(),
                'coherence': eu.coherence(),
                'tension': eu.tension(),
                'children': eu.children if eu.level == 1 else [],
                'sample': eu.texts[:3]
            }
            for eu in system.get_top_level_eus(30)
        ]
    }

    output_path = Path("/app/test_eu/results/streaming_hierarchical.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
