"""
Frame Emergence - Level 3 Hierarchy

After Claims → Sub-events → Events,
now merge Events into Frames (broader narratives).

Example potential frames:
- "Hong Kong 2025" (Wang Fuk Court Fire + Jimmy Lai Trial)
- "Trump Presidency 2.0" (BBC lawsuit + Brown response + Venezuela)

Run inside container:
    docker exec herenews-app python /app/test_eu/frame_emergence.py
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
import time

from load_graph import load_snapshot, GraphSnapshot


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

PG_HOST = os.environ.get("POSTGRES_HOST", "herenews-postgres")
PG_DB = os.environ.get("POSTGRES_DB", "herenews")
PG_USER = os.environ.get("POSTGRES_USER", "herenews_user")
PG_PASS = os.environ.get("POSTGRES_PASSWORD", "herenews_pass")

# Thresholds at each level (lower = broader groupings)
SIM_THRESHOLD = 0.70      # Claims → Sub-events
LLM_THRESHOLD = 0.55
EVENT_MERGE_THRESHOLD = 0.60   # Sub-events → Events
FRAME_MERGE_THRESHOLD = 0.50   # Events → Frames (even broader)


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


def llm_check(prompt: str) -> bool:
    """Generic LLM yes/no check with retry"""
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
    """Check if two claims are about same specific event"""
    prompt = f"""Are these claims about the same news story/event? Answer YES or NO.

Claim 1: {text1[:250]}
Claim 2: {text2[:250]}

Same story?"""
    return llm_check(prompt)


def llm_same_story(summary1: str, summary2: str) -> bool:
    """Check if two sub-events are part of same broader story"""
    prompt = f"""Are these two descriptions part of the SAME broader news event/story?
They might be different aspects (e.g., casualties vs investigation) of the same incident.
Answer YES or NO.

Description 1: {summary1[:300]}

Description 2: {summary2[:300]}

Same broader story?"""
    return llm_check(prompt)


def llm_same_narrative(desc1: str, desc2: str) -> bool:
    """Check if two events are part of same broader narrative/frame"""
    prompt = f"""Are these two events part of the SAME broader narrative or connected theme?
Consider: same geographic region, same political context, same crisis, same actors over time.
Answer YES or NO.

Event 1: {desc1[:400]}

Event 2: {desc2[:400]}

Same broader narrative?"""
    return llm_check(prompt)


def llm_generate_frame_name(descriptions: List[str]) -> str:
    """Generate a name for a frame based on its events"""
    combined = "\n".join([f"- {d[:200]}" for d in descriptions[:5]])
    prompt = f"""These events belong to the same broader narrative. Generate a SHORT name (3-6 words) that captures the common theme.

Events:
{combined}

Name for this narrative frame (3-6 words):"""

    try:
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 20,
                "temperature": 0.3
            },
            timeout=60.0
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip().strip('"')
    except:
        return "Unnamed Frame"


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@dataclass
class EU:
    id: str
    level: int = 0  # 0 = sub-event, 1 = event, 2 = frame
    claim_ids: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    page_ids: Set[str] = field(default_factory=set)
    embedding: Optional[List[float]] = None
    internal_corr: int = 0
    internal_contra: int = 0
    children: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    generated_name: Optional[str] = None  # LLM-generated name for higher levels

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
        if self.generated_name:
            return self.generated_name
        return self.texts[0][:50] + "..." if self.texts else "empty"

    def description(self) -> str:
        """Return a description for narrative checking"""
        if self.generated_name:
            return f"{self.generated_name}: {self.texts[0][:150]}"
        return self.texts[0][:200] if self.texts else "empty"


class ThreeLevelSystem:
    def __init__(self, snapshot: GraphSnapshot):
        self.snapshot = snapshot
        self.eus: Dict[str, EU] = {}
        self.claim_to_eu: Dict[str, str] = {}
        self.sub_counter = 0
        self.event_counter = 0
        self.frame_counter = 0
        self.llm_calls = 0
        self.total_claims = 0
        self.merges = 0

    def process_claim(self, claim_id: str, text: str, page_id: str, embedding: List[float]) -> str:
        """Phase 1: Process claim into sub-event (level 0)"""
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
        """Phase 2: Merge sub-events (level 0) into events (level 1)"""
        print("\n--- Phase 2: Merging sub-events into events ---")

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
                    event.internal_corr += sub.internal_corr
                    event.internal_contra += sub.internal_contra
                    if sub.embedding:
                        all_embs.append(sub.embedding)
                    sub.parent_id = event.id

                event.embedding = np.mean(all_embs, axis=0).tolist()
                self.eus[event.id] = event
                events_created += 1

                print(f"  Created {event.id}: {len(group)} sub-events, {event.size()} claims")

        print(f"Events created: {events_created}")

    def merge_into_frames(self, min_size: int = 10):
        """Phase 3: Merge events (level 1) into frames (level 2)"""
        print("\n--- Phase 3: Merging events into frames ---")

        # Get significant events that aren't already parented
        candidates = [
            eu for eu in self.eus.values()
            if eu.level == 1 and eu.size() >= min_size and eu.parent_id is None
        ]

        # Also include large unparented sub-events (they're effectively events)
        large_subs = [
            eu for eu in self.eus.values()
            if eu.level == 0 and eu.size() >= min_size and eu.parent_id is None
        ]
        candidates.extend(large_subs)

        print(f"Event-level candidates: {len(candidates)}")

        if len(candidates) < 2:
            return

        candidates.sort(key=lambda x: x.size(), reverse=True)

        used = set()
        frames_created = 0
        llm_frame_calls = 0

        for eu in candidates:
            if eu.id in used:
                continue

            group = [eu]
            used.add(eu.id)

            for other in candidates:
                if other.id in used:
                    continue

                sim = cosine_sim(eu.embedding, other.embedding)

                if sim >= FRAME_MERGE_THRESHOLD:
                    llm_frame_calls += 1
                    self.llm_calls += 1
                    if llm_same_narrative(eu.description(), other.description()):
                        group.append(other)
                        used.add(other.id)

            if len(group) > 1:
                self.frame_counter += 1

                # Generate frame name
                descriptions = [e.description() for e in group]
                frame_name = llm_generate_frame_name(descriptions)

                frame = EU(
                    id=f"frame_{self.frame_counter}",
                    level=2,
                    claim_ids=[],
                    texts=[],
                    page_ids=set(),
                    children=[e.id for e in group],
                    generated_name=frame_name
                )

                all_embs = []
                for event in group:
                    frame.claim_ids.extend(event.claim_ids)
                    frame.texts.extend(event.texts[:3])
                    frame.page_ids |= event.page_ids
                    frame.internal_corr += event.internal_corr
                    frame.internal_contra += event.internal_contra
                    if event.embedding:
                        all_embs.append(event.embedding)
                    event.parent_id = frame.id

                frame.embedding = np.mean(all_embs, axis=0).tolist()
                self.eus[frame.id] = frame
                frames_created += 1

                print(f"\n  Created FRAME: \"{frame_name}\"")
                print(f"    {len(group)} events, {frame.size()} total claims")
                for event in group:
                    lvl = "[EVENT]" if event.level == 1 else "[sub]"
                    print(f"      └─ {lvl} {event.label()[:40]} ({event.size()} claims)")

        print(f"\nFrames created: {frames_created}")
        print(f"LLM calls for frame merging: {llm_frame_calls}")

    def get_hierarchy(self) -> Dict:
        """Return full hierarchy for visualization"""
        frames = [eu for eu in self.eus.values() if eu.level == 2]
        orphan_events = [eu for eu in self.eus.values() if eu.level == 1 and eu.parent_id is None]
        orphan_subs = [eu for eu in self.eus.values() if eu.level == 0 and eu.parent_id is None]

        return {
            'frames': frames,
            'orphan_events': orphan_events,
            'orphan_subs': orphan_subs
        }

    def stats(self) -> Dict:
        level0 = [eu for eu in self.eus.values() if eu.level == 0]
        level1 = [eu for eu in self.eus.values() if eu.level == 1]
        level2 = [eu for eu in self.eus.values() if eu.level == 2]

        return {
            'claims': self.total_claims,
            'sub_events': len(level0),
            'events': len(level1),
            'frames': len(level2),
            'largest': max(eu.size() for eu in self.eus.values()) if self.eus else 0,
            'merges': self.merges,
            'llm_calls': self.llm_calls
        }


def run_simulation(snapshot: GraphSnapshot):
    print("Loading cached embeddings...")
    cached = load_cached_embeddings()
    print(f"  Found {len(cached)} cached embeddings")

    all_claims = list(snapshot.claims.keys())
    random.seed(42)
    random.shuffle(all_claims)

    system = ThreeLevelSystem(snapshot)

    print(f"\n{'='*70}")
    print(f"Phase 1: Streaming {len(all_claims)} claims into sub-events")
    print(f"{'='*70}\n")

    for i, cid in enumerate(all_claims):
        claim = snapshot.claims[cid]
        system.process_claim(cid, claim.text, claim.page_id or "?", cached[cid])

        if (i + 1) % 300 == 0:
            s = system.stats()
            print(f"[{i+1:4d}] SubEvents:{s['sub_events']:3d}")

    print(f"\n{'='*70}")
    print("Phase 2: Merging sub-events into events")
    print(f"{'='*70}")

    system.merge_into_events(min_size=3)

    print(f"\n{'='*70}")
    print("Phase 3: Merging events into frames")
    print(f"{'='*70}")

    system.merge_into_frames(min_size=10)

    # Final report
    print(f"\n{'='*70}")
    print("FINAL THREE-LEVEL HIERARCHY")
    print(f"{'='*70}\n")

    s = system.stats()
    print(f"Total claims: {s['claims']}")
    print(f"Sub-events (level 0): {s['sub_events']}")
    print(f"Events (level 1): {s['events']}")
    print(f"Frames (level 2): {s['frames']}")
    print(f"LLM calls: {s['llm_calls']}")

    hierarchy = system.get_hierarchy()

    # Print frames first
    if hierarchy['frames']:
        print(f"\n{'='*70}")
        print("EMERGED FRAMES")
        print(f"{'='*70}\n")

        for frame in sorted(hierarchy['frames'], key=lambda x: x.size(), reverse=True):
            state = "ACTIVE ⚡" if frame.tension() > 0.1 else "STABLE"
            print(f"[FRAME] \"{frame.label()}\"")
            print(f"        {frame.size()} claims, {len(frame.page_ids)} pages, {state}")
            print(f"        coherence={frame.coherence():.0%}, mass={frame.mass():.1f}")

            for child_id in frame.children:
                child = system.eus.get(child_id)
                if child:
                    lvl = "EVENT" if child.level == 1 else "sub"
                    cstate = "⚡" if child.tension() > 0.1 else "✓"
                    print(f"        └─ [{lvl}] {child.label()[:45]} ({child.size()} claims) {cstate}")
            print()

    # Print orphan events
    if hierarchy['orphan_events']:
        print(f"\n{'='*70}")
        print("UNFRAMED EVENTS (could form frames with more data)")
        print(f"{'='*70}\n")

        for event in sorted(hierarchy['orphan_events'], key=lambda x: x.size(), reverse=True)[:10]:
            state = "ACTIVE ⚡" if event.tension() > 0.1 else "STABLE"
            print(f"[EVENT] {event.label()[:50]}")
            print(f"        {event.size()} claims, {len(event.page_ids)} pages, {state}")

    return system


def main():
    print("=" * 70)
    print("Frame Emergence (Three-Level Hierarchy)")
    print("=" * 70 + "\n")

    snapshot = load_snapshot()
    print(f"Loaded {len(snapshot.claims)} claims\n")

    system = run_simulation(snapshot)

    # Save
    output = {
        'stats': system.stats(),
        'frames': [
            {
                'id': eu.id,
                'name': eu.generated_name,
                'size': eu.size(),
                'pages': len(eu.page_ids),
                'mass': eu.mass(),
                'coherence': eu.coherence(),
                'tension': eu.tension(),
                'children': eu.children,
                'sample': eu.texts[:5]
            }
            for eu in system.eus.values() if eu.level == 2
        ],
        'events': [
            {
                'id': eu.id,
                'level': eu.level,
                'size': eu.size(),
                'parent': eu.parent_id,
                'sample': eu.texts[:3]
            }
            for eu in system.eus.values() if eu.level == 1
        ]
    }

    output_path = Path("/app/test_eu/results/frame_emergence.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
