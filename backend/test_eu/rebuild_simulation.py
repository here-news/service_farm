#!/usr/bin/env python3
"""
Experiment: Rebuild Simulation

Instead of migration, simulate rebuilding events from scratch by streaming
existing claims through the fractal event system.

Approach:
1. Clear events (in simulation - don't actually delete)
2. Stream claims ordered by created_at
3. Let events emerge naturally with proper EU hierarchy
4. Compare emerged events to original events

Run inside container:
    docker exec herenews-app python /app/test_eu/rebuild_simulation.py
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict
import httpx
import psycopg2
from psycopg2.extras import execute_values

from load_graph import load_snapshot, GraphSnapshot


# Config
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

PG_HOST = os.environ.get("POSTGRES_HOST", "herenews-postgres")
PG_DB = os.environ.get("POSTGRES_DB", "herenews")
PG_USER = os.environ.get("POSTGRES_USER", "herenews_user")
PG_PASS = os.environ.get("POSTGRES_PASSWORD", "herenews_pass")

SIM_THRESHOLD = 0.70
LLM_THRESHOLD = 0.55
EVENT_MERGE_THRESHOLD = 0.60


def get_pg_connection():
    return psycopg2.connect(
        host=PG_HOST,
        database=PG_DB,
        user=PG_USER,
        password=PG_PASS
    )


def load_cached_embeddings() -> Dict[str, List[float]]:
    """Load existing embeddings from PostgreSQL"""
    conn = get_pg_connection()
    cur = conn.cursor()

    cur.execute("SELECT claim_id, embedding FROM core.claim_embeddings")
    rows = cur.fetchall()

    cur.close()
    conn.close()

    embeddings = {}
    for claim_id, emb in rows:
        # pgvector returns as string, parse it
        if isinstance(emb, str):
            emb = [float(x) for x in emb.strip('[]').split(',')]
        embeddings[claim_id] = list(emb)

    return embeddings


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0


def llm_same_event(text1: str, text2: str) -> bool:
    """LLM verification for borderline similarity"""
    prompt = f"""Are these two claims about the SAME specific event/incident?

Claim 1: {text1[:200]}
Claim 2: {text2[:200]}

Answer only YES or NO."""

    try:
        response = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0
            },
            timeout=30
        )
        result = response.json()
        return 'YES' in result['choices'][0]['message']['content'].upper()
    except:
        return False


@dataclass
class StreamClaim:
    """Claim ready for streaming"""
    id: str
    text: str
    page_id: str
    embedding: List[float]
    original_event_id: Optional[str] = None


@dataclass
class EmergentEU:
    """EU that emerges from streaming"""
    id: str
    level: int  # 1=sub-event, 2=event
    embedding: List[float]
    claim_ids: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    page_ids: Set[str] = field(default_factory=set)
    children: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None

    def size(self) -> int:
        return len(self.claim_ids)

    def update_embedding(self, new_embedding: List[float]):
        """Running centroid update"""
        n = len(self.claim_ids)
        if n == 1:
            self.embedding = new_embedding
        else:
            self.embedding = [
                (self.embedding[i] * (n - 1) + new_embedding[i]) / n
                for i in range(len(new_embedding))
            ]


class FractalEventSimulator:
    """Simulates the fractal event system"""

    def __init__(self):
        self.sub_events: Dict[str, EmergentEU] = {}
        self.events: Dict[str, EmergentEU] = {}
        self.claim_to_sub: Dict[str, str] = {}
        self.sub_to_event: Dict[str, str] = {}

        self.stats = {
            'claims_processed': 0,
            'sub_events_created': 0,
            'events_created': 0,
            'merges': 0,
            'llm_calls': 0
        }

    def process_claim(self, claim: StreamClaim) -> Dict:
        """Process a single claim through the fractal system"""
        self.stats['claims_processed'] += 1

        best_sub, best_sim = self._find_best_sub_event(claim.embedding)

        if best_sub and best_sim >= SIM_THRESHOLD:
            return self._absorb_into_sub(claim, best_sub)

        elif best_sub and best_sim >= LLM_THRESHOLD:
            self.stats['llm_calls'] += 1
            if llm_same_event(claim.text, best_sub.texts[0]):
                return self._absorb_into_sub(claim, best_sub)

        return self._create_sub_event(claim)

    def _find_best_sub_event(self, embedding: List[float]) -> Tuple[Optional[EmergentEU], float]:
        best_sub = None
        best_sim = 0.0

        for sub in self.sub_events.values():
            sim = cosine_similarity(embedding, sub.embedding)
            if sim > best_sim:
                best_sim = sim
                best_sub = sub

        return best_sub, best_sim

    def _absorb_into_sub(self, claim: StreamClaim, sub: EmergentEU) -> Dict:
        sub.claim_ids.append(claim.id)
        sub.texts.append(claim.text)
        sub.page_ids.add(claim.page_id)
        sub.update_embedding(claim.embedding)
        self.claim_to_sub[claim.id] = sub.id

        return {'action': 'absorbed', 'sub_event_id': sub.id, 'sub_event_size': sub.size()}

    def _create_sub_event(self, claim: StreamClaim) -> Dict:
        sub_id = f"sub_{len(self.sub_events)}"
        sub = EmergentEU(
            id=sub_id,
            level=1,
            embedding=claim.embedding.copy(),
            claim_ids=[claim.id],
            texts=[claim.text],
            page_ids={claim.page_id}
        )
        self.sub_events[sub_id] = sub
        self.claim_to_sub[claim.id] = sub_id
        self.stats['sub_events_created'] += 1

        return {'action': 'created_sub', 'sub_event_id': sub_id}

    def merge_pass(self) -> int:
        """Merge sub-events into events"""
        merges = 0
        unassigned = [s for s in self.sub_events.values() if s.id not in self.sub_to_event]

        for sub in unassigned:
            best_event, best_sim = self._find_best_event(sub.embedding)

            if best_event and best_sim >= EVENT_MERGE_THRESHOLD:
                self._add_sub_to_event(sub, best_event)
                merges += 1
            elif sub.size() >= 3:
                self._create_event_from_sub(sub)
                merges += 1

        self.stats['merges'] += merges
        return merges

    def _find_best_event(self, embedding: List[float]) -> Tuple[Optional[EmergentEU], float]:
        best_event = None
        best_sim = 0.0

        for event in self.events.values():
            sim = cosine_similarity(embedding, event.embedding)
            if sim > best_sim:
                best_sim = sim
                best_event = event

        return best_event, best_sim

    def _add_sub_to_event(self, sub: EmergentEU, event: EmergentEU):
        event.children.append(sub.id)
        event.claim_ids.extend(sub.claim_ids)
        event.page_ids.update(sub.page_ids)
        sub.parent_id = event.id
        self.sub_to_event[sub.id] = event.id

        n = len(event.children)
        event.embedding = [
            (event.embedding[i] * (n - 1) + sub.embedding[i]) / n
            for i in range(len(event.embedding))
        ]

    def _create_event_from_sub(self, sub: EmergentEU):
        event_id = f"event_{len(self.events)}"
        event = EmergentEU(
            id=event_id,
            level=2,
            embedding=sub.embedding.copy(),
            claim_ids=sub.claim_ids.copy(),
            page_ids=sub.page_ids.copy(),
            children=[sub.id]
        )
        self.events[event_id] = event
        sub.parent_id = event_id
        self.sub_to_event[sub.id] = event_id
        self.stats['events_created'] += 1


def compare_with_original(simulator: FractalEventSimulator, claims: List[StreamClaim]) -> Dict:
    """Compare emerged events with original event assignments"""

    original_events = defaultdict(set)
    for claim in claims:
        if claim.original_event_id:
            original_events[claim.original_event_id].add(claim.id)

    emerged_events = defaultdict(set)
    for event in simulator.events.values():
        for claim_id in event.claim_ids:
            emerged_events[event.id].add(claim_id)

    results = {
        'original_event_count': len(original_events),
        'emerged_event_count': len(emerged_events),
        'original_claims_in_events': sum(len(c) for c in original_events.values()),
        'emerged_claims_in_events': sum(len(c) for c in emerged_events.values()),
        'alignments': []
    }

    for orig_id, orig_claims in original_events.items():
        best_overlap = 0
        best_emerged = None

        for emg_id, emg_claims in emerged_events.items():
            overlap = len(orig_claims & emg_claims)
            if overlap > best_overlap:
                best_overlap = overlap
                best_emerged = emg_id

        if best_emerged:
            emg_claims = emerged_events[best_emerged]
            precision = best_overlap / len(emg_claims) if emg_claims else 0
            recall = best_overlap / len(orig_claims) if orig_claims else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

            results['alignments'].append({
                'original': orig_id,
                'emerged': best_emerged,
                'original_size': len(orig_claims),
                'emerged_size': len(emg_claims),
                'overlap': best_overlap,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

    if results['alignments']:
        results['avg_f1'] = sum(a['f1'] for a in results['alignments']) / len(results['alignments'])
        results['avg_precision'] = sum(a['precision'] for a in results['alignments']) / len(results['alignments'])
        results['avg_recall'] = sum(a['recall'] for a in results['alignments']) / len(results['alignments'])

    return results


def main():
    print("=" * 70)
    print("REBUILD SIMULATION: Stream Claims → Emergent Events")
    print("=" * 70)

    # Load graph snapshot
    print("\n📥 Loading graph snapshot...")
    snapshot = load_snapshot()

    # Load cached embeddings
    print("📊 Loading cached embeddings...")
    embeddings = load_cached_embeddings()
    print(f"   Found {len(embeddings)} cached embeddings")

    # Build claims for streaming
    claims = []
    for claim in snapshot.claims.values():
        if claim.id in embeddings and claim.text:
            # Find original event
            original_event = None
            for event_id, event in snapshot.events.items():
                if claim.id in event.claim_ids:
                    original_event = event_id
                    break

            claims.append(StreamClaim(
                id=claim.id,
                text=claim.text,
                page_id=claim.page_id or "",
                embedding=embeddings[claim.id],
                original_event_id=original_event
            ))

    print(f"\n📋 Ready to stream {len(claims)} claims")

    # Initialize simulator
    simulator = FractalEventSimulator()

    # Stream claims
    print("\n🌊 Streaming claims...")
    start_time = time.time()

    batch_size = 100
    for i in range(0, len(claims), batch_size):
        batch = claims[i:i+batch_size]

        for claim in batch:
            simulator.process_claim(claim)

        merges = simulator.merge_pass()

        elapsed = time.time() - start_time
        print(f"   [{i+len(batch):4d}/{len(claims)}] "
              f"sub-events: {len(simulator.sub_events):3d}, "
              f"events: {len(simulator.events):2d}, "
              f"merges: {merges}, "
              f"time: {elapsed:.1f}s")

    # Final merge pass
    simulator.merge_pass()

    total_time = time.time() - start_time

    # Results
    print("\n" + "=" * 70)
    print("REBUILD RESULTS")
    print("=" * 70)

    print(f"\n⏱️  Total time: {total_time:.1f}s ({len(claims)/total_time:.1f} claims/sec)")

    print(f"\n📊 Statistics:")
    print(f"   Claims processed: {simulator.stats['claims_processed']}")
    print(f"   Sub-events created: {simulator.stats['sub_events_created']}")
    print(f"   Events created: {simulator.stats['events_created']}")
    print(f"   Total merges: {simulator.stats['merges']}")
    print(f"   LLM calls: {simulator.stats['llm_calls']}")

    # Hierarchy summary
    print(f"\n🏗️  Hierarchy:")
    orphan_subs = sum(1 for s in simulator.sub_events.values() if s.id not in simulator.sub_to_event)
    print(f"   Level 1 (sub-events): {len(simulator.sub_events)} ({orphan_subs} orphaned)")
    print(f"   Level 2 (events): {len(simulator.events)}")

    if simulator.events:
        avg_subs = sum(len(e.children) for e in simulator.events.values()) / len(simulator.events)
        avg_claims = sum(len(e.claim_ids) for e in simulator.events.values()) / len(simulator.events)
        print(f"   Avg sub-events per event: {avg_subs:.1f}")
        print(f"   Avg claims per event: {avg_claims:.1f}")

    # Compare with original
    print(f"\n🔍 Comparison with Original Events:")
    comparison = compare_with_original(simulator, claims)
    print(f"   Original events: {comparison['original_event_count']}")
    print(f"   Emerged events: {comparison['emerged_event_count']}")

    if 'avg_f1' in comparison:
        print(f"   Alignment F1: {comparison['avg_f1']:.2%}")
        print(f"   Precision: {comparison['avg_precision']:.2%}")
        print(f"   Recall: {comparison['avg_recall']:.2%}")

    # Top emerged events
    print(f"\n🏆 Top Emerged Events:")
    top_events = sorted(simulator.events.values(), key=lambda e: len(e.claim_ids), reverse=True)[:5]
    for event in top_events:
        sample_text = simulator.sub_events[event.children[0]].texts[0][:60] if event.children else "(no text)"
        print(f"   {event.id}: {len(event.claim_ids)} claims, {len(event.children)} sub-events")
        print(f"      Sample: {sample_text}...")

    # Save results
    results = {
        'stats': simulator.stats,
        'hierarchy': {
            'sub_events': len(simulator.sub_events),
            'events': len(simulator.events),
            'orphan_subs': orphan_subs
        },
        'comparison': comparison,
        'time_seconds': total_time,
        'claims_per_second': len(claims) / total_time
    }

    with open('/app/test_eu/rebuild_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Results saved to rebuild_results.json")


if __name__ == "__main__":
    main()
