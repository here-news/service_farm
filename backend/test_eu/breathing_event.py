"""
Breathing Event System - Live Streaming Event Emergence

A living system where:
1. Claims stream in continuously
2. Claims find their position (cluster into EUs)
3. EUs "breathe" - grow, absorb, increase mass
4. LLM curates internally (causal relationships, coherence)
5. System emits decisions (stabilized, contradiction, ready to publish)

Run inside container:
    docker exec herenews-app python /app/test_eu/breathing_event.py

Or for SSE streaming endpoint:
    docker exec herenews-app python /app/test_eu/breathing_event.py --serve
"""

import os
import json
import random
import numpy as np
import asyncio
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Optional, AsyncGenerator
from datetime import datetime
from pathlib import Path
import httpx
import psycopg2
import time
from enum import Enum

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


# ============================================================================
# EVENT TYPES (for streaming)
# ============================================================================

class EventType(str, Enum):
    CLAIM_ABSORBED = "claim_absorbed"      # Claim joined existing EU
    EU_CREATED = "eu_created"              # New EU spawned
    EU_MERGED = "eu_merged"                # Two EUs merged into parent
    CONTRADICTION_FOUND = "contradiction"   # Conflicting claims detected
    CAUSAL_FOUND = "causal_found"          # Causal relationship detected
    EVENT_STABILIZED = "stabilized"         # Event went ACTIVE → STABLE
    EVENT_ACTIVATED = "activated"           # Event went STABLE → ACTIVE
    COHERENCE_IMPROVED = "coherence_improved"  # Internal consolidation
    MASS_THRESHOLD = "mass_threshold"       # EU crossed mass threshold


@dataclass
class StreamEvent:
    """An event emitted by the breathing system"""
    type: EventType
    timestamp: str
    eu_id: str
    data: Dict

    def to_json(self) -> str:
        return json.dumps({
            'type': self.type.value,
            'timestamp': self.timestamp,
            'eu_id': self.eu_id,
            'data': self.data
        })


# ============================================================================
# EU DATA STRUCTURES
# ============================================================================

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
    causal_links: List[Dict] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_activity: str = field(default_factory=lambda: datetime.now().isoformat())

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

    def state(self) -> str:
        return "ACTIVE" if self.tension() > 0.1 else "STABLE"

    def label(self) -> str:
        return self.texts[0][:60] + "..." if self.texts else "empty"

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'level': self.level,
            'size': self.size(),
            'pages': len(self.page_ids),
            'mass': round(self.mass(), 2),
            'coherence': round(self.coherence(), 2),
            'tension': round(self.tension(), 2),
            'state': self.state(),
            'label': self.label(),
            'children': self.children,
            'causal_links': self.causal_links[:5],
            'created_at': self.created_at,
            'last_activity': self.last_activity
        }


# ============================================================================
# LLM HELPERS
# ============================================================================

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


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def llm_call(prompt: str, max_tokens: int = 10) -> str:
    """Generic LLM call with retry"""
    for attempt in range(3):
        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0
                },
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except:
            if attempt < 2:
                time.sleep(2)
            else:
                return ""


def llm_same_event(text1: str, text2: str) -> bool:
    prompt = f"""Are these claims about the same news story/event? Answer YES or NO.

Claim 1: {text1[:250]}
Claim 2: {text2[:250]}

Same story?"""
    return llm_call(prompt, 5).upper().startswith("YES")


def llm_same_story(summary1: str, summary2: str) -> bool:
    prompt = f"""Are these two descriptions part of the SAME broader news event/story?
Answer YES or NO.

Description 1: {summary1[:300]}
Description 2: {summary2[:300]}

Same broader story?"""
    return llm_call(prompt, 5).upper().startswith("YES")


def llm_detect_causal(text: str) -> Optional[Dict]:
    """Detect causal language in a claim"""
    prompt = f"""Does this claim contain causal language (caused, led to, resulted in, because, after, following)?
If YES, return the causal phrase. If NO, say NO.

Claim: {text[:400]}

Format: YES: "phrase" or NO"""

    result = llm_call(prompt, 30)
    if result.upper().startswith("YES"):
        phrase = result.split(":", 1)[1].strip().strip('"') if ":" in result else ""
        return {"has_causal": True, "phrase": phrase}
    return None


# ============================================================================
# BREATHING EVENT SYSTEM
# ============================================================================

class BreathingEventSystem:
    """
    A living event system that processes claims and emits stream events.
    """

    def __init__(self, snapshot: GraphSnapshot):
        self.snapshot = snapshot
        self.eus: Dict[str, EU] = {}
        self.claim_to_eu: Dict[str, str] = {}
        self.sub_counter = 0
        self.event_counter = 0
        self.llm_calls = 0

        # Stream of events
        self.event_queue: List[StreamEvent] = []

        # Thresholds for decisions
        self.mass_thresholds = [1.0, 5.0, 10.0, 20.0]
        self.eu_mass_crossed: Dict[str, Set[float]] = {}

    def emit(self, event_type: EventType, eu_id: str, data: Dict):
        """Emit a stream event"""
        event = StreamEvent(
            type=event_type,
            timestamp=datetime.now().isoformat(),
            eu_id=eu_id,
            data=data
        )
        self.event_queue.append(event)
        return event

    def process_claim(self, claim_id: str, text: str, page_id: str,
                      embedding: List[float], check_causal: bool = True) -> List[StreamEvent]:
        """
        Process a single claim - the core "breathing" action.
        Returns list of events that occurred.
        """
        events = []

        # Find best matching EU (level 0, no parent)
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
            # ABSORB into existing EU
            old_state = best_eu.state()
            old_coherence = best_eu.coherence()
            old_mass = best_eu.mass()

            best_eu.claim_ids.append(claim_id)
            best_eu.texts.append(text)
            best_eu.page_ids.add(page_id)
            best_eu.last_activity = datetime.now().isoformat()

            # Update embedding (running centroid)
            old_emb = np.array(best_eu.embedding)
            new_emb = np.array(embedding)
            n = len(best_eu.claim_ids)
            best_eu.embedding = ((old_emb * (n - 1) + new_emb) / n).tolist()

            # Check for internal relationships
            claim = self.snapshot.claims.get(claim_id)
            if claim:
                for corr_id in claim.corroborates_ids:
                    if corr_id in best_eu.claim_ids:
                        best_eu.internal_corr += 1
                for contra_id in claim.contradicts_ids:
                    if contra_id in best_eu.claim_ids:
                        best_eu.internal_contra += 1
                        # Emit contradiction event
                        events.append(self.emit(
                            EventType.CONTRADICTION_FOUND,
                            best_eu.id,
                            {
                                'claim_id': claim_id,
                                'contradicts': contra_id,
                                'text': text[:100],
                                'tension': best_eu.tension()
                            }
                        ))

            self.claim_to_eu[claim_id] = best_eu.id

            # Emit absorption event
            events.append(self.emit(
                EventType.CLAIM_ABSORBED,
                best_eu.id,
                {
                    'claim_id': claim_id,
                    'text': text[:100],
                    'similarity': round(best_sim, 3),
                    'new_size': best_eu.size(),
                    'new_mass': round(best_eu.mass(), 2)
                }
            ))

            # Check state transitions
            new_state = best_eu.state()
            if old_state == "STABLE" and new_state == "ACTIVE":
                events.append(self.emit(
                    EventType.EVENT_ACTIVATED,
                    best_eu.id,
                    {'reason': 'contradiction_introduced', 'tension': best_eu.tension()}
                ))
            elif old_state == "ACTIVE" and new_state == "STABLE":
                events.append(self.emit(
                    EventType.EVENT_STABILIZED,
                    best_eu.id,
                    {'coherence': best_eu.coherence(), 'size': best_eu.size()}
                ))

            # Check coherence improvement
            if best_eu.coherence() > old_coherence + 0.05:
                events.append(self.emit(
                    EventType.COHERENCE_IMPROVED,
                    best_eu.id,
                    {'old': round(old_coherence, 2), 'new': round(best_eu.coherence(), 2)}
                ))

            # Check mass thresholds
            new_mass = best_eu.mass()
            if best_eu.id not in self.eu_mass_crossed:
                self.eu_mass_crossed[best_eu.id] = set()
            for threshold in self.mass_thresholds:
                if old_mass < threshold <= new_mass and threshold not in self.eu_mass_crossed[best_eu.id]:
                    self.eu_mass_crossed[best_eu.id].add(threshold)
                    events.append(self.emit(
                        EventType.MASS_THRESHOLD,
                        best_eu.id,
                        {'threshold': threshold, 'mass': round(new_mass, 2), 'label': best_eu.label()}
                    ))

            target_eu = best_eu

        else:
            # CREATE new EU
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
            self.eu_mass_crossed[new_eu.id] = set()

            events.append(self.emit(
                EventType.EU_CREATED,
                new_eu.id,
                {
                    'claim_id': claim_id,
                    'text': text[:100],
                    'initial_mass': round(new_eu.mass(), 2)
                }
            ))

            target_eu = new_eu

        # Optional: Check for causal language
        if check_causal and random.random() < 0.2:  # Sample 20%
            self.llm_calls += 1
            causal = llm_detect_causal(text)
            if causal and causal.get('has_causal'):
                target_eu.causal_links.append({
                    'claim_id': claim_id,
                    'phrase': causal.get('phrase', ''),
                    'detected_at': datetime.now().isoformat()
                })
                events.append(self.emit(
                    EventType.CAUSAL_FOUND,
                    target_eu.id,
                    {
                        'claim_id': claim_id,
                        'phrase': causal.get('phrase', ''),
                        'text': text[:100]
                    }
                ))

        return events

    def try_merge_events(self, min_size: int = 5) -> List[StreamEvent]:
        """
        Periodically try to merge sub-events into events.
        Called after batches of claims.
        """
        events = []

        candidates = [
            eu for eu in self.eus.values()
            if eu.level == 0 and eu.size() >= min_size and eu.parent_id is None
        ]

        if len(candidates) < 2:
            return events

        candidates.sort(key=lambda x: x.size(), reverse=True)

        used = set()

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
                # Create parent event
                self.event_counter += 1
                parent = EU(
                    id=f"event_{self.event_counter}",
                    level=1,
                    claim_ids=[],
                    texts=[],
                    page_ids=set(),
                    children=[sub.id for sub in group]
                )

                all_embs = []
                for sub in group:
                    parent.claim_ids.extend(sub.claim_ids)
                    parent.texts.extend(sub.texts[:2])
                    parent.page_ids |= sub.page_ids
                    parent.internal_corr += sub.internal_corr
                    parent.internal_contra += sub.internal_contra
                    if sub.embedding:
                        all_embs.append(sub.embedding)
                    sub.parent_id = parent.id

                parent.embedding = np.mean(all_embs, axis=0).tolist()
                self.eus[parent.id] = parent
                self.eu_mass_crossed[parent.id] = set()

                events.append(self.emit(
                    EventType.EU_MERGED,
                    parent.id,
                    {
                        'children': [s.id for s in group],
                        'child_labels': [s.label()[:40] for s in group],
                        'total_claims': parent.size(),
                        'mass': round(parent.mass(), 2),
                        'coherence': round(parent.coherence(), 2)
                    }
                ))

        return events

    def get_status(self) -> Dict:
        """Get current system status"""
        level0 = [eu for eu in self.eus.values() if eu.level == 0]
        level1 = [eu for eu in self.eus.values() if eu.level == 1]
        active = [eu for eu in self.eus.values() if eu.state() == "ACTIVE"]

        return {
            'total_claims': len(self.claim_to_eu),
            'sub_events': len(level0),
            'events': len(level1),
            'active_eus': len(active),
            'llm_calls': self.llm_calls,
            'top_eus': [eu.to_dict() for eu in sorted(self.eus.values(), key=lambda x: x.mass(), reverse=True)[:10]]
        }


# ============================================================================
# STREAMING SIMULATION
# ============================================================================

def run_breathing_simulation(snapshot: GraphSnapshot, delay: float = 0.1):
    """
    Simulate the breathing event system with live output.
    """
    print("=" * 70)
    print("BREATHING EVENT SYSTEM - Live Simulation")
    print("=" * 70)
    print()

    # Load embeddings
    print("Loading cached embeddings...")
    cached = load_cached_embeddings()
    print(f"  Found {len(cached)} cached embeddings")

    # Prepare claims in random order
    all_claims = list(snapshot.claims.keys())
    random.seed(42)
    random.shuffle(all_claims)

    system = BreathingEventSystem(snapshot)

    print(f"\nStarting stream of {len(all_claims)} claims...")
    print("=" * 70)
    print()

    total_events = 0

    for i, cid in enumerate(all_claims):
        claim = snapshot.claims[cid]

        # Process claim
        events = system.process_claim(
            cid,
            claim.text,
            claim.page_id or "?",
            cached[cid],
            check_causal=(i % 10 == 0)  # Check causal every 10th claim
        )

        # Print events
        for event in events:
            total_events += 1
            icon = {
                EventType.CLAIM_ABSORBED: "🔵",
                EventType.EU_CREATED: "🟢",
                EventType.EU_MERGED: "🟡",
                EventType.CONTRADICTION_FOUND: "🔴",
                EventType.CAUSAL_FOUND: "🟣",
                EventType.EVENT_STABILIZED: "✅",
                EventType.EVENT_ACTIVATED: "⚡",
                EventType.COHERENCE_IMPROVED: "📈",
                EventType.MASS_THRESHOLD: "🎯"
            }.get(event.type, "•")

            if event.type == EventType.CLAIM_ABSORBED:
                print(f"{icon} [{event.eu_id}] +claim (size={event.data['new_size']}, mass={event.data['new_mass']})")
            elif event.type == EventType.EU_CREATED:
                print(f"{icon} NEW [{event.eu_id}]: {event.data['text'][:50]}...")
            elif event.type == EventType.EU_MERGED:
                print(f"{icon} MERGED → [{event.eu_id}]: {len(event.data['children'])} sub-events, {event.data['total_claims']} claims")
                for label in event.data['child_labels']:
                    print(f"      └─ {label}")
            elif event.type == EventType.CONTRADICTION_FOUND:
                print(f"{icon} CONTRADICTION in [{event.eu_id}]: tension={event.data['tension']:.0%}")
            elif event.type == EventType.CAUSAL_FOUND:
                print(f"{icon} CAUSAL in [{event.eu_id}]: \"{event.data['phrase']}\"")
            elif event.type == EventType.EVENT_STABILIZED:
                print(f"{icon} STABILIZED [{event.eu_id}]: coherence={event.data['coherence']:.0%}")
            elif event.type == EventType.EVENT_ACTIVATED:
                print(f"{icon} ACTIVATED [{event.eu_id}]: tension introduced")
            elif event.type == EventType.MASS_THRESHOLD:
                print(f"{icon} MASS THRESHOLD [{event.eu_id}]: crossed {event.data['threshold']} (now {event.data['mass']})")

        # Periodically try merges
        if (i + 1) % 100 == 0:
            merge_events = system.try_merge_events(min_size=3)
            for event in merge_events:
                total_events += 1
                print(f"🟡 MERGED → [{event.eu_id}]: {len(event.data['children'])} sub-events, {event.data['total_claims']} claims")

            # Status update
            status = system.get_status()
            print(f"\n--- [{i+1}/{len(all_claims)}] Status: {status['sub_events']} sub-events, {status['events']} events, {status['active_eus']} active ---\n")

        # Small delay for "breathing" effect
        if delay > 0:
            time.sleep(delay)

    # Final merge pass
    print("\n" + "=" * 70)
    print("Final merge pass...")
    merge_events = system.try_merge_events(min_size=3)
    for event in merge_events:
        print(f"🟡 MERGED → [{event.eu_id}]: {event.data['total_claims']} claims")

    # Final status
    print("\n" + "=" * 70)
    print("FINAL STATUS")
    print("=" * 70)

    status = system.get_status()
    print(f"\nTotal claims processed: {status['total_claims']}")
    print(f"Sub-events: {status['sub_events']}")
    print(f"Events: {status['events']}")
    print(f"Active EUs: {status['active_eus']}")
    print(f"LLM calls: {status['llm_calls']}")
    print(f"Stream events emitted: {total_events}")

    print("\nTop EUs by mass:")
    for eu in status['top_eus'][:10]:
        state = "⚡" if eu['state'] == "ACTIVE" else "✓"
        print(f"  {state} [{eu['id']}] mass={eu['mass']:.1f} size={eu['size']} coh={eu['coherence']:.0%} - {eu['label'][:40]}")

    return system


def main():
    import sys

    snapshot = load_snapshot()
    print(f"Loaded {len(snapshot.claims)} claims\n")

    # Run simulation with minimal delay
    system = run_breathing_simulation(snapshot, delay=0.0)

    # Save final state
    output = {
        'status': system.get_status(),
        'events': [e.to_json() for e in system.event_queue[-100:]]  # Last 100 events
    }

    output_path = Path("/app/test_eu/results/breathing_event.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
