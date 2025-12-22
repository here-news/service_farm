"""
Causal/Consequence Relationship Detection Between Events

Can we detect:
- Causal links: "fire caused investigation"
- Consequence links: "trial led to international pressure"
- Temporal sequence: "announcement → reaction → policy"

Run inside container:
    docker exec herenews-app python /app/test_eu/causal_emergence.py
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


def llm_detect_causal_relationship(event1_desc: str, event2_desc: str) -> Dict:
    """Detect if there's a causal/consequence relationship between events"""
    prompt = f"""Analyze these two news events for causal or consequence relationships.

Event A: {event1_desc[:400]}

Event B: {event2_desc[:400]}

Questions:
1. Did Event A CAUSE or LEAD TO Event B? (causal)
2. Did Event B CAUSE or LEAD TO Event A? (reverse causal)
3. Is Event B a REACTION or RESPONSE to Event A? (consequence)
4. Is Event A a REACTION or RESPONSE to Event B? (reverse consequence)
5. Are they TEMPORALLY SEQUENCED parts of the same story? (sequence)
6. Are they PARALLEL/CONCURRENT developments? (parallel)

Return JSON:
{{
  "relationship": "causal|consequence|sequence|parallel|none",
  "direction": "A_to_B|B_to_A|bidirectional|none",
  "confidence": 0.0-1.0,
  "explanation": "brief reason"
}}"""

    for attempt in range(3):
        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0
                },
                timeout=60.0
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("```")[1].replace("json", "").strip()
            return json.loads(content)
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                return {"relationship": "error", "direction": "none", "confidence": 0, "explanation": str(e)}


def llm_detect_causal_language(text: str) -> Dict:
    """Detect causal language patterns in a single claim"""
    prompt = f"""Analyze this news claim for causal language patterns.

Claim: {text[:500]}

Look for:
- Causal words: "caused", "led to", "resulted in", "because of", "due to"
- Consequence words: "as a result", "following", "in response to", "after"
- Attribution: "blamed", "credited", "responsible for"

Return JSON:
{{
  "has_causal_language": true/false,
  "causal_type": "cause|effect|attribution|none",
  "causal_phrase": "the specific phrase" or null,
  "references_other_event": true/false
}}"""

    for attempt in range(3):
        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 150,
                    "temperature": 0
                },
                timeout=60.0
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("```")[1].replace("json", "").strip()
            return json.loads(content)
        except:
            if attempt < 2:
                time.sleep(2)
            else:
                return {"has_causal_language": False, "causal_type": "none"}


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
    children: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    causal_claims: List[Dict] = field(default_factory=list)  # Claims with causal language

    def size(self) -> int:
        return len(self.claim_ids)

    def label(self) -> str:
        return self.texts[0][:50] + "..." if self.texts else "empty"

    def description(self) -> str:
        return " ".join(self.texts[:3])[:500]


class CausalSystem:
    def __init__(self, snapshot: GraphSnapshot):
        self.snapshot = snapshot
        self.eus: Dict[str, EU] = {}
        self.claim_to_eu: Dict[str, str] = {}
        self.sub_counter = 0
        self.event_counter = 0
        self.llm_calls = 0

        # Causal relationship tracking
        self.event_relationships: List[Dict] = []

    def process_claim(self, claim_id: str, text: str, page_id: str, embedding: List[float]) -> str:
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
        candidates = [
            eu for eu in self.eus.values()
            if eu.level == 0 and eu.size() >= min_size and eu.parent_id is None
        ]

        if len(candidates) < 2:
            return

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

    def detect_causal_language_in_claims(self, sample_size: int = 50):
        """Scan claims for causal language patterns"""
        print("\n--- Detecting causal language in claims ---")

        # Get sample of claims from larger EUs
        large_eus = [eu for eu in self.eus.values() if eu.size() >= 5]
        all_claims = []
        for eu in large_eus:
            for i, cid in enumerate(eu.claim_ids[:10]):
                all_claims.append((cid, eu.texts[i] if i < len(eu.texts) else "", eu.id))

        sample = random.sample(all_claims, min(sample_size, len(all_claims)))

        causal_claims = []
        for cid, text, eu_id in sample:
            result = llm_detect_causal_language(text)
            self.llm_calls += 1

            if result.get('has_causal_language'):
                causal_claims.append({
                    'claim_id': cid,
                    'eu_id': eu_id,
                    'text': text[:100],
                    'causal_type': result.get('causal_type'),
                    'causal_phrase': result.get('causal_phrase'),
                    'references_other': result.get('references_other_event')
                })

                # Store in EU
                eu = self.eus.get(eu_id)
                if eu:
                    eu.causal_claims.append(result)

        print(f"Found {len(causal_claims)} claims with causal language out of {len(sample)}")
        return causal_claims

    def detect_inter_event_relationships(self):
        """Detect causal/consequence relationships between events"""
        print("\n--- Detecting relationships between events ---")

        # Get significant events
        events = [
            eu for eu in self.eus.values()
            if (eu.level == 1) or (eu.level == 0 and eu.size() >= 15 and eu.parent_id is None)
        ]

        print(f"Analyzing {len(events)} events for inter-relationships...")

        relationships = []

        # Check pairs that might be related (some entity overlap or moderate similarity)
        for i, e1 in enumerate(events):
            for e2 in events[i+1:]:
                # Skip if too similar (same event) or too different
                sim = cosine_sim(e1.embedding, e2.embedding)
                if sim > 0.7 or sim < 0.3:
                    continue

                # Check for relationship
                self.llm_calls += 1
                result = llm_detect_causal_relationship(e1.description(), e2.description())

                if result.get('relationship') not in ['none', 'error'] and result.get('confidence', 0) > 0.5:
                    relationships.append({
                        'event1_id': e1.id,
                        'event1_label': e1.label(),
                        'event2_id': e2.id,
                        'event2_label': e2.label(),
                        'relationship': result.get('relationship'),
                        'direction': result.get('direction'),
                        'confidence': result.get('confidence'),
                        'explanation': result.get('explanation')
                    })

                    print(f"\n  Found: {result.get('relationship')} ({result.get('direction')})")
                    print(f"    {e1.label()[:40]}")
                    print(f"    → {e2.label()[:40]}")
                    print(f"    {result.get('explanation', '')[:60]}")

        self.event_relationships = relationships
        return relationships


def run_analysis(snapshot: GraphSnapshot):
    print("Loading cached embeddings...")
    cached = load_cached_embeddings()
    print(f"  Found {len(cached)} cached embeddings")

    all_claims = list(snapshot.claims.keys())
    random.seed(42)
    random.shuffle(all_claims)

    system = CausalSystem(snapshot)

    # Build hierarchy
    print(f"\n{'='*70}")
    print("Building hierarchy...")
    print(f"{'='*70}\n")

    for cid in all_claims:
        claim = snapshot.claims[cid]
        system.process_claim(cid, claim.text, claim.page_id or "?", cached[cid])

    system.merge_into_events(min_size=3)

    level0 = [eu for eu in system.eus.values() if eu.level == 0]
    level1 = [eu for eu in system.eus.values() if eu.level == 1]
    print(f"Sub-events: {len(level0)}, Events: {len(level1)}")

    # Detect causal language
    print(f"\n{'='*70}")
    print("Phase 1: Causal language detection in claims")
    print(f"{'='*70}")

    causal_claims = system.detect_causal_language_in_claims(sample_size=100)

    # Analyze causal language patterns
    print(f"\n{'='*70}")
    print("CAUSAL LANGUAGE ANALYSIS")
    print(f"{'='*70}\n")

    if causal_claims:
        by_type = {}
        for c in causal_claims:
            t = c.get('causal_type', 'unknown')
            by_type[t] = by_type.get(t, 0) + 1

        print("Causal language types found:")
        for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"  {t}: {count}")

        print("\nSample claims with causal language:")
        for c in causal_claims[:10]:
            print(f"\n  [{c['causal_type']}] {c['text'][:80]}...")
            if c.get('causal_phrase'):
                print(f"    Phrase: \"{c['causal_phrase']}\"")
            if c.get('references_other'):
                print(f"    References other event: YES")

    # Detect inter-event relationships
    print(f"\n{'='*70}")
    print("Phase 2: Inter-event relationship detection")
    print(f"{'='*70}")

    relationships = system.detect_inter_event_relationships()

    # Summary
    print(f"\n{'='*70}")
    print("RELATIONSHIP SUMMARY")
    print(f"{'='*70}\n")

    if relationships:
        by_rel = {}
        for r in relationships:
            t = r.get('relationship')
            by_rel[t] = by_rel.get(t, 0) + 1

        print("Relationship types found:")
        for t, count in sorted(by_rel.items(), key=lambda x: -x[1]):
            print(f"  {t}: {count}")

        print("\nAll detected relationships:")
        for r in relationships:
            arrow = "→" if r['direction'] == 'A_to_B' else "←" if r['direction'] == 'B_to_A' else "↔"
            print(f"\n  [{r['relationship']}] (conf: {r['confidence']:.0%})")
            print(f"    {r['event1_label'][:40]}")
            print(f"    {arrow} {r['event2_label'][:40]}")
            print(f"    {r['explanation'][:70]}")
    else:
        print("No causal/consequence relationships detected between events.")
        print("This may indicate events in our dataset are largely independent.")

    return system, causal_claims, relationships


def main():
    print("=" * 70)
    print("Causal/Consequence Relationship Detection")
    print("=" * 70 + "\n")

    snapshot = load_snapshot()
    print(f"Loaded {len(snapshot.claims)} claims\n")

    system, causal_claims, relationships = run_analysis(snapshot)

    # Save
    output = {
        'causal_claims': causal_claims[:50],
        'relationships': relationships,
        'summary': {
            'claims_with_causal_language': len(causal_claims),
            'inter_event_relationships': len(relationships),
            'llm_calls': system.llm_calls
        }
    }

    output_path = Path("/app/test_eu/results/causal_emergence.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
