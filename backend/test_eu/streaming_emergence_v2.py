"""
Streaming Emergence v2

More adaptive thresholds + better anomaly handling.

Key changes:
- Lower initial threshold (0.70)
- More aggressive LLM verification range
- Track clustering rate over time
- Don't stop on anomaly, just report

Run inside container:
    docker exec herenews-app python /app/test_eu/streaming_emergence_v2.py
"""

import os
import json
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from pathlib import Path
import httpx

from load_graph import load_snapshot, GraphSnapshot


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# More permissive thresholds
SIM_THRESHOLD = 0.70  # Lower direct merge threshold
LLM_THRESHOLD = 0.55  # Lower LLM check threshold


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    response = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=60.0
    )
    response.raise_for_status()
    data = response.json()
    embeddings = [None] * len(texts)
    for item in data["data"]:
        embeddings[item["index"]] = item["embedding"]
    return embeddings


def llm_same_event(text1: str, text2: str) -> bool:
    prompt = f"""Are these claims about the same news story/event? Consider if they describe the same incident, person's situation, or ongoing story. Answer YES or NO.

Claim 1: {text1[:250]}
Claim 2: {text2[:250]}

Same story?"""

    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5,
            "temperature": 0
        },
        timeout=20.0
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip().upper().startswith("YES")


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@dataclass
class EU:
    id: str
    claim_ids: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    page_ids: Set[str] = field(default_factory=set)
    embedding: Optional[List[float]] = None
    internal_corr: int = 0
    internal_contra: int = 0

    def size(self) -> int:
        return len(self.claim_ids)

    def coherence(self) -> float:
        total = self.internal_corr + self.internal_contra
        return self.internal_corr / total if total > 0 else 1.0

    def mass(self) -> float:
        return self.size() * 0.1 * (0.5 + self.coherence()) * (1 + 0.1 * len(self.page_ids))

    def label(self) -> str:
        return self.texts[0][:50] + "..." if self.texts else "empty"


class StreamingSystem:
    def __init__(self, snapshot: GraphSnapshot):
        self.snapshot = snapshot
        self.eus: Dict[str, EU] = {}
        self.claim_to_eu: Dict[str, str] = {}
        self.eu_counter = 0
        self.llm_calls = 0
        self.llm_yes = 0
        self.total_claims = 0
        self.merge_history = []  # Track merge rate over time

    def process_claim(self, claim_id: str, text: str, page_id: str, embedding: List[float]) -> tuple:
        """Returns (eu_id, action) where action is 'new', 'merge', or 'llm_merge'"""
        self.total_claims += 1

        best_eu = None
        best_sim = 0.0

        for eu in self.eus.values():
            if eu.embedding:
                sim = cosine_sim(embedding, eu.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_eu = eu

        action = 'new'
        should_merge = False

        if best_sim >= SIM_THRESHOLD:
            should_merge = True
            action = 'merge'
        elif best_sim >= LLM_THRESHOLD and best_eu:
            self.llm_calls += 1
            if llm_same_event(text, best_eu.texts[0]):
                should_merge = True
                action = 'llm_merge'
                self.llm_yes += 1

        if should_merge and best_eu:
            best_eu.claim_ids.append(claim_id)
            best_eu.texts.append(text)
            best_eu.page_ids.add(page_id)

            # Update embedding
            old_emb = np.array(best_eu.embedding)
            new_emb = np.array(embedding)
            n = len(best_eu.claim_ids)
            best_eu.embedding = ((old_emb * (n - 1) + new_emb) / n).tolist()

            # Update links
            claim = self.snapshot.claims.get(claim_id)
            if claim:
                for corr_id in claim.corroborates_ids:
                    if corr_id in best_eu.claim_ids:
                        best_eu.internal_corr += 1
                for contra_id in claim.contradicts_ids:
                    if contra_id in best_eu.claim_ids:
                        best_eu.internal_contra += 1

            self.claim_to_eu[claim_id] = best_eu.id
            self.merge_history.append(1)
            return best_eu.id, action
        else:
            self.eu_counter += 1
            new_eu = EU(
                id=f"eu_{self.eu_counter}",
                claim_ids=[claim_id],
                texts=[text],
                page_ids={page_id},
                embedding=embedding
            )
            self.eus[new_eu.id] = new_eu
            self.claim_to_eu[claim_id] = new_eu.id
            self.merge_history.append(0)
            return new_eu.id, 'new'

    def merge_rate(self, window: int = 20) -> float:
        """Recent merge rate"""
        if len(self.merge_history) < window:
            return sum(self.merge_history) / len(self.merge_history) if self.merge_history else 0
        return sum(self.merge_history[-window:]) / window

    def stats(self) -> Dict:
        sizes = [eu.size() for eu in self.eus.values()]
        multi = [eu for eu in self.eus.values() if eu.size() > 1]
        return {
            'claims': self.total_claims,
            'eus': len(self.eus),
            'multi': len(multi),
            'largest': max(sizes) if sizes else 0,
            'avg_size': sum(sizes) / len(sizes) if sizes else 0,
            'llm_calls': self.llm_calls,
            'llm_yes_rate': self.llm_yes / self.llm_calls if self.llm_calls else 0,
            'merge_rate': self.merge_rate()
        }

    def top_eus(self, n: int = 5) -> List[EU]:
        return sorted(self.eus.values(), key=lambda x: x.size(), reverse=True)[:n]


def run_simulation(snapshot: GraphSnapshot, num_claims: int = 150):
    """Run with detailed progress"""

    all_claims = list(snapshot.claims.keys())
    random.seed(42)  # Reproducible
    random.shuffle(all_claims)
    sample = all_claims[:num_claims]

    print(f"Getting embeddings for {num_claims} claims...")
    texts = [snapshot.claims[c].text for c in sample]
    embeddings = get_embeddings_batch(texts)
    claim_emb = dict(zip(sample, embeddings))

    system = StreamingSystem(snapshot)

    print(f"\n{'='*70}")
    print(f"Streaming {num_claims} claims (thresholds: sim={SIM_THRESHOLD}, llm={LLM_THRESHOLD})")
    print(f"{'='*70}\n")

    for i, cid in enumerate(sample):
        claim = snapshot.claims[cid]
        page_id = claim.page_id or "?"

        eu_id, action = system.process_claim(cid, claim.text, page_id, claim_emb[cid])

        # Progress every 10
        if (i + 1) % 10 == 0:
            s = system.stats()
            merge_indicator = "📈" if s['merge_rate'] > 0.3 else "📉" if s['merge_rate'] < 0.1 else "➡️"

            print(f"[{i+1:3d}] EUs:{s['eus']:3d} Multi:{s['multi']:2d} Max:{s['largest']:2d} "
                  f"MergeRate:{s['merge_rate']:.0%} {merge_indicator} LLM:{s['llm_calls']}({s['llm_yes_rate']:.0%}yes)")

            # Show growing EUs
            for eu in system.top_eus(3):
                if eu.size() >= 2:
                    print(f"      [{eu.size():2d}] {eu.label()[:40]} (m={eu.mass():.1f})")
            print()

    # Final
    print(f"{'='*70}")
    print("FINAL STATE")
    print(f"{'='*70}\n")

    s = system.stats()
    print(f"Claims: {s['claims']}")
    print(f"EUs: {s['eus']} ({s['multi']} multi-claim)")
    print(f"Largest: {s['largest']}")
    print(f"Avg size: {s['avg_size']:.1f}")
    print(f"LLM calls: {s['llm_calls']} ({s['llm_yes_rate']:.0%} positive)")

    print(f"\n{'='*70}")
    print("TOP EUs BY SIZE")
    print(f"{'='*70}\n")

    for eu in system.top_eus(12):
        if eu.size() >= 2:
            print(f"[{eu.size()} claims, {len(eu.page_ids)} pages] mass={eu.mass():.2f}")
            for t in eu.texts[:3]:
                print(f"  - {t[:65]}...")
            print()

    return system


def main():
    print("=" * 70)
    print("Streaming Emergence v2")
    print("=" * 70 + "\n")

    snapshot = load_snapshot()
    print(f"Loaded {len(snapshot.claims)} claims\n")

    system = run_simulation(snapshot, num_claims=150)

    # Save
    output = {
        'stats': system.stats(),
        'top_eus': [
            {'size': eu.size(), 'pages': len(eu.page_ids), 'mass': eu.mass(), 'texts': eu.texts[:5]}
            for eu in system.top_eus(20)
        ]
    }

    output_path = Path("/app/test_eu/results/streaming_v2.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
