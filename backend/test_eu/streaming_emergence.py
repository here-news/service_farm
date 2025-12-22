"""
Streaming Emergence Simulation

Simulate claims arriving one-by-one (random order, some from same pages).
Watch EU structure grow in real-time.

Stop anytime if anomaly detected.

Run inside container:
    docker exec herenews-app python /app/test_eu/streaming_emergence.py
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

# Thresholds
SIM_THRESHOLD = 0.75
LLM_THRESHOLD = 0.65


def get_embedding(text: str) -> List[float]:
    """Get embedding for single text"""
    response = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBEDDING_MODEL, "input": [text]},
        timeout=30.0
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for batch"""
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
    """Ask LLM if two claims are same event"""
    prompt = f"""Are these two claims about the same specific news event? Answer YES or NO only.

Claim 1: {text1[:200]}
Claim 2: {text2[:200]}

Same event?"""

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


def cosine_sim(a: List[float], b: List[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@dataclass
class StreamingEU:
    """EU that grows over time"""
    id: str
    claim_ids: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    page_ids: Set[str] = field(default_factory=set)
    embedding: Optional[List[float]] = None

    # Metrics
    internal_corr: int = 0
    internal_contra: int = 0

    def size(self) -> int:
        return len(self.claim_ids)

    def coherence(self) -> float:
        total = self.internal_corr + self.internal_contra
        return self.internal_corr / total if total > 0 else 1.0

    def source_diversity(self) -> int:
        return len(self.page_ids)

    def mass(self) -> float:
        return self.size() * 0.1 * (0.5 + self.coherence()) * (1 + 0.1 * self.source_diversity())

    def label(self) -> str:
        if self.texts:
            return self.texts[0][:50] + "..."
        return f"EU_{self.id}"


class StreamingEUSystem:
    """System that processes claims as they arrive"""

    def __init__(self, snapshot: GraphSnapshot):
        self.snapshot = snapshot
        self.eus: Dict[str, StreamingEU] = {}
        self.claim_to_eu: Dict[str, str] = {}
        self.eu_counter = 0
        self.llm_calls = 0
        self.total_claims = 0

    def process_claim(self, claim_id: str, text: str, page_id: str, embedding: List[float]) -> str:
        """Process a new claim. Returns EU id it joined/created."""
        self.total_claims += 1

        # Find best matching EU
        best_eu = None
        best_sim = 0.0

        for eu in self.eus.values():
            if eu.embedding:
                sim = cosine_sim(embedding, eu.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_eu = eu

        should_merge = False

        if best_sim >= SIM_THRESHOLD:
            should_merge = True
        elif best_sim >= LLM_THRESHOLD and best_eu:
            # LLM verify
            self.llm_calls += 1
            should_merge = llm_same_event(text, best_eu.texts[0])

        if should_merge and best_eu:
            # Join existing EU
            best_eu.claim_ids.append(claim_id)
            best_eu.texts.append(text)
            best_eu.page_ids.add(page_id)

            # Update embedding (running average)
            old_emb = np.array(best_eu.embedding)
            new_emb = np.array(embedding)
            n = len(best_eu.claim_ids)
            best_eu.embedding = ((old_emb * (n - 1) + new_emb) / n).tolist()

            # Update corr/contra
            claim = self.snapshot.claims.get(claim_id)
            if claim:
                for corr_id in claim.corroborates_ids:
                    if corr_id in [c for c in best_eu.claim_ids if c != claim_id]:
                        best_eu.internal_corr += 1
                for contra_id in claim.contradicts_ids:
                    if contra_id in [c for c in best_eu.claim_ids if c != claim_id]:
                        best_eu.internal_contra += 1

            self.claim_to_eu[claim_id] = best_eu.id
            return best_eu.id
        else:
            # Create new EU
            self.eu_counter += 1
            new_eu = StreamingEU(
                id=f"eu_{self.eu_counter}",
                claim_ids=[claim_id],
                texts=[text],
                page_ids={page_id},
                embedding=embedding
            )
            self.eus[new_eu.id] = new_eu
            self.claim_to_eu[claim_id] = new_eu.id
            return new_eu.id

    def stats(self) -> Dict:
        """Get current system stats"""
        sizes = [eu.size() for eu in self.eus.values()]
        multi_claim = [eu for eu in self.eus.values() if eu.size() > 1]

        return {
            'total_claims': self.total_claims,
            'total_eus': len(self.eus),
            'multi_claim_eus': len(multi_claim),
            'largest_eu': max(sizes) if sizes else 0,
            'avg_eu_size': sum(sizes) / len(sizes) if sizes else 0,
            'llm_calls': self.llm_calls
        }

    def top_eus(self, n: int = 5) -> List[StreamingEU]:
        """Get top N EUs by size"""
        return sorted(self.eus.values(), key=lambda x: x.size(), reverse=True)[:n]

    def anomaly_check(self) -> Optional[str]:
        """Check for anomalies. Returns description if found."""
        stats = self.stats()

        # Check for giant EU (> 50% of claims)
        if stats['largest_eu'] > stats['total_claims'] * 0.5 and stats['total_claims'] > 20:
            return f"ANOMALY: Giant EU with {stats['largest_eu']}/{stats['total_claims']} claims"

        # Check for too many singletons (> 80% after 50 claims)
        if stats['total_claims'] > 50:
            singleton_rate = (stats['total_eus'] - stats['multi_claim_eus']) / stats['total_eus']
            if singleton_rate > 0.85:
                return f"ANOMALY: Too many singletons ({singleton_rate:.0%})"

        return None


def run_streaming_simulation(snapshot: GraphSnapshot, num_claims: int = 100, batch_size: int = 10):
    """Run streaming simulation with progress output"""

    # Get random sample of claims
    all_claim_ids = list(snapshot.claims.keys())
    random.shuffle(all_claim_ids)
    sample_ids = all_claim_ids[:num_claims]

    # Get all embeddings upfront (simulate having them cached)
    print(f"Pre-computing embeddings for {num_claims} claims...")
    texts = [snapshot.claims[cid].text for cid in sample_ids]
    embeddings = get_embeddings_batch(texts)

    claim_embeddings = dict(zip(sample_ids, embeddings))

    # Create system
    system = StreamingEUSystem(snapshot)

    print(f"\n{'='*70}")
    print("Streaming Emergence Simulation")
    print(f"{'='*70}")
    print(f"Processing {num_claims} claims in random order...\n")

    # Process claims one by one
    for i, claim_id in enumerate(sample_ids):
        claim = snapshot.claims[claim_id]
        page_id = claim.page_id or "unknown"

        eu_id = system.process_claim(
            claim_id,
            claim.text,
            page_id,
            claim_embeddings[claim_id]
        )

        # Progress update every batch_size claims
        if (i + 1) % batch_size == 0:
            stats = system.stats()
            print(f"[{i+1:3d}] EUs: {stats['total_eus']:3d} | Multi: {stats['multi_claim_eus']:2d} | "
                  f"Largest: {stats['largest_eu']:2d} | LLM: {stats['llm_calls']}")

            # Show top EUs
            top = system.top_eus(3)
            for eu in top:
                if eu.size() > 1:
                    print(f"      └─ [{eu.size():2d}] {eu.label()[:45]}... (coh={eu.coherence():.0%}, mass={eu.mass():.2f})")

            # Anomaly check
            anomaly = system.anomaly_check()
            if anomaly:
                print(f"\n⚠️  {anomaly}")
                print("Stopping simulation.")
                break

            print()

    # Final report
    print(f"\n{'='*70}")
    print("Final State")
    print(f"{'='*70}")

    stats = system.stats()
    print(f"\nClaims processed: {stats['total_claims']}")
    print(f"EUs created: {stats['total_eus']}")
    print(f"Multi-claim EUs: {stats['multi_claim_eus']}")
    print(f"Largest EU: {stats['largest_eu']} claims")
    print(f"Avg EU size: {stats['avg_eu_size']:.1f}")
    print(f"LLM calls: {stats['llm_calls']}")

    print(f"\n{'='*70}")
    print("Top EUs")
    print(f"{'='*70}")

    for eu in system.top_eus(10):
        if eu.size() >= 2:
            print(f"\n[{eu.size()} claims, {eu.source_diversity()} pages] mass={eu.mass():.2f}, coh={eu.coherence():.0%}")
            print(f"  {eu.label()}")
            if eu.size() <= 5:
                for t in eu.texts[1:3]:
                    print(f"    - {t[:60]}...")

    return system


def main():
    print("=" * 70)
    print("Streaming EU Emergence")
    print("=" * 70)

    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set")
        return

    snapshot = load_snapshot()
    print(f"Loaded {len(snapshot.claims)} claims from {len(snapshot.pages)} pages\n")

    # Run simulation
    system = run_streaming_simulation(snapshot, num_claims=100, batch_size=10)

    # Save results
    output_path = Path("/app/test_eu/results/streaming_emergence.json")

    output = {
        'stats': system.stats(),
        'top_eus': [
            {
                'size': eu.size(),
                'pages': eu.source_diversity(),
                'mass': eu.mass(),
                'coherence': eu.coherence(),
                'sample': eu.texts[:3]
            }
            for eu in system.top_eus(15)
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
