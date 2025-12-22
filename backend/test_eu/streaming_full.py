"""
Full Streaming Emergence (All 1215 Claims)

Saves embeddings to PostgreSQL for reuse.
Streams all claims and watches EU structure grow.

Run inside container:
    docker exec herenews-app python /app/test_eu/streaming_full.py
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


def save_embeddings(embeddings: Dict[str, List[float]]):
    """Save embeddings to PostgreSQL"""
    if not embeddings:
        return

    conn = get_pg_connection()
    cur = conn.cursor()

    # Prepare data - convert list to pgvector format
    data = [(cid, emb) for cid, emb in embeddings.items()]

    execute_values(
        cur,
        """
        INSERT INTO core.claim_embeddings (claim_id, embedding)
        VALUES %s
        ON CONFLICT (claim_id) DO UPDATE SET embedding = EXCLUDED.embedding
        """,
        data,
        template="(%s, %s::vector)"
    )

    conn.commit()
    cur.close()
    conn.close()


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings from OpenAI"""
    response = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=120.0
    )
    response.raise_for_status()
    data = response.json()
    embeddings = [None] * len(texts)
    for item in data["data"]:
        embeddings[item["index"]] = item["embedding"]
    return embeddings


def llm_same_event(text1: str, text2: str) -> bool:
    """LLM verification for borderline similarity"""
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
        except Exception as e:
            if attempt < 2:
                import time
                time.sleep(2)
            else:
                print(f"    LLM timeout, skipping merge")
                return False


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

    def tension(self) -> float:
        total = self.internal_corr + self.internal_contra
        return self.internal_contra / total if total > 0 else 0.0

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
        self.merges = 0

    def process_claim(self, claim_id: str, text: str, page_id: str, embedding: List[float]) -> Tuple[str, str]:
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
            self.merges += 1
            best_eu.claim_ids.append(claim_id)
            best_eu.texts.append(text)
            best_eu.page_ids.add(page_id)

            # Update embedding (running centroid)
            old_emb = np.array(best_eu.embedding)
            new_emb = np.array(embedding)
            n = len(best_eu.claim_ids)
            best_eu.embedding = ((old_emb * (n - 1) + new_emb) / n).tolist()

            # Update internal links
            claim = self.snapshot.claims.get(claim_id)
            if claim:
                for corr_id in claim.corroborates_ids:
                    if corr_id in best_eu.claim_ids:
                        best_eu.internal_corr += 1
                for contra_id in claim.contradicts_ids:
                    if contra_id in best_eu.claim_ids:
                        best_eu.internal_contra += 1

            self.claim_to_eu[claim_id] = best_eu.id
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
            return new_eu.id, 'new'

    def stats(self) -> Dict:
        sizes = [eu.size() for eu in self.eus.values()]
        multi = [eu for eu in self.eus.values() if eu.size() > 1]
        large = [eu for eu in self.eus.values() if eu.size() >= 5]

        return {
            'claims': self.total_claims,
            'eus': len(self.eus),
            'multi': len(multi),
            'large': len(large),
            'largest': max(sizes) if sizes else 0,
            'merges': self.merges,
            'merge_rate': self.merges / self.total_claims if self.total_claims else 0,
            'llm_calls': self.llm_calls,
            'llm_yes_rate': self.llm_yes / self.llm_calls if self.llm_calls else 0
        }

    def top_eus(self, n: int = 10) -> List[EU]:
        return sorted(self.eus.values(), key=lambda x: x.size(), reverse=True)[:n]


def run_full_simulation(snapshot: GraphSnapshot):
    """Run on all claims"""

    all_claim_ids = list(snapshot.claims.keys())
    n_claims = len(all_claim_ids)

    # Load cached embeddings
    print("Loading cached embeddings from PostgreSQL...")
    cached = load_cached_embeddings()
    print(f"  Found {len(cached)} cached embeddings")

    # Find claims needing embeddings
    need_embedding = [cid for cid in all_claim_ids if cid not in cached]
    print(f"  Need to compute {len(need_embedding)} new embeddings")

    # Compute missing embeddings in batches
    if need_embedding:
        print("\nComputing missing embeddings...")
        batch_size = 100
        new_embeddings = {}

        for i in range(0, len(need_embedding), batch_size):
            batch_ids = need_embedding[i:i+batch_size]
            batch_texts = [snapshot.claims[cid].text for cid in batch_ids]

            print(f"  Batch {i//batch_size + 1}/{(len(need_embedding)-1)//batch_size + 1}...")
            embeddings = get_embeddings_batch(batch_texts)

            for cid, emb in zip(batch_ids, embeddings):
                new_embeddings[cid] = emb
                cached[cid] = emb

        # Save to PostgreSQL
        print(f"  Saving {len(new_embeddings)} new embeddings to PostgreSQL...")
        save_embeddings(new_embeddings)

    # Shuffle for random arrival order
    random.seed(42)
    random.shuffle(all_claim_ids)

    # Create system
    system = StreamingSystem(snapshot)

    print(f"\n{'='*70}")
    print(f"Streaming {n_claims} claims")
    print(f"{'='*70}\n")

    # Process all claims
    report_interval = 100

    for i, cid in enumerate(all_claim_ids):
        claim = snapshot.claims[cid]
        page_id = claim.page_id or "?"

        system.process_claim(cid, claim.text, page_id, cached[cid])

        # Progress report
        if (i + 1) % report_interval == 0 or i == n_claims - 1:
            s = system.stats()
            print(f"[{i+1:4d}/{n_claims}] EUs:{s['eus']:3d} Multi:{s['multi']:2d} Large(5+):{s['large']:2d} "
                  f"Max:{s['largest']:2d} MergeRate:{s['merge_rate']:.0%} LLM:{s['llm_calls']}")

            # Top 3 EUs
            for eu in system.top_eus(3):
                if eu.size() >= 3:
                    state = "⚡" if eu.tension() > 0.2 else "✓"
                    print(f"      {state} [{eu.size():2d}] {eu.label()[:40]} (m={eu.mass():.1f}, coh={eu.coherence():.0%})")
            print()

    # Final report
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}\n")

    s = system.stats()
    print(f"Total claims: {s['claims']}")
    print(f"Total EUs: {s['eus']}")
    print(f"Multi-claim EUs (2+): {s['multi']}")
    print(f"Large EUs (5+): {s['large']}")
    print(f"Largest EU: {s['largest']} claims")
    print(f"Overall merge rate: {s['merge_rate']:.0%}")
    print(f"LLM calls: {s['llm_calls']} ({s['llm_yes_rate']:.0%} positive)")

    print(f"\n{'='*70}")
    print("TOP 20 EUs")
    print(f"{'='*70}\n")

    for eu in system.top_eus(20):
        if eu.size() >= 2:
            state = "ACTIVE" if eu.tension() > 0.1 else "STABLE"
            print(f"[{eu.size():2d} claims, {len(eu.page_ids):2d} pages] mass={eu.mass():.2f} coh={eu.coherence():.0%} [{state}]")
            print(f"    {eu.texts[0][:70]}...")
            if eu.internal_contra > 0:
                print(f"    ⚡ {eu.internal_contra} contradictions (tension={eu.tension():.0%})")
            print()

    return system


def main():
    print("=" * 70)
    print("Full Streaming Emergence (All Claims)")
    print("=" * 70 + "\n")

    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set")
        return

    snapshot = load_snapshot()
    print(f"Loaded {len(snapshot.claims)} claims from {len(snapshot.pages)} pages\n")

    system = run_full_simulation(snapshot)

    # Save results
    output = {
        'stats': system.stats(),
        'top_eus': [
            {
                'size': eu.size(),
                'pages': len(eu.page_ids),
                'mass': eu.mass(),
                'coherence': eu.coherence(),
                'tension': eu.tension(),
                'corr': eu.internal_corr,
                'contra': eu.internal_contra,
                'sample_texts': eu.texts[:5]
            }
            for eu in system.top_eus(30)
        ]
    }

    output_path = Path("/app/test_eu/results/streaming_full.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
