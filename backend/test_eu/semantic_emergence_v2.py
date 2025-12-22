"""
Semantic Emergence v2

Lower thresholds + LLM verification for borderline merges.

Run inside container:
    docker exec herenews-app python /app/test_eu/semantic_emergence_v2.py
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
import httpx

from load_graph import load_snapshot, GraphSnapshot


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"  # Light model for verification


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from OpenAI API"""
    response = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
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
    """Ask LLM if two claims are about the same event"""
    prompt = f"""Are these two claims about the same specific event? Answer only YES or NO.

Claim 1: {text1[:200]}

Claim 2: {text2[:200]}

Same event?"""

    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 10,
            "temperature": 0
        },
        timeout=30.0
    )
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"].strip().upper()
    return answer.startswith("YES")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@dataclass
class SemanticEU:
    id: str
    claim_ids: List[str]
    texts: List[str]
    embedding: Optional[List[float]] = None


def semantic_cluster_with_llm(
    claims: Dict[str, str],
    sim_threshold: float = 0.75,
    llm_threshold: float = 0.65,
    use_llm: bool = True
) -> List[SemanticEU]:
    """
    Cluster with embedding similarity + LLM verification for borderline.

    - similarity >= sim_threshold: merge directly
    - llm_threshold <= similarity < sim_threshold: ask LLM
    - similarity < llm_threshold: don't merge
    """

    claim_ids = list(claims.keys())
    texts = list(claims.values())

    print(f"Getting embeddings for {len(texts)} claims...")
    embeddings = get_embeddings(texts)
    claim_embeddings = dict(zip(claim_ids, embeddings))

    clusters: List[SemanticEU] = []
    llm_calls = 0

    for cid in claim_ids:
        emb = claim_embeddings[cid]
        text = claims[cid]

        best_cluster = None
        best_sim = 0.0

        for cluster in clusters:
            if cluster.embedding:
                sim = cosine_similarity(emb, cluster.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = cluster

        should_merge = False

        if best_sim >= sim_threshold:
            should_merge = True
        elif use_llm and best_sim >= llm_threshold and best_cluster:
            # Ask LLM
            llm_calls += 1
            should_merge = llm_same_event(text, best_cluster.texts[0])
            if llm_calls <= 5:  # Log first few
                print(f"  LLM check (sim={best_sim:.2f}): {should_merge}")

        if should_merge and best_cluster:
            best_cluster.claim_ids.append(cid)
            best_cluster.texts.append(text)
            cluster_embs = [claim_embeddings[c] for c in best_cluster.claim_ids]
            best_cluster.embedding = np.mean(cluster_embs, axis=0).tolist()
        else:
            new_cluster = SemanticEU(
                id=f"sem_{len(clusters)}",
                claim_ids=[cid],
                texts=[text],
                embedding=emb
            )
            clusters.append(new_cluster)

    print(f"LLM calls made: {llm_calls}")
    return clusters


def run_experiment(snapshot: GraphSnapshot, sample_size: int = 50):
    """Run on Wang Fuk Court sample"""

    # Get sample claims
    wfc_claims = {}
    for cid, claim in snapshot.claims.items():
        text_lower = claim.text.lower()
        if 'wang fuk' in text_lower or 'tai po' in text_lower or 'hong kong' in text_lower and 'fire' in text_lower:
            wfc_claims[cid] = claim.text
            if len(wfc_claims) >= sample_size:
                break

    print(f"Sample: {len(wfc_claims)} claims\n")

    # Test different configurations
    configs = [
        {"sim_threshold": 0.80, "llm_threshold": 0.65, "use_llm": False, "name": "Embedding only (0.80)"},
        {"sim_threshold": 0.75, "llm_threshold": 0.60, "use_llm": False, "name": "Embedding only (0.75)"},
        {"sim_threshold": 0.80, "llm_threshold": 0.65, "use_llm": True, "name": "Embedding + LLM verify"},
    ]

    results = {}

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Config: {config['name']}")
        print(f"{'='*60}")

        clusters = semantic_cluster_with_llm(
            wfc_claims,
            sim_threshold=config["sim_threshold"],
            llm_threshold=config["llm_threshold"],
            use_llm=config["use_llm"]
        )

        # Count cluster sizes
        sizes = [len(c.claim_ids) for c in clusters]
        multi_claim = [c for c in clusters if len(c.claim_ids) > 1]

        print(f"\nClusters: {len(clusters)}")
        print(f"Multi-claim clusters: {len(multi_claim)}")
        print(f"Largest: {max(sizes)} claims")

        # Show top clusters
        for c in sorted(clusters, key=lambda x: len(x.claim_ids), reverse=True)[:5]:
            if len(c.claim_ids) > 1:
                print(f"\n  [{len(c.claim_ids)} claims]")
                for t in c.texts[:3]:
                    print(f"    - {t[:65]}...")

        results[config["name"]] = {
            "clusters": len(clusters),
            "multi_claim": len(multi_claim),
            "largest": max(sizes)
        }

    return results


def main():
    print("=" * 60)
    print("Semantic Emergence v2 (Embedding + LLM)")
    print("=" * 60)

    if not OPENAI_API_KEY:
        print("\nError: OPENAI_API_KEY not set")
        return

    snapshot = load_snapshot()
    print(f"Loaded {len(snapshot.claims)} claims")

    results = run_experiment(snapshot, sample_size=50)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"  {name}: {r['clusters']} clusters, largest={r['largest']}")


if __name__ == "__main__":
    main()
