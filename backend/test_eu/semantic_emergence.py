"""
Semantic Emergence (Small Scale Test)

Uses embeddings + optional LLM to let claims organically merge.

1. Get embeddings for claims (on demand)
2. Find semantically similar claims
3. Merge into proto-events
4. Compute event embedding
5. Check if events should merge

Run inside container:
    docker exec herenews-app python /app/test_eu/semantic_emergence.py
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path

# For embeddings
import httpx

from load_graph import load_snapshot, GraphSnapshot


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from OpenAI API"""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")

    response = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": EMBEDDING_MODEL,
            "input": texts
        },
        timeout=60.0
    )
    response.raise_for_status()
    data = response.json()

    # Sort by index to maintain order
    embeddings = [None] * len(texts)
    for item in data["data"]:
        embeddings[item["index"]] = item["embedding"]

    return embeddings


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@dataclass
class SemanticEU:
    """EU with embedding"""
    id: str
    claim_ids: List[str]
    texts: List[str]
    embedding: Optional[List[float]] = None

    def combined_text(self) -> str:
        """Combined text for embedding"""
        return " ".join(self.texts[:5])  # Limit for embedding


def semantic_cluster(
    claims: Dict[str, str],  # claim_id -> text
    similarity_threshold: float = 0.85
) -> List[SemanticEU]:
    """
    Cluster claims by semantic similarity.

    Simple greedy approach:
    1. Get all embeddings
    2. For each claim, find most similar existing cluster
    3. If similarity > threshold, join cluster
    4. Otherwise, start new cluster
    """

    claim_ids = list(claims.keys())
    texts = list(claims.values())

    print(f"Getting embeddings for {len(texts)} claims...")
    embeddings = get_embeddings(texts)

    # Build claim -> embedding map
    claim_embeddings = dict(zip(claim_ids, embeddings))

    # Greedy clustering
    clusters: List[SemanticEU] = []

    for cid in claim_ids:
        emb = claim_embeddings[cid]
        text = claims[cid]

        # Find best matching cluster
        best_cluster = None
        best_sim = 0.0

        for cluster in clusters:
            if cluster.embedding:
                sim = cosine_similarity(emb, cluster.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = cluster

        if best_sim >= similarity_threshold and best_cluster:
            # Join existing cluster
            best_cluster.claim_ids.append(cid)
            best_cluster.texts.append(text)
            # Update cluster embedding (average)
            cluster_embs = [claim_embeddings[c] for c in best_cluster.claim_ids]
            best_cluster.embedding = np.mean(cluster_embs, axis=0).tolist()
        else:
            # Start new cluster
            new_cluster = SemanticEU(
                id=f"sem_{len(clusters)}",
                claim_ids=[cid],
                texts=[text],
                embedding=emb
            )
            clusters.append(new_cluster)

    return clusters


def merge_similar_clusters(
    clusters: List[SemanticEU],
    similarity_threshold: float = 0.80
) -> List[SemanticEU]:
    """
    Merge clusters that are semantically similar.
    This is the "event merging" step.
    """

    if len(clusters) <= 1:
        return clusters

    merged = []
    used = set()

    # Sort by size (largest first)
    clusters = sorted(clusters, key=lambda x: len(x.claim_ids), reverse=True)

    for i, cluster in enumerate(clusters):
        if i in used:
            continue

        # Find similar clusters to merge
        to_merge = [cluster]
        used.add(i)

        for j, other in enumerate(clusters):
            if j in used:
                continue

            sim = cosine_similarity(cluster.embedding, other.embedding)
            if sim >= similarity_threshold:
                to_merge.append(other)
                used.add(j)

        # Merge
        if len(to_merge) == 1:
            merged.append(cluster)
        else:
            combined = SemanticEU(
                id=f"merged_{len(merged)}",
                claim_ids=[],
                texts=[]
            )
            all_embs = []
            for c in to_merge:
                combined.claim_ids.extend(c.claim_ids)
                combined.texts.extend(c.texts)
                if c.embedding:
                    all_embs.append(c.embedding)

            combined.embedding = np.mean(all_embs, axis=0).tolist()
            merged.append(combined)

    return merged


def run_small_experiment(snapshot: GraphSnapshot, sample_size: int = 50):
    """Run semantic emergence on a small sample"""

    # Get Wang Fuk Court claims as our test set
    wfc_claims = {}
    for cid, claim in snapshot.claims.items():
        text_lower = claim.text.lower()
        if 'wang fuk' in text_lower or 'tai po' in text_lower:
            wfc_claims[cid] = claim.text
            if len(wfc_claims) >= sample_size:
                break

    # If not enough, add more fire-related
    if len(wfc_claims) < sample_size:
        for cid, claim in snapshot.claims.items():
            if cid not in wfc_claims and 'fire' in claim.text.lower():
                wfc_claims[cid] = claim.text
                if len(wfc_claims) >= sample_size:
                    break

    print(f"Sample: {len(wfc_claims)} claims")

    # Initial clustering
    print("\n--- Initial Clustering (threshold=0.85) ---")
    clusters = semantic_cluster(wfc_claims, similarity_threshold=0.85)
    print(f"Initial clusters: {len(clusters)}")

    for c in sorted(clusters, key=lambda x: len(x.claim_ids), reverse=True)[:10]:
        print(f"\n  Cluster {c.id}: {len(c.claim_ids)} claims")
        print(f"    Sample: {c.texts[0][:80]}...")

    # Merge similar clusters
    print("\n--- Merging Similar Clusters (threshold=0.80) ---")
    merged = merge_similar_clusters(clusters, similarity_threshold=0.80)
    print(f"After merge: {len(merged)} clusters")

    for c in sorted(merged, key=lambda x: len(x.claim_ids), reverse=True)[:10]:
        print(f"\n  Cluster {c.id}: {len(c.claim_ids)} claims")
        for t in c.texts[:3]:
            print(f"    - {t[:70]}...")

    return merged


def main():
    print("=" * 60)
    print("Semantic Emergence (Small Scale)")
    print("=" * 60)

    if not OPENAI_API_KEY:
        print("\nError: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY=your_key")
        return

    snapshot = load_snapshot()
    print(f"\nLoaded {len(snapshot.claims)} claims")

    clusters = run_small_experiment(snapshot, sample_size=50)

    # Save results
    output_path = Path("/app/test_eu/results/semantic_emergence.json")
    output = {
        'clusters': [
            {
                'id': c.id,
                'claim_count': len(c.claim_ids),
                'sample_texts': c.texts[:5]
            }
            for c in sorted(clusters, key=lambda x: len(x.claim_ids), reverse=True)
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
