"""
Semantic Hierarchy

Build a hierarchy of EUs:
1. Claims cluster into sub-events (semantic similarity)
2. Sub-events cluster into events (LLM: "same broader story?")
3. Events could cluster into narratives (optional)

Run inside container:
    docker exec herenews-app python /app/test_eu/semantic_hierarchy.py
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import httpx

from load_graph import load_snapshot, GraphSnapshot


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"


def get_embeddings(texts: List[str]) -> List[List[float]]:
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


def llm_same_story(summary1: str, summary2: str) -> bool:
    """Ask LLM if two sub-events are part of the same broader story"""
    prompt = f"""Are these two descriptions part of the SAME broader news story/event? Answer YES or NO.

Description 1: {summary1[:300]}

Description 2: {summary2[:300]}

Same story?"""

    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
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


def llm_summarize(texts: List[str]) -> str:
    """Get LLM to summarize a cluster of claims"""
    combined = "\n".join([f"- {t[:150]}" for t in texts[:5]])
    prompt = f"""Summarize these claims in one sentence (max 30 words):

{combined}

Summary:"""

    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0
        },
        timeout=30.0
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@dataclass
class EU:
    id: str
    level: int  # 0=claim, 1=sub-event, 2=event
    claim_ids: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)  # Child EU ids
    embedding: Optional[List[float]] = None
    summary: str = ""


def build_hierarchy(claims: Dict[str, str]) -> Dict[str, EU]:
    """Build semantic hierarchy"""

    all_eus: Dict[str, EU] = {}

    # Level 0: Claims as leaf EUs
    claim_ids = list(claims.keys())
    texts = list(claims.values())

    print("Getting embeddings...")
    embeddings = get_embeddings(texts)

    for i, cid in enumerate(claim_ids):
        eu = EU(
            id=f"L0_{cid}",
            level=0,
            claim_ids=[cid],
            texts=[texts[i]],
            embedding=embeddings[i]
        )
        all_eus[eu.id] = eu

    # Level 1: Cluster into sub-events (embedding similarity)
    print("\nLevel 1: Clustering claims into sub-events...")

    l0_eus = [eu for eu in all_eus.values() if eu.level == 0]
    used = set()
    l1_counter = 0

    # Greedy clustering
    for eu in l0_eus:
        if eu.id in used:
            continue

        # Find similar claims
        cluster = [eu]
        used.add(eu.id)

        for other in l0_eus:
            if other.id in used:
                continue
            sim = cosine_similarity(eu.embedding, other.embedding)
            if sim >= 0.75:
                cluster.append(other)
                used.add(other.id)

        # Create sub-event
        l1_counter += 1
        sub_event = EU(
            id=f"L1_{l1_counter}",
            level=1,
            claim_ids=[c for eu in cluster for c in eu.claim_ids],
            texts=[t for eu in cluster for t in eu.texts],
            children=[eu.id for eu in cluster],
            embedding=np.mean([eu.embedding for eu in cluster], axis=0).tolist()
        )
        all_eus[sub_event.id] = sub_event

    l1_eus = [eu for eu in all_eus.values() if eu.level == 1]
    print(f"  Created {len(l1_eus)} sub-events")

    # Get summaries for sub-events with multiple claims
    print("  Getting summaries...")
    for eu in l1_eus:
        if len(eu.claim_ids) > 1:
            eu.summary = llm_summarize(eu.texts)
        else:
            eu.summary = eu.texts[0][:100]

    # Level 2: Cluster sub-events into events (LLM verification)
    print("\nLevel 2: Clustering sub-events into events...")

    used = set()
    l2_counter = 0

    # Only consider multi-claim sub-events for merging
    significant_l1 = [eu for eu in l1_eus if len(eu.claim_ids) >= 2]
    print(f"  Significant sub-events: {len(significant_l1)}")

    for eu in significant_l1:
        if eu.id in used:
            continue

        cluster = [eu]
        used.add(eu.id)

        for other in significant_l1:
            if other.id in used:
                continue

            # Use LLM to check if same story
            if llm_same_story(eu.summary, other.summary):
                cluster.append(other)
                used.add(other.id)

        # Create event
        l2_counter += 1
        event = EU(
            id=f"L2_{l2_counter}",
            level=2,
            claim_ids=[c for sub in cluster for c in sub.claim_ids],
            texts=[t for sub in cluster for t in sub.texts],
            children=[sub.id for sub in cluster]
        )

        # Get event summary
        sub_summaries = [sub.summary for sub in cluster]
        if len(sub_summaries) > 1:
            event.summary = llm_summarize(sub_summaries)
        else:
            event.summary = sub_summaries[0]

        event.embedding = np.mean([sub.embedding for sub in cluster], axis=0).tolist()
        all_eus[event.id] = event

    # Also add single-claim sub-events as events
    for eu in l1_eus:
        if eu.id not in used:
            l2_counter += 1
            event = EU(
                id=f"L2_{l2_counter}",
                level=2,
                claim_ids=eu.claim_ids,
                texts=eu.texts,
                children=[eu.id],
                embedding=eu.embedding,
                summary=eu.summary
            )
            all_eus[event.id] = event

    l2_eus = [eu for eu in all_eus.values() if eu.level == 2]
    print(f"  Created {len(l2_eus)} events")

    return all_eus


def display_hierarchy(all_eus: Dict[str, EU]):
    """Display the hierarchy"""

    print(f"\n{'='*60}")
    print("Semantic Hierarchy")
    print(f"{'='*60}")

    l2_eus = sorted(
        [eu for eu in all_eus.values() if eu.level == 2],
        key=lambda x: len(x.claim_ids),
        reverse=True
    )

    for event in l2_eus[:10]:
        print(f"\n[EVENT] {event.summary}")
        print(f"  Claims: {len(event.claim_ids)}")

        # Show sub-events
        for child_id in event.children:
            child = all_eus.get(child_id)
            if child and child.level == 1 and len(child.claim_ids) > 1:
                print(f"    └─ [SUB] {child.summary} ({len(child.claim_ids)} claims)")


def main():
    print("=" * 60)
    print("Semantic Hierarchy")
    print("=" * 60)

    snapshot = load_snapshot()

    # Get sample: Wang Fuk Court + some other events for contrast
    sample_claims = {}

    # Wang Fuk Court
    for cid, claim in snapshot.claims.items():
        text_lower = claim.text.lower()
        if 'wang fuk' in text_lower or ('tai po' in text_lower and 'fire' in text_lower):
            sample_claims[cid] = claim.text
            if len(sample_claims) >= 40:
                break

    # Add some Jimmy Lai for contrast
    for cid, claim in snapshot.claims.items():
        if cid not in sample_claims and 'jimmy lai' in claim.text.lower():
            sample_claims[cid] = claim.text
            if len(sample_claims) >= 55:
                break

    # Add some Charlie Kirk
    for cid, claim in snapshot.claims.items():
        if cid not in sample_claims and 'charlie kirk' in claim.text.lower():
            sample_claims[cid] = claim.text
            if len(sample_claims) >= 70:
                break

    print(f"\nSample: {len(sample_claims)} claims (mixed topics)\n")

    all_eus = build_hierarchy(sample_claims)

    display_hierarchy(all_eus)

    # Save
    output_path = Path("/app/test_eu/results/semantic_hierarchy.json")
    l2_eus = [eu for eu in all_eus.values() if eu.level == 2]

    output = {
        'events': [
            {
                'summary': eu.summary,
                'claims': len(eu.claim_ids),
                'sub_events': len([c for c in eu.children if all_eus.get(c) and all_eus[c].level == 1])
            }
            for eu in sorted(l2_eus, key=lambda x: len(x.claim_ids), reverse=True)
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
