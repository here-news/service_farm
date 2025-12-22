"""
Taxonomy Emergence Analysis

When does an EU need a taxonomy tag?
Hypothesis: Taxonomy crystallizes at higher levels, not from day one.

Run inside container:
    docker exec herenews-app python /app/test_eu/taxonomy_emergence.py
"""

import os
import json
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
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


# Standard taxonomy categories (for evaluation)
TAXONOMY_CATEGORIES = [
    "Politics",
    "Crime & Justice",
    "Disaster & Emergency",
    "Business & Finance",
    "Technology",
    "International Relations",
    "Human Rights",
    "Entertainment",
    "Health",
    "Science",
    "Environment"
]


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


def llm_classify_taxonomy(text: str, categories: List[str]) -> Dict:
    """Classify text into taxonomy categories with confidence"""
    cats_str = ", ".join(categories)
    prompt = f"""Classify this news content into ONE primary category and optionally ONE secondary category.
Categories: {cats_str}

Content: {text[:500]}

Return JSON: {{"primary": "category", "secondary": "category or null", "confidence": 0.0-1.0}}"""

    for attempt in range(3):
        try:
            response = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0
                },
                timeout=60.0
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()
            # Parse JSON
            if content.startswith("```"):
                content = content.split("```")[1].replace("json", "").strip()
            return json.loads(content)
        except:
            if attempt < 2:
                time.sleep(2)
            else:
                return {"primary": "Unknown", "secondary": None, "confidence": 0.0}


def llm_assess_taxonomy_usefulness(text: str, size: int) -> Dict:
    """Assess whether taxonomy is useful at this level"""
    prompt = f"""For this news content cluster ({size} claims), assess taxonomy usefulness.

Content sample: {text[:400]}

Questions:
1. Is the content specific enough that taxonomy adds value? (or is it already self-describing)
2. Would taxonomy help users find this content?
3. Would taxonomy help connect this to related content?

Return JSON: {{"needs_taxonomy": true/false, "reason": "brief explanation", "specificity": "high/medium/low"}}"""

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
                return {"needs_taxonomy": False, "reason": "error", "specificity": "unknown"}


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
    taxonomy: Optional[Dict] = None
    taxonomy_usefulness: Optional[Dict] = None

    def size(self) -> int:
        return len(self.claim_ids)

    def label(self) -> str:
        return self.texts[0][:50] + "..." if self.texts else "empty"


class TaxonomySystem:
    def __init__(self, snapshot: GraphSnapshot):
        self.snapshot = snapshot
        self.eus: Dict[str, EU] = {}
        self.claim_to_eu: Dict[str, str] = {}
        self.sub_counter = 0
        self.event_counter = 0
        self.llm_calls = 0

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

    def analyze_taxonomy_at_levels(self):
        """Analyze taxonomy usefulness at different levels and sizes"""
        print("\n" + "=" * 70)
        print("TAXONOMY ANALYSIS AT DIFFERENT LEVELS")
        print("=" * 70)

        # Sample different sizes
        size_buckets = {
            'singleton': [eu for eu in self.eus.values() if eu.level == 0 and eu.size() == 1],
            'small (2-4)': [eu for eu in self.eus.values() if eu.level == 0 and 2 <= eu.size() <= 4],
            'medium (5-15)': [eu for eu in self.eus.values() if eu.level == 0 and 5 <= eu.size() <= 15],
            'large (16+)': [eu for eu in self.eus.values() if eu.level == 0 and eu.size() >= 16],
            'events (L1)': [eu for eu in self.eus.values() if eu.level == 1]
        }

        results = {}

        for bucket_name, eus in size_buckets.items():
            if not eus:
                continue

            print(f"\n--- {bucket_name.upper()} ({len(eus)} EUs) ---")

            # Sample up to 5 from each bucket
            sample = random.sample(eus, min(5, len(eus)))

            bucket_results = []

            for eu in sample:
                sample_text = " ".join(eu.texts[:3])

                # Get taxonomy classification
                taxonomy = llm_classify_taxonomy(sample_text, TAXONOMY_CATEGORIES)
                self.llm_calls += 1

                # Assess usefulness
                usefulness = llm_assess_taxonomy_usefulness(sample_text, eu.size())
                self.llm_calls += 1

                eu.taxonomy = taxonomy
                eu.taxonomy_usefulness = usefulness

                result = {
                    'id': eu.id,
                    'size': eu.size(),
                    'level': eu.level,
                    'taxonomy': taxonomy,
                    'usefulness': usefulness
                }
                bucket_results.append(result)

                needs = "✓ NEEDS" if usefulness.get('needs_taxonomy') else "✗ NO NEED"
                print(f"\n  {eu.id} ({eu.size()} claims)")
                print(f"    {eu.label()[:50]}")
                print(f"    Taxonomy: {taxonomy.get('primary')} (conf: {taxonomy.get('confidence', 0):.0%})")
                print(f"    {needs}: {usefulness.get('reason', '')[:60]}")
                print(f"    Specificity: {usefulness.get('specificity', 'unknown')}")

            results[bucket_name] = bucket_results

        return results

    def compute_taxonomy_emergence_pattern(self, results: Dict) -> Dict:
        """Analyze when taxonomy becomes useful"""
        pattern = {}

        for bucket_name, bucket_results in results.items():
            if not bucket_results:
                continue

            needs_taxonomy = sum(1 for r in bucket_results if r['usefulness'].get('needs_taxonomy'))
            total = len(bucket_results)

            avg_confidence = np.mean([r['taxonomy'].get('confidence', 0) for r in bucket_results])

            high_specificity = sum(1 for r in bucket_results if r['usefulness'].get('specificity') == 'high')

            pattern[bucket_name] = {
                'needs_taxonomy_rate': needs_taxonomy / total if total > 0 else 0,
                'avg_confidence': avg_confidence,
                'high_specificity_rate': high_specificity / total if total > 0 else 0,
                'sample_size': total
            }

        return pattern


def run_analysis(snapshot: GraphSnapshot):
    print("Loading cached embeddings...")
    cached = load_cached_embeddings()
    print(f"  Found {len(cached)} cached embeddings")

    all_claims = list(snapshot.claims.keys())
    random.seed(42)
    random.shuffle(all_claims)

    system = TaxonomySystem(snapshot)

    # Phase 1: Build hierarchy
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

    # Phase 2: Taxonomy analysis
    results = system.analyze_taxonomy_at_levels()

    # Phase 3: Pattern analysis
    print(f"\n{'='*70}")
    print("TAXONOMY EMERGENCE PATTERN")
    print(f"{'='*70}\n")

    pattern = system.compute_taxonomy_emergence_pattern(results)

    print(f"{'Bucket':<20} {'Needs Tax.':<12} {'Avg Conf.':<12} {'High Spec.':<12}")
    print("-" * 56)

    for bucket, stats in pattern.items():
        print(f"{bucket:<20} {stats['needs_taxonomy_rate']:.0%}          "
              f"{stats['avg_confidence']:.0%}          {stats['high_specificity_rate']:.0%}")

    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}\n")

    # Find the transition point
    for bucket, stats in pattern.items():
        if stats['needs_taxonomy_rate'] >= 0.5:
            print(f"Taxonomy becomes useful at: {bucket}")
            break

    print("\nHypothesis validation:")
    print("  - Singletons: Should NOT need taxonomy (content is self-describing)")
    print("  - Small clusters: Probably NOT (still specific)")
    print("  - Medium clusters: MAYBE (pattern starting to emerge)")
    print("  - Large clusters / Events: YES (need category for navigation)")

    return system, results, pattern


def main():
    print("=" * 70)
    print("Taxonomy Emergence Analysis")
    print("=" * 70 + "\n")

    snapshot = load_snapshot()
    print(f"Loaded {len(snapshot.claims)} claims\n")

    system, results, pattern = run_analysis(snapshot)

    # Save
    output = {
        'pattern': pattern,
        'samples': {
            bucket: [
                {
                    'id': r['id'],
                    'size': r['size'],
                    'taxonomy': r['taxonomy'],
                    'needs_taxonomy': r['usefulness'].get('needs_taxonomy'),
                    'specificity': r['usefulness'].get('specificity')
                }
                for r in bucket_results
            ]
            for bucket, bucket_results in results.items()
        },
        'llm_calls': system.llm_calls
    }

    output_path = Path("/app/test_eu/results/taxonomy_emergence.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
